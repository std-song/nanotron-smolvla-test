from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator

import torch
import yaml


def add_repo_paths(nanotron_root: Path, lerobot_src: Path) -> None:
    for path in (nanotron_root / "src", lerobot_src):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_torchrun_env() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def infinite_dummy_batches(config, batch_size: int, device: torch.device) -> Iterator[dict[str, torch.Tensor]]:
    vocab_size = int(config["dummy_data"].get("language_vocab_size", 49_152))
    state_dim = int(config["model"]["max_state_dim"])
    action_dim = int(config["model"]["max_action_dim"])
    chunk_size = int(config["model"]["chunk_size"])
    language_length = int(config["model"].get("tokenizer_max_length", 48))
    image_shape = tuple(config["model"].get("image_shape", [3, 512, 512]))

    while True:
        yield {
            "image": torch.rand((batch_size, *image_shape), device=device, dtype=torch.float32),
            "image_mask": torch.ones((batch_size,), device=device, dtype=torch.bool),
            "language_tokens": torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, language_length),
                device=device,
                dtype=torch.long,
            ),
            "language_attention_mask": torch.ones((batch_size, language_length), device=device, dtype=torch.bool),
            "state": torch.randn((batch_size, state_dim), device=device, dtype=torch.float32),
            "action": torch.randn((batch_size, chunk_size, action_dim), device=device, dtype=torch.float32),
            "action_is_pad": torch.zeros((batch_size, chunk_size), device=device, dtype=torch.bool),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, default=Path("../lerobot/src"))
    args = parser.parse_args()

    nanotron_root = Path(__file__).resolve().parents[2]
    add_repo_paths(nanotron_root, (nanotron_root / args.lerobot_src).resolve())

    from nanotron import distributed as dist
    from nanotron.models import build_model
    from nanotron.parallel import ParallelContext
    from nanotron.parallel.parameters import sanity_check
    from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
    from nanotron.parallel.pipeline_parallel.context_manager import attach_pipeline_state_to_model
    from nanotron.parallel.pipeline_parallel.state import PipelineTrainBatchState
    from nanotron.parallel.data_parallel.utils import sync_gradients_across_dp

    from modeling import SmolVLANanotronModel, SmolVLANanotronModelConfig, convert_parameters_to_nanotron

    ensure_torchrun_env()
    if not torch.cuda.is_available():
        raise RuntimeError("This minimal SmolVLA/Nanotron training example requires CUDA.")

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(int(config.get("seed", 42)))
    torch.cuda.manual_seed_all(int(config.get("seed", 42)))

    parallelism = config["parallelism"]
    if int(parallelism.get("tp", 1)) != 1 or int(parallelism.get("pp", 1)) != 1:
        raise ValueError("The minimal SmolVLA example currently supports tp=1 and pp=1 only.")

    parallel_context = ParallelContext(
        tensor_parallel_size=int(parallelism.get("tp", 1)),
        pipeline_parallel_size=int(parallelism.get("pp", 1)),
        data_parallel_size=int(parallelism.get("dp", int(os.environ["WORLD_SIZE"]))),
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

    model_config = SmolVLANanotronModelConfig(**config["model"])
    model = build_model(
        model_builder=lambda: SmolVLANanotronModel(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=getattr(torch, config.get("dtype", "bfloat16")),
        target_pp_ranks=[0],
        device=device,
    )
    convert_parameters_to_nanotron(model)
    sanity_check(model)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check SmolVLA freeze/train flags.")

    optimizer_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-10)),
        fused=bool(optimizer_cfg.get("fused", True)),
    )

    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    micro_batch_size = int(config["tokens"]["micro_batch_size"])
    grad_accum = int(config["tokens"].get("batch_accumulation_per_replica", 1))
    train_steps = int(config["tokens"]["train_steps"])
    clip_grad = optimizer_cfg.get("clip_grad", None)
    data_iter = infinite_dummy_batches(config, micro_batch_size, device)

    if dist.get_rank(parallel_context.world_pg) == 0:
        local_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(f"SmolVLA/Nanotron minimal training: params={local_params:,}, trainable={trainable:,}")

    for step in range(1, train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = pipeline_engine.train_batch_iter(
            model=model,
            pg=parallel_context.pp_pg,
            batch=(next(data_iter) for _ in range(grad_accum)),
            nb_microbatches=grad_accum,
            grad_accumulator=None,
        )

        if parallel_context.dp_pg.size() > 1:
            sync_gradients_across_dp(
                module=model,
                dp_pg=parallel_context.dp_pg,
                reduce_op=dist.ReduceOp.AVG,
                reduce_scatter=False,
                grad_accumulator=None,
            )

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(clip_grad))
        optimizer.step()

        loss = torch.stack([out["loss"] for out in outputs]).sum().detach()
        if parallel_context.dp_pg.size() > 1:
            dist.all_reduce(loss, group=parallel_context.dp_pg, op=dist.ReduceOp.AVG)

        if dist.get_rank(parallel_context.world_pg) == 0:
            print(f"step={step} loss={loss.item():.6f}")

    # Make sure the last kernels have finished before torchrun tears down NCCL.
    torch.cuda.synchronize(device)
    parallel_context.destroy()


if __name__ == "__main__":
    main()

