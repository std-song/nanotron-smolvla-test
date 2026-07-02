from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from .config import load_config
from .data import build_data_iterator
from .model import SmolVLANanotronModel, convert_parameters_to_nanotron


def add_repo_paths(nanotron_src: Path | None, lerobot_src: Path | None) -> None:
    for path in (nanotron_src, lerobot_src):
        if path is None:
            continue
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--nanotron-src", type=Path, default=None)
    parser.add_argument("--lerobot-src", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_repo_paths(args.nanotron_src, args.lerobot_src)

    from nanotron import distributed as dist
    from nanotron.models import build_model
    from nanotron.parallel import ParallelContext
    from nanotron.parallel.data_parallel.utils import sync_gradients_across_dp
    from nanotron.parallel.parameters import sanity_check
    from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine

    ensure_torchrun_env()
    if not torch.cuda.is_available():
        raise RuntimeError("Nanotron-SmVLA training requires CUDA.")

    cfg = load_config(args.config_file)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.parallelism.tp != 1 or cfg.parallelism.pp != 1:
        raise ValueError("Stage A/B supports tp=1 and pp=1 only. Later stages will split SmolVLA for TP/PP.")

    parallel_context = ParallelContext(
        tensor_parallel_size=cfg.parallelism.tp,
        pipeline_parallel_size=cfg.parallelism.pp,
        data_parallel_size=cfg.parallelism.dp,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

    model = build_model(
        model_builder=lambda: SmolVLANanotronModel(
            config=cfg.model,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=getattr(torch, cfg.dtype),
        target_pp_ranks=[0],
        device=device,
    )
    convert_parameters_to_nanotron(model)
    sanity_check(model)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check SmolVLA freeze/train flags.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
        fused=cfg.optimizer.fused,
    )

    data_iter = build_data_iterator(cfg.model, cfg.data, cfg.tokens, device)
    pipeline_engine = AllForwardAllBackwardPipelineEngine()

    if dist.get_rank(parallel_context.world_pg) == 0:
        local_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(
            f"Nanotron-SmVLA training: data={cfg.data.kind} "
            f"params={local_params:,}, trainable={trainable:,}"
        )

    for step in range(1, cfg.tokens.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = pipeline_engine.train_batch_iter(
            model=model,
            pg=parallel_context.pp_pg,
            batch=(next(data_iter) for _ in range(cfg.tokens.batch_accumulation_per_replica)),
            nb_microbatches=cfg.tokens.batch_accumulation_per_replica,
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

        if cfg.optimizer.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.optimizer.clip_grad))
        optimizer.step()

        loss = torch.stack([out["loss"] for out in outputs]).sum().detach()
        if parallel_context.dp_pg.size() > 1:
            dist.all_reduce(loss, group=parallel_context.dp_pg, op=dist.ReduceOp.AVG)
        if dist.get_rank(parallel_context.world_pg) == 0:
            print(f"step={step} loss={loss.item():.6f}")

    torch.cuda.synchronize(device)
    parallel_context.destroy()


if __name__ == "__main__":
    main()
