from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict
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


def _disk_used_percent(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.used / usage.total * 100.0


def _init_tracker(cfg, enabled: bool):
    if not enabled or cfg.tracking.backend.lower() != "wandb":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("tracking.backend=wandb requires the wandb package in the active environment") from exc
    if not cfg.tracking.project:
        raise ValueError("tracking.project is required when tracking.backend=wandb")
    return wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        name=cfg.tracking.name,
        tags=cfg.tracking.tags,
        mode=cfg.tracking.mode,
        config=asdict(cfg),
    )


def _log_tracker(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def _finish_tracker(run) -> None:
    if run is not None:
        run.finish()



def _sync_trainable_gradients_across_dp(trainable_params: list[torch.nn.Parameter], dp_pg, dist_module) -> None:
    for param in trainable_params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist_module.all_reduce(param.grad, group=dp_pg, op=dist_module.ReduceOp.AVG)


def _checkpoint_path(path: str | None) -> Path | None:
    return Path(path).expanduser() if path else None


def _ranked_checkpoint_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}_rank_{rank:03d}{path.suffix}")


def _local_checkpoint_path(path: Path, parallel_context, dist_module) -> Path:
    if parallel_context.tp_pg.size() == 1:
        return path
    return _ranked_checkpoint_path(path, dist_module.get_rank(parallel_context.world_pg))

def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_optimizer: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
    }
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


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
    from nanotron.parallel.parameters import sanity_check
    from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine

    ensure_torchrun_env()
    if not torch.cuda.is_available():
        raise RuntimeError("Nanotron-SmVLA training requires CUDA.")

    cfg = load_config(args.config_file)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.parallelism.pp != 1:
        raise ValueError("Pipeline parallelism is not implemented for SmolVLA yet; use pp=1.")
    if cfg.parallelism.tp != 1 and not cfg.model.expert_tensor_parallel:
        raise ValueError("tp>1 requires model.expert_tensor_parallel=true for the TP1 expert-only path.")
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

    resume_path = _checkpoint_path(cfg.checkpoint.resume_from)
    start_step = 0
    if resume_path is not None:
        local_resume_path = _local_checkpoint_path(resume_path, parallel_context, dist)
        start_step = _load_checkpoint(local_resume_path, model, optimizer, device)
    dp_rank = dist.get_rank(parallel_context.dp_pg)
    dp_size = parallel_context.dp_pg.size()
    data_iter = build_data_iterator(
        cfg.model,
        cfg.data,
        cfg.tokens,
        device,
        data_rank=dp_rank,
        data_world_size=dp_size,
    )
    pipeline_engine = AllForwardAllBackwardPipelineEngine()
    is_rank0 = dist.get_rank(parallel_context.world_pg) == 0
    tracker = _init_tracker(cfg, is_rank0)

    if is_rank0:
        local_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(
            f"Nanotron-SmVLA training: data={cfg.data.kind} "
            f"dp={dp_size}, tp={cfg.parallelism.tp}, pp={cfg.parallelism.pp}, "
            f"params={local_params:,}, trainable={trainable:,}, expert_tp={cfg.model.expert_tensor_parallel}"
        )
        if resume_path is not None:
            print(f"checkpoint_resumed path={resume_path} step={start_step}")

    try:
        for step in range(start_step + 1, cfg.tokens.train_steps + 1):
            step_started_at = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            outputs = pipeline_engine.train_batch_iter(
                model=model,
                pg=parallel_context.pp_pg,
                batch=(next(data_iter) for _ in range(cfg.tokens.batch_accumulation_per_replica)),
                nb_microbatches=cfg.tokens.batch_accumulation_per_replica,
                grad_accumulator=None,
            )

            if parallel_context.dp_pg.size() > 1:
                _sync_trainable_gradients_across_dp(trainable_params, parallel_context.dp_pg, dist)

            grad_norm = None
            if cfg.optimizer.clip_grad is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.optimizer.clip_grad))
            optimizer.step()

            loss = torch.stack([out["loss"] for out in outputs]).sum().detach()
            if parallel_context.dp_pg.size() > 1:
                dist.all_reduce(loss, group=parallel_context.dp_pg, op=dist.ReduceOp.AVG)

            if is_rank0:
                step_time = time.perf_counter() - step_started_at
                print(f"step={step} loss={loss.item():.6f}")

                if step % max(int(cfg.tracking.log_every), 1) == 0:
                    samples_seen = (
                        step
                        * cfg.tokens.micro_batch_size
                        * cfg.tokens.batch_accumulation_per_replica
                        * cfg.parallelism.dp
                    )
                    metrics = {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step_time_sec": step_time,
                        "train/samples_seen": samples_seen,
                        "system/disk_used_percent": _disk_used_percent(Path.cwd()),
                    }
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = float(grad_norm.detach().cpu())
                    _log_tracker(tracker, metrics, step)


            if cfg.checkpoint.output_dir and cfg.checkpoint.save_every and step % int(cfg.checkpoint.save_every) == 0:
                checkpoint_path = Path(cfg.checkpoint.output_dir) / f"step_{step:06d}.pt"
                local_checkpoint_path = _local_checkpoint_path(checkpoint_path, parallel_context, dist)
                if is_rank0 or parallel_context.tp_pg.size() > 1:
                    _save_checkpoint(local_checkpoint_path, model, optimizer, step, cfg.checkpoint.save_optimizer)
                if parallel_context.world_pg.size() > 1:
                    dist.barrier(parallel_context.world_pg)
                if is_rank0:
                    print(f"checkpoint_saved path={checkpoint_path}")
                    if parallel_context.tp_pg.size() > 1:
                        print(f"checkpoint_saved_tp_shards pattern={checkpoint_path.stem}_rank_*.pt")
                    _log_tracker(tracker, {"checkpoint/saved": 1, "checkpoint/step": step}, step)
    finally:
        _finish_tracker(tracker)

    torch.cuda.synchronize(device)
    parallel_context.destroy()


if __name__ == "__main__":
    main()