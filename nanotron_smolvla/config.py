from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParallelismConfig:
    dp: int = 1
    pp: int = 1
    tp: int = 1


@dataclass
class SmolVLAModelConfig:
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    train_expert_only: bool = True
    freeze_vision_encoder: bool = True
    train_state_proj: bool = True
    attention_mode: str = "cross_attn"
    num_vlm_layers: int = 2
    num_expert_layers: int = 2
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 8
    tokenizer_max_length: int = 48
    image_key: str | None = None
    image_shape: tuple[int, int, int] = (3, 512, 512)
    device: str = "cuda"
    compile_model: bool = False

    # Generic Nanotron paths expect these attributes to exist.
    vocab_size: int = 1
    num_attention_heads: int = 1
    num_key_value_heads: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmolVLAModelConfig":
        data = dict(data)
        if "image_shape" in data and isinstance(data["image_shape"], list):
            data["image_shape"] = tuple(data["image_shape"])
        return cls(**data)


@dataclass
class DummyDataConfig:
    language_vocab_size: int = 49_152


@dataclass
class LeRobotDataConfig:
    repo_id: str = "lerobot/pusht"
    root: str | None = None
    revision: str | None = None
    episodes: list[int] | None = None
    video_backend: str | None = None
    num_workers: int = 0
    max_samples: int | None = None
    shuffle: bool = True


@dataclass
class DataConfig:
    kind: str = "dummy"
    dummy: DummyDataConfig = field(default_factory=DummyDataConfig)
    lerobot: LeRobotDataConfig = field(default_factory=LeRobotDataConfig)


@dataclass
class TokensConfig:
    train_steps: int = 2
    micro_batch_size: int = 1
    batch_accumulation_per_replica: int = 1


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_grad: float | None = 10.0
    fused: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerConfig":
        data = dict(data)
        if "betas" in data and isinstance(data["betas"], list):
            data["betas"] = tuple(data["betas"])
        return cls(**data)


@dataclass
class CheckpointConfig:
    output_dir: str | None = None
    save_every: int | None = None
    resume_from: str | None = None
    save_optimizer: bool = True


@dataclass
class TrackingConfig:
    backend: str = "disabled"
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str | None = None
    log_every: int = 1


@dataclass
class TrainConfig:
    seed: int = 42
    dtype: str = "bfloat16"
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    model: SmolVLAModelConfig = field(default_factory=SmolVLAModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokens: TokensConfig = field(default_factory=TokensConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return value if value is not None else {}


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_raw = _section(raw, "data")
    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        dtype=str(raw.get("dtype", "bfloat16")),
        parallelism=ParallelismConfig(**_section(raw, "parallelism")),
        model=SmolVLAModelConfig.from_dict(_section(raw, "model")),
        data=DataConfig(
            kind=str(data_raw.get("kind", "dummy")),
            dummy=DummyDataConfig(**data_raw.get("dummy", {})),
            lerobot=LeRobotDataConfig(**data_raw.get("lerobot", {})),
        ),
        tokens=TokensConfig(**_section(raw, "tokens")),
        optimizer=OptimizerConfig.from_dict(_section(raw, "optimizer")),
        checkpoint=CheckpointConfig(**_section(raw, "checkpoint")),
        tracking=TrackingConfig(**_section(raw, "tracking")),
    )
