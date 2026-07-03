from __future__ import annotations

from itertools import cycle
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .config import DataConfig, SmolVLAModelConfig, TokensConfig


def _pad_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        return tensor[..., :target_dim]
    out = torch.zeros(*tensor.shape[:-1], target_dim, dtype=tensor.dtype, device=tensor.device)
    out[..., : tensor.shape[-1]] = tensor
    return out


def _to_device_batch(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return result


def infinite_dummy_batches(
    model_config: SmolVLAModelConfig,
    data_config: DataConfig,
    tokens_config: TokensConfig,
    device: torch.device,
    data_rank: int = 0,
    data_world_size: int = 1,
) -> Iterator[dict[str, torch.Tensor]]:
    vocab_size = int(data_config.dummy.language_vocab_size)
    batch_size = int(tokens_config.micro_batch_size)
    state_dim = int(model_config.max_state_dim)
    action_dim = int(model_config.max_action_dim)
    chunk_size = int(model_config.chunk_size)
    language_length = int(model_config.tokenizer_max_length)
    image_shape = tuple(model_config.image_shape)

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


class LeRobotBatchAdapter:
    def __init__(self, model_config: SmolVLAModelConfig, device: torch.device):
        self.model_config = model_config
        self.device = device
        self.preprocessor = None
        self.camera_key = None

    def build(
        self,
        data_config: DataConfig,
        tokens_config: TokensConfig,
        data_rank: int = 0,
        data_world_size: int = 1,
    ):
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.policies.factory import dataset_to_policy_features
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        from lerobot.utils.collate import lerobot_collate_fn
        from lerobot.utils.constants import ACTION
        from lerobot.configs import FeatureType

        cfg = data_config.lerobot
        ds_meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root, revision=cfg.revision)
        features = dataset_to_policy_features(ds_meta.features)

        policy_cfg = SmolVLAConfig(
            device=self.model_config.device,
            vlm_model_name=self.model_config.vlm_model_name,
            load_vlm_weights=self.model_config.load_vlm_weights,
            train_expert_only=self.model_config.train_expert_only,
            freeze_vision_encoder=self.model_config.freeze_vision_encoder,
            train_state_proj=self.model_config.train_state_proj,
            attention_mode=self.model_config.attention_mode,
            num_vlm_layers=self.model_config.num_vlm_layers,
            num_expert_layers=self.model_config.num_expert_layers,
            self_attn_every_n_layers=self.model_config.self_attn_every_n_layers,
            expert_width_multiplier=self.model_config.expert_width_multiplier,
            max_state_dim=self.model_config.max_state_dim,
            max_action_dim=self.model_config.max_action_dim,
            chunk_size=self.model_config.chunk_size,
            n_action_steps=self.model_config.chunk_size,
            tokenizer_max_length=self.model_config.tokenizer_max_length,
            compile_model=self.model_config.compile_model,
        )
        policy_cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        policy_cfg.input_features = {key: ft for key, ft in features.items() if key not in policy_cfg.output_features}
        policy_cfg.validate_features()

        image_keys = list(policy_cfg.image_features.keys())
        if not image_keys:
            raise ValueError(f"Dataset {cfg.repo_id} has no visual observation keys for SmolVLA")
        self.camera_key = self.model_config.image_key or image_keys[0]
        if self.camera_key not in image_keys:
            raise ValueError(f"Configured image_key={self.camera_key!r} not in dataset image keys: {image_keys}")

        delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
            episodes=cfg.episodes,
            delta_timestamps=delta_timestamps,
            revision=cfg.revision,
            video_backend=cfg.video_backend,
            return_uint8=True,
        )
        if cfg.max_samples is not None:
            dataset = Subset(dataset, list(range(min(cfg.max_samples, len(dataset)))))

        self.preprocessor, _ = make_smolvla_pre_post_processors(policy_cfg, dataset_stats=ds_meta.stats)
        collate_fn = lerobot_collate_fn if ds_meta.has_language_columns else None
        sampler = None
        shuffle = cfg.shuffle
        if data_world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=data_world_size,
                rank=data_rank,
                shuffle=cfg.shuffle,
                drop_last=False,
            )
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=tokens_config.micro_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
        )
        return dataloader

    def adapt(self, batch: dict) -> dict[str, torch.Tensor]:
        from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        assert self.preprocessor is not None
        assert self.camera_key is not None

        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
                batch[key] = value.to(dtype=torch.float32) / 255.0
        processed = self.preprocessor(batch)
        processed = _to_device_batch(processed, self.device)

        image = processed[self.camera_key]
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with shape [B,C,H,W] or [B,T,C,H,W], got {tuple(image.shape)}")

        target_hw = tuple(self.model_config.image_shape[-2:])
        if tuple(image.shape[-2:]) != target_hw:
            image = F.interpolate(image.float(), size=target_hw, mode="bilinear", align_corners=False)

        state = _pad_last_dim(processed[OBS_STATE].float(), self.model_config.max_state_dim)
        action = _pad_last_dim(processed[ACTION].float(), self.model_config.max_action_dim)
        if action.ndim == 2:
            action = action[:, None, :].expand(-1, self.model_config.chunk_size, -1)

        batch_size = action.shape[0]
        action_steps = action.shape[1]
        action_is_pad = processed.get("action_is_pad")
        if action_is_pad is None:
            action_is_pad = torch.zeros((batch_size, action_steps), dtype=torch.bool, device=self.device)
        else:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool)

        return {
            "image": image.float(),
            "image_mask": torch.ones((image.shape[0],), dtype=torch.bool, device=self.device),
            "language_tokens": processed[OBS_LANGUAGE_TOKENS].long(),
            "language_attention_mask": processed[OBS_LANGUAGE_ATTENTION_MASK].bool(),
            "state": state,
            "action": action,
            "action_is_pad": action_is_pad,
        }


def build_data_iterator(
    model_config: SmolVLAModelConfig,
    data_config: DataConfig,
    tokens_config: TokensConfig,
    device: torch.device,
    data_rank: int = 0,
    data_world_size: int = 1,
) -> Iterator[dict[str, torch.Tensor]]:
    if data_config.kind == "dummy":
        return infinite_dummy_batches(model_config, data_config, tokens_config, device)
    if data_config.kind == "lerobot":
        adapter = LeRobotBatchAdapter(model_config=model_config, device=device)
        dataloader = adapter.build(data_config, tokens_config, data_rank=data_rank, data_world_size=data_world_size)

        def iterator():
            for raw_batch in cycle(dataloader):
                yield adapter.adapt(raw_batch)

        return iterator()
    raise ValueError(f"Unsupported data.kind={data_config.kind!r}; expected 'dummy' or 'lerobot'")
