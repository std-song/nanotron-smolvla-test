from __future__ import annotations

from typing import Union

import torch
from torch import nn

from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

from .config import SmolVLAModelConfig


def convert_parameters_to_nanotron(module: nn.Module) -> None:
    """Convert regular torch Parameters in-place so Nanotron optimizers can track metadata."""

    memo: dict[int, NanotronParameter] = {}
    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is None or isinstance(param, NanotronParameter):
                continue
            param_id = id(param)
            if param_id not in memo:
                memo[param_id] = NanotronParameter(param, requires_grad=param.requires_grad)
            child._parameters[name] = memo[param_id]


class SmolVLALossModule(nn.Module):
    def __init__(self, config: SmolVLAModelConfig):
        super().__init__()

        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
        from lerobot.utils.constants import ACTION, OBS_STATE

        image_key = config.image_key or "observation.image"
        self.image_key = image_key

        smolvla_config = SmolVLAConfig(
            input_features={
                image_key: PolicyFeature(type=FeatureType.VISUAL, shape=config.image_shape),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(config.max_state_dim,)),
            },
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(config.max_action_dim,)),
            },
            device=config.device,
            vlm_model_name=config.vlm_model_name,
            load_vlm_weights=config.load_vlm_weights,
            train_expert_only=config.train_expert_only,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_state_proj=config.train_state_proj,
            attention_mode=config.attention_mode,
            num_vlm_layers=config.num_vlm_layers,
            num_expert_layers=config.num_expert_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            n_action_steps=config.chunk_size,
            tokenizer_max_length=config.tokenizer_max_length,
            resize_imgs_with_padding=(config.image_shape[1], config.image_shape[2]),
            compile_model=config.compile_model,
        )
        smolvla_config.validate_features()
        self.model = VLAFlowMatching(smolvla_config)

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        model_dtype = next(self.model.parameters()).dtype
        state = state.to(dtype=model_dtype)
        action = action.to(dtype=model_dtype)
        with torch.autocast(
            device_type="cuda",
            dtype=model_dtype,
            enabled=model_dtype in (torch.float16, torch.bfloat16),
        ):
            losses = self.model(
                images=[image],
                img_masks=[image_mask],
                lang_tokens=language_tokens,
                lang_masks=language_attention_mask,
                state=state,
                actions=action,
            )
        valid = (~action_is_pad).unsqueeze(-1)
        denom = (valid.sum() * losses.shape[-1]).clamp_min(1)
        return {"loss": ((losses * valid).sum() / denom).float()}


class SmolVLANanotronModel(NanotronModel):
    def __init__(
        self,
        config: SmolVLAModelConfig,
        parallel_context: ParallelContext,
        parallel_config=None,
        random_states=None,
    ):
        super().__init__()
        self.config = config
        self.parallel_context = parallel_context
        self.parallel_config = parallel_config
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))

        self.loss = PipelineBlock(
            p2p=self.p2p,
            module_builder=SmolVLALossModule,
            module_kwargs={"config": config},
            module_input_keys={
                "image",
                "image_mask",
                "language_tokens",
                "language_attention_mask",
                "state",
                "action",
                "action_is_pad",
            },
            module_output_keys={"loss"},
        )

    def forward(
        self,
        image: Union[torch.Tensor, TensorPointer],
        image_mask: Union[torch.Tensor, TensorPointer],
        language_tokens: Union[torch.Tensor, TensorPointer],
        language_attention_mask: Union[torch.Tensor, TensorPointer],
        state: Union[torch.Tensor, TensorPointer],
        action: Union[torch.Tensor, TensorPointer],
        action_is_pad: Union[torch.Tensor, TensorPointer],
    ) -> dict[str, Union[torch.Tensor, TensorPointer]]:
        return self.loss(
            image=image,
            image_mask=image_mask,
            language_tokens=language_tokens,
            language_attention_mask=language_attention_mask,
            state=state,
            action=action,
            action_is_pad=action_is_pad,
        )

    @torch.no_grad()
    def init_model_randomly(self, config):
        # SmolVLA/Transformers initialize their own modules during construction.
        return None

    def get_block_compute_costs(self):
        return {SmolVLALossModule: 1}

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        return 0.0, 0.0
