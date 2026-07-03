from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from nanotron import distributed as dist
from nanotron.models import NanotronModel
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.sharded_parameters import SplitConfig, create_sharded_parameter_from_config
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_reduce_sum

from .config import SmolVLAModelConfig


class _AllGatherLastDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, pg) -> torch.Tensor:
        ctx.pg = pg
        ctx.rank = dist.get_rank(pg)
        ctx.world_size = pg.size()
        if ctx.world_size == 1:
            return tensor
        chunks = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(chunks, tensor.contiguous(), group=pg)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return grad_output, None
        local = grad_output.chunk(ctx.world_size, dim=-1)[ctx.rank].contiguous()
        return local, None


def _all_gather_last_dim(tensor: torch.Tensor, pg) -> torch.Tensor:
    return _AllGatherLastDim.apply(tensor, pg)


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


class GatheredColumnParallelLinear(nn.Module):
    """Shard a linear layer's output dimension, then gather the full output for legacy SmolVLA code."""

    def __init__(self, source: nn.Linear, pg):
        super().__init__()
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = pg.size()
        if source.out_features % self.world_size != 0:
            raise ValueError(f"out_features={source.out_features} must be divisible by tp={self.world_size}")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.local_out_features = source.out_features // self.world_size
        self.weight = nn.Parameter(
            torch.empty(
                self.local_out_features,
                source.in_features,
                device=source.weight.device,
                dtype=source.weight.dtype,
            ),
            requires_grad=source.weight.requires_grad,
        )
        if source.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.empty(self.local_out_features, device=source.bias.device, dtype=source.bias.dtype),
                requires_grad=source.bias.requires_grad,
            )
        with torch.no_grad():
            start = self.rank * self.local_out_features
            end = start + self.local_out_features
            self.weight.copy_(source.weight[start:end])
            if self.bias is not None:
                self.bias.copy_(source.bias[start:end])
        split_config = SplitConfig(split_dim=0)
        self.weight = create_sharded_parameter_from_config(self.weight, pg=pg, split_config=split_config)
        if self.bias is not None:
            self.bias = create_sharded_parameter_from_config(self.bias, pg=pg, split_config=SplitConfig(split_dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.weight, self.bias)
        return _all_gather_last_dim(local, self.pg).clone()


class SlicedInputRowParallelLinear(nn.Module):
    """Shard a linear layer's input dimension and all-reduce the partial outputs."""

    def __init__(self, source: nn.Linear, pg):
        super().__init__()
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = pg.size()
        if source.in_features % self.world_size != 0:
            raise ValueError(f"in_features={source.in_features} must be divisible by tp={self.world_size}")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.local_in_features = source.in_features // self.world_size
        self.weight = nn.Parameter(
            torch.empty(
                source.out_features,
                self.local_in_features,
                device=source.weight.device,
                dtype=source.weight.dtype,
            ),
            requires_grad=source.weight.requires_grad,
        )
        # None of the current expert row-parallel targets use bias, but keep the general case correct.
        if source.bias is not None and self.rank == 0:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=source.bias.requires_grad)
        else:
            self.bias = None
        with torch.no_grad():
            start = self.rank * self.local_in_features
            end = start + self.local_in_features
            self.weight.copy_(source.weight[:, start:end])
        self.weight = create_sharded_parameter_from_config(self.weight, pg=pg, split_config=SplitConfig(split_dim=1))
        if self.bias is not None:
            self.bias = NanotronParameter(self.bias, requires_grad=self.bias.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = self.rank * self.local_in_features
        end = start + self.local_in_features
        out = F.linear(x[..., start:end], self.weight, self.bias)
        if self.world_size > 1:
            out = differentiable_all_reduce_sum(out, group=self.pg).clone()
        return out


def _set_child_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def enable_expert_tensor_parallel(model: nn.Module, tp_pg) -> int:
    """Replace trainable SmolVLA expert linears with TP shims that preserve full hidden shapes."""

    if tp_pg.size() == 1:
        return 0
    expert = model.vlm_with_expert.lm_expert
    replaced = 0
    for name, child in list(expert.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not any(param.requires_grad for param in child.parameters(recurse=False)):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
            replacement = GatheredColumnParallelLinear(child, tp_pg)
        elif leaf in {"o_proj", "down_proj"}:
            replacement = SlicedInputRowParallelLinear(child, tp_pg)
        else:
            continue
        _set_child_module(expert, name, replacement)
        replaced += 1
    return replaced


class SmolVLALossModule(nn.Module):
    def __init__(self, config: SmolVLAModelConfig, parallel_context: ParallelContext | None = None):
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
        self.tp_replaced_linears = 0
        if config.expert_tensor_parallel:
            if parallel_context is None:
                raise ValueError("expert_tensor_parallel requires a ParallelContext")
            if not config.train_expert_only:
                raise ValueError("expert_tensor_parallel currently requires train_expert_only=true")
            self.tp_replaced_linears = enable_expert_tensor_parallel(self.model, parallel_context.tp_pg)

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
            module_kwargs={"config": config, "parallel_context": parallel_context},
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
