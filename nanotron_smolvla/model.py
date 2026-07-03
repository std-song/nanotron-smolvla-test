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

def run_vlm_with_expert_layer_range(
    vlm_with_expert: nn.Module,
    inputs_embeds: list[torch.Tensor | None],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values=None,
    use_cache: bool | None = False,
    fill_kv_cache: bool | None = False,
    layer_start: int = 0,
    layer_end: int | None = None,
    apply_final_norm: bool = True,
):
    """Run a contiguous SmolVLA VLM/expert layer range.

    PP0 uses the full range, preserving LeRobot's current forward behavior. The
    range arguments are the future PP stage boundary.
    """

    models = [vlm_with_expert.get_vlm_model().text_model, vlm_with_expert.lm_expert]
    model_layers = vlm_with_expert.get_model_layers(models)
    batch_size = None
    for hidden_states in inputs_embeds:
        if hidden_states is not None:
            batch_size = hidden_states.shape[0]
            break
    if batch_size is None:
        raise ValueError("At least one input embedding tensor is required")

    num_layers = vlm_with_expert.num_vlm_layers
    if layer_end is None:
        layer_end = num_layers
    if not 0 <= layer_start <= layer_end <= num_layers:
        raise ValueError(f"Invalid layer range [{layer_start}, {layer_end}) for {num_layers} layers")

    head_dim = vlm_with_expert.vlm.config.text_config.head_dim
    for layer_idx in range(layer_start, layer_end):
        if (
            fill_kv_cache
            or "cross" not in vlm_with_expert.attention_mode
            or (
                vlm_with_expert.self_attn_every_n_layers > 0
                and layer_idx % vlm_with_expert.self_attn_every_n_layers == 0
            )
        ):
            att_outputs, past_key_values = vlm_with_expert.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )
        else:
            att_outputs, past_key_values = vlm_with_expert.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                past_key_values=past_key_values,
            )

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            if layer is None:
                outputs_embeds.append(hidden_states)
                continue

            end = start + hidden_states.shape[1]
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            att_out = att_output[:, start:end]
            out_emb = layer.self_attn.o_proj(att_out)
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            out_emb = layer.post_attention_layernorm(out_emb)
            out_emb = layer.mlp(out_emb)
            out_emb += after_first_residual
            outputs_embeds.append(out_emb)
            start = end if len(att_outputs) == 1 else 0

        inputs_embeds = outputs_embeds

    if apply_final_norm:
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                outputs_embeds.append(models[i].norm(hidden_states))
            else:
                outputs_embeds.append(None)
        inputs_embeds = outputs_embeds

    return inputs_embeds, past_key_values

class SmolVLALossModule(nn.Module):
    def __init__(self, config: SmolVLAModelConfig, parallel_context: ParallelContext | None = None):
        super().__init__()

        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks
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
        self._make_att_2d_masks = make_att_2d_masks
        self.model = VLAFlowMatching(smolvla_config)
        self.tp_replaced_linears = 0
        if config.expert_tensor_parallel:
            if parallel_context is None:
                raise ValueError("expert_tensor_parallel requires a ParallelContext")
            if not config.train_expert_only:
                raise ValueError("expert_tensor_parallel currently requires train_expert_only=true")
            self.tp_replaced_linears = enable_expert_tensor_parallel(self.model, parallel_context.tp_pg)

    def _sample_flow_inputs(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = self.model.sample_noise(action.shape, action.device)
        time = self.model.sample_time(action.shape[0], action.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action
        u_t = noise - action
        return x_t, u_t, time

    def _embed_training_inputs(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            [image], [image_mask], language_tokens, language_attention_mask, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, time)
        return prefix_embs, prefix_pad_masks, prefix_att_masks, suffix_embs, suffix_pad_masks, suffix_att_masks

    def _build_attention_inputs(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = self._make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        return att_2d_masks, position_ids

    def _run_backbone(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        (_, suffix_out), _ = run_vlm_with_expert_layer_range(
            self.model.vlm_with_expert,
            inputs_embeds=[prefix_embs, suffix_embs],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            fill_kv_cache=False,
            layer_start=0,
            layer_end=None,
            apply_final_norm=True,
        )
        return suffix_out[:, -self.model.config.chunk_size :]

    def _compute_flow_loss(self, suffix_out: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.model.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

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
            x_t, u_t, time = self._sample_flow_inputs(action)
            (
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                suffix_embs,
                suffix_pad_masks,
                suffix_att_masks,
            ) = self._embed_training_inputs(
                image,
                image_mask,
                language_tokens,
                language_attention_mask,
                state,
                x_t,
                time,
            )
            attention_mask, position_ids = self._build_attention_inputs(
                prefix_pad_masks,
                prefix_att_masks,
                suffix_pad_masks,
                suffix_att_masks,
            )
            suffix_out = self._run_backbone(prefix_embs, suffix_embs, attention_mask, position_ids)
            losses = self._compute_flow_loss(suffix_out, u_t)
        valid = (~action_is_pad).unsqueeze(-1)
        denom = (valid.sum() * losses.shape[-1]).clamp_min(1)
        return {"loss": ((losses * valid).sum() / denom).float()}


class SmolVLAPPStage0Module(nn.Module):
    """First PP stage: flow sampling, input embedding, and early VLM/expert layers."""

    def __init__(self, config: SmolVLAModelConfig, parallel_context: ParallelContext | None = None):
        super().__init__()
        self.loss_module = SmolVLALossModule(config, parallel_context)
        self.split_layer = config.pipeline_split_layer or max(1, config.num_vlm_layers // 2)
        if not 0 < self.split_layer < config.num_vlm_layers:
            raise ValueError(
                f"pipeline_split_layer must be in [1, {config.num_vlm_layers - 1}] for pp=2; "
                f"got {self.split_layer}"
            )

    def forward(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        language_tokens: torch.Tensor,
        language_attention_mask: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        model_dtype = next(self.loss_module.model.parameters()).dtype
        state = state.to(dtype=model_dtype)
        action = action.to(dtype=model_dtype)
        with torch.autocast(
            device_type="cuda",
            dtype=model_dtype,
            enabled=model_dtype in (torch.float16, torch.bfloat16),
        ):
            x_t, u_t, time = self.loss_module._sample_flow_inputs(action)
            (
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                suffix_embs,
                suffix_pad_masks,
                suffix_att_masks,
            ) = self.loss_module._embed_training_inputs(
                image,
                image_mask,
                language_tokens,
                language_attention_mask,
                state,
                x_t,
                time,
            )
            attention_mask, position_ids = self.loss_module._build_attention_inputs(
                prefix_pad_masks,
                prefix_att_masks,
                suffix_pad_masks,
                suffix_att_masks,
            )
            prefix_embs, suffix_embs = run_vlm_with_expert_layer_range(
                self.loss_module.model.vlm_with_expert,
                inputs_embeds=[prefix_embs, suffix_embs],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                fill_kv_cache=False,
                layer_start=0,
                layer_end=self.split_layer,
                apply_final_norm=False,
            )[0]
        return {
            "prefix_embs": prefix_embs,
            "suffix_embs": suffix_embs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "u_t": u_t,
        }


class SmolVLAPPStage1Module(nn.Module):
    """Second PP stage: late VLM/expert layers, final norm, and flow loss."""

    def __init__(self, config: SmolVLAModelConfig, parallel_context: ParallelContext | None = None):
        super().__init__()
        self.loss_module = SmolVLALossModule(config, parallel_context)
        self.split_layer = config.pipeline_split_layer or max(1, config.num_vlm_layers // 2)
        if not 0 < self.split_layer < config.num_vlm_layers:
            raise ValueError(
                f"pipeline_split_layer must be in [1, {config.num_vlm_layers - 1}] for pp=2; "
                f"got {self.split_layer}"
            )

    def forward(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        u_t: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        model_dtype = next(self.loss_module.model.parameters()).dtype
        with torch.autocast(
            device_type="cuda",
            dtype=model_dtype,
            enabled=model_dtype in (torch.float16, torch.bfloat16),
        ):
            (_, suffix_out), _ = run_vlm_with_expert_layer_range(
                self.loss_module.model.vlm_with_expert,
                inputs_embeds=[prefix_embs, suffix_embs],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                fill_kv_cache=False,
                layer_start=self.split_layer,
                layer_end=None,
                apply_final_norm=True,
            )
            suffix_out = suffix_out[:, -self.loss_module.model.config.chunk_size :]
            losses = self.loss_module._compute_flow_loss(suffix_out, u_t)
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
        self.use_pipeline_split = parallel_context.pp_pg.size() > 1

        if self.use_pipeline_split:
            self.stage0 = PipelineBlock(
                p2p=self.p2p,
                module_builder=SmolVLAPPStage0Module,
                module_kwargs={"config": config, "parallel_context": parallel_context},
                module_input_keys={
                    "image",
                    "image_mask",
                    "language_tokens",
                    "language_attention_mask",
                    "state",
                    "action",
                },
                module_output_keys={
                    "prefix_embs",
                    "suffix_embs",
                    "attention_mask",
                    "position_ids",
                    "u_t",
                },
            )
            self.stage1 = PipelineBlock(
                p2p=self.p2p,
                module_builder=SmolVLAPPStage1Module,
                module_kwargs={"config": config, "parallel_context": parallel_context},
                module_input_keys={
                    "prefix_embs",
                    "suffix_embs",
                    "attention_mask",
                    "position_ids",
                    "u_t",
                    "action_is_pad",
                },
                module_output_keys={"loss"},
            )
        else:
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
        if self.use_pipeline_split:
            hidden = self.stage0(
                image=image,
                image_mask=image_mask,
                language_tokens=language_tokens,
                language_attention_mask=language_attention_mask,
                state=state,
                action=action,
            )
            hidden["action_is_pad"] = action_is_pad
            return self.stage1(**hidden)
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
        return {SmolVLALossModule: 1, SmolVLAPPStage0Module: 2, SmolVLAPPStage1Module: 1}

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        return 0.0, 0.0
