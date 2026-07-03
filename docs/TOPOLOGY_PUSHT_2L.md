# PushT Two-Layer SmolVLA Topology Audit

Generated on AutoDL from:

```bash
python scripts/inspect_smolvla_topology.py \
  --config-file configs/smolvla_pusht_2dp_autodl.yaml \
  --nanotron-src /root/autodl-tmp/nanotron-minimal/nanotron/src \
  --lerobot-src /root/autodl-tmp/smolvla-minimal/lerobot/src \
  --max-lines 300
```

Remote copy: `/root/autodl-tmp/nanotron-smolvla-project/artifacts_topology_2tp_plan.txt`.

## Summary

```text
num_vlm_layers=2
num_expert_layers=2
train_expert_only=True
freeze_vision_encoder=True
attention_mode=cross_attn
loss_module: params=226,429,216 trainable=13,916,512
vlm_with_expert: params=224,794,064 trainable=12,281,360
vlm_text_model: params=66,974,400 trainable=0
lm_expert: params=12,281,360 trainable=12,281,360
```

## Top-Level Trainable Projections

```text
state_proj: in=32 out=960 bias=True trainable=True dtype=torch.float32
action_in_proj: in=32 out=720 bias=True trainable=True dtype=torch.float32
action_time_mlp_in: in=1440 out=720 bias=True trainable=True dtype=torch.float32
action_time_mlp_out: in=720 out=720 bias=True trainable=True dtype=torch.float32
action_out_proj: in=720 out=32 bias=True trainable=True dtype=torch.float32
```

These can stay replicated for the first 2TP implementation.

## Frozen VLM Text Layers

The reduced VLM text model has two layers and 14 linear modules. All are frozen in the current smoke config.

```text
layers.0.self_attn.q_proj in=960 out=960 dtype=torch.bfloat16
layers.0.self_attn.k_proj in=960 out=320 dtype=torch.bfloat16
layers.0.self_attn.v_proj in=960 out=320 dtype=torch.bfloat16
layers.0.self_attn.o_proj in=960 out=960 dtype=torch.bfloat16
layers.0.mlp.gate_proj in=960 out=2560 dtype=torch.bfloat16
layers.0.mlp.up_proj in=960 out=2560 dtype=torch.bfloat16
layers.0.mlp.down_proj in=2560 out=960 dtype=torch.bfloat16
layers.1.self_attn.q_proj in=960 out=960 dtype=torch.bfloat16
layers.1.self_attn.k_proj in=960 out=320 dtype=torch.bfloat16
layers.1.self_attn.v_proj in=960 out=320 dtype=torch.bfloat16
layers.1.self_attn.o_proj in=960 out=960 dtype=torch.bfloat16
layers.1.mlp.gate_proj in=960 out=2560 dtype=torch.bfloat16
layers.1.mlp.up_proj in=960 out=2560 dtype=torch.bfloat16
layers.1.mlp.down_proj in=2560 out=960 dtype=torch.bfloat16
```

## Trainable Expert Layers

The expert model has two layers and 14 trainable linear modules.

```text
layers.0.self_attn.q_proj in=720 out=960 dtype=torch.bfloat16
layers.0.self_attn.k_proj in=720 out=320 dtype=torch.bfloat16
layers.0.self_attn.v_proj in=720 out=320 dtype=torch.bfloat16
layers.0.self_attn.o_proj in=960 out=720 dtype=torch.bfloat16
layers.0.mlp.gate_proj in=720 out=2048 dtype=torch.bfloat16
layers.0.mlp.up_proj in=720 out=2048 dtype=torch.bfloat16
layers.0.mlp.down_proj in=2048 out=720 dtype=torch.bfloat16
layers.1.self_attn.q_proj in=720 out=960 dtype=torch.bfloat16
layers.1.self_attn.k_proj in=320 out=320 dtype=torch.float32
layers.1.self_attn.v_proj in=320 out=320 dtype=torch.float32
layers.1.self_attn.o_proj in=960 out=720 dtype=torch.bfloat16
layers.1.mlp.gate_proj in=720 out=2048 dtype=torch.bfloat16
layers.1.mlp.up_proj in=720 out=2048 dtype=torch.bfloat16
layers.1.mlp.down_proj in=2048 out=720 dtype=torch.bfloat16
```

The second expert layer has cross-attention `k_proj` and `v_proj` reshaped to `320 -> 320` with float32 dtype. The first TP replacement pass must handle these separately from ordinary expert self-attention projections.
