# Tensor and Pipeline Parallel Plan

This note captures the next implementation stages after the verified 2DP PushT smoke run.

## Current state

- `nanotron_smolvla.model.SmolVLANanotronModel` wraps the whole LeRobot `VLAFlowMatching` module inside one Nanotron `PipelineBlock`.
- Data parallelism is verified with `dp=2,tp=1,pp=1`.
- DP gradient synchronization is project-local and only touches trainable parameters.
- SmolVLA internals are still standard LeRobot/Transformers modules, so setting `tp=2` would not shard any model weights yet.

## SmolVLA topology to shard

`VLAFlowMatching` has three useful regions:

- Replicated input/output glue:
  - `state_proj`
  - `action_in_proj`
  - `action_time_mlp_in`
  - `action_time_mlp_out`
  - `action_out_proj`
- Mostly replicated frozen VLM assets:
  - vision encoder
  - connector
  - tokenizer/processor
  - language embeddings, while `train_expert_only=True`
- Transformer layers that should become TP-aware:
  - VLM text layers in `vlm_with_expert.get_vlm_model().text_model.layers`
  - expert layers in `vlm_with_expert.lm_expert.layers`

Inside each trainable transformer layer, the first TP targets are:

- attention projections: `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP projections: commonly `mlp.gate_proj`, `up_proj`, `down_proj`

Nanotron's reference pattern is:

- q/k/v or merged qkv: `TensorParallelColumnLinear`
- attention output: `TensorParallelRowLinear`
- MLP gate/up: `TensorParallelColumnLinear`
- MLP down: `TensorParallelRowLinear`
- embeddings: `TensorParallelEmbedding`, deferred until VLM text training is enabled

## Stage TP0: topology audit

Before modifying model code, run `scripts/inspect_smolvla_topology.py` on AutoDL with the same config used for smoke tests. Keep the output under `artifacts/` if it is useful for review.

Expected result: a stable list of all `nn.Linear` modules, their shapes, and whether they are trainable.

## Stage TP1: expert-only 2TP

Goal: verify `dp=1,tp=2,pp=1` while keeping the frozen VLM path replicated.

Planned scope:

- Require `train_expert_only=True` for the first TP implementation.
- Replace trainable expert-layer attention and MLP linears with project-local TP shims first, then migrate the shims toward native Nanotron TP layers once the SmolVLA attention path is head-local.
- Keep action/state projection layers replicated.
- Keep vision encoder, connector, language embeddings, and frozen VLM text layers replicated.
- Load full checkpoint weights on each TP rank, then copy the local shard into each TP module.
- Save checkpoints in the current full-state format first, then add a sharded checkpoint format later.

Validation ladder:

1. `tp=1` equivalence smoke: replacement path enabled but TP size is 1.
2. `tp=2` dummy smoke: 2 steps, no PushT data dependency.
3. `tp=2` PushT smoke: 5 steps. Verified on AutoDL with `configs/smolvla_pusht_2tp_autodl.yaml`.
4. `tp=2` PushT stability: 50 steps plus checkpoint save/resume. Verified with TP rank-local checkpoint shards.

## Stage TP2: 2DP x 2TP

Goal: verify `dp=2,tp=2,pp=1` on four GPUs.

The current DP synchronization must skip TP-sharded parameters or synchronize them only across each parameter's DP replica group. This means the trainable parameter sync code needs to distinguish:

- replicated parameters, synchronized over DP
- TP-sharded parameters, synchronized across matching TP ranks within DP groups

Nanotron's own sharded parameter metadata should be used here instead of name-based rules.

## Stage PP0: pipeline split design

Pipeline parallelism is not a config flip yet because `VLAFlowMatching.forward` builds prefix/suffix embeddings, calls all VLM/expert layers, and computes loss in one module.

The likely PP split is:

- Stage 0: image/language/state/action embedding and first group of VLM/expert layers
- Stage 1: remaining VLM/expert layers, final norm, action projection, MSE loss

For the current two-layer smoke model, PP has little benefit and mainly validates mechanics. A useful PP test probably needs a larger `num_vlm_layers`.

## Stage PP1: 2PP smoke

Goal: verify `dp=1,tp=1,pp=2`.

Required code changes:

- Extract a SmolVLA layer runner that can execute a contiguous layer range.
- Pass prefix/suffix hidden states, masks, position ids, and optional KV cache across pipeline stages.
- Move loss computation to the last stage.

## Final target

The final multi-card target is:

```text
dp=2,tp=2,pp=2
```

This requires eight GPUs. The practical route is:

1. 2DP is done.
2. 2TP expert-only.
3. 2DP x 2TP.
4. 2PP with TP disabled.
5. 2DP x 2TP x 2PP.
