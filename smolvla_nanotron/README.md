# Minimal SmolVLA training with Nanotron primitives

This example runs LeRobot's `SmolVLA` loss through Nanotron's distributed process groups,
model builder, parameter metadata checks, and pipeline engine. It intentionally starts with
random dummy robot batches so the training framework bridge can be validated before wiring a
real `LeRobotDataset`.

## What this proves

- Nanotron can initialize a SmolVLA wrapper as a `NanotronModel`.
- Nanotron's `PipelineBlock` / `AllForwardAllBackwardPipelineEngine` can drive SmolVLA forward and backward.
- Nanotron-compatible parameter metadata can be attached to the wrapped SmolVLA parameters.
- A minimal optimizer step works on CUDA with bf16 autocast.

## Current scope

- Supported: CUDA, `tp=1`, `pp=1`, `dp=1` for the smoke test.
- Data: synthetic images, language tokens, state, and actions.
- Model: LeRobot `VLAFlowMatching` with a reduced SmolVLM backbone (`num_vlm_layers=2`).
- Not yet supported: Nanotron tensor parallel or pipeline parallel inside SmolVLA internals.
- Not yet included: checkpoint save/load and a real LeRobot dataset collator.

## Files

- `modeling.py`: SmolVLA-to-Nanotron model wrapper.
- `run_train.py`: minimal Nanotron-style training entrypoint.
- `config_minimal.yaml`: generic online/cache-backed config.
- `config_offline_autodl.yaml`: the exact offline AutoDL config used for verification.
- `RESULTS.md`: remote GPU verification notes and outputs.

## Run

From the `nanotron` repository root:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/path/to/nanotron/src:/path/to/lerobot/src:${PYTHONPATH:-}

torchrun --nproc_per_node=1 examples/smolvla_nanotron/run_train.py \
  --config-file examples/smolvla_nanotron/config_minimal.yaml \
  --lerobot-src /path/to/lerobot/src
```

On the verified AutoDL machine, the offline run used:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/smolvla-minimal/conda-py312

export HF_HOME=/root/autodl-tmp/smolvla-minimal/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/nanotron-minimal/nanotron/src:/root/autodl-tmp/smolvla-minimal/lerobot/src:/root/autodl-tmp/nanotron-smolvla-bridge

cd /root/autodl-tmp/nanotron-smolvla-bridge
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_train.py \
  --config-file config_offline_autodl.yaml \
  --lerobot-src /root/autodl-tmp/smolvla-minimal/lerobot/src
```

## Next steps

1. Replace `infinite_dummy_batches` with a LeRobot dataset collator that emits the same tensor keys.
2. Decide which SmolVLA parameters to train: expert-only, state projection, or full VLM fine-tuning.
3. Add checkpoint save/load in Nanotron format.
4. Only after the single-card path is stable, split SmolVLA internals into Nanotron TP/PP-aware blocks.
