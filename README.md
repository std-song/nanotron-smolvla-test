# Nanotron-SmVLA Test Project

This repository contains a minimal but structured bridge for training LeRobot SmolVLA through Nanotron primitives.

## Current milestones

- Stage A: Refactor the original smoke test into a small Python project.
- Stage B: Add a real `LeRobotDataset` backend and verify single-GPU training on the PushT dataset.

The original smoke-test files and first verification notes are preserved in `smolvla_nanotron/`.
The active project code lives in `nanotron_smolvla/`.

## Layout

```text
nanotron_smolvla/
  config.py        # YAML config dataclasses
  data.py          # dummy and LeRobotDataset batch iterators
  model.py         # SmolVLA wrapped as a NanotronModel
  train.py         # Nanotron-style training entrypoint
configs/
  smolvla_dummy_1gpu.yaml
  smolvla_pusht_1gpu_autodl.yaml
scripts/
  run_dummy_1gpu.sh
  run_pusht_1gpu_autodl.sh
smolvla_nanotron/
  # archived initial smoke-test script and verification notes
```

## Stage A: dummy smoke test

```bash
export NANOTRON_SRC=/path/to/nanotron/src
export LEROBOT_SRC=/path/to/lerobot/src
bash scripts/run_dummy_1gpu.sh configs/smolvla_dummy_1gpu.yaml
```

## Stage B: real LeRobotDataset smoke test on PushT

This uses the same small PushT dataset that was previously used for the official SmolVLA training smoke test.
On the AutoDL host, the config points to the cached SmolVLM snapshot and the downloaded local PushT dataset at `/root/autodl-tmp/smolvla-minimal/cache/huggingface/lerobot/kuai-zi/pusht-bucket1`.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/smolvla-minimal/conda-py312

cd /root/autodl-tmp/nanotron-smolvla-project
bash scripts/run_pusht_1gpu_autodl.sh configs/smolvla_pusht_1gpu_autodl.yaml
```

The default verification config is deliberately tiny:

```yaml
data:
  kind: lerobot
  lerobot:
    repo_id: kuai-zi/smolvla-pusht-autodl
    root: /root/autodl-tmp/smolvla-minimal/cache/huggingface/lerobot/kuai-zi/pusht-bucket1
    max_samples: 64
tokens:
  train_steps: 2
  micro_batch_size: 1
```

## Important limitations

- Stage A/B supports `dp=1,tp=1,pp=1` only.
- Tensor parallel and pipeline parallel are not implemented inside SmolVLA yet.
- The model uses a reduced SmolVLM backbone (`num_vlm_layers=2`) for fast smoke tests.
- The PushT run is a framework/data-path validation, not a useful policy training run.

## Verified first smoke-test result

The initial dummy run on AutoDL RTX 3090 produced:

```text
Reducing the number of VLM layers to 2 ...
SmolVLA/Nanotron minimal training: params=226,429,216, trainable=13,916,512
step=1 loss=2.707207
step=2 loss=2.689403
```

See `smolvla_nanotron/RESULTS.md` for the original result details.

## Verified Stage B result

The real PushT `LeRobotDataset` run was verified on the AutoDL RTX 3090 host with the downloaded dataset at `/root/autodl-tmp/smolvla-minimal/cache/huggingface/lerobot/kuai-zi/pusht-bucket1`.
The data adapter resizes real images to the configured SmolVLM input size before forwarding them through SmolVLA.

```text
Reducing the number of VLM layers to 2 ...
Nanotron-SmVLA training: data=lerobot params=226,429,216, trainable=13,916,512
step=1 loss=1.369288
step=2 loss=1.393214
```

At verification time, `/root/autodl-tmp` remained at 64% usage.
