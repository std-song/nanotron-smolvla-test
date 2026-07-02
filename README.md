# Nanotron-SmVLA Test Project

This repository contains a minimal but structured bridge for training LeRobot SmolVLA through Nanotron primitives.

## Current milestones

- Stage A: Refactor the original smoke test into a small Python project.
- Stage B: Add a real `LeRobotDataset` backend and verify single-GPU training on the PushT dataset.

The active project code lives in `nanotron_smolvla/`.
The original prototype is preserved under `archive/initial_smolvla_nanotron_smoke/` for historical reference only.

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
archive/
  initial_smolvla_nanotron_smoke/
    # archived initial prototype and first verification notes
```

## Repository Roles

- `nanotron_smolvla/` is the active training package. New work, including checkpointing, W&B tracking, 2DP, 2TP, and 2PP, should be implemented here.
- `configs/` contains runnable YAML configs for AutoDL and smoke validations.
- `scripts/` contains launcher scripts for the AutoDL environment.
- `archive/initial_smolvla_nanotron_smoke/` is read-only historical reference from the first minimal bridge prototype.

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
  train_steps: 50
  micro_batch_size: 1
checkpoint:
  output_dir: outputs/pusht_1gpu
  save_every: 25
  resume_from: null
tracking:
  backend: wandb
  project: nanotron-smolvla-test
  name: pusht-1gpu-checkpoint
  mode: online
```


## W&B tracking

Tracking is optional and only initialized on rank 0. The current W&B metrics are:

- `train/loss`
- `train/lr`
- `train/grad_norm`
- `train/step_time_sec`
- `train/samples_seen`
- `system/disk_used_percent`
- `checkpoint/saved`

For online tracking on AutoDL, log in once before running:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/smolvla-minimal/conda-py312
wandb login
```

The 5-step smoke config uses offline mode and can be synced later:

```bash
bash scripts/run_pusht_1gpu_autodl.sh configs/smolvla_pusht_wandb_5step_autodl.yaml
wandb sync /root/autodl-tmp/nanotron-smolvla-project/wandb/offline-run-YYYYMMDD_HHMMSS-xxxx
```

## Important limitations

- Stage A-C supports `dp=1,tp=1,pp=1` only.
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

See `archive/initial_smolvla_nanotron_smoke/RESULTS.md` for the original result details.

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

## Verified Stage C1 result

Checkpoint save/resume was verified on the same AutoDL RTX 3090 host. The 50-step PushT run saved checkpoints at steps 25 and 50:

```text
step=25 loss=0.981450
checkpoint_saved path=outputs/pusht_1gpu/step_000025.pt
...
step=50 loss=1.168826
checkpoint_saved path=outputs/pusht_1gpu/step_000050.pt
```

The resume config loaded `outputs/pusht_1gpu/step_000050.pt` and continued training:

```text
checkpoint_resumed path=outputs/pusht_1gpu/step_000050.pt step=50
step=51 loss=0.723549
step=52 loss=0.971373
```

Each checkpoint was about 486 MB, and `/root/autodl-tmp` usage was 66% after the run.

## Verified Stage C2a result

W&B metric logging was verified on AutoDL in offline mode with the 5-step PushT smoke config. The run wrote local W&B data under `wandb/offline-run-20260702_161947-v1s1bin5` and reported these metrics:

```text
step=1 loss=1.369288
step=2 loss=1.393214
step=3 loss=1.315108
step=4 loss=1.526516
step=5 loss=1.235370
wandb: Run summary:
wandb: system/disk_used_percent 65.84843
wandb: train/grad_norm 8.8125
wandb: train/loss 1.23537
wandb: train/lr 0.0001
wandb: train/samples_seen 5
wandb: train/step_time_sec 0.05516
```

The AutoDL environment had `wandb` installed, but `wandb status` showed `api_key: null`, so online sync requires `wandb login` first.
