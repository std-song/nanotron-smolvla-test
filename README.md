# Nanotron-SmVLA Test Project

This repository contains a minimal but structured bridge for training LeRobot SmolVLA through Nanotron primitives.

## Current milestones

- Stage A: Refactor the original smoke test into a small Python project.
- Stage B: Add a real `LeRobotDataset` backend and verify single-GPU training on the PushT dataset.
- Stage D: Verify 2-way data parallel training on one 2-GPU AutoDL instance.
- Stage TP1: Verify expert-only 2-way tensor parallel training on one 2-GPU AutoDL instance.
- Stage TP2: Verify combined 2DP x 2TP expert-only training on one 4-GPU AutoDL instance.
- Stage PP0: Refactor SmolVLA training forward into pipeline-ready pieces while preserving pp=1 behavior.
- Stage PP1: Verify 2-stage pipeline parallel training, checkpoint save, and checkpoint resume on a 4-GPU AutoDL instance using 2 GPUs.

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
  smolvla_pusht_2dp_autodl.yaml
  smolvla_pusht_2tp_autodl.yaml
  smolvla_pusht_2tp_resume_autodl.yaml
  smolvla_pusht_2dp_2tp_autodl.yaml
  smolvla_pusht_2dp_2tp_resume_autodl.yaml
  smolvla_pusht_pp0_1gpu_autodl.yaml
  smolvla_pusht_pp0_2tp_autodl.yaml
  smolvla_pusht_pp1_2pp_autodl.yaml
  smolvla_pusht_pp1_2pp_50step_autodl.yaml
  smolvla_pusht_pp1_2pp_50step_ckpt_autodl.yaml
  smolvla_pusht_pp1_2pp_resume_autodl.yaml
scripts/
  run_dummy_1gpu.sh
  run_pusht_1gpu_autodl.sh
  run_pusht_2dp_autodl.sh
  run_pusht_2tp_autodl.sh
  run_pusht_2dp_2tp_autodl.sh
  run_pusht_pp1_2pp_autodl.sh
  inspect_smolvla_topology.py
docs/
  TP_PP_PLAN.md
  TOPOLOGY_PUSHT_2L.md
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



## Stage D: 2DP smoke test

The 2DP launcher uses `torchrun --nproc_per_node=2` with `dp=2,tp=1,pp=1`. The LeRobot dataloader uses a `DistributedSampler` so each DP rank reads a shard, while rank 0 owns W&B logging and checkpoint writes.

```bash
cd /root/autodl-tmp/nanotron-smolvla-project
bash scripts/run_pusht_2dp_autodl.sh configs/smolvla_pusht_2dp_autodl.yaml
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

- Stage A-C supports `dp=1,tp=1,pp=1`; Stage D verifies `dp=2,tp=1,pp=1`.
- Expert-only TP is verified for `dp=1,tp=2,pp=1` and `dp=2,tp=2,pp=1`.
- PP1 is verified for `dp=1,tp=1,pp=2`; TP+PP composition is still the next stage.
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

## Verified Stage D result

2-way data parallel training was verified on a cloned AutoDL instance with two RTX 3090 GPUs, driver/CUDA compatible with `torch 2.11.0+cu130`. The stable run used `configs/smolvla_pusht_2dp_autodl.yaml` and `scripts/run_pusht_2dp_autodl.sh`.

```text
Nanotron-SmVLA training: data=lerobot dp=2, tp=1, pp=1, params=226,429,216, trainable=13,916,512
step=25 loss=1.031807
checkpoint_saved path=outputs/pusht_2dp/step_000025.pt
...
step=50 loss=1.095483
checkpoint_saved path=outputs/pusht_2dp/step_000050.pt
wandb: Run summary:
wandb: checkpoint/step 50
wandb: train/grad_norm 6.75
wandb: train/loss 1.09548
wandb: train/samples_seen 100
```

The resume config loaded the 2DP checkpoint and continued training:

```text
checkpoint_resumed path=outputs/pusht_2dp/step_000050.pt step=50
step=51 loss=0.753152
step=52 loss=1.073192
wandb: train/samples_seen 104
```

Each 2DP checkpoint was about 486 MB. `/root/autodl-tmp` usage was 68% after the stable and resume runs.
The DP path uses project-local gradient synchronization over trainable parameters only. Missing gradients are treated as zero, which avoids all-reducing frozen or conditionally unused SmolVLA parameters.

## Next stage: 2TP planning

The 2TP/2PP route is documented in `docs/TP_PP_PLAN.md`, with the current two-layer PushT topology recorded in `docs/TOPOLOGY_PUSHT_2L.md`. Before replacing any SmolVLA internals with Nanotron tensor-parallel layers, inspect the real module topology on AutoDL:

```bash
cd /root/autodl-tmp/nanotron-smolvla-project
python scripts/inspect_smolvla_topology.py \
  --config-file configs/smolvla_pusht_2dp_autodl.yaml \
  --nanotron-src /root/autodl-tmp/nanotron-minimal/nanotron/src \
  --lerobot-src /root/autodl-tmp/smolvla-minimal/lerobot/src \
  --max-lines 300
```

## Verified Stage TP1 result

Expert-only tensor parallel training was verified on the same two-GPU AutoDL CUDA 13 instance with `dp=1,tp=2,pp=1`. The first TP implementation shards trainable expert `nn.Linear` weights while preserving full SmolVLA hidden shapes for the existing LeRobot forward path.

```text
Nanotron-SmVLA training: data=lerobot dp=1, tp=2, pp=1, params=220,290,336, trainable=7,777,632, expert_tp=True
step=1 loss=1.368408
step=2 loss=1.368272
step=3 loss=1.324518
step=4 loss=1.532223
step=5 loss=1.279573
wandb: Run summary:
wandb: system/disk_used_percent 67.74519
wandb: train/grad_norm 8.0625
wandb: train/loss 1.27957
wandb: train/samples_seen 5
```

The W&B offline run was written to `/root/autodl-tmp/nanotron-smolvla-project/wandb/wandb/offline-run-20260703_102901-zrhjfdk0`.

## Verified Stage TP1 stability result

The expert-only 2TP path was extended to 50 PushT steps with TP-sharded checkpoint save/resume. Because TP ranks hold different parameter shards, TP checkpoints are saved per world rank:

```text
outputs/pusht_2tp/step_000050_rank_000.pt 450M
outputs/pusht_2tp/step_000050_rank_001.pt 450M
```

The 50-step run produced:

```text
step=25 loss=0.966962
checkpoint_saved path=outputs/pusht_2tp/step_000025.pt
checkpoint_saved_tp_shards pattern=step_000025_rank_*.pt
...
step=50 loss=1.173062
checkpoint_saved path=outputs/pusht_2tp/step_000050.pt
checkpoint_saved_tp_shards pattern=step_000050_rank_*.pt
wandb: train/samples_seen 50
wandb: system/disk_used_percent 69.50324
```

The resume config loaded `outputs/pusht_2tp/step_000050.pt`, which resolves to each rank's local shard file, then continued training:

```text
checkpoint_resumed path=outputs/pusht_2tp/step_000050.pt step=50
step=51 loss=0.690763
step=52 loss=0.956361
wandb: system/disk_used_percent 71.26103
```

## Verified Stage TP2 result

Combined data parallel and tensor parallel training was verified on a four-GPU AutoDL CUDA 13 instance with `dp=2,tp=2,pp=1`. Nanotron groups DP ranks by matching TP shard, so DP gradient synchronization averages rank pairs that own the same expert shard.

The 5-step smoke produced:

```text
Nanotron-SmVLA training: data=lerobot dp=2, tp=2, pp=1, params=220,290,336, trainable=7,777,632, expert_tp=True
step=1 loss=1.406795
step=2 loss=1.425695
step=3 loss=1.378253
step=4 loss=1.493057
step=5 loss=1.241209
wandb: train/samples_seen 10
```

The 50-step run with checkpointing produced:

```text
step=25 loss=1.032271
checkpoint_saved path=outputs/pusht_2dp_2tp/step_000025.pt
checkpoint_saved_tp_shards pattern=step_000025_rank_*.pt
...
step=50 loss=1.096941
checkpoint_saved path=outputs/pusht_2dp_2tp/step_000050.pt
checkpoint_saved_tp_shards pattern=step_000050_rank_*.pt
wandb: train/samples_seen 100
wandb: system/disk_used_percent 74.9023
```

The four rank-local checkpoint shards were created as expected:

```text
outputs/pusht_2dp_2tp/step_000050_rank_000.pt 450M
outputs/pusht_2dp_2tp/step_000050_rank_001.pt 450M
outputs/pusht_2dp_2tp/step_000050_rank_002.pt 450M
outputs/pusht_2dp_2tp/step_000050_rank_003.pt 450M
```

The resume config loaded `outputs/pusht_2dp_2tp/step_000050.pt`, which resolves to each world rank's shard file, then continued training:

```text
checkpoint_resumed path=outputs/pusht_2dp_2tp/step_000050.pt step=50
step=51 loss=0.727421
step=52 loss=1.068907
wandb: train/samples_seen 104
wandb: system/disk_used_percent 78.292
```

## Verified Stage PP0 result

PP0 refactored the SmolVLA training path into pipeline-ready pieces while keeping a single Nanotron `PipelineBlock` and `pp=1`. The extracted helper `run_vlm_with_expert_layer_range(...)` currently runs the full layer range; PP1 can split that range across pipeline stages.

The 1GPU PP0 smoke matched the previous single-GPU loss trace:

```text
Nanotron-SmVLA training: data=lerobot dp=1, tp=1, pp=1, params=226,429,216, trainable=13,916,512, expert_tp=False
step=1 loss=1.369288
step=2 loss=1.393214
step=3 loss=1.315108
step=4 loss=1.526516
step=5 loss=1.235370
```

The 2TP PP0 smoke matched the previous expert-only 2TP loss trace:

```text
Nanotron-SmVLA training: data=lerobot dp=1, tp=2, pp=1, params=220,290,336, trainable=7,777,632, expert_tp=True
step=1 loss=1.368408
step=2 loss=1.368272
step=3 loss=1.324518
step=4 loss=1.532223
step=5 loss=1.279573
```

## Verified Stage PP1 result

PP1 splits the two-layer SmolVLA smoke backbone into two Nanotron `PipelineBlock`s. Stage 0 performs flow input sampling, prefix/suffix embedding, attention mask construction, and layer range `[0, 1)`. Stage 1 receives the hidden states through Nanotron PP communication, runs layer range `[1, end)`, applies final norms, and computes the flow-matching loss.

The 5-step smoke was verified on the four-GPU AutoDL CUDA 13 instance using two RTX 3090 GPUs:

```text
Nanotron-SmVLA training: data=lerobot dp=1, tp=1, pp=2, params=226,429,216, trainable=13,916,512, expert_tp=False
step=1 loss=1.369288
step=2 loss=1.393000
step=3 loss=1.330252
step=4 loss=1.538532
step=5 loss=1.261307
```

Run command:

```bash
cd /root/autodl-tmp/nanotron-smolvla-project
bash scripts/run_pusht_pp1_2pp_autodl.sh configs/smolvla_pusht_pp1_2pp_autodl.yaml
```

## Verified Stage PP1 stability and checkpoint result

The 50-step PP1 stability config completed successfully:

```text
step=25 loss=0.979418
...
step=50 loss=1.160684
EXIT_CODE:0
wandb: train/samples_seen 50
wandb: system/disk_used_percent 78.29293
```

PP checkpointing saves rank-local shards, because each pipeline rank owns a different stage module:

```text
checkpoint_saved path=outputs/pusht_pp1_2pp/step_000025.pt
checkpoint_saved_rank_shards pattern=step_000025_rank_*.pt
checkpoint_saved path=outputs/pusht_pp1_2pp/step_000050.pt
checkpoint_saved_rank_shards pattern=step_000050_rank_*.pt
```

The verified shard files were:

```text
outputs/pusht_pp1_2pp/step_000025_rank_000.pt 433M
outputs/pusht_pp1_2pp/step_000025_rank_001.pt 433M
outputs/pusht_pp1_2pp/step_000050_rank_000.pt 433M
outputs/pusht_pp1_2pp/step_000050_rank_001.pt 433M
```

The resume config loaded each rank's local shard from `outputs/pusht_pp1_2pp/step_000050.pt` and continued training:

```text
checkpoint_resumed path=outputs/pusht_pp1_2pp/step_000050.pt step=50
step=51 loss=0.734147
step=52 loss=1.026340
EXIT_CODE:0
wandb: system/disk_used_percent 81.66815
```

Run commands:

```bash
cd /root/autodl-tmp/nanotron-smolvla-project
bash scripts/run_pusht_pp1_2pp_autodl.sh configs/smolvla_pusht_pp1_2pp_50step_autodl.yaml
bash scripts/run_pusht_pp1_2pp_autodl.sh configs/smolvla_pusht_pp1_2pp_50step_ckpt_autodl.yaml
bash scripts/run_pusht_pp1_2pp_autodl.sh configs/smolvla_pusht_pp1_2pp_resume_autodl.yaml
```
