# Nanotron-SmVLA Verification Results

## Remote machine

- Provider: AutoDL / SeeTaCloud SSH endpoint
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB
- System disk after verification: `8%` used
- Data disk after verification: `64%` used
- Verification directory: `/root/autodl-tmp/nanotron-smolvla-bridge`

## Reused remote environments

- Nanotron source: `/root/autodl-tmp/nanotron-minimal/nanotron/src`
- SmolVLA / LeRobot env: `/root/autodl-tmp/smolvla-minimal/conda-py312`
- LeRobot source: `/root/autodl-tmp/smolvla-minimal/lerobot/src`
- Offline SmolVLM snapshot:
  `/root/autodl-tmp/smolvla-minimal/cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467`

## Command

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

## Output

```text
Reducing the number of VLM layers to 2 ...
SmolVLA/Nanotron minimal training: params=226,429,216, trainable=13,916,512
step=1 loss=2.707207
step=2 loss=2.689403
```

## Notes

- The first remote run failed because the machine could not reach Hugging Face to resolve processor metadata.
- The successful run used `config_offline_autodl.yaml`, pointing `vlm_model_name` to the local cached SmolVLM snapshot.
- The bf16 Nanotron initialization required explicit state/action dtype alignment and CUDA autocast around the SmolVLA forward pass.
