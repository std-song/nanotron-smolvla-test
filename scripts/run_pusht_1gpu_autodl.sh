#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-configs/smolvla_pusht_1gpu_autodl.yaml}
NANOTRON_SRC=${NANOTRON_SRC:-/root/autodl-tmp/nanotron-minimal/nanotron/src}
LEROBOT_SRC=${LEROBOT_SRC:-/root/autodl-tmp/smolvla-minimal/lerobot/src}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/smolvla-minimal/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/autodl-tmp/smolvla-minimal/cache/huggingface/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/root/autodl-tmp/smolvla-minimal/cache/huggingface/transformers}
export TORCH_HOME=${TORCH_HOME:-/root/autodl-tmp/smolvla-minimal/cache/torch}
export TMPDIR=${TMPDIR:-/root/autodl-tmp/smolvla-minimal/tmp}
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$PWD:$NANOTRON_SRC:$LEROBOT_SRC:${PYTHONPATH:-}"

torchrun --nproc_per_node=1 -m nanotron_smolvla.train \
  --config-file "$CONFIG_FILE" \
  --nanotron-src "$NANOTRON_SRC" \
  --lerobot-src "$LEROBOT_SRC"
