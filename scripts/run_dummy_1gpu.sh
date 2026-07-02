#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-configs/smolvla_dummy_1gpu.yaml}
NANOTRON_SRC=${NANOTRON_SRC:-/root/autodl-tmp/nanotron-minimal/nanotron/src}
LEROBOT_SRC=${LEROBOT_SRC:-/root/autodl-tmp/smolvla-minimal/lerobot/src}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONPATH="$PWD:$NANOTRON_SRC:$LEROBOT_SRC:${PYTHONPATH:-}"

torchrun --nproc_per_node=1 -m nanotron_smolvla.train \
  --config-file "$CONFIG_FILE" \
  --nanotron-src "$NANOTRON_SRC" \
  --lerobot-src "$LEROBOT_SRC"
