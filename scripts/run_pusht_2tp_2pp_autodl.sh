#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/smolvla_pusht_2tp_2pp_autodl.yaml}"
export NANOTRON_SRC="${NANOTRON_SRC:-/root/autodl-tmp/nanotron-minimal/nanotron/src}"
export LEROBOT_SRC="${LEROBOT_SRC:-/root/autodl-tmp/smolvla-minimal/lerobot/src}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="${PWD}:${NANOTRON_SRC}:${LEROBOT_SRC}:${PYTHONPATH:-}"
export WANDB_DIR="${WANDB_DIR:-${PWD}/wandb}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29641}"

torchrun --nproc_per_node=4 \
  -m nanotron_smolvla.train \
  --config-file "${CONFIG_PATH}" \
  --nanotron-src "${NANOTRON_SRC}" \
  --lerobot-src "${LEROBOT_SRC}"