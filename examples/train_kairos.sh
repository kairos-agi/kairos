#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-kairos/configs/kairos_4b_train_config.py}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/accelerate_zero1.yaml}"

LAUNCH_ARGS=(--config_file "${ACCELERATE_CONFIG}")
if [[ -n "${NUM_PROCESSES:-}" ]]; then
  LAUNCH_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

accelerate launch "${LAUNCH_ARGS[@]}" \
  examples/train_kairos.py \
  --config "${CONFIG}" \
  "$@"
