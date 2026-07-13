#!/usr/bin/env bash
# Launch hora-style RMA (phase 1 PPO → phase 2 ProprioAdapt) with multi-env eval.
# Usage: ./scripts/rma/run_rma.sh [seed] [device] [run_name]
set -euo pipefail

SEED="${1:-0}"
DEVICE="${2:-cuda:0}"
RUN_NAME="${3:-rma_paramrand_pm25_seed${SEED}}"
ARGS_FILE="${ARGS_FILE:-configs/rma/rma_paramrand_pm25.yaml}"

python -m scripts.rma.rma_training_dr \
  --args-file "${ARGS_FILE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --run-name "${RUN_NAME}"
