#!/usr/bin/env bash
set -euo pipefail

ARGS_FILE="scripts/smooth_policy/amp_history/configs/td3/td3_no_alignment.yaml"
MOTION_WEIGHT=0.01
BASE_LOG_DIR="runs/td3/updated_training/density_sweep"

# Base density = 3000 (sysid canonical paddle_density).
# 1:1 ratio: puck_density == paddle_density for all runs.
# Multipliers: 0.75x, 1x, 1.25x, 1.5x, 2x
declare -a MULTIPLIERS=(0.75 1.0 1.25 1.5 2.0)
declare -a DENSITIES=(2250 3000 3750 4500 6000)
declare -a LABELS=("d2250" "d3000" "d3750" "d4500" "d6000")
declare -a GPUS=(1 2 3)

NUM_GPUS=${#GPUS[@]}

echo "=== Density Sweep: ${#DENSITIES[@]} runs across GPUs ${GPUS[*]} ==="

batch_pids=()

for i in "${!DENSITIES[@]}"; do
  gpu_idx=$(( i % NUM_GPUS ))
  device="cuda:${GPUS[$gpu_idx]}"
  density=${DENSITIES[$i]}
  label=${LABELS[$i]}

  log_dir="${BASE_LOG_DIR}/${label}"
  run_name="mw001_${label}"

  echo "Launching density=${density} on ${device} -> ${log_dir}"
  python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
    --args-file "${ARGS_FILE}" \
    --motion-reward-weight ${MOTION_WEIGHT} \
    --paddle-density ${density} \
    --puck-density ${density} \
    --enable-puck-delay-interpolation True \
    --device "${device}" \
    --log-parent-dir "${log_dir}" \
    --run-name "${run_name}" &

  batch_pids+=($!)

  # When a full batch fills the GPUs, wait before launching the next batch
  if (( (i + 1) % NUM_GPUS == 0 )); then
    echo "--- Waiting for batch (PIDs: ${batch_pids[*]}) ---"
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    batch_pids=()
  fi
done

# Wait for any remaining jobs in the final partial batch
if (( ${#batch_pids[@]} > 0 )); then
  echo "--- Waiting for final batch (PIDs: ${batch_pids[*]}) ---"
  for pid in "${batch_pids[@]}"; do wait "$pid"; done
fi

echo "=== All density sweep runs complete ==="
