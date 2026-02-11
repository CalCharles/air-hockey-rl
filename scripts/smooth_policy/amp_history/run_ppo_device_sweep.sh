#!/bin/bash
# Run PPO AMP training across multiple CUDA devices.
#
# Usage:
#   ./run_ppo_device_sweep.sh --args-file <path/to/args.yaml> [additional args]
#
# Examples:
#   ./run_ppo_device_sweep.sh --args-file scripts/smooth_policy/amp_history/configs/pid/amp_better_reward.yaml
#   ./run_ppo_device_sweep.sh --args-file scripts/smooth_policy/amp_history/configs/pid/amp_better_reward.yaml --num-iterations 300 --learning-rate 5e-5

set -euo pipefail

# Array of CUDA devices to sweep
CUDA_DEVICES=(0 1 2 3)

# Capture all passed arguments
EXTRA_ARGS=("$@")

# Require --args-file so each run starts from a config file.
if [[ ! " $* " =~ [[:space:]]--args-file[[:space:]] ]]; then
    echo "Error: --args-file is required."
    echo "Usage: $0 --args-file <path/to/args.yaml> [additional args]"
    exit 1
fi

# Respect user-provided run names; otherwise create per-device defaults.
HAS_RUN_NAME=false
if [[ " $* " =~ [[:space:]]--run-name[[:space:]] ]]; then
    HAS_RUN_NAME=true
fi

# Base command for PPO AMP training
BASE_CMD=(python scripts/smooth_policy/amp_history/amp_training/amp_training_lsgan.py)

# Launch one run per GPU
for CUDA_DEV in "${CUDA_DEVICES[@]}"; do
    echo "Starting PPO AMP training on cuda:${CUDA_DEV}"

    if [[ "${HAS_RUN_NAME}" == true ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${BASE_CMD[@]}" \
            "${EXTRA_ARGS[@]}" \
            --device cuda:0 \
            &
    else
        CUDA_VISIBLE_DEVICES="${CUDA_DEV}" "${BASE_CMD[@]}" \
            "${EXTRA_ARGS[@]}" \
            --device cuda:0 \
            --run-name "ppo_cuda_${CUDA_DEV}" \
            &
    fi

    # Small delay to stagger process starts
    sleep 2
done

echo "All experiments launched. Use 'jobs' or 'nvidia-smi' to monitor progress."
echo "Logs will be saved to runs/default_training/..."

# Wait for all background jobs to complete
wait

echo "All experiments completed."
