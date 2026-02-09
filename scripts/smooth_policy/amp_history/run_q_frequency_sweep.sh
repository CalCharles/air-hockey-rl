#!/bin/bash
# Run SAC training with different Q-network update frequencies on different CUDA devices
#
# Usage:
#   ./run_q_frequency_sweep.sh [additional args for python script]
#
# Examples:
#   ./run_q_frequency_sweep.sh --args-file configs/sac/sac_puck_juggle.yaml
#   ./run_q_frequency_sweep.sh --args-file configs/sac/sac_amp.yaml --total-timesteps 500000
#   ./run_q_frequency_sweep.sh --args-file configs/sac/sac_puck_juggle.yaml --disc-reward-weight 0

# Array of Q frequencies to test
Q_FREQUENCIES=(4 8 16 32)

# Array of CUDA devices (0 to 3)
CUDA_DEVICES=(0 1 2 3)

# Capture all passed arguments
EXTRA_ARGS="$@"

# Base command
BASE_CMD="python scripts/smooth_policy/amp_history/amp_training/amp_training_lsgan_sac.py"

# Launch each experiment on a different GPU
for i in "${!Q_FREQUENCIES[@]}"; do
    Q_FREQ=${Q_FREQUENCIES[$i]}
    CUDA_DEV=${CUDA_DEVICES[$i]}
    
    echo "Starting q_frequency=$Q_FREQ on cuda:$CUDA_DEV"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEV $BASE_CMD \
        $EXTRA_ARGS \
        --q-frequency $Q_FREQ \
        --device cuda:0 \
        --run-name "q_freq_${Q_FREQ}" \
        &
    
    # Small delay to stagger process starts
    sleep 2
done

echo "All experiments launched. Use 'jobs' or 'nvidia-smi' to monitor progress."
echo "Logs will be saved to runs/default_training/..."

# Wait for all background jobs to complete
wait

echo "All experiments completed."
