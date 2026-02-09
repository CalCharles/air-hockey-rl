#!/bin/bash

# Loop through cuda devices 0 to 3 and start them in parallel
for i in {0..3}
do
    echo "Starting training on cuda:$i"
    python scripts/smooth_policy/amp_history/amp_training/amp_training_lsgan.py --args-file scripts/smooth_policy/amp_history/configs/pid/amp_better_reward.yaml \
        --device cuda:$i &
    
    # Store the process ID
    pids[$i]=$!
done

echo "All training jobs launched in parallel!"
echo "Process IDs: ${pids[@]}"

# Wait for all background processes to complete
for i in {0..3}
do
    wait ${pids[$i]}
    echo "Training on cuda:$i completed with exit code: $?"
done

echo "All training runs completed!"
