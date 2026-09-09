#!/bin/bash

source .venv/bin/activate

BASE_ARGS="configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml"

# Sweep over (batch_size, q_updates, actor_updates_per_iteration)
# Inverse relationship: larger batch → fewer updates per episode end
# Current config: batch=512, q_updates=25, actor_updates=6
CONFIGS=(
  #  batch  q_updates  actor_updates
#   "512      25         6"    # baseline — current config
  "1024     12         3"    # 2x batch, ~half updates
  "2048     6          2"    # 4x batch, ~quarter updates
  "4096     3          1"    # 8x batch, minimal updates
)

for config in "${CONFIGS[@]}"; do
  read -r batch_size q_updates actor_updates <<< "$config"

  run_name="sweep_bs${batch_size}_q${q_updates}_a${actor_updates}"

  echo "Submitting: batch=$batch_size q_updates=$q_updates actor_updates=$actor_updates"

  .venv/bin/python -m scripts.td3.td3_training_dr \
    --args-file "$BASE_ARGS" \
    --batch-size "$batch_size" \
    --q-updates "$q_updates" \
    --actor-updates-per-iteration "$actor_updates" \
    --sbatch \
    --sbatch-run-name "$run_name" \
    --sbatch-partition gh \
    --sbatch-time 12:00:00

done