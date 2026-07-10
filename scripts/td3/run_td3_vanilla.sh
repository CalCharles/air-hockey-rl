#!/bin/bash

source /work/10993/rohanpatel01/vista/air-hockey-rl/.venv/bin/activate
# cd /work/10993/rohanpatel01/vista/air-hockey-rl-dev

cd /work/10993/rohanpatel01/vista/air-hockey-rl

BASE_ARGS="configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml"

for seed in 41 42 43 44 45; do
    
    run_name="td3_baseline_no_exploration"
    echo "Launching Run: ${run_name}"

    /work/10993/rohanpatel01/vista/air-hockey-rl/.venv/bin/python -m scripts.td3.td3_training_dr \
    --args-file "$BASE_ARGS" \
    --params_cache_path "saved/${run_name}" \
    --sbatch \
    --sbatch-run-name "$run_name" \
    --sbatch-partition gh \
    --sbatch-time 6:00:00 \
    --run_name "$run_name" \
    --seed "$seed" \
    --exploration-primitive-chance 0.0

    # Run in terminal and not via sbatch
    # /work/10993/rohanpatel01/vista/air-hockey-rl/.venv/bin/python -m scripts.td3.td3_training_dr \
    # --args-file "$BASE_ARGS" \
    # --log_parent_dir "runs/debug/${run_name}" \
    # --params_cache_path "saved/${run_name}" \
    # --eval_id_ood_out_dir "results/${run_name}" \
    # --run_name "$run_name" \
    # --seed "$seed"

done