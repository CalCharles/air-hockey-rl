#!/bin/bash

source .venv/bin/activate

BASE_ARGS="configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml"

run_sweep() {
  local use_transformer=$1

  # Submit seed: 5 when more are finished bc we can only submit 40 jobs at a time
  # 2 4 6 8 10 12 14 16 32
  # 1 2 3 4

  for context_len in 2 4 6 8 10 12 14 16 32; do
    for seed in 1 2; do
    
      run_name="sweep_transformer_${use_transformer}_ctx_${context_len}"

      if [ "$use_transformer" = "true" ]; then
        transformer_flag="--use-transformer"
      else
        transformer_flag="--no-use-transformer"
      fi

      .venv/bin/python -m scripts.td3.td3_training_dr \
        --args-file "$BASE_ARGS" \
        $transformer_flag \
        --use-history \
        --context-len "$context_len" \
        --sbatch \
        --sbatch-run-name "$run_name" \
        --sbatch-partition gh \
        --sbatch-time 12:00:00 \
        --run_name "$run_name" \
        --seed "$seed"

    done
  done
}

# TODO: Note that we're sweeping over use_transformer={true, false} as well
# run_sweep true
run_sweep false