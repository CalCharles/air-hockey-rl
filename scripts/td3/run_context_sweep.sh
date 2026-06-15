#!/bin/bash

source .venv/bin/activate

BASE_ARGS="configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml"

run_sweep() {
  local use_transformer=$1

  for context_len in 2 4 16 32; do
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
      --sbatch-time 12:00:00

  done
}

run_sweep true
run_sweep false