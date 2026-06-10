#!/bin/bash

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1
echo "Successfully restricted session to GPUs: $CUDA_VISIBLE_DEVICES"

BASE_ARGS="configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml"

run_sweep() {
  local use_transformer=$1
  local device=$2

  #   # 2 4 16 
  for context_len in 2 4 16 32; do
    run_name="sweep_transformer_${use_transformer}_ctx_${context_len}"
    log_dir="runs/td3/zeroshot_paramrand/sweeps/${run_name}/seed0"

    # Build the transformer flag conditionally
    if [ "$use_transformer" = "true" ]; then
      transformer_flag="--use-transformer"
    else
      transformer_flag="--no-use-transformer"
    fi

    .venv/bin/python -m cProfile -o "profile_${run_name}.out" \
      -m scripts.td3.td3_training_dr \
      --args-file "$BASE_ARGS" \
      $transformer_flag \
      --context-len "$context_len" \
      --log-parent-dir "$log_dir" \
      --run-name "$run_name" \
      --device "$device"
  done
}

run_sweep true cuda:0 &
run_sweep false cuda:1 &

wait