#!/usr/bin/env bash
# 5 seeds x {paper CRL, c0 switches} on GPUs 0/1, one process per run (CUDA-graph updates).
# Logs: runs/crl/<name>_s<seed>/train.log ; TensorBoard in the same directories.
set -euo pipefail
cd "$(dirname "$0")/../.."
launch() {  # preset seed gpu
  local dir="runs/crl/crl_$1_s$2"
  mkdir -p "$dir"
  nohup .venv/bin/python -m scripts.goal_conditioned_rl.train_crl --config "configs/crl/juggle_crl_$1.yaml" \
    --seed "$2" --device "cuda:$3" --run-dir "$dir" > "$dir/train.log" 2>&1 &
  echo "$1 seed $2 -> cuda:$3 pid $!"
}
launch paper 0 0; launch paper 1 0; launch paper 2 0; launch c0 0 0; launch c0 1 0
launch paper 3 1; launch paper 4 1; launch c0 2 1; launch c0 3 1; launch c0 4 1
