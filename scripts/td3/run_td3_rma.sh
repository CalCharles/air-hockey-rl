#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=python
fi

CONFIG="configs/td3/zeroshot_paramrand/td3_rma_pm25.yaml"
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(41 42 43 44 45)
fi

for seed in "${SEEDS[@]}"; do
    run_name="td3_rma_pm25_seed${seed}"
    "$PYTHON" -m scripts.td3.td3_training_dr \
        --args-file "$CONFIG" \
        --params-cache-path "saved/${run_name}" \
        --sbatch \
        --sbatch-run-name "$run_name" \
        --sbatch-partition gh \
        --sbatch-time 12:00:00 \
        --run-name "$run_name" \
        --seed "$seed"
done
