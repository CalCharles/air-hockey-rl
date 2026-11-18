#!/usr/bin/env bash
# Train the clean-sim sustained-juggling recipe (the winning `long_discount` trial of
# docs/juggling-iterations.md: clean_upper args, gamma 0.995, 1M steps, actor warm-started
# from latest_models/canonical/hist2_motion0_v2) over several seeds, all at once, alternating
# GPUs 0 and 1. Runs in the background; follow the per-seed logs below.
#
#   bash scripts/td3/run_clean_juggle_seeds.sh             # launch seeds 0-4
#   bash scripts/td3/run_clean_juggle_seeds.sh --dry-run   # show commands only
#
# Override with env vars: SEEDS="0 1 2" GPUS="0 1" ROOT=runs/td3/other bash scripts/td3/run_clean_juggle_seeds.sh
# Afterwards, test each seed on 32 fresh starts with a 20k-step cap:
#   .venv/bin/python -m scripts.td3.juggle_survival --run-dir $ROOT/seed<S> --scenario-set clean \
#     --episodes 32 --horizon 20000 --seed-base 200000 --out-dir $ROOT/seed<S>/final_clean
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=${ROOT:-runs/td3/clean_juggle_g0995}
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1

mkdir -p "$ROOT"
for i in "${!SEED_LIST[@]}"; do
    seed=${SEED_LIST[$i]}
    gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
    if [[ -e "$ROOT/seed$seed" ]]; then
        echo "skip seed $seed: $ROOT/seed$seed already exists"
        continue
    fi
    cmd=(.venv/bin/python -u -m scripts.td3.td3_training
         --args-file configs/td3/sustained/clean_upper.yaml
         --total-timesteps 1000000 --gamma 0.995 --seed "$seed"
         --run-name "clean_juggle_g0995_seed$seed"
         --log-parent-dir "$ROOT/seed$seed")
    if (( DRY_RUN )); then
        echo "CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
        continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/airhockey-mpl \
        nohup "${cmd[@]}" > "$ROOT/seed$seed.log" 2>&1 &
    echo "launched seed $seed on GPU $gpu (pid $!): tail -f $ROOT/seed$seed.log"
done
(( DRY_RUN )) || echo "tensorboard: tensorboard --logdir $ROOT"
