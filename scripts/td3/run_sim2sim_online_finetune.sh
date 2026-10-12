#!/usr/bin/env bash
# Train the sim2sim in-env baselines: 5 hist4 seeds in each of the 5 sampled target envs (25 runs x 2M steps),
# all at once on GPUs 0 and 1. Runs in the background; follow the log below. See the header of
# scripts/td3/sim2sim_online_finetune.py for the whole experiment.
#
#   bash scripts/td3/run_sim2sim_online_finetune.sh            # launch
#   bash scripts/td3/run_sim2sim_online_finetune.sh --dry-run  # show envs + commands only
#
# Extra arguments go to the python script (e.g. --jobs-per-gpu 7, --total-timesteps 1000000).
# Re-running skips finished runs. Afterwards:
#   .venv/bin/python -m scripts.td3.sim2sim_online_finetune --zero-shot-and-online-fine-tune --gpus 0 1
#   .venv/bin/python -m scripts.td3.sim2sim_online_finetune --results
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=runs/td3/sim2sim_online_finetune
mkdir -p "$ROOT"
LOG="$ROOT/train_launcher_$(date +%Y%m%d-%H%M%S).log"

if [[ " $* " == *" --dry-run "* ]]; then
    exec .venv/bin/python -m scripts.td3.sim2sim_online_finetune --train --root "$ROOT" --gpus 0 1 "$@"
fi

nohup .venv/bin/python -u -m scripts.td3.sim2sim_online_finetune --train --root "$ROOT" --gpus 0 1 "$@" > "$LOG" 2>&1 &
echo "launched (pid $!): 25 hist4 runs on GPUs 0 and 1"
echo "  launcher log : tail -f $LOG"
echo "  per-run logs : $ROOT/train/env<E>/sim2sim_hist4_env<E>_seed<S>.stdout.log"
echo "  tensorboard  : tensorboard --logdir $ROOT/train"
