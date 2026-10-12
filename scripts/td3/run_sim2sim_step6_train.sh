#!/usr/bin/env bash
# Training for the sim2sim step-6 direct-force experiment (both use direct force control and the
# canonical hist4 TD3 recipe, 3M steps, final checkpoint only; one run per seed, round-robin over GPUs):
#   oracle: domain randomization over the identify_ood_gap step-6 box
#           (configs/td3/sim2sim_step6/td3_dr_step6_direct_force_hist4.yaml)
#   source: domain randomization over +-25 % around sysid-v2, the policy that is evaluated zero-shot and
#           fine-tuned in the target envs (configs/td3/sim2sim_step6/td3_dr25_direct_force_hist4.yaml)
# Runs land in runs/td3/sim2sim_step6_direct_force/<oracle|source>/; the 5 target envs drawn from the
# step-6 box are in runs/td3/sim2sim_step6_direct_force/envs.json.
#
#   bash scripts/td3/run_sim2sim_step6_train.sh oracle                 # seeds 0 1 2 on GPUs 0 1
#   SEEDS="0 1 2" GPUS="2 3" bash scripts/td3/run_sim2sim_step6_train.sh source
#
# Extra arguments go to td3_training_dr (e.g. --total-timesteps 1000000).
set -euo pipefail

cd "$(dirname "$0")/../.."
KIND=${1:?usage: run_sim2sim_step6_train.sh oracle|source [extra td3_training_dr args]}
shift
case "$KIND" in
    oracle) ARGS_FILE=configs/td3/sim2sim_step6/td3_dr_step6_direct_force_hist4.yaml; NAME=dr_step6_direct_force_hist4 ;;
    source) ARGS_FILE=configs/td3/sim2sim_step6/td3_dr25_direct_force_hist4.yaml; NAME=dr25_direct_force_hist4 ;;
    *) echo "unknown kind '$KIND' (oracle|source)" >&2; exit 1 ;;
esac
ROOT=runs/td3/sim2sim_step6_direct_force/$KIND
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a GPUS <<< "${GPUS:-0 1}"
mkdir -p "$ROOT"

i=0
for S in "${SEEDS[@]}"; do
    RUN=${NAME}_seed${S}
    DIR=$ROOT/$RUN
    if [[ -f $DIR/model.pth ]]; then
        echo "[skip ] $RUN: finished (model.pth exists)"
        continue
    fi
    if [[ -e $DIR ]]; then
        echo "[stop ] $DIR exists but is unfinished; move it away first" >&2
        exit 1
    fi
    GPU=${GPUS[$(( i % ${#GPUS[@]} ))]}; i=$((i + 1))
    nohup .venv/bin/python -u -m scripts.td3.td3_training_dr --args-file "$ARGS_FILE" \
        --seed "$S" --run-name "$RUN" --device "cuda:$GPU" --log-parent-dir "$DIR" "$@" \
        > "$DIR.stdout.log" 2>&1 &
    echo "launched $RUN on cuda:$GPU (pid $!): tail -f $DIR.stdout.log"
done
echo "TensorBoard: tensorboard --logdir $ROOT"
