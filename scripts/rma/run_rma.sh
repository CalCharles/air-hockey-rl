#!/usr/bin/env bash
# Launch hora-style RMA (phase 1 PPO → phase 2 ProprioAdapt) with multi-env eval.
#
# Comment / uncomment the two launch blocks below to switch between:
#   - sbatch (cluster)
#   - local terminal smoke / full run
#
# Usage:
#   ./scripts/rma/run_rma.sh
#   ./scripts/rma/run_rma.sh 0 cuda:0 rma_paramrand_pm25
set -euo pipefail

DEVICE="${2:-cuda:0}"
RUN_NAME="${3:-rma_paramrand_pm25}"
ARGS_FILE="${ARGS_FILE:-configs/rma/rma_paramrand_pm25.yaml}"
SBATCH_TIME="${SBATCH_TIME:-16:00:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gh}"

# Same venv as the TD3 vista jobs / RMA slurm template.
VENV_PYTHON="${VENV_PYTHON:-/work/10993/rohanpatel01/vista/air-hockey-rl/.venv/bin/python}"

cd /work/10993/rohanpatel01/vista/air-hockey-rl-rma


# ---------------------------------------------------------------------------
# Cluster: submit via sbatch (patches scripts/rma/vista_template.slurm)
# ---------------------------------------------------------------------------

for seed in 1 2 3 4 5; do

  echo "Launching Run: ${RUN_NAME} (seed=${seed}, device=${DEVICE})"

  "${VENV_PYTHON}" -m scripts.rma.rma_training_dr \
    --args-file "${ARGS_FILE}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --run-name "${RUN_NAME}" \
    --sbatch \
    --sbatch-run-name "${RUN_NAME}" \
    --sbatch-partition "${SBATCH_PARTITION}" \
    --sbatch-time "${SBATCH_TIME}"

done

# ---------------------------------------------------------------------------
# Local: run in this terminal (smoke defaults — few timesteps)
# Comment out the short-budget flags (or bump them) for a full local run.
# ---------------------------------------------------------------------------
SEED=0
# "${VENV_PYTHON}" -m scripts.rma.rma_training_dr \
#   --args-file "${ARGS_FILE}" \
#   --seed "${SEED}" \
#   --device "${DEVICE}" \
#   --run-name "${RUN_NAME}" \
#   --log-parent-dir "runs/rma/debug/${RUN_NAME}" \
#   --max-agent-steps 2048 \
#   --adaptation-max-agent-steps 512 \
#   --save-frequency 1 \
#   --adaptation-save-interval 256 \
#   --num-envs 2 \
#   --horizon-length 64 \
#   --minibatch-size 64 \
#   --eval-n-envs 2 \
#   --eval-eps-per-env 1
