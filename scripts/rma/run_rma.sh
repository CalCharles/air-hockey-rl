#!/usr/bin/env bash
# Launch hora-style RMA (phase 1 PPO → phase 2 ProprioAdapt) with multi-env eval.
#
# Local:
#   ./scripts/rma/run_rma.sh [seed] [device] [run_name]
#
# Cluster (sbatch via the TD3-style helper; uses scripts/rma/vista_template.slurm):
#   SBATCH=1 ./scripts/rma/run_rma.sh [seed] [device] [run_name]
#   SBATCH_TIME=12:00:00 SBATCH=1 ./scripts/rma/run_rma.sh 0 cuda:0 rma_paramrand_pm25
set -euo pipefail

SEED="${1:-0}"
DEVICE="${2:-cuda:0}"
RUN_NAME="${3:-rma_paramrand_pm25_seed${SEED}}"
ARGS_FILE="${ARGS_FILE:-configs/rma/rma_paramrand_pm25.yaml}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gh}"

# Same venv as the TD3 vista jobs / RMA slurm template.
VENV_PYTHON="${VENV_PYTHON:-/work/10993/rohanpatel01/vista/air-hockey-rl/.venv/bin/python}"

cd /work/10993/rohanpatel01/vista/air-hockey-rl-rma

if [[ "${SBATCH:-0}" == "1" ]]; then
  "${VENV_PYTHON}" -m scripts.rma.rma_training_dr \
    --args-file "${ARGS_FILE}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --run-name "${RUN_NAME}" \
    --sbatch \
    --sbatch-run-name "${RUN_NAME}" \
    --sbatch-partition "${SBATCH_PARTITION}" \
    --sbatch-time "${SBATCH_TIME}"
else
  "${VENV_PYTHON}" -m scripts.rma.rma_training_dr \
    --args-file "${ARGS_FILE}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --run-name "${RUN_NAME}"
fi
