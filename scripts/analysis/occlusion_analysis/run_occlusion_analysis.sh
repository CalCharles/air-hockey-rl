#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
DATA_DIR="${1:-/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/}"
RUN_TAG="${2:-}"
SAMPLING_MODE="${3:-first}"
NUM_TRAJ="${4:-40}"

cd "${REPO_ROOT}"

ANALYZER="scripts/analysis/occlusion_analysis/analyze_occlusion_patterns.py"
PLOTTER="scripts/analysis/occlusion_analysis/plot_occlusion_results.py"
OUTPUT_ROOT="scripts/analysis/occlusion_analysis/output"

RUN_ARGS=(
  --data-dir "${DATA_DIR}"
  --num-trajectories "${NUM_TRAJ}"
  --sampling-mode "${SAMPLING_MODE}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ -n "${RUN_TAG}" ]]; then
  RUN_ARGS+=(--run-tag "${RUN_TAG}")
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
  "${PYTHON}" "${ANALYZER}" "${RUN_ARGS[@]}"
else
  uv run python "${ANALYZER}" "${RUN_ARGS[@]}"
fi

if [[ -n "${RUN_TAG}" ]]; then
  OUT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/${RUN_TAG}"
else
  OUT_DIR="$(ls -td "${REPO_ROOT}/${OUTPUT_ROOT}"/* | head -n 1)"
fi

if [[ -x ".venv/bin/python" ]]; then
  "${PYTHON}" "${PLOTTER}" --output-dir "${OUT_DIR}"
else
  uv run python "${PLOTTER}" --output-dir "${OUT_DIR}"
fi

echo "Done. Results are in: ${OUT_DIR}"

