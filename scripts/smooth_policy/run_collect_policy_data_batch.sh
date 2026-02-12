#!/usr/bin/env bash
set -euo pipefail

# Batch-run smooth policy data collection for a fixed timestep budget.
# For each configured run directory:
#   - resolves the model checkpoint
#   - finds a nearby config.yaml
#   - infers hidden size from model weights
#   - calls collect_policy_data.py with --total-timesteps only

REPO_ROOT="/home/air-hockey/daliu/air-hockey-rl"
COLLECT_SCRIPT="${REPO_ROOT}/scripts/smooth_policy/collect_policy_data.py"
TIMESTEPS=20000
DEVICE="cpu"
OUT_ROOT="${REPO_ROOT}/runs/policy_data_collection/batch_20000"

if [[ -d "${REPO_ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[FATAL] Could not find python or python3 in PATH."
  exit 1
fi

mkdir -p "${OUT_ROOT}"

declare -a RUN_DIRS=(
  # AMP
  "runs/amp_with_actions/ppo/test_with_history_scale_fixed/runr1/checkpoint_300"
  "runs/amp_with_actions/ppo/test_with_history/run/checkpoint_480"
  "runs/amp_training/pid/test/checkpoint_610"
  "runs/amp_training/pid/testr1/checkpoint_630"
  "runs/sac_amp/test/test_runs"

  # Non-AMP
  "runs/sac_test/pid/amp"
  "runs/sac_test/pid/first_puck_juggle"
  "runs/sac_test/pid/second_task_only/checkpoint_570000"
  "runs/sac_test/pid/third_juggle_no_autotune"
  "runs/sac_training/puck_juggle_purer1"
  "runs/sac_training/puck_juggle_purer2"
  "runs/sac_training/puck_juggle_purer3/checkpoint_960000"
  "runs/sac_training/puck_juggle_purer4/checkpoint_980000"
)

resolve_model_path() {
  local run_dir="$1"
  local model_path=""
  local latest_ckpt=""

  if [[ -f "${run_dir}/model.pth" ]]; then
    model_path="${run_dir}/model.pth"
  else
    latest_ckpt="$(find "${run_dir}" -maxdepth 1 -type d -name "checkpoint_*" | sort -V | tail -n 1 || true)"
    if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}/model.pth" ]]; then
      model_path="${latest_ckpt}/model.pth"
    fi
  fi

  if [[ -z "${model_path}" ]]; then
    return 1
  fi
  printf "%s\n" "${model_path}"
}

resolve_config_path() {
  local start_dir="$1"
  local cur="${start_dir}"
  local cfg=""
  local parent=""
  local _i=0

  for _i in $(seq 1 10); do
    if [[ -f "${cur}/config.yaml" ]]; then
      cfg="${cur}/config.yaml"
      break
    fi
    parent="$(dirname "${cur}")"
    if [[ "${parent}" == "${cur}" ]]; then
      break
    fi
    cur="${parent}"
  done

  if [[ -z "${cfg}" ]]; then
    return 1
  fi
  printf "%s\n" "${cfg}"
}

infer_hidden_size() {
  local model_path="$1"
  "${PYTHON_BIN}" - "${model_path}" <<'PY'
import sys
import torch

model_path = sys.argv[1]
state = torch.load(model_path, map_location="cpu")

for key in ("actor.0.weight", "actor_mean.0.weight"):
    if key in state:
        print(int(state[key].shape[0]))
        break
else:
    raise KeyError(
        f"Could not infer hidden size from known keys in {model_path}. "
        "Expected one of: actor.0.weight, actor_mean.0.weight"
    )
PY
}

safe_name() {
  local path="$1"
  # Keep names unique and readable in output directories.
  printf "%s\n" "${path#runs/}" | tr '/\\' '_'
}

declare -a FAILED=()

for rel_dir in "${RUN_DIRS[@]}"; do
  abs_dir="${REPO_ROOT}/${rel_dir}"

  if [[ ! -d "${abs_dir}" ]]; then
    echo "[SKIP] missing directory: ${rel_dir}"
    FAILED+=("${rel_dir} (missing directory)")
    continue
  fi

  echo "=================================================="
  echo "[RUN] ${rel_dir}"

  if ! model_path="$(resolve_model_path "${abs_dir}")"; then
    echo "[FAIL] could not resolve model.pth for ${rel_dir}"
    FAILED+=("${rel_dir} (model not found)")
    continue
  fi

  if ! config_path="$(resolve_config_path "${abs_dir}")"; then
    echo "[FAIL] could not resolve config.yaml for ${rel_dir}"
    FAILED+=("${rel_dir} (config not found)")
    continue
  fi

  if ! hidden_size="$(infer_hidden_size "${model_path}")"; then
    echo "[FAIL] could not infer hidden size from ${model_path}"
    FAILED+=("${rel_dir} (hidden size inference failed)")
    continue
  fi

  out_dir="${OUT_ROOT}/$(safe_name "${rel_dir}")"
  mkdir -p "${out_dir}"

  echo "model: ${model_path}"
  echo "config: ${config_path}"
  echo "hidden: ${hidden_size}"
  echo "out: ${out_dir}"

  if ! "${PYTHON_BIN}" "${COLLECT_SCRIPT}" \
    --model "${model_path}" \
    --config-path "${config_path}" \
    --save-dir "${out_dir}" \
    --total-timesteps "${TIMESTEPS}" \
    --agent-hidden-size "${hidden_size}" \
    --device "${DEVICE}"; then
    echo "[FAIL] collector failed for ${rel_dir}"
    FAILED+=("${rel_dir} (collector failed)")
    continue
  fi
done

echo "=================================================="
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Completed with failures (${#FAILED[@]}):"
  for item in "${FAILED[@]}"; do
    echo " - ${item}"
  done
  exit 1
fi

echo "All collections completed successfully."
