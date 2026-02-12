#!/usr/bin/env bash
set -euo pipefail

# Run from repo root by default.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/real_configs/rollout_config.yaml}"
TIMESTEPS="${TIMESTEPS:-150}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-data/constant}"
AUTO_GIF="${AUTO_GIF:-1}"  # 1 => pass --auto-gif, 0 => disable

# Optional: choose a python executable (defaults to python on PATH).
PYTHON_BIN="${PYTHON_BIN:-python}"

run_case() {
  local ax="$1"
  local ay="$2"
  local label="$3"
  local save_path="${BASE_SAVE_DIR}/${label}"

  echo "============================================================"
  echo "Running action (${ax}, ${ay}) -> ${save_path}"
  echo "============================================================"

  cmd=(
    "$PYTHON_BIN" scripts/real/rollout_constant.py
    --config-path "$CONFIG_PATH"
    --timesteps "$TIMESTEPS"
    --action "$ax" "$ay"
    --clip
    --save-path "$save_path"
  )

  if [[ "$AUTO_GIF" == "1" ]]; then
    cmd+=(--auto-gif)
  fi

  "${cmd[@]}"
}

# Base actions
run_case "-0.02" "0.0" "action_0p02_0p00"
run_case "-0.05" "0.0" "action_0p05_0p00"
run_case "-0.10" "0.0" "action_0p10_0p00"
run_case "-0.25" "0.0" "action_0p25_0p00"
run_case "-0.50" "0.0" "action_0p50_0p00"

# Flipped (x, y) -> (y, x)
run_case "0.0" "-0.02" "action_0p00_0p02"
run_case "0.0" "-0.05" "action_0p00_0p05"
run_case "0.0" "-0.10" "action_0p00_0p10"
run_case "0.0" "-0.25" "action_0p00_0p25"
run_case "0.0" "-0.50" "action_0p00_0p50"

echo "All action sweeps completed."
