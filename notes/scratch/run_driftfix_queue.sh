#!/usr/bin/env bash
# Queue runner for residual RL drift-fix experiments.
# Chains: wait for current training -> eval -> start next training -> eval -> ...
#
# Usage: bash notes/scratch/run_driftfix_queue.sh

set -e
set -u

REPO="/home/air-hockey/daliu/air-hockey-rl"
cd "$REPO"

PY=".venv/bin/python"
DRIFTFIX_DIR="scripts/smooth_policy/amp_history/configs/td3/sim2sim/diagnose/long/driftfix"
RUN_BASE="runs/td3/sim2sim/hist2_motion0_to_combined/residual_diagnose/long/driftfix"
TARGET_CFG="scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined.yaml"
DEVICE="cuda:1"

# Each entry is "config_name" (without .yaml) — corresponds to log dir name.
QUEUE=(
  "wd1e2_rs015"
  "scale_sched_15to05"
  "no_per_rs015"
  "q_wd1e3_rs015"
  "wd1e3_scale_sched_15to05"
)

run_eval () {
  local name="$1"
  local rd="$RUN_BASE/$name/seed0"
  echo "[queue] eval: $rd"
  bash scripts/smooth_policy/eval_all_ckpts_residual.sh "$rd" "$TARGET_CFG" "$DEVICE" \
    >> "$RUN_BASE/${name}_eval.log" 2>&1
}

run_training () {
  local name="$1"
  local cfg="$DRIFTFIX_DIR/${name}.yaml"
  local logf="$RUN_BASE/${name}.log"
  if [ ! -f "$cfg" ]; then
    echo "[queue] missing config: $cfg" >&2
    return 1
  fi
  mkdir -p "$RUN_BASE"
  echo "[queue] training: $name (config $cfg)"
  $PY -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
    --args-file "$cfg" \
    > "$logf" 2>&1
}

for name in "${QUEUE[@]}"; do
  run_training "$name"
  run_eval "$name"
done
echo "[queue] all done"
