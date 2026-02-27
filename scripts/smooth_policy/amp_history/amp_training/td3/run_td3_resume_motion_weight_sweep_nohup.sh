#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/air-hockey/daliu/air-hockey-rl"
TRAIN_SCRIPT="$REPO_ROOT/scripts/smooth_policy/amp_history/amp_training/td3/amp_training_td3.py"
ARGS_FILE="$REPO_ROOT/scripts/smooth_policy/amp_history/configs/td3/td3_standard_resume_checkpoint.yaml"
LOG_BASE="$REPO_ROOT/runs/td3/final/task_only_resume_from_checkpoint"

weights=(0.1 0.25 0.5 1.0 2.0 5.0)
gpus=(1   2    3   1   2   3)

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Training script not found: $TRAIN_SCRIPT"
  exit 1
fi

if [[ ! -f "$ARGS_FILE" ]]; then
  echo "Args file not found: $ARGS_FILE"
  exit 1
fi

cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "venv/bin/activate"
fi

for i in "${!weights[@]}"; do
  w="${weights[$i]}"
  gpu="${gpus[$i]}"
  w_tag="${w//./p}"
  run_dir="$LOG_BASE/motion_w${w_tag}_gpu${gpu}"
  run_name="td3_resume_motion_w${w_tag}_gpu${gpu}"
  nohup_log="$run_dir/nohup.out"

  mkdir -p "$run_dir"

  cmd=(
    python "$TRAIN_SCRIPT"
    --args-file "$ARGS_FILE"
    --motion-reward-weight "$w"
    --device "cuda:${gpu}"
    --log-parent-dir "$run_dir"
    --run-name "$run_name"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN: ${cmd[*]} > $nohup_log 2>&1 &"
  else
    nohup "${cmd[@]}" > "$nohup_log" 2>&1 &
    echo "Started: motion_reward_weight=$w on cuda:$gpu -> $run_dir"
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run complete. No training jobs were started."
else
  echo "All resume runs launched."
  echo "Check status: ps -fu \"$USER\" | grep amp_training_td3.py"
fi
