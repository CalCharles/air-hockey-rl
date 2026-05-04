#!/usr/bin/env bash
# Eval pipeline for Part B (v30 family). Sequential per-checkpoint
# deterministic eval against sim2sim_combined.yaml on GPU 3.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

LOG_DIR=notes/scratch/partB_pipeline_logs
mkdir -p "$LOG_DIR"

DEVICE="${1:-cuda:3}"
TARGET=scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined.yaml

RUN_DIRS=(
  "runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v30_explore_full/seed0"
  "runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v30_explore_full/seed1"
  "runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v30_explore_lite/seed0"
  "runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v30_explore_lite/seed1"
  "runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v30_explore_directional_only/seed0"
  "runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v30_explore_directional_only/seed1"
)

PIPELINE_LOG="$LOG_DIR/eval_pipeline.log"
echo "[eval] start $(date -u +%FT%TZ) device=$DEVICE" | tee -a "$PIPELINE_LOG"

for run in "${RUN_DIRS[@]}"; do
  base=$(echo "$run" | tr / _)
  log="$LOG_DIR/eval_${base}.log"
  echo "[eval] -> $run $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  bash scripts/smooth_policy/eval_all_ckpts_residual.sh "$run" "$TARGET" "$DEVICE" > "$log" 2>&1
  rc=$?
  echo "[eval] $run exit=$rc $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
done

echo "[eval] all done $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
