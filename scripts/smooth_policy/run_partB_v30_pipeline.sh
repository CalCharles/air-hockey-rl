#!/usr/bin/env bash
# Sequential pipeline for Part B (v30 family) on GPU 3.
# 3 variants x 2 seeds = 6 runs, 300k each. ~4h total wall clock.
# Output dir: notes/scratch/partB_pipeline_logs/<run>.log
# Each run also writes its own training log via td3_training.py to its run dir.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

LOG_DIR=notes/scratch/partB_pipeline_logs
mkdir -p "$LOG_DIR"

CFG_DIR=scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50

CONFIGS=(
  "td3_residual_v30_explore_full.yaml"
  "td3_residual_v30_explore_full_seed1.yaml"
  "td3_residual_v30_explore_lite.yaml"
  "td3_residual_v30_explore_lite_seed1.yaml"
  "td3_residual_v30_explore_directional_only.yaml"
  "td3_residual_v30_explore_directional_only_seed1.yaml"
)

PIPELINE_LOG="$LOG_DIR/pipeline.log"
echo "[pipeline] start $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"

for cfg in "${CONFIGS[@]}"; do
  base="${cfg%.yaml}"
  run_log="$LOG_DIR/${base}.log"
  echo "[pipeline] launching $cfg -> $run_log $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"

  .venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
    --args-file "$CFG_DIR/$cfg" > "$run_log" 2>&1
  rc=$?
  echo "[pipeline] $cfg exit=$rc $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  if [ $rc -ne 0 ]; then
    echo "[pipeline] STOPPING — non-zero exit on $cfg" | tee -a "$PIPELINE_LOG"
    exit $rc
  fi
done

echo "[pipeline] all 6 done $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
