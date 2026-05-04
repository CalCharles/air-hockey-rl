#!/usr/bin/env bash
# Sequential pipeline for v30_explore_lite seeds 2/3/4 on GPU 3.
# 3 runs x 300k = ~2h total wall clock.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

LOG_DIR=notes/scratch/partB_pipeline_logs
mkdir -p "$LOG_DIR"

CFG_DIR=scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50

CONFIGS=(
  "td3_residual_v30_explore_lite_seed2.yaml"
  "td3_residual_v30_explore_lite_seed3.yaml"
  "td3_residual_v30_explore_lite_seed4.yaml"
)

PIPELINE_LOG="$LOG_DIR/v30_lite_reseed_pipeline.log"
echo "[reseed] start $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"

for cfg in "${CONFIGS[@]}"; do
  base="${cfg%.yaml}"
  run_log="$LOG_DIR/${base}.log"
  echo "[reseed] launching $cfg -> $run_log $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"

  .venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
    --args-file "$CFG_DIR/$cfg" > "$run_log" 2>&1
  rc=$?
  echo "[reseed] $cfg exit=$rc $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  if [ $rc -ne 0 ]; then
    echo "[reseed] STOPPING — non-zero exit on $cfg" | tee -a "$PIPELINE_LOG"
    exit $rc
  fi
done

echo "[reseed] all 3 done $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
