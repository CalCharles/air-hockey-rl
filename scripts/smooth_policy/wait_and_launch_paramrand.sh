#!/usr/bin/env bash
# Wait for the in-flight cuda:1 isolation run (only_action_attenuation, the
# only run in the gpu1 queue of the 2026-05-09 23:47 UTC isolation sweep)
# to exit, then launch the 2M-step paramrand-pm25 run on cuda:1 in the
# background.
#
# Usage: ./wait_and_launch_paramrand.sh
#
# This script is itself meant to be nohup'd in the background:
#   nohup bash scripts/smooth_policy/wait_and_launch_paramrand.sh \
#     > notes/scratch/zeroshot_paramrand_logs/_wait_and_launch.out 2>&1 &
#
# Detection: greps the gpu1 pipeline log for the exit line of the
# `only_action_attenuation` run (written by run_zeroshot_ablations_700k.sh
# when the python invocation returns). Polls every 30s; logs each poll so
# the wait is visible in the output.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

UPSTREAM_LOG="notes/scratch/zeroshot_ablation_700k_logs/pipeline_gpu1.log"
SENTINEL_PATTERN="\[gpu1\] only_action_attenuation exit="
ARGS_FILE="scripts/smooth_policy/amp_history/configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml"
LOG_DIR="notes/scratch/zeroshot_paramrand_logs"
RUN_LOG="$LOG_DIR/paramrand_pm25.log"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"

echo "[wait_and_launch] start $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
echo "[wait_and_launch] watching $UPSTREAM_LOG for /$SENTINEL_PATTERN/" | tee -a "$PIPELINE_LOG"

# Poll loop. Bash's `until` + grep gives a clean exit when the sentinel
# appears. 30-second poll cadence is plenty given the 2h+ wait.
poll=0
until grep -Eq "$SENTINEL_PATTERN" "$UPSTREAM_LOG" 2>/dev/null; do
  if (( poll % 20 == 0 )); then  # log every 10 minutes
    echo "[wait_and_launch] still waiting at $(date -u +%FT%TZ) (poll=$poll)" \
      | tee -a "$PIPELINE_LOG"
  fi
  sleep 30
  poll=$((poll + 1))
done

EXIT_LINE=$(grep -E "$SENTINEL_PATTERN" "$UPSTREAM_LOG" | tail -1)
echo "[wait_and_launch] cuda:1 freed at $(date -u +%FT%TZ): $EXIT_LINE" \
  | tee -a "$PIPELINE_LOG"

# Sanity: confirm no python process is still using cuda:1 from the old run.
# (Pipeline writes the exit line BEFORE python is fully reaped in some
# edge cases. Wait until ps shows no `td3_zeroshot_only_action_attenuation`
# process alive, capped at 60s.)
wait_extra=0
while pgrep -f "td3_zeroshot_only_action_attenuation" > /dev/null 2>&1 \
      && [ $wait_extra -lt 60 ]; do
  sleep 1
  wait_extra=$((wait_extra + 1))
done
if [ $wait_extra -gt 0 ]; then
  echo "[wait_and_launch] waited extra ${wait_extra}s for python to exit" \
    | tee -a "$PIPELINE_LOG"
fi

# Verify cuda:1 actually free (memory check).
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
echo "[wait_and_launch] cuda:1 memory.used=${mem_used} MiB" | tee -a "$PIPELINE_LOG"

echo "[wait_and_launch] launching paramrand_pm25 (2M steps, td3_training_dr) -> $RUN_LOG $(date -u +%FT%TZ)" \
  | tee -a "$PIPELINE_LOG"

# Launch the paramrand run. Uses the new td3_training_dr entrypoint.
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training_dr \
  --args-file "$ARGS_FILE" > "$RUN_LOG" 2>&1
rc=$?
echo "[wait_and_launch] paramrand_pm25 exit=$rc $(date -u +%FT%TZ)" \
  | tee -a "$PIPELINE_LOG"
