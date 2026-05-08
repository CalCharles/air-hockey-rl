#!/usr/bin/env bash
# Sequential pipeline for the 12-cell warp075_p30 CQL recipe campaign.
# Usage: ./run_warp075_campaign.sh <gpu_id>
#   gpu_id matches _queue_gpu{N}_campaign.txt and the device set in each YAML.
#
# Plan: 4 Phase-A (env difficulty) + 8 Phase-B (CQL hyperparam grid) cells, 300k each,
# distributed across 4 GPUs (3 cells per GPU). Phase-C/D 1M extensions launched by hand
# after Phase-A/B finish.
#
# Logs:
#   notes/scratch/sim2sim_redesign_logs/campaign/<name>.log
#   notes/scratch/sim2sim_redesign_logs/campaign/pipeline_gpu{N}.log
# Run dirs: runs/td3/sim2sim_redesign/residual_warp075_p30/<name>/seed0/
# Continues on failure.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

GPU_ID="${1:?usage: $0 <gpu_id>}"
CFG_DIR="scripts/smooth_policy/amp_history/configs/td3/sim2sim/warp075_p30_residual"
QUEUE_FILE="${CFG_DIR}/_queue_gpu${GPU_ID}_campaign.txt"
LOG_DIR="notes/scratch/sim2sim_redesign_logs/campaign"
PIPELINE_LOG="$LOG_DIR/pipeline_gpu${GPU_ID}.log"

mkdir -p "$LOG_DIR"
[ -f "$QUEUE_FILE" ] || { echo "ERROR: queue $QUEUE_FILE missing" >&2; exit 1; }

declare -a OK=() FAIL=()
echo "[gpu${GPU_ID}] start $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] queue: $(tr '\n' ' ' < "$QUEUE_FILE")" | tee -a "$PIPELINE_LOG"

while IFS= read -r name; do
  [ -z "$name" ] && continue
  cfg="${CFG_DIR}/${name}.yaml"
  run_log="${LOG_DIR}/${name}.log"
  [ -f "$cfg" ] || { echo "[gpu${GPU_ID}] SKIP $name — config missing" | tee -a "$PIPELINE_LOG"; FAIL+=("$name(missing)"); continue; }
  echo "[gpu${GPU_ID}] launching $name $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  .venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
    --args-file "$cfg" > "$run_log" 2>&1
  rc=$?
  echo "[gpu${GPU_ID}] $name exit=$rc $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  if [ $rc -eq 0 ]; then OK+=("$name"); else FAIL+=("$name(exit=$rc)"); fi
done < "$QUEUE_FILE"

echo "[gpu${GPU_ID}] DONE $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] OK (${#OK[@]}): ${OK[*]:-none}" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] FAIL (${#FAIL[@]}): ${FAIL[*]:-none}" | tee -a "$PIPELINE_LOG"
[ ${#FAIL[@]} -gt 0 ] && exit 1 || exit 0
