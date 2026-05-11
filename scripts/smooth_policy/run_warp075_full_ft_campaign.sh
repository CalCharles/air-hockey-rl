#!/usr/bin/env bash
# Sequential pipeline for the 12-cell warp075_p30 full-FT + CQL recipe campaign.
# Usage: ./run_warp075_full_ft_campaign.sh <gpu_id>
#   gpu_id matches _queue_gpu{N}_full_ft.txt and the device set in each YAML.
#
# Plan: 4 cell variants (A baseline, B +CQL, C +CQL+actor2+n5, D +CQL+actor2+n5+fulllr)
# × 3 replicas (seed0 + seed1 on canonical warp075_p30 + seed0 on env_mild_p10).
# 12 cells × 500k each, ~80 min wall per cell, ~10.5h per GPU on 2 GPUs.
#
# GPU 2: A and B variants (low-LR FT recipe family)
# GPU 3: C and D variants (residual canonical knobs ported to full-FT)
#
# Logs:
#   notes/scratch/sim2sim_full_ft_logs/campaign/<name>.log
#   notes/scratch/sim2sim_full_ft_logs/campaign/pipeline_gpu{N}.log
# Run dirs: runs/td3/sim2sim_full_ft_warp075_p30/<name>/seed{0,1}/
# Continues on per-cell failure.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

GPU_ID="${1:?usage: $0 <gpu_id>}"
CFG_DIR="scripts/smooth_policy/amp_history/configs/td3/sim2sim/warp075_p30_full_ft"
QUEUE_FILE="${CFG_DIR}/_queue_gpu${GPU_ID}_full_ft.txt"
LOG_DIR="notes/scratch/sim2sim_full_ft_logs/campaign"
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
