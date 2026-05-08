#!/usr/bin/env bash
# Sequential pipeline for the post-Polyak-fix residual rerun.
# Usage: ./run_post_polyak_fix.sh <gpu_id> [suffix]
#   gpu_id matches _queue_gpu{N}{suffix}.txt and the device set in each YAML.
#   suffix (optional): empty for the 300k coarse pass, "_1M" for the 1M
#                       holistic-trajectory extension. Default empty.
#
# Plan (300k pass): 5 runs across 4 GPUs, 1 seed each at 300k. See
#   notes/scratch/experiments/2026-05-06_18-29_post-polyak-fix-rerun.md
# for the full design (3 axes, isolated knobs against fix_v27_baseline).
#
# 1M pass (2026-05-07): same 5 configs at total_timesteps=1M, fresh log dirs
# under runs/td3/sim2sim/post_polyak_fix_1M/, to test trajectory holding vs
# collapse over the full budget.
#
# Logs:
#   notes/scratch/post_polyak_fix_logs/<name>.log              (per-run stdout+stderr)
#   notes/scratch/post_polyak_fix_logs/pipeline_gpu{N}{S}.log  (per-GPU pipeline status)
# Run output dirs:
#   runs/td3/sim2sim/post_polyak_fix{,_1M}/<name>/seed0/
#
# Continues on failure so a single bad run doesn't tank the rest.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

GPU_ID="${1:?usage: $0 <gpu_id> [suffix]}"
SUFFIX="${2:-}"
CFG_DIR="scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/post_polyak_fix"
QUEUE_FILE="${CFG_DIR}/_queue_gpu${GPU_ID}${SUFFIX}.txt"
LOG_DIR="notes/scratch/post_polyak_fix_logs"
PIPELINE_LOG="$LOG_DIR/pipeline_gpu${GPU_ID}${SUFFIX}.log"

if [ ! -f "$QUEUE_FILE" ]; then
  echo "ERROR: queue file $QUEUE_FILE not found" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

declare -a OK_RUNS=()
declare -a FAIL_RUNS=()

echo "[gpu${GPU_ID}] start $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] queue: $(tr '\n' ' ' < "$QUEUE_FILE")" | tee -a "$PIPELINE_LOG"

while IFS= read -r name; do
  [ -z "$name" ] && continue
  cfg="${CFG_DIR}/${name}.yaml"
  run_log="${LOG_DIR}/${name}.log"
  if [ ! -f "$cfg" ]; then
    echo "[gpu${GPU_ID}] SKIP $name — config $cfg missing" | tee -a "$PIPELINE_LOG"
    FAIL_RUNS+=("$name(missing-config)")
    continue
  fi
  echo "[gpu${GPU_ID}] launching $name -> $run_log $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  .venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
    --args-file "$cfg" > "$run_log" 2>&1
  rc=$?
  echo "[gpu${GPU_ID}] $name exit=$rc $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  if [ $rc -eq 0 ]; then
    OK_RUNS+=("$name")
  else
    FAIL_RUNS+=("$name(exit=$rc)")
    echo "[gpu${GPU_ID}] continuing despite failure on $name" | tee -a "$PIPELINE_LOG"
  fi
done < "$QUEUE_FILE"

echo "" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] DONE $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] succeeded (${#OK_RUNS[@]}): ${OK_RUNS[*]:-none}" | tee -a "$PIPELINE_LOG"
echo "[gpu${GPU_ID}] failed    (${#FAIL_RUNS[@]}): ${FAIL_RUNS[*]:-none}" | tee -a "$PIPELINE_LOG"

if [ ${#FAIL_RUNS[@]} -gt 0 ]; then exit 1; fi
