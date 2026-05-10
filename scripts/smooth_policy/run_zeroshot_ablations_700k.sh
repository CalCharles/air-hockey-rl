#!/usr/bin/env bash
# Sequential pipeline for the +200k extension of the zero-shot sim2real
# ablation sweep (500k -> 700k continuations + 1 fresh 700k run that
# replaces the broken `no_obs_delay`). Adapted from
# scripts/smooth_policy/run_zeroshot_ablations.sh.
#
# Usage: ./run_zeroshot_ablations_700k.sh <gpu_id>
#   gpu_id: 0 or 1 (matches _queue_gpu{N}.txt and the device set in each TD3 args YAML)
#
# Per-run wall clock (extrapolated from 500k base = 1h45m):
#   - 200k continuation ~ 42m
#   - fresh 700k        ~ 2h27m
# Pipeline totals: cuda:0 ~5h15m (1 fresh + 4 cont), cuda:1 ~4h54m (7 cont).
#
# Logs:
#   notes/scratch/zeroshot_ablation_700k_logs/<name>.log     (per-run stdout+stderr)
#   notes/scratch/zeroshot_ablation_700k_logs/pipeline_gpu{N}.log (per-GPU pipeline status)
# Run output dirs:
#   runs/td3/zeroshot_ablations_700k/<name>/seed0/
#
# Continues on failure so a single bad run (e.g. OOM) doesn't tank the rest.

set -u
cd /home/air-hockey/daliu/air-hockey-rl

GPU_ID="${1:?usage: $0 <gpu_id>}"
QUEUE_FILE="scripts/smooth_policy/amp_history/configs/td3/zeroshot_ablations_700k/_queue_gpu${GPU_ID}.txt"
CFG_DIR="scripts/smooth_policy/amp_history/configs/td3/zeroshot_ablations_700k"
LOG_DIR="notes/scratch/zeroshot_ablation_700k_logs"
PIPELINE_LOG="$LOG_DIR/pipeline_gpu${GPU_ID}.log"

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
  cfg="${CFG_DIR}/td3_zeroshot_${name}.yaml"
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
