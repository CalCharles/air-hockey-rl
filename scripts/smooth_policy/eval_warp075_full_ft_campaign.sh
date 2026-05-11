#!/usr/bin/env bash
# Per-checkpoint eval driver for the warp075 full-FT campaign.
# Iterates the cells assigned to a given GPU, picks the right target config
# per cell (canonical warp075_p30 vs cross-env env_mild_p10), and invokes
# eval_all_ckpts_residual.sh on each run dir.
#
# Usage:
#   ./eval_warp075_full_ft_campaign.sh <gpu_id>
#
# Logs: notes/scratch/sim2sim_full_ft_logs/eval/eval_pipeline_gpu{N}.log
#       (per-cell stdout: notes/scratch/sim2sim_full_ft_logs/eval/<cell>_<seed>.log)

set -u
cd /home/air-hockey/daliu/air-hockey-rl

GPU_ID="${1:?usage: $0 <gpu_id>}"
DEVICE="cuda:${GPU_ID}"
RUN_ROOT="runs/td3/sim2sim_full_ft_warp075_p30"
LOG_DIR="notes/scratch/sim2sim_full_ft_logs/eval"
PIPELINE_LOG="$LOG_DIR/eval_pipeline_gpu${GPU_ID}.log"
TARGET_P30="scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_warp075_p30.yaml"
TARGET_P10="scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_warp075_p10.yaml"

mkdir -p "$LOG_DIR"

# Cell list per GPU. Mirrors the training pipeline assignment.
case "$GPU_ID" in
  2)
    CELLS=(
      "A_baseline/seed0:p30"
      "A_baseline/seed1:p30"
      "B_cql20/seed0:p30"
      "B_cql20/seed1:p30"
      "A_baseline_p10/seed0:p10"
      "B_cql20_p10/seed0:p10"
    ) ;;
  3)
    CELLS=(
      "C_cql20_actor2_n5/seed0:p30"
      "C_cql20_actor2_n5/seed1:p30"
      "D_cql20_actor2_n5_fulllr/seed0:p30"
      "D_cql20_actor2_n5_fulllr/seed1:p30"
      "C_cql20_actor2_n5_p10/seed0:p10"
      "D_cql20_actor2_n5_fulllr_p10/seed0:p10"
    ) ;;
  *) echo "ERR: unknown gpu_id $GPU_ID" >&2; exit 1 ;;
esac

declare -a OK=() FAIL=()
echo "[evalgpu${GPU_ID}] start $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"

for entry in "${CELLS[@]}"; do
  cell="${entry%:*}"
  tag="${entry##*:}"
  if [ "$tag" = "p10" ]; then TARGET="$TARGET_P10"; else TARGET="$TARGET_P30"; fi
  run_dir="$RUN_ROOT/$cell"
  cell_label="${cell//\//_}"
  cell_log="$LOG_DIR/${cell_label}.log"

  if [ ! -d "$run_dir" ]; then
    echo "[evalgpu${GPU_ID}] SKIP $cell — run_dir missing" | tee -a "$PIPELINE_LOG"
    FAIL+=("$cell(missing)"); continue
  fi
  echo "[evalgpu${GPU_ID}] eval $cell target=$tag $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  bash scripts/smooth_policy/eval_all_ckpts_residual.sh "$run_dir" "$TARGET" "$DEVICE" \
    > "$cell_log" 2>&1
  rc=$?
  echo "[evalgpu${GPU_ID}] $cell exit=$rc $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
  if [ $rc -eq 0 ]; then OK+=("$cell"); else FAIL+=("$cell(exit=$rc)"); fi
done

echo "[evalgpu${GPU_ID}] DONE $(date -u +%FT%TZ)" | tee -a "$PIPELINE_LOG"
echo "[evalgpu${GPU_ID}] OK (${#OK[@]}): ${OK[*]:-none}" | tee -a "$PIPELINE_LOG"
echo "[evalgpu${GPU_ID}] FAIL (${#FAIL[@]}): ${FAIL[*]:-none}" | tee -a "$PIPELINE_LOG"
[ ${#FAIL[@]} -gt 0 ] && exit 1 || exit 0
