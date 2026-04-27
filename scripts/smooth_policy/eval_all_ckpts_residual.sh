#!/usr/bin/env bash
# Per-checkpoint sim2sim eval driver for a residual run.
#
# Usage:
#   bash scripts/smooth_policy/eval_all_ckpts_residual.sh <run_dir> <target_config> <device>
# Example:
#   bash scripts/smooth_policy/eval_all_ckpts_residual.sh \
#     runs/td3/sim2sim/hist2_motion0_to_combined/residual_diagnose/long/combo_400k/seed0 \
#     scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined.yaml \
#     cuda:1
#
# Loops every checkpoint_<step>/ + the final model.pth, writes eval_combined_ckpt_<step>/
# (and eval_combined_final/) under the run dir.

set -euo pipefail

RUN_DIR="$1"
TARGET="$2"
DEVICE="${3:-cuda:0}"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERR: run dir not found: $RUN_DIR" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES_ORIG="${CUDA_VISIBLE_DEVICES:-}"
DEVICE_IDX="${DEVICE##cuda:}"
export CUDA_VISIBLE_DEVICES="$DEVICE_IDX"

PY=".venv/bin/python"

for ckpt_dir in $(ls -1 "$RUN_DIR" | grep -E "^checkpoint_[0-9]+$" | sort -t_ -k2 -n); do
    step="${ckpt_dir##checkpoint_}"
    out="$RUN_DIR/eval_combined_ckpt_${step}"
    if [ -f "$out/metrics.json" ]; then
        echo "skip step=$step (already evaluated)"
        continue
    fi
    echo "eval step=$step"
    "$PY" scripts/smooth_policy/sim2sim_eval.py \
        --checkpoint "$RUN_DIR/$ckpt_dir/model.pth" \
        --target-config "$TARGET" \
        --n-episodes 50 \
        --seed 0 \
        --out-dir "$out"
done

# Final model
out="$RUN_DIR/eval_combined_final"
if [ -f "$RUN_DIR/model.pth" ] && [ ! -f "$out/metrics.json" ]; then
    echo "eval final"
    "$PY" scripts/smooth_policy/sim2sim_eval.py \
        --checkpoint "$RUN_DIR/model.pth" \
        --target-config "$TARGET" \
        --n-episodes 50 \
        --seed 0 \
        --out-dir "$out"
fi

echo "done: $RUN_DIR"
