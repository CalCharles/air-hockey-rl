#!/usr/bin/env bash
# Per-checkpoint EMA-actor eval driver. Like eval_all_ckpts_residual.sh but
# for `model_ema.pth` files (saved when residual_ema_decay is set in training).
# Outputs to `eval_combined_ema_ckpt_<step>/` to keep separate from online eval.
#
# Usage:
#   bash scripts/smooth_policy/eval_all_ckpts_residual_ema.sh <run_dir> <target_config> <device>

set -euo pipefail

RUN_DIR="$1"
TARGET="$2"
DEVICE="${3:-cuda:0}"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERR: run dir not found: $RUN_DIR" >&2
    exit 1
fi

DEVICE_IDX="${DEVICE##cuda:}"
export CUDA_VISIBLE_DEVICES="$DEVICE_IDX"

PY=".venv/bin/python"

for ckpt_dir in $(ls -1 "$RUN_DIR" | grep -E "^checkpoint_[0-9]+$" | sort -t_ -k2 -n); do
    step="${ckpt_dir##checkpoint_}"
    out="$RUN_DIR/eval_combined_ema_ckpt_${step}"
    if [ -f "$out/metrics.json" ]; then
        echo "skip step=$step (already evaluated)"
        continue
    fi
    if [ ! -f "$RUN_DIR/$ckpt_dir/model_ema.pth" ]; then
        echo "skip step=$step (no model_ema.pth)"
        continue
    fi
    echo "eval EMA step=$step"
    "$PY" scripts/smooth_policy/sim2sim_eval.py \
        --checkpoint "$RUN_DIR/$ckpt_dir/model_ema.pth" \
        --target-config "$TARGET" \
        --n-episodes 50 \
        --seed 0 \
        --out-dir "$out"
done

# Final EMA model
out="$RUN_DIR/eval_combined_ema_final"
if [ -f "$RUN_DIR/model_ema.pth" ] && [ ! -f "$out/metrics.json" ]; then
    echo "eval EMA final"
    "$PY" scripts/smooth_policy/sim2sim_eval.py \
        --checkpoint "$RUN_DIR/model_ema.pth" \
        --target-config "$TARGET" \
        --n-episodes 50 \
        --seed 0 \
        --out-dir "$out"
fi

echo "done EMA eval: $RUN_DIR"
