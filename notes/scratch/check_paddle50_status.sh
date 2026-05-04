#!/usr/bin/env bash
# Quick-status check for the paddle50 sim2sim campaign.
# Prints the latest training step from each log + per-checkpoint eval table.
set -u

REPO=/home/air-hockey/daliu/air-hockey-rl
ROOT="$REPO/runs/td3/sim2sim/hist2_motion0_to_paddle50"

echo "==== TRAINING STATUS ===="
for log in "$ROOT"/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    last_step=$(grep -oE "Step [0-9]+:" "$log" | tail -1 | grep -oE "[0-9]+")
    is_done=$(grep -c "Saving model" "$log" 2>/dev/null || echo 0)
    echo "  $name: latest step=${last_step:-?}  saves=${is_done}"
done

echo
echo "==== EVAL STATUS ===="
.venv/bin/python "$REPO/notes/scratch/aggregate_paddle50_results.py" 2>&1 | head -100
