#!/usr/bin/env bash
# Rolling per-checkpoint eval driver for the paddle50 sim2sim campaign.
#
# Loops the eval_all_ckpts_residual.sh script over both runs every 5
# minutes until both runs are done (or this is killed). Each iteration
# is idempotent (the eval script skips ckpts that already have
# metrics.json). Eval runs on cuda:3 to leave cuda:1/2 free for training.
#
# Usage:
#   bash notes/scratch/rolling_eval_paddle50.sh > /tmp/rolling_eval_paddle50.log 2>&1 &
#
# To stop: kill the bash process (parent of the eval children).

set -u

REPO=/home/air-hockey/daliu/air-hockey-rl
TARGET="$REPO/scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined.yaml"
DEVICE="cuda:3"

while true; do
    # Discover all variant/seed* dirs that contain checkpoint_* dirs.
    for run_dir in $(find "$REPO/runs/td3/sim2sim/hist2_motion0_to_paddle50" -maxdepth 2 -mindepth 2 -type d 2>/dev/null); do
        if ls "$run_dir"/checkpoint_* 1>/dev/null 2>&1; then
            label="${run_dir#$REPO/runs/td3/sim2sim/hist2_motion0_to_paddle50/}"
            echo "=== $(date -u +%H:%M:%S) :: $label ==="
            bash "$REPO/scripts/smooth_policy/eval_all_ckpts_residual.sh" \
                "$run_dir" "$TARGET" "$DEVICE" 2>&1 | tail -40
        fi
    done
    sleep 300
done
