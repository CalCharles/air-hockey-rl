#!/usr/bin/env bash
# Launches seed-2 of v3 (cuda:1) and full_ft (cuda:2) once seed-1 of both completes.
# Self-launches both as background processes; waits for both to finish.
set -u

cd /home/air-hockey/daliu/air-hockey-rl

ROOT=runs/td3/sim2sim/hist2_motion0_to_paddle50

# Wait for seed-1 of both to be done.
echo "$(date) :: waiting for v3 seed1 and ft seed1 to finish..."
while true; do
    [ -f "$ROOT/residual_v3_no_per_qwd/seed1/model.pth" ] && \
    [ -f "$ROOT/full_ft/seed1/model.pth" ] && break
    sleep 30
done
echo "$(date) :: seed1 done — launching seed2..."

# Launch seed-2 of v3 (cuda:1) and full_ft (cuda:2) in parallel.
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v3_no_per_qwd_seed2.yaml \
  > "$ROOT/residual_v3_no_per_qwd_seed2.log" 2>&1 &
V3_PID=$!
echo "v3 seed2 launched, PID=$V3_PID"

.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_full_ft_seed2.yaml \
  > "$ROOT/full_ft_seed2.log" 2>&1 &
FT_PID=$!
echo "ft seed2 launched, PID=$FT_PID"

wait
echo "$(date) :: seed2 done"
