#!/usr/bin/env bash
# Launch v2 (rs=0.25) on cuda:1 and v3 (no_per+q_wd) on cuda:2 in parallel.
# Both 300k. Run after v1+full_ft finish.
set -u

cd /home/air-hockey/daliu/air-hockey-rl

# v2 — rs=0.25 (more head room for harder env)
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v2_rs025.yaml \
  > runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v2_rs025_seed0.log 2>&1 &
V2_PID=$!
echo "v2 launched, PID=$V2_PID"

# v3 — no_per + q_wd (different recipe — pre-recency_top50 winner)
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v3_no_per_qwd.yaml \
  > runs/td3/sim2sim/hist2_motion0_to_paddle50/residual_v3_no_per_qwd_seed0.log 2>&1 &
V3_PID=$!
echo "v3 launched, PID=$V3_PID"

echo "both launched. PIDs: $V2_PID $V3_PID"
wait
echo "both finished"
