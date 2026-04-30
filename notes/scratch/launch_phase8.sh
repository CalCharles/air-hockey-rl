#!/usr/bin/env bash
# Auto-launch Phase 8 after Phase 7 finishes.
# Phase 8: v15 (sf=0.1), v16 (sf=0.2 + smallbuf), v17 (sf=0.2 + age_decay)
# Each tests if v13 (sf=0.2 winner) can be improved further.
set -u

cd /home/air-hockey/daliu/air-hockey-rl
ROOT=runs/td3/sim2sim/hist2_motion0_to_paddle50

echo "$(date) :: waiting for Phase 7 to finish (v13 seed1+2 + v14)..."
while true; do
    [ -f "$ROOT/residual_v13_top20_baseline/seed1/model.pth" ] && \
    [ -f "$ROOT/residual_v13_top20_baseline/seed2/model.pth" ] && \
    [ -f "$ROOT/residual_v14_top20_window100_age_1e4/seed0/model.pth" ] && break
    sleep 60
done
echo "$(date) :: Phase 7 done — launching Phase 8..."

.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v15_top10_baseline.yaml \
  > "$ROOT/residual_v15_top10_baseline_seed0.log" 2>&1 &
V15=$!
echo "v15 PID=$V15"

.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v16_top20_smallbuf.yaml \
  > "$ROOT/residual_v16_top20_smallbuf_seed0.log" 2>&1 &
V16=$!
echo "v16 PID=$V16"

.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v17_top20_age_1e4.yaml \
  > "$ROOT/residual_v17_top20_age_1e4_seed0.log" 2>&1 &
V17=$!
echo "v17 PID=$V17"

wait
echo "$(date) :: Phase 8 complete"
