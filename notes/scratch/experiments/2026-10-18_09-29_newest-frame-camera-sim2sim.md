# Sim2sim: hist4 juggle policies under a newest-frame camera (queue removed)

- **Date**: 2026-10-18 09:29 UTC
- **Status**: done
- **Run dirs**: policies `runs/td3/sysid_v2_real_workspace/hist4_seed{0..4}/juggle_sysid_v2_hist4/checkpoint_1975000`; output `runs/td3/sysid_v2_real_workspace/newest_frame_sim2sim/{grab0,grab25,grab50,grab75,grab100,lagcam}/` (`summary.md`, `results.json`)
- **Configs**:
  - source: `configs/new_juggle/sysid_v2_hist4.yaml`
  - targets: `configs/new_juggle/sysid_v2_hist4_puck_grab{0,25,50,75,100}.yaml` (new) and `sysid_v2_hist4_puck_lagcam.yaml` (reference: today's queued-camera model)
- **Eval script**: `scripts/td3/extras/eval_occlusion_map_sim2sim.py --episodes 40` (seeds 1000–1039, paired across envs and policies)
- **Follows**: [`2026-10-18_08-03_puck-camera-lag-sim2sim.md`](2026-10-18_08-03_puck-camera-lag-sim2sim.md), [`2026-10-18_07-51_real-camera-frame-staleness.md`](2026-10-18_07-51_real-camera-frame-staleness.md)

## Question

Suppose the robot reads the newest camera frame each step (grab thread or `CAP_PROP_BUFFERSIZE=1`) instead of the oldest frame in the V4L2 queue. Does the policy recover the source-sim return?

## Model

After the fix, the puck observation is the newest 30 fps capture at or before (now − camera latency).
- Its age is the camera's own latency (exposure, USB, decode) plus 0–33 ms.
- Consecutive frames are still ~33 / 66 ms apart.

This is the existing `puck_obs_lag_seconds` + `puck_obs_lag_frame_period: 1/30` mode, with the lag set to the camera latency. No code changes were needed.
- The camera latency is unknown. The measured 151 ms includes the queue. So it is swept over 0 / 25 / 50 / 75 / 100 ms.
- The 0 ms case uses 1 ms, because the sim turns the lag and the frame grid off at exactly 0.
- The paddle stays at the shared 25 ms delay, matching fresh telemetry.
- The history frame spacing (sim 25 ms vs robot ~49 ms) is **not** modelled in either env.

## Results

Each cell is the mean ± SE across the 5 policies; each policy's value is the mean of its 40 paired episodes. Source sim: **110.6 ± 6.2**, episode length 180.5.

| Target: puck-only lag | Return | Δ vs source (paired) | Episode length |
|---|---|---|---|
| newest frame, 0 ms latency (lag 0–33 ms) | **98.0 ± 6.3** | −12.6 ± 1.0 | 163.5 |
| newest frame, 25 ms (25–58 ms) | **76.5 ± 5.1** | −34.1 ± 4.2 | 127.5 |
| newest frame, 50 ms (50–83 ms) | **61.1 ± 2.1** | −49.4 ± 4.6 | 108.1 |
| newest frame, 75 ms (75–108 ms) | **52.2 ± 1.5** | −58.4 ± 6.9 | 93.4 |
| newest frame, 100 ms (100–133 ms) | **46.3 ± 1.2** | −64.3 ± 6.0 | 84.4 |
| queued camera today (120–153 ms, `lagcam`) | **47.9 ± 1.5** | −62.7 ± 5.7 | 84.9 |

Every one of the 5 policies loses in every target. Per-policy numbers are in each `summary.md`.

## Conclusion

- **Removing the queue alone recovers the return only if the camera's own latency is small.**
  - At ≤ 25 ms it recovers 29–51 of the ~63 lost.
  - At 50 ms it recovers 13.
  - At ≥ 75 ms it recovers nothing measurable. The return levels off at ~46–52 from about 75 ms on, the same as today's queued camera.
- **Even a zero-latency camera costs −12.6 ± 1.0** (−11 %). That comes from the 0–33 ms frame age plus the uneven spacing.
- **The loss is steep at small lags.** The first 25 ms of latency costs about as much as the next 75 ms. These policies have essentially no tolerance for a puck–paddle timing skew.
- **What to do:** the on-robot latency measurement after the fix decides how much the fix alone buys. In every plausible webcam case (50–100 ms) the policy still needs to train with the measured puck lag (`puck_obs_lag_seconds` + frame grid, ideally randomized around it), with the history spacing fixed at the same time.
- **Confidence:** one seed set; 40 paired episodes; paired SEs are 1–7.
