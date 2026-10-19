# Real camera frames: stale, in order, and unevenly spaced

- **Date**: 2026-10-18 UTC
- **Status**: done (analysis of existing data; the on-robot fix test is still open)
- **Data**: the four canonical 2026-10-14 hist4 real eval runs in `results/eval_4_tasks_of_policies_hist4_2026-09-19/`
- **Scripts and figure**: `results/eval_4_tasks_of_policies_hist4_2026-09-19/camera_staleness/` (`stale1.py`, `stale2.py`, `stale_fig.py`, `juggle_trajectory_data0_camera_lag.png`)
- **Follows**: [`2026-10-15_19-43_real-rollout-sim2real-gap-sources.md`](2026-10-15_19-43_real-rollout-sim2real-gap-sources.md), §1 (camera latency ≈ 120 ms)

## Question

The camera read may be returning buffered, old frames. If so, does that show up in the saved data, and would it look like the puck jumping back and then snapping forward?

## Mechanism

`save_collect` (`airhockey/sims/real/control_parameters.py`) calls `cap.read()` once per 49 ms step.

- The call returns in 0.5 ms (5th–95th percentile 0.4–0.6 ms). It never waits for the camera, so there is always a frame already queued.
- `CAP_PROP_BUFFERSIZE=1` is commented out at `airhockey/sims/air_hockey_real.py:1325`.
- The V4L2 queue is first-in, first-out. Each read returns the oldest unread frame, so frames always come out **in time order, just late**.
- A buffered camera therefore produces a roughly constant delay, never a step backward. A backward snap would need frames to come out of order, which this queue does not do.

## Findings

1. **No backward snaps.** In 869 free-flight puck steps (logged visible, > 25 cm from the paddle, away from the walls), the puck never moved backward along its track. The lowest along-track speed was +0.29 m/s.
2. **The camera runs at ~30 fps and is read at 20 Hz, so frame spacing alternates ~33 / 66 ms.**
   - 82 % of consecutive free-flight steps have a 1:2 or 2:1 distance ratio; only 4 % are within 20 % of 1:1.
   - The per-step speed in the observation therefore zigzags by about ±50 %.
   - Any 4-step difference spans two short and two long gaps (~198 ms), so the oldest-to-newest history difference is unaffected. The individual per-step differences are not.
3. **The camera image lags the telemetry by 120–150 ms.**
   - Per-frame latency, from 146 fast-paddle frames comparing the image paddle with telemetry: median 151 ms, interquartile range 126–182 ms.
   - The earlier all-frame fit gave 120 ms.
   - Consecutive frames differ by about ±20 ms, which is the 33 / 66 ms phase. Only 4 of 83 consecutive-frame changes exceed 40 ms, which is within detector noise.
4. **The paddle observation is not delayed.** It comes from robot telemetry read 0.5 ms before the frame, so the policy sees a fresh paddle next to a ~120–150 ms old puck. The sim delays both by 25 ms.

## Why it matters

- **The relative puck-to-paddle position is wrong by the lag × puck speed.** That is 12–15 cm at 1 m/s and 25–30 cm at 2 m/s. The contact distance (paddle radius + puck radius) is 8.3 cm.
- **The per-step velocity cue zigzags**, unlike in the sim, where frames are evenly spaced.

## What we cannot tell from the logs

How much of the 120–150 ms is queue depth and how much is the camera and USB pipeline itself. A USB webcam typically adds 50–100 ms on its own. The test has to run on the robot:
1. `v4l2-ctl -d /dev/video0 --get-parm` to read the frame rate
2. set `CAP_PROP_BUFFERSIZE=1`, or run a grab thread that keeps only the newest frame
3. re-run the image-vs-telemetry lag scan

Whatever latency remains after that should go into the sim as a puck-only observation delay.
