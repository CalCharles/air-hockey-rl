# Sources of sim-vs-real mismatch in the 2026-10-14 hist4 zero-shot real rollouts

- **Date**: 2026-10-15 19:43 UTC
- **Status**: done (analysis only, no fixes applied)
- **Run dirs**: `results/eval_4_tasks_of_policies_hist4_2026-09-19/` — `juggle_sysid/data_20261014-201743`, `puck_goal_sysid/data_20261014-210939`, `puck_goal_vel_sysid/data_20261014-211942`, `puck_vel_sysid/data_20261014-220834` (sim-vs-real table: `results_table.md` there)
- **Configs**: real `configs/real_configs/rollout_td3_config_hist4.yaml`; sim reference `configs/new_juggle/tasks_v2/sim_sysid_juggle.yaml` with `hist_len: 4`

## Question

The hist4 sysid-v2 policies drop sharply sim → real: juggle 130 → 48, puck_goal 8/10 → 1/10, puck_goal_vel 6/10 → 0/10, puck_vel 23.7 → 1.4. Which things happen on the real system that the simulator does not model? We had a starting suspicion that the camera-tracked puck stutters: it freezes, jumps on re-acquire, and so on.

## Setup

- **Data:** all 47 `episode_hdf5/*/*.hdf5` files in the four run folders (juggle 10, puck_goal 11, puck_goal_vel 11, puck_vel 15), 3,437 policy steps. Each step has:
  - the logged puck `(x, y, occluded)` and `pose` (the paddle observation)
  - `paddle_actual` (actual TCP pose and speed)
  - `paddle_cmd` (pre- and post-filter servoL target)
  - `policy_action`, `timing`, and the 320×240 camera frame `train_img` (the same frames as `ground_truth_camera.mp4`, 1 per step)
- **Independent puck detector:** an HSV red-blob detector run on every saved frame. A pixel→obs-frame homography was fitted with RANSAC on the logged visible detections (2,933 / 2,975 inliers). On frames where both detectors see the puck they agree to a median 2.2 mm (p90 5.1 mm).
- **Camera latency:** a dark-blob paddle tracker in the image was compared with the robot telemetry `pose` at different time lags.
- **Paddle replay:** each episode's real `policy_action` sequence was run through the Box2D paddle (sysid v2 gains, hist_len 4, puck disabled), starting from the real start pose. A one-step test also drove the sim PID with the robot's own post-filter target from the real state at every step.
- **Sim reference** (`airhockey/sims/airhockey_box2d.py`):
  - observation corruption: iid Gaussian puck noise σ = 1 cm, random occlusion at 5 % onset per step anywhere on the table (geometric run length, at most 7 frames, holds the last visible position), and an observation delay of 25 ms applied to puck and paddle alike
  - paddle: speed cap `max_paddle_vel` = 2 m/s in every direction

Analysis scripts were run from the session scratchpad and are not committed. They used the numbering `a1_obs.py` (occlusion and timing), `a3_phantom.py`, `a4_indep.py` / `a5_compare.py` (independent detector), `a6_paddle.py` / `a7_pid.py` (paddle replay), `a8_latency.py` / `a9_vis.py` (camera latency) and `a10_misc.py`.

## Results

Ranked by likely impact on transfer.

### 1. Camera latency ≈ 120 ms; sim models 25 ms

Robot telemetry is read about 0.5 ms before each camera frame arrives (`timing`: frame received − step start = 0.55 ms median). Yet the paddle *in the image* sits where telemetry placed it about 120 ms earlier.

The paddle tracker was fitted against the telemetry at different lags. The table gives the mean residual after removing a constant offset (paddle-top parallax and calibration):

| lag (ms) | 0 | 40 | 80 | 100 | **120** | 140 | 160 |
|---|---|---|---|---|---|---|---|
| residual (mm) | 52.8 | 44.4 | 38.5 | 36.8 | **36.2** | 36.9 | 39.1 |

A visual check backs this up. On the fastest paddle frame of six episodes, a circle drawn at the telemetry position from 120 ms earlier lands exactly on the paddle in the image. The circle at the current telemetry position is about one paddle-width ahead.

**Consequences:**
- The policy gets a fresh paddle position but a puck position about 2.4 control steps old.
- At 1.5 m/s that is about 18 cm of error in where the puck is relative to the paddle.
- The sim has no such skew: its 25 ms delay applies to both objects.
- The likely cause is frames queuing in the OpenCV/V4L2 capture buffer, since `cap.read()` in `save_collect` returns the oldest queued frame. This has not been verified on the robot.

### 2. The real paddle is 2–3× slower than the sim paddle under policy actions

Policy actions are saturated (|a| > 0.95) 47 % of the time.

| | Real (`paddle_actual`) | Sim, same action sequence |
|---|---|---|
| Speed: median / p95 / max (m/s) | 0.41 / 0.87 / 1.09 | 0.89 / 2.12 / 2.43 |
| p99 \|vx\| / p99 \|vy\| (m/s) | 0.91 / 0.50 | x reaches the 2.0 cap in 1 step; y ≈ 1.4 |
| Peak lag from command to velocity (steps) | mostly 3 (24 eps) or 2 (17 eps) | mostly 1 (28) or 0 (17) |
| Position gap after 5 / 10 / 20 identical actions | — | median 24 / 39 / 41 cm |

- **The gap is in the tracking model itself.** In the one-step test (sim PID driven by the real post-filter target from the real state), sim steps are a median 2.2× longer than real steps, and the median one-step error is 28.5 mm (n = 2,994).
- **Why the PID sysid didn't flag it.** That sysid recorded at hist_len 1 ([`2026-09-10_03-39`](2026-09-10_03-39_paddle-pid-cmaes-sysid.md), [`2026-09-19_01-35`](2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md)). Its per-step error is averaged over trials that spend most of their time at rest against the workspace limits, and the real peaks it measured were already ~0.8–1.0 m/s in x and 0.49 m/s in y ([`2026-09-10_04-39`](2026-09-10_04-39_paddle-pid-overlays-per-trial.md)).
- **Direction-dependent limits.** The real speed limits differ by direction (x− > x+ ≫ y). The sim cap is the same in every direction.
- **Code-level difference, not quantified separately.** The real controller anchors each new target, and the hist-filter history, on UR's *target* TCP pose (`getTargetTCPPose`, with servoL `lookahead` 0.2 s). The sim anchors on the actual Box2D paddle position.

A slower paddle means weaker hits. That fits the puck_vel (23.7 → 1.4) and goal-task collapses.

### 3. Most "occlusions" are detector misses, clustered in specific places on the table

- **Rate:** 12.8 % of steps are logged `occluded`. In 309 of those 441 frames (70 %) the puck is plainly visible to the independent detector. Only 153 frames have no red blob at all.
- **Where the misses are** (obs x, robot end = +x):

| x bin | −0.4…−0.2 | −0.2…0.0 | 0.0…+0.2 | +0.2…+0.4 | +0.8…+1.0 | elsewhere |
|---|---|---|---|---|---|---|
| visible-but-logged-occluded frames | 59 | 138 | 29 | 23 | 53 | 7 |

  The center cluster is the red/black "WINDCHILL" logo: the puck drops out frame after frame while it crosses the logo.
- **Run lengths** (1…10 frames): 87, 28, 7, 11, 7, 9, 2, 4, 1, 1, plus 4 runs over 10 frames. Mean 2.7, maximum 34. The sim's runs are capped at 7.
- **Jump on re-acquire.** While the puck is missing, the observation repeats the last position. When it comes back, the median jump is 9.1 cm after a 1-frame run, 11.0 cm after 2, 17.3 cm after 3, and 19.5 cm after 4 or more (p90 44 cm). This is the freeze-then-jump stutter seen in the camera view.
- **Speed has little effect.** The miss rate is 9–14 % at every puck speed up to 2 m/s, and blob elongation stays at 1.06–1.16. Motion blur is not the mechanism. (The "high-speed" spike in a first-pass speed binning was an artefact of the phantom jumps below.)

### 4. A phantom puck at (0.90, −0.30), flagged as visible

- **What happens:** the logged detector sometimes reports a puck **with `occluded = 0`** at a fixed spot, about (0.894–0.902, −0.298) in the obs frame. That is pixel ≈ (297, 147) in the 320×240 frame: the robot-end rail, where there is printed lettering and ceiling glare.
- **How often:** 49 visible frames plus 33 frames holding the phantom while occluded. It appears in 14 episodes: juggle 2, puck_goal 1, puck_vel 11 of 15.
- **How it starts:** the real puck is lost on the logo. `_select_component_centroid` then picks the candidate closest to the history, and the rail artefact is the only candidate.
- **What the policy sees:** the puck teleports about 1.1 m and back, sometimes flipping between the two for several steps (e.g. juggle `trajectory_data0` steps 22–27, puck_vel `trajectory_data3` steps 4–26).
- **What 40 visible frames look like:** in 40 logged-visible frames the logged position is more than 8 cm from the actual puck. All of the ones listed are this phantom.

### 5. Every episode starts with an 8-step zero-action hold

All 47 episodes begin with exactly 8 steps (~0.4 s) of `policy_action = 0` (the post-reset transition hold) while the puck is already moving from the reset strike. The sim policy controls the paddle from step 0, and its start-state distribution is different.

### Checked and not a problem (or minor)

- **No stale or duplicated camera frames.** Consecutive frames always differ (mean abs diff ≥ 1.75). Only 1 of 2,349 moving visible frames repeated the previous position exactly.
- **Real detection noise is lower than sim's.** It is about 2 mm against the independent detector, plus 4 mm quantization (`_pixel_to_robot_xy`: one detector pixel × 4 × 1 mm). The sim uses σ = 1 cm.
- **Loop timing is tight.** Steps are 49.10 ± 0.50 ms (p99 49.1, max 71.9) against the sim's 50 ms. Frame → detection takes a median 13 ms; frame → servoL takes a median 19 ms (p99 36).
- **Not checked here:** puck free-flight, wall and paddle–puck collision dynamics. Sysid v2 covers those.

## Conclusion

The sim misses three large effects:
- **Puck observation lag:** the puck observation trails the paddle observation by about 120 ms.
- **Paddle speed:** the real paddle is about half as fast in x and a third as fast in y under saturated actions, and it reacts 1–2 steps later.
- **Detector errors tied to the table:** long, place-specific dropouts (the logo and the robot end), plus a phantom puck flagged as valid.

All three are unmodelled. The sim's uniform 5 % occlusion and its 1 cm noise do not resemble the real error process.

Confidence:
- **High** for (1) the camera latency (visual overlay plus lag scan) and (3)/(4) the detector errors (verified frame by frame).
- **High** that the sim paddle is much faster than the real one under these actions (2). The exact split between tracking gains, speed caps and target anchoring is untested.
- Which effect dominates each task's failure is untested.

## Next

1. On the robot, check the latency directly: set `CAP_PROP_BUFFERSIZE=1`, or run a grab thread that keeps only the newest frame. If latency stays above ~50 ms, add a separate puck-only observation delay to the sim.
2. Refit or cap the sim paddle on policy-like bang-bang motions at hist_len 4: direction-dependent speed caps (~1.0 / 0.85 / 0.5 m/s) and onset latency. Consider anchoring the sim target on a lagged setpoint, as the robot does.
3. Fix the detector: handle the logo region and mask the rail artefact. Also reject detections that imply impossible jumps instead of reporting them as visible. Alternatively, train with occlusion that depends on position and comes in bursts, plus occasional phantom outliers.
4. Re-evaluate in sim with (1)–(3) injected to see which effect explains most of the zero-shot drop.
