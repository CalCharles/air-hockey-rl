# Actual-TCP drift in long real episodes

Investigated 2026-11-01. In long episodes (more than about 2000 steps) the paddle
starts touching the robot-end wall. The workspace clip is working: every commanded
target stays in bounds. The problem is that the actual TCP drifts away from the
command over the episode, and nothing in software bounds the actual position.

Run that showed it:

```
python -m scripts.td3.extras.async_td3_real_eval --agent history \
  --config configs/real_configs/rollout_td3_config_hist4_T10000.yaml \
  --model-path data/new_sim_best_policies/night8__ft_term03_nopretanh_s41__swa1200k \
  --collector-device cpu --eval-episodes 3 --episode-gif-max-frames 2000 --append \
  --first-reset-side right \
  --data-root-dir real_runs/eval_new_sim_T10000/s41
```

Data: `real_runs/eval_new_sim_T10000/s41/data_20261101-161124/episode_hdf5/`.

## How the bounds are enforced in code

- Workspace in robot frame: `REAL_WORKSPACE_LIMS = (-0.83, -0.45, -0.33, 0.39)`
  (`airhockey/sims/air_hockey_real.py:72`), plus the chamfered far corners
  (`effective_x_max` in `airhockey/sims/real/coordinate_transform.py`).
- Robot-end wall: `ROBOT_END_WALL_TCP_X = -0.42154`, the TCP x with the paddle
  touching the wall (`airhockey/sims/real/table_calibration.py`). That leaves 2.85 cm
  of clearance at x_max = -0.45.
- Each step, the filtered servoL target is clipped again with `clip_limits` after
  `filter_update` (`air_hockey_real.py:2255`). Nothing in that path depends on the
  step count.
- The wider reset workspace (`set_reset_workspace`) is always restored in the
  `finally` block of `scripts/td3/helper/real_reset_runner.py:310`.
- `isPoseWithinSafetyLimits` is called on the commanded pose.

**Every check runs on the commanded pose. None of them check the actual TCP.**

## What the logs show

HDF5 fields used: `paddle_cmd[:, 6:8]` is the post-filter, post-clip servoL target
(robot frame). `paddle_actual[:, 0:4]` is the actual TCP xy and velocity (robot
frame). `pose` is the paddle position in the table/observation frame
(robot x + 1.3359).

### Max x per 250-step window, episode 1 (2437 steps)

| Steps | Max commanded x | Max actual x |
|---|---|---|
| 0–250 | −0.4500 | −0.4497 |
| 500–750 | −0.4500 | −0.4401 |
| 1000–1250 | −0.4500 | −0.4330 |
| 1500–1750 | −0.4500 | −0.4272 |
| 1750–2000 | −0.4500 | **−0.4215** (touching the wall) |
| 2250–2437 | −0.4500 | −0.4149 |

Episode 3 (2666 steps, ended in an e-stop) follows the same pattern and reaches
**−0.4035** at steps 2500–2666, about 1.8 cm past the point where the paddle touches
the wall.

### Offset between actual and commanded position, paddle nearly still (|v| < 0.05 m/s)

Median of (actual − commanded), per 500 steps:

| Episode | 0–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500+ |
|---|---|---|---|---|---|---|
| 1 x | +0.000 | +0.009 | +0.018 | +0.022 | +0.038 | — |
| 1 y | +0.000 | −0.017 | −0.012 | +0.011 | −0.023 | — |
| 3 x | +0.000 | +0.021 | +0.044 | +0.030 | −0.022 (n=5) | +0.046 |
| 3 y | +0.000 | +0.002 | +0.010 | −0.010 | −0.017 | −0.025 |

Over all steps (not just the slow ones), episode 1's median offset grows to
x +0.025 and y −0.030 m by the end.

### Other observations

- **The commands stay in bounds.** No commanded target goes past −0.45 at any step,
  and the safety check never failed.
- **The whole reachable range shifts, not just the edge.** The lowest actual x also
  rises, from about −0.79 to −0.74. So this isn't overshoot at the boundary; the
  whole range is offset.
- **The drift rate is roughly linear:** about 1.5 cm in x per 1000 steps (about 50 s).
  With 2.85 cm of clearance, the paddle reaches the wall at around 1800–2000 steps,
  which matches what was seen on the robot.
- **The offset is back to 0 at the start of every episode.** The reset calls
  `forceModeStop`, moves the arm, and re-applies the z force.
- **The loop isn't slowing down.** The step time stays at about 49 ms median and
  54 ms p95 for the whole episode.
- **y drifts too**, by about 2–3 cm, so the side walls are also at risk on long
  episodes.
- **Sim-to-real side effect:** the observation uses the actual TCP
  (`use_actual_tcp_for_state`). The policy sees itself drifting toward the wall
  while its commands are held at the clip limit.

## Likely cause (not yet confirmed on hardware)

`apply_negative_z_force` (`airhockey/sims/real/robot_control.py:5`) calls

```
ctrl.forceMode(force_frame, [0, 0, 1, 0, 0, 0], [0, 0, ±5, 0, 0, 0], 2,
               [2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
```

According to the UR script manual, for non-compliant axes (x and y here) the
`limits` entries are the *maximum allowed deviation* between the actual TCP and the
programmed TCP. At 2.0 m, x/y deviation is effectively unbounded. Small offsets from
puck hits, friction and the downward press can then build up without being
corrected. This fits the evidence: the drift is only in x/y, it grows steadily, and
it clears when force mode is restarted at reset.

## Possible fixes (none implemented)

1. **Tighten the x/y deviation limits** in `force_limits` to a few millimetres, so the
   UR controller holds x/y to the programmed path. This is the cleanest fix. Test it
   at low speed first, because forceMode may protective-stop when the limit is hit.
2. **Re-apply force mode periodically** in long episodes, or re-anchor when the
   actual-vs-commanded offset grows. This resets the drift the same way a reset does.
3. **Add a software guard on the actual TCP position:** stop, hold or re-anchor when
   the actual TCP is within some margin of a wall, or clip commands using the
   actual-vs-commanded offset.

To check a fix, rerun a long episode and track the median of
`paddle_actual[:, 0] - paddle_cmd[:, 6]` over time. It should stay near 0.

Related: [`paddle-clamping-coverage-gap.md`](paddle-clamping-coverage-gap.md)
(forceMode timeout/coverage behaviour).
