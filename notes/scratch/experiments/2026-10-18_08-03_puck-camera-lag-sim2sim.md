# Sim2sim: the real camera's 120–150 ms puck lag, put into the sim

- **Date**: 2026-10-18 08:03 UTC
- **Status**: done
- **Run dirs**: policies `runs/td3/sysid_v2_real_workspace/hist4_seed{0..4}/juggle_sysid_v2_hist4/checkpoint_1975000`; output `runs/td3/sysid_v2_real_workspace/puck_lag135_sim2sim/` and `.../puck_lagcam_sim2sim/` (`summary.md`, `results.json`)
- **Configs**:
  - source: `configs/new_juggle/sysid_v2_hist4.yaml`
  - targets: `configs/new_juggle/sysid_v2_hist4_puck_lag135.yaml` (constant 135 ms) and `configs/new_juggle/sysid_v2_hist4_puck_lagcam.yaml` (120 ms + 30 fps frame grid)
- **Eval script**: `scripts/td3/extras/eval_occlusion_map_sim2sim.py` (a generic paired source/target eval despite its name)
- **Follows**: [`2026-10-18_07-51_real-camera-frame-staleness.md`](2026-10-18_07-51_real-camera-frame-staleness.md) and [`2026-10-15_19-43_real-rollout-sim2real-gap-sources.md`](2026-10-15_19-43_real-rollout-sim2real-gap-sources.md) §1

## Question

On the robot, the camera puck trails the robot telemetry by about 120–150 ms. The paddle observation is fresh telemetry. The sim delays puck and paddle together by 25 ms, so there is no relative lag. How much do the hist4 juggle policies lose when the sim puck observation lags by the real amount?

## Sim model

Two new options in `airhockey/sims/airhockey_box2d.py`. Both default to 0, which means off and leaves the code path unchanged. Checked: with the options off, the 10 seed0 returns match the occlusion-map eval exactly, and the physics trajectory is the same with the lag on.

- **`puck_obs_lag_seconds`**
  - The sim keeps a short trace of the true puck state (position and velocity) after every physics sub-step.
  - The reported puck is the true puck from this many seconds earlier, linearly interpolated between sub-steps.
  - It applies to the puck only, on top of the shared `delay_seconds`, so this value is the puck-vs-paddle relative lag.
  - As on the robot, the lagged puck also feeds reward and termination.
- **`puck_obs_lag_frame_period`**
  - When set, the lagged time is snapped to a camera frame grid with a random phase per episode.
  - With 1/30 s and a 120 ms lag, the lag varies over 120–153 ms and consecutive frames are about 33 or 67 ms apart, as with the real 30 fps camera read at 20 Hz.
- Both use their own random number generator, so the other random draws are unchanged.

## Eval

- 5 hist4 policies × 10 episodes per env (the real-feasible count)
- paired reset seeds 1000–1009
- return = summed env reward
- each cell = mean ± standard error across the 5 policies

## Results

| | Source sim | + 135 ms puck lag | + 120 ms lag on a 30 fps grid |
|---|---|---|---|
| **Return** | **108.7 ± 8.6** | **45.8 ± 3.5** | **49.6 ± 1.7** |
| Δ vs source (paired) | – | −62.9 ± 10.3 | −59.1 ± 7.9 |
| Episode length | 170.8 ± 12.4 | 80.8 ± 5.6 | 84.5 ± 1.8 |

| Policy | Source | 135 ms | 30 fps camera |
|---|---|---|---|
| hist4_seed0 | 116.0 | 37.5 | 49.6 |
| hist4_seed1 | 137.1 | 41.9 | 53.2 |
| hist4_seed2 | 94.2 | 43.0 | 43.3 |
| hist4_seed3 | 107.8 | 58.0 | 50.1 |
| hist4_seed4 | 88.5 | 48.6 | 51.8 |

**Reward check.** Re-scored with the *true* puck (steps with the true puck in the upper half), the 135 ms target gets 43.3, against 45.8 from the env reward. The source gets 108.7 under both. So the loss is the policy failing, not delayed scoring.

## Conclusion

- **The puck lag alone costs more than half the return**: 109 → 46–50, −54 to −58 %. Every policy loses 37–95, and episodes are half as long. A constant lag and the 30 fps camera model give the same answer, so the lag itself matters, not the uneven frame spacing.
- **This is the largest single sim-to-real factor found so far.** The occlusion map cost −17 %, and the real start states and the 8-step hold cost nothing.
- **It lands at the real-robot level.** The 2026-10-14 real eval scored 48, against 130 in sim. That was a different hist4 bundle, so the comparison is not like-for-like, but the size of the drop matches.
- **Confidence.** One seed set of 10 episodes, but the effect is about 6× the standard error of the paired difference.

## Also found: the sim's observation history is spaced 25 ms apart, the robot's 49 ms

With `enable_observation_delay: true`, every control step is split at `delay_seconds` into two physics segments. `get_singleagent_transition` appends a puck and a paddle history frame after *each* segment.

- **Sim:** the 5-frame observation history is about 25 ms apart and spans ~100 ms.
- **Robot:** it appends one camera frame per 49 ms step, so it spans ~200 ms.

Checked on a sim rollout: history length grows by 2 per step. The real robot therefore shows the policy per-frame displacements about 2× larger than it was trained on. This was not modelled in these runs.

## Next

1. **On the robot:** set `CAP_PROP_BUFFERSIZE=1` or use a newest-frame grab thread, and re-measure the lag. Whatever remains is the real lag to train with.
2. **Train with `puck_obs_lag_seconds`.** Randomize it, e.g. 100–160 ms, or use the camera-grid mode. Then check that the target loss disappears.
3. **Fix or model the history frame spacing:** one frame per control step, as on the robot. Then re-evaluate.
