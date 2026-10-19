# Sim eval from the real-robot episode start states

- **Date**: 2026-10-18 07:38 UTC
- **Status**: done
- **Run dirs**: policies `runs/td3/sysid_v2_real_workspace/hist4_seed{0..4}/juggle_sysid_v2_hist4/checkpoint_1975000`; output `runs/td3/sysid_v2_real_workspace/real_start_sim2sim/` (`summary.md`, `results.json`, `real_starts.json`)
- **Config**: `configs/new_juggle/sysid_v2_hist4.yaml` (unchanged; the start states are injected at reset)
- **Script**: `scripts/td3/extras/eval_real_start_sim2sim.py`
- **Related**: [`2026-10-15_19-43_real-rollout-sim2real-gap-sources.md`](2026-10-15_19-43_real-rollout-sim2real-gap-sources.md), [`2026-10-16_02-05_occlusion-map-sim2sim.md`](2026-10-16_02-05_occlusion-map-sim2sim.md)

## Question

The sim's juggle reset is broad. The puck spawns uniformly over the far ⅔ of the table at 0–0.5 m/s in a random direction, and the paddle spawns uniformly in the workspace. The real robot starts every episode from the end of the scripted reset strike, which is much narrower:

- the puck is mid-table (x −0.53…−0.07)
- it moves at 0.5–1.5 m/s, always toward the far end
- the paddle is at rest in one of two corners, (0.67, ±0.37)

Are the real starts harder for the hist4 juggle policies than the sim's own starts?

## Setup

- **Start states.** There is one start state per real episode: 40 unique episodes from the four canonical 2026-10-14 runs (juggle, puck_goal, puck_goal_vel, puck_vel). All four tasks use the same reset policy.
  - Three puck_vel episodes are dropped because their first detection is the BRUNSWICK phantom. Phantom frames are also left out of the fit.
  - The puck's position and velocity come from a linear fit to its first 5 visible frames, after removing the sim's 0.73 m/s² down-table acceleration from x.
  - A two-frame difference is unreliable. The per-step puck speed alternates between about 0.6 and 1.2 m/s, which suggests camera frames alternately ~33 and ~66 ms apart, i.e. a ~30 fps camera sampled at 20 Hz.
  - The paddle start is its telemetry pose at step 0.
- **Check.** With noise and occlusion off and zero actions, the sim puck spawned from a real start follows the real track to within about 1 cm over the first 6 steps (3 episodes checked).
- **Conditions.** All three use the same reset seeds 1000–1039 for every policy and condition.
  - `sim`: the sim's own reset.
  - `real_puck`: real puck position and velocity, with a sim-random paddle.
  - `real_puck_paddle`: real puck and real paddle position. The +y corner (y = 0.38) is slightly outside the sim workspace (y_max 0.352).
- **Policy start.** The policy acts from step 0. The real 8-step zero-action hold is not reproduced.

## Results

Each cell is the mean ± standard error across the 5 policies. Each policy's value is the mean of its 40 episodes.

| | Sim reset | Real puck start | Real puck + paddle start |
|---|---|---|---|
| **Return** | **110.6 ± 6.2** | **120.8 ± 5.9** | **124.0 ± 4.1** |
| Δ vs sim reset (paired) | – | +10.2 ± 1.6 | +13.5 ± 3.1 |
| Episode length | 180.5 ± 9.4 | 181.9 ± 8.2 | 188.6 ± 6.0 |

| Policy | Sim reset | Real puck | Real puck + paddle |
|---|---|---|---|
| hist4_seed0 | 115.3 ± 7.6 | 122.1 ± 6.8 | 122.7 ± 6.6 |
| hist4_seed1 | 121.5 ± 6.1 | 132.4 ± 6.0 | 129.6 ± 7.0 |
| hist4_seed2 | 110.8 ± 6.6 | 126.1 ± 6.8 | 122.3 ± 6.4 |
| hist4_seed3 | 86.7 ± 8.2 | 98.2 ± 7.8 | 110.6 ± 8.4 |
| hist4_seed4 | 118.5 ± 7.1 | 125.2 ± 6.7 | 135.1 ± 5.9 |

## Conclusion

- **The real start distribution does not explain the sim-to-real drop.** In sim, the real starts are slightly *easier* than the sim's own reset: +10 to +13 return, and all 5 policies gain.
- **The likely reason is that the real reset gives the policy time.** The puck is sent away from the robot and comes back under gravity. The sim reset sometimes spawns the puck near the robot end, moving toward the paddle.
- **Caveats.**
  - The 8-step zero-action hold is not reproduced.
  - The start state is the camera's view of the puck, which is ~120 ms old according to the gap-sources note. Both effects only delay the first decision while the puck is still moving away, so they are unlikely to flip the sign.
  - One seed set.

## Next

The remaining candidates for the real drop are still the camera latency / stale frames and the slower real paddle.
