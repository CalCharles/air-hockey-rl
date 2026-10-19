# Sim2sim: does the real-robot puck-occlusion map hurt the hist4 juggle policies?

- **Date**: 2026-10-16 02:05 UTC
- **Status**: done
- **Run dirs**: policies `runs/td3/sysid_v2_real_workspace/hist4_seed{0..4}/juggle_sysid_v2_hist4/checkpoint_1975000`; eval output `runs/td3/sysid_v2_real_workspace/occlusion_map_sim2sim/` (`summary.md`, `results.json`)
- **Configs**: source `configs/new_juggle/sysid_v2_hist4.yaml`; target `configs/new_juggle/sysid_v2_hist4_occlusion_map.yaml` (= source + `occlusion_map_path`); map `configs/new_juggle/occlusion_maps/real_20261014_no_ends.json`
- **Related**: [`2026-10-15_19-43_real-rollout-sim2real-gap-sources.md`](2026-10-15_19-43_real-rollout-sim2real-gap-sources.md) (where the position-dependent detector dropouts were found)

## Question

The real puck detector drops the puck mostly in one strip of the table: the logo and the band below it, x ≈ −0.3…+0.1. The sim models occlusion as a uniform 5 %. If we add the measured position-dependent occlusion to the sim, how much do the hist4 sysid-v2 juggle policies lose?

## Setup

- **The map.** P(occluded | true puck in a 10 cm cell) comes from the real hist4 evals of 2026-10-14:
  - all six run folders, all tasks, 57 unique episodes (5 duplicated HDF5s removed)
  - the table ends (x < −0.80 and x > +0.78) are excluded: the camera's field of view and the robot arm, not the detector
  - cells with fewer than 5 frames, and everything outside the grid, have p = 0
  - written by `scripts/visualization/plot_occlusion_heatmaps.py --x-min -0.80 --x-max 0.78 --save-map …`
- **The sim model.** A new option in `airhockey/sims/airhockey_box2d.py`, `occlusion_map_path` (default `None`; with it unset no new code runs):
  - each time the *true* Box2D puck position changes cell, the sim draws `occluded = rng.random() < p_cell` and keeps that state while the puck stays in the cell
  - while occluded, the observation reports the last reported puck position with the occluded flag set to 1, as the real env does
  - it is OR-combined with the existing uniform occlusion (`random_occlusion_rate` 0.05), which stays on in **both** envs
  - it uses its own random number generator (seeded from the reset seed), so the other random draws are unchanged
  - checked: source and target give identical observations until the first map occlusion; repeated entries into the 94 % cell occlude 94.2 % of the time
- **The eval.** `scripts/td3/extras/eval_occlusion_map_sim2sim.py`:
  - 5 policies (hist4 seeds 0–4) × **10 episodes**, to match what is feasible on the real robot
  - the same reset seeds 1000–1009 in both envs and for every policy
  - return = summed env reward
  - the table shows the mean ± standard error **across the 5 policies**, where each policy's value is the mean of its 10 episodes

## Results

| | source sim | target sim (+ occlusion map) | target − source (paired) |
|---|---|---|---|
| **Return** | **108.7 ± 8.6** | **90.0 ± 7.4** | **−18.7 ± 5.3** |
| Episode length | 170.8 ± 12.4 | 146.2 ± 9.5 | |

| Policy | Source | Target | Δ |
|---|---|---|---|
| hist4_seed0 | 116.0 ± 15.0 | 81.5 ± 12.7 | −34.5 |
| hist4_seed1 | 137.1 ± 10.6 | 111.4 ± 18.3 | −25.7 |
| hist4_seed2 | 94.2 ± 15.9 | 86.8 ± 16.2 | −7.4 |
| hist4_seed3 | 107.8 ± 20.7 | 101.0 ± 21.4 | −6.8 |
| hist4_seed4 | 88.5 ± 18.9 | 69.3 ± 17.4 | −19.2 |

Occlusion of the newest puck frame in the observation:

| | Occluded steps | Runs | Mean run | Max run | Run lengths 1, 2, …, 7, ≥ 8 |
|---|---|---|---|---|---|
| Source | 8.3 % | 591 | 1.20 | 5 | 501, 69, 17, 0, 4, 0, 0, 0 |
| Target | 18.5 % | 763 | 1.78 | 15 | 487, 159, 41, 19, 27, 19, 2, 9 |

**Calibration.** Over 128 cells with at least 20 visited steps, the map model's realised occlusion rate is within 3.5 percentage points of the map on average (step-weighted). That matches the expected sampling noise.

## Conclusion

- **The map costs about 17 % of return.** Adding the real position-dependent occlusion lowers juggle return by 18.7 ± 5.3. Every one of the 5 policies loses (−7 to −35), and episodes are shorter (171 → 146 steps).
- **It is a real cost, but well below the observed sim-to-real drop.** The 2026-10-14 real eval of a different hist4 bundle went from 130 in sim to 48 on the robot. That bundle is not these policies, so this is not a like-for-like comparison. Occlusion alone does not account for the gap. The camera latency (~120 ms) and the slower real paddle from the gap-sources note remain unmodelled.
- **The target is somewhat harsher than the real robot in total occlusion.** The target has 18.5 % occluded steps; real juggle had 13.4 %, *including* the table ends. That is because the uniform 5 % stays on top of a map that already contains the real background rate (about 1.7 % on the open table).
- **The target is milder than the real robot in long occlusions.** The target's longest run is 15 frames; the real maximum was 34.
- **Confidence.** 10 episodes per policy gives large per-policy standard errors (±11–21). The pooled paired Δ is the reliable number. One evaluation seed set; phantom detections are not modelled.

## Next

1. A target with the uniform occlusion off (map only), to match the real total occlusion rate.
2. Add the camera latency, as a puck-only observation delay, and the real paddle speed limits, then see how much of the real drop the combination reproduces.
3. Train with the map (and later the latency) on, and check whether the target-env loss disappears.
