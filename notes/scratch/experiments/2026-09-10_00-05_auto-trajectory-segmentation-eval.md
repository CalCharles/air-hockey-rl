# Automatic trajectory segmentation (free fall / wall / paddle) — tool + 10-trajectory evaluation set

- **Date**: 2026-09-10 00:05 UTC
- **Status**: done
- **Run dirs**: `sysid/auto_segments/mouse_sample10/` (eval set, gitignored), `sysid/auto_segments/wall_collision_teleop_data0/` (second dataset smoke test)
- **Code**: `scripts/sysid/segment_trajectories.py`, `scripts/sysid/helper/{trajectory_segmentation,segment_rendering}.py`
- **Doc**: `notes/docs/environments/real-world/trajectory-auto-segmentation.md`

## Question
Can the sysid inputs (puck free-flight clips, wall bounces, paddle impacts) be cut out of raw
recordings automatically instead of by hand, and does it work on the shared mouse-teleop
dataset (`shared/mouse_state_data_all_new_len_gt130_take100_trim30/trimmed_hdf5`, 100 files)?

## Setup
Split-fit event detector (damped free-flight model fitted on 4 usable frames before / after
every candidate split; velocity jump `dv > 0.35 m/s` or position jump `> 6 cm` = event),
geometric classification (wall: near a wall line + normal component reversal; paddle: puck path
within `r_paddle + r_puck + 6 cm`; opponent: on the `x > 0` half; else unknown), lag-aware merge of
an unknown jump into a classified event ≤ 4 frames later. Physics `g = −0.661, γ = 0.178`
(`sysid_best_params.yaml`). Eval set = 10 files drawn with `random.Random(0)` from the 100;
every file gets a GIF (top-down + camera + timeline) and a diagnostic PNG.

## Results

**Frame calibration (the main finding).** The dataset's `puck` and `pose` are not in the same
frame. Measured free-flight acceleration is +0.71 m/s² in the logged puck frame (robot at
`x > 0`, the "base" convention of `puck-system-id.md`), so the puck is mirrored into the sim
frame. `pose_x = robot_x − 1.2` places the paddle on the wrong side of the table centre: using
it directly, the puck passed *through* the logged paddle without any velocity change in
95/100 files and only 2 paddle collisions were detected in the whole dataset. Camera frames
confirm the arm hits the puck at raw puck x ≈ +0.26 while `pose` says −0.77. A data-driven
calibration (`x_sim = sign·pose_x + offset`, maximising impulses-at-contact minus
free-flight-inside-paddle) gives `−1·pose_x − 1.08` on the 100 files (1283/2114 on-table
impulses explained, 148 pass-through frames) and `−1·pose_x − 0.02` on the curated
`sysid/paddle_puck_collision` clips, i.e. the older recordings log the physical paddle
directly. A second effect: the tracked puck lags the pose by ≈ 1 frame, so at impact it
appears ~3 cm inside the paddle (also true in the curated clips, mean apparent centre
distance 3.3 cm vs the physical 8.3 cm).

**Eval set (seed 0), segments per trajectory:**

| trajectory | frames | free_fall | wall | paddle | opponent | unknown | rest | occluded | longest free-fall (fr) | free-fall fit rms (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| data108 | 149 | 4 | 2 | 1 | 0 | 0 | 0 | 0 | 42 | 1.22 |
| data183 | 179 | 12 | 6 | 1 | 2 | 1 | 1 | 0 | 21 | 0.78 |
| data191 | 152 | 8 | 5 | 2 | 0 | 0 | 0 | 0 | 45 | 1.13 |
| data213 | 245 | 12 | 4 | 5 | 0 | 0 | 3 | 0 | 37 | 1.35 |
| data216 | 107 | 5 | 3 | 1 | 0 | 0 | 0 | 0 | 24 | 1.46 |
| data23 | 180 | 7 | 3 | 2 | 1 | 0 | 0 | 1 | 52 | 1.79 |
| data251 | 333 | 14 | 5 | 6 | 0 | 1 | 1 | 0 | 36 | 1.12 |
| data252 | 197 | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 78 | 2.08 |
| data256 | 154 | 7 | 3 | 2 | 1 | 0 | 0 | 0 | 37 | 1.08 |
| data9 | 185 | 6 | 2 | 2 | 1 | 0 | 0 | 1 | 36 | 1.28 |
| **total** | 1881 | 78 | 33 | 24 | 5 | 2 | 5 | 2 | | |

Frames: free_fall 1498 · wall 179 · paddle 152 · opponent 20 · unknown 8 · rest 22 · occluded 2.

**Whole dataset (100 files, 20 372 frames):** free_fall 884 segments (602 with ≥ 10 frames,
352 with ≥ 20), paddle 386 events, wall 398, opponent 46, unknown 25; long free-fall fit rms
median 1.1 cm (p90 2.3 cm). Remaining unknowns sit 14–18 cm from the calibrated paddle
(robot half), i.e. mostly fast paddle hits just outside the contact slack.

**Spot checks against the camera (data213):** paddle impacts at frames 42–48 and 126–131 and
the far-end human hits are where the labels say; the two lag-induced "unknown then paddle"
pairs merged correctly after the absorb step. Visual QA of the remaining nine GIFs is left
to the user — that is what the eval set is for.

**Older recording (`wall_collision_teleop/trajectory_data0`, 1402 frames, no camera):** runs
with the same defaults; calibration picks a near-stationary paddle mapping; 54 free-fall,
17 wall, 6 paddle, 6 opponent, 17 unknown, 12 occluded.

## Conclusion
The segmenter is usable as-is for harvesting free-flight clips and wall bounces, and for
paddle impacts once the paddle frame is calibrated — which the tool now does automatically
and records in every output. Anyone using `pose` from the mouse dataset for anything
geometric (contact checks, replay in sim) must apply the same mapping. Thresholds were tuned
on this dataset only (single camera / loop rate); re-check `dv_threshold` on data with a
different tracker cadence.

## Next
- Confirm the paddle mapping's physical origin (`robot_x − 0.1` in the raw puck frame) in the
  real-env code path, and fix the logger so `pose` is written in the puck frame.
- Feed the auto-extracted free-fall segments into `sysid/puck_grid_search.py` (needs its
  `−gx` sign convention and raw-frame inputs — the slices are written in the sim frame, so
  mirror x back or run the fit with `gx = +0.661`).
- Joint (g, γ) refit on the ~350 long free-fall segments: the per-segment quadratic
  acceleration spread (−0.46 … −0.97 m/s²) suggests the tilt is not uniform across the table.
