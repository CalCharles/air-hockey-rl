# Automatic trajectory segmentation (free fall / wall / paddle)

Tool that chunks a real split-schema HDF5 recording into contiguous, non-overlapping
segments labelled by what the puck is doing, so sysid inputs (puck free-flight
clips, wall bounces, paddle impacts) no longer have to be hand-labelled.

| What | Where |
|------|-------|
| Library (load, calibrate frames, detect + classify events, build segments, export) | `sysid/common/trajectory_segmentation.py` |
| Rendering (GIF with camera panel + timeline, diagnostic PNG) | `sysid/common/segment_rendering.py` |
| CLI | `sysid/common/segment_trajectories.py` |
| Evaluation set (10 random mouse-teleop trajectories, seed 0) | `sysid/common/runs/mouse_dataset/segmentation_eval/` (gitignored, local; produced by stage 0 of `sysid/common/run_puck_wall_sysid.py --eval-sample 10` — see [`sysid-pipeline.md`](sysid-pipeline.md)) |
| Evaluation writeup | [`notes/scratch/experiments/2026-09-10_00-05_auto-trajectory-segmentation-eval.md`](../../../scratch/experiments/2026-09-10_00-05_auto-trajectory-segmentation-eval.md) |

## Usage

```bash
# 10 random trajectories from the shared mouse dataset (the canonical eval set)
python sysid/common/segment_trajectories.py \
    --input-dir shared/mouse_state_data_all_new_len_gt130_take100_trim30/trimmed_hdf5 \
    --sample 10 --seed 0 --out sysid/common/runs/mouse_dataset/segmentation_eval

# a single older recording without camera images (the 2026-04 curated clips now live in the
# archive /data2/air_hockey/sysid_legacy_20260918/wall_collision_teleop/)
python sysid/common/segment_trajectories.py \
    --inputs /data2/air_hockey/sysid_legacy_20260918/wall_collision_teleop/trajectory_data0.hdf5 \
    --out sysid/common/runs/wall_collision_teleop_data0/segmentation_eval --no-hdf5
```

Per trajectory the tool writes `<out>/<stem>/`:

- `segments.json` — ordered segment list (`label`, inclusive `start`/`end` frame,
  `n_frames`, `duration_s`, `valid_fraction`, `meta`), all detected events with their
  metrics, the frame calibration and the config used.
- `segmentation.gif` — top-down table (sim frame, robot on the right, +y up, i.e. the
  overhead camera's orientation) with a label-coloured puck trail, the camera image when
  the file has one, a banner with the current label and event metrics, and a colour
  timeline of the whole trajectory. 10 fps = half real time.
- `segmentation.png` — x(t), y(t), local speed, split-fit velocity jump `dv` and
  puck–paddle distance with label bands. Fastest way to audit a whole trajectory.
- `segments/<label>_<start>_<end>.hdf5` — split-schema slice per free-fall / wall /
  paddle segment (same datasets as the source minus `image` unless `--include-images`;
  attrs carry label, frame range, source path and the axis calibration).

`<out>/summary.md` tabulates segment counts per trajectory; `<out>/sample_manifest.json`
records the sampled files, seed, calibration and config. Any
`SegmentationConfig` field can be overridden with `--cfg key=value`.

## Labels

| Label | Meaning | Segment extent |
|---|---|---|
| `free_fall` | puck moving under gravity + damping only — the input for puck sysid fits | everything between events; short occlusion gaps (≤ `max_occlusion_gap` = 10 frames) are bridged, the `puck` occluded flag is preserved in the slice |
| `wall_collision` | bounce off a wall; `meta.events[].wall` ∈ x+ / x- / y+ / y- (x± are the end walls in the sim frame, y± the side walls) | last clean pre-impact frame → first clean post-impact frame, ± `collision_pad` (1) |
| `paddle_collision` | puck hit or pushed by the robot paddle | same |
| `opponent_hit` | velocity jump on the opponent half (x > 0) away from the paddle — a human hitting / catching the puck | same |
| `unknown_impulse` | velocity jump on the robot half that is neither near a wall nor within paddle contact | same |
| `rest` | ≥ 3 frames below 0.08 m/s | run |
| `occluded` | occlusion gap longer than `max_occlusion_gap` | run |

Event metadata: `speed_pre`, `speed_post`, `speed_ratio`, `dv`, `dp`, `gap_frames`
(occluded / stale frames spanned by the impact), `paddle_min_dist`,
`paddle_approach_speed`, `wall`, `wall_normal_pre/post`. Free-fall metadata: full-segment
damped-model fit `fit_rms_m`, `v0`, `speed0`, `n_usable`.

## Method

1. **Stale samples.** The tracker runs asynchronously from the 20 Hz loop; a repeated
   puck reading while the puck moves is a stale duplicate and is dropped from all fits
   (≥ 4 identical readings count as a genuine rest).
2. **Split fits.** For every usable frame *s*, the damped free-flight model
   `a = g − γ v` (linear LSQ in `(p0, v0)` for fixed `g, γ` from
   `sysid_best_params.yaml`) is fitted to the 4 usable frames before *s* and the 4
   from *s* on; both are evaluated at `t_s`. The velocity jump `dv` and position jump
   `dp` between them are the event statistic. Raw finite-difference velocities are
   useless here: camera/loop aliasing makes consecutive-frame speeds alternate between
   ~1× and ~2× the true value, whereas the 4-point fits are stable (median `dv` in free
   flight 0.08 m/s, wall bounces 0.7–2 m/s).
3. **Events.** Frames with `dv > 0.35 m/s` or `dp > 6 cm` are grouped; the peak of each
   group is the event. Occlusion gaps are handled automatically because the pre/post
   windows straddle them.
4. **Classification.** Paddle if the puck path across the impact window comes within
   `r_paddle + r_puck + 0.06 m` of the paddle (generous because the tracked puck lags the
   robot pose by ≈ 1 frame, so it visibly "penetrates" the paddle by ~3 cm at impact);
   wall if the puck is within 8 cm of a wall line and the normal velocity component
   reverses; opponent if neither and the puck is on the opponent half; otherwise
   unknown. An unknown event within 4 frames of a classified one is absorbed by it (the
   lag makes the velocity jump show up before the geometric contact).
5. **Segments.** Event windows are painted over a free-fall background; rests and long
   occlusions are cut out; contiguous runs of one label become segments.

## Frame calibration — read this before trusting `pose`

The recordings do **not** share one coordinate convention, and the tool calibrates it
from the data on every run (printed on the first line, stored in every output):

- **Puck.** The tracker logs the puck in "base" coordinates with the robot at `x > 0` and
  the puck accelerating towards `+x` (measured ≈ +0.71 m/s² on the mouse dataset, +0.51
  on `wall_collision_teleop`). That is the convention documented in
  [`puck-system-id.md`](puck-system-id.md) (its `gx = −0.661` is a *deceleration*
  parameter). The tool works in the **sim frame** (robot at `x < 0`, physical
  acceleration `gravity_x = −0.661`), so it mirrors the puck x axis whenever the measured
  free-flight acceleration has the opposite sign to `gravity_x` (`puck_x_sign = −1`).
- **Paddle.** `pose` is *not* the physical paddle in the puck frame for the mouse dataset:
  there `pose_x = robot_x − 1.2` while the physical paddle sits at ≈ `robot_x − 0.1` in
  the raw puck frame (verified frame-by-frame against the camera images: the arm hits
  the puck at raw x ≈ +0.26 while `pose` says −0.77). Using `pose` directly, the puck
  passed straight through the logged paddle in 95 / 100 trajectories and only 2 paddle
  collisions were found. In the older recordings (the 2026-04 `wall_collision_teleop` and
  curated `paddle_puck_collision` clips, archived under `/data2/air_hockey/sysid_legacy_20260918/`) `pose` *is* the physical paddle in the raw
  puck frame. The tool therefore fits `x_sim(paddle) = sign · pose_x + offset` by
  maximising `(#on-table velocity jumps within paddle contact) − 0.5 · (#free-flight
  frames inside the paddle)` over sign ∈ {±1} and a 2 cm offset grid. Results:
  mouse dataset `−1 · pose_x − 1.08` (1283 / 2114 impulses explained, 148
  pass-through frames over 100 files); curated clips `−1 · pose_x − 0.02`; a
  near-stationary paddle (`wall_collision_teleop`) is degenerate but the chosen mapping
  places it correctly. Override with `--puck-x-sign`, `--paddle-x-sign`,
  `--paddle-x-offset` if a dataset is known.

## Known limitations

- Single event per impact window: a paddle hit immediately followed by a wall bounce
  within ~2 frames merges into one event labelled by the stronger jump.
- The classifier only sees the tracked puck and the (calibrated) pose; a human hand on
  the robot half still lands in `unknown_impulse`, and a paddle hit whose contact frames
  are occluded can be attributed to the nearby end wall.
- The free-fall fit uses the canonical `g = −0.661, γ = 0.178`; on the mouse dataset the
  per-segment quadratic acceleration is −0.46 … −0.97 m/s² and long segments fit to
  ~1.1 cm rms (p90 2.3 cm). Segments with rms above ~3 cm deserve a look before being
  used for sysid.
- `fit_window = 4` with `max_window_span = 14`: an event within 3 usable frames of the
  recording boundary is not evaluated.
