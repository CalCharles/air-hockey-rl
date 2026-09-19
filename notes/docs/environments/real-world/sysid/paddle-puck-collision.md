# Sysid section: paddle–puck collision → `puck_restitution` / `paddle_restitution`, `puck_density` (CMA-ES)

The Box2D paddle–puck contact (`CollisionForceListener` in `airhockey/sims/airhockey_box2d.py`)
pins the post-impact *relative* normal speed to `e · approach speed` with a momentum-conserving
reduced-mass impulse. For a head-on hit that gives

```
v_out = −v_in + (1 + e) · r / (r + 1) · (u_p + v_in)        r = m_paddle / m_puck
```

(`v_in` puck speed towards the paddle, `u_p` paddle speed towards the puck, `e = max(puck_restitution,
paddle_restitution)`). This page describes how `e` and `r` are identified from scripted collision
recordings, and why the data only pin their product — the **gain** `(1+e)·r/(r+1)`.

| What | Where |
|---|---|
| Folder | `sysid/paddle_puck_collision/` — `code/`, `data/<session>` → the recording session, `results/<name>/` ([`README.md`](../../../../../sysid/paddle_puck_collision/README.md)) |
| Code (all of it) | `sysid/paddle_puck_collision/code/` |
| Entrypoint | `sysid/paddle_puck_collision/code/fit_collision_cmaes.py --input-dir <session> --out sysid/paddle_puck_collision/results/<name>` |
| Qualitative view | `sysid/paddle_puck_collision/code/render_collisions.py --input-dir <session> --fit-dir <fit out>` — side-by-side videos (camera · real on the table · sim replay) per collision, `<fit out>/videos/` |
| Tests | `pytest sysid/paddle_puck_collision/code/tests -q` (synthetic recordings + sim-generated collisions) |
| Offset evaluation (no fitting) | `sysid/paddle_puck_collision/code/evaluate_offset_collisions.py --input-dir <change_angle session> --fit-dir <fit out>` → `<results>/offset_<session>/` — every oblique hit replayed with the fitted parameters, puck launched straight in the real lane at the measured contact offset; exit speed / angle errors, plots, videos. Result 2026-09-18: speed to 7.5 % up to 5 cm offset, angle 1.9 × too large (frictionless contact), grazes ≥ 6.5 cm unreliable — [`2026-09-18_23-30`](../../../../scratch/experiments/2026-09-18_23-30_offset-paddle-puck-collisions-replay.md) |
| Reference run | `sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision/` (gitignored; regenerated 2026-09-18, identical optimum) — result in [`2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md`](../../../../scratch/experiments/2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md) |
| Background | [`2026-09-10_02-49_paddle-puck-mass-ratio.md`](../../../../scratch/experiments/2026-09-10_02-49_paddle-puck-mass-ratio.md) (why the collision depends on the mass ratio, why `paddle_density` must stay) |

## Data: scripted collision sessions

`configs/robot_data_collection/paddle_motion_config.yaml` on the robot machine, experiment
`puck_collision`. The operator releases the puck at the far end of the tilted table from one
of three positions (`top`, `3/4`, `1/2` — the approach speed at the paddle is ≈ 1.1 / 1.0 /
0.65 m/s); when the tracked puck crosses `trigger_x_obs` the robot executes 20 steps of the
constant action `(−delta, 0)` (`delta` ∈ {0, 0.33, 0.66, 1.00} → paddle plateau speed ≈ 0 / 0.3 /
0.6 / 1.0 m/s up the table) followed by 20 zero-action steps. Twelve conditions, three clean
repeats intended; the 2026-09-10 session has 44 files because bad takes were re-done but kept.

Per file (`collision_<idx>_h<height>_delta<d>_y<offset>.hdf5`): `arm_puck_track` (camera frames
while waiting for the puck), `train_vals` (the canonical 35-column real-world row per step:
`cur_time`, `pose_x/y`, `speed_x/y` in the robot frame, `puck_x/y`, `puck_occluded`), `train_img`,
and attrs with the condition, trigger, controller limits and outcome flags. `manifest.json`
only lists the trials since the last restart of the session — everything is read from attrs.

**Frames.** The puck is in the observation frame (robot end at `x ≈ +0.9`, puck accelerating
towards `+x`). The paddle is `pose_x + center_offset_constant` (1.2) in that frame — verified by
the stationary-paddle trials, whose puck reverses `85.5 ± 7.4 mm` from the shifted pose vs
`r_paddle + r_puck = 82.5 mm`. The camera frames lag the robot clock by **≈ 0.10 s** (calibrated
per session, see below); the stored image and the puck detection share that lag, so "the puck
looks inside the paddle" in the images is expected.

## Measuring one collision (`speeds.py`)

The 30 Hz tracker is sampled by the 20 Hz loop (a frame is read 0–33 ms after capture, sometimes
twice, and the tracker holds its last value while it loses the puck — the arm crosses the camera
view right at impact), so frame differences alias badly. Speeds come from model fits:

1. usable frames = not occluded and not an exact repeat of the previous sample;
2. change-point search: for every usable split, fit the damped free-flight model
   `a = g − γ v` (`fit_damped` from `sysid/common/trajectory_segmentation.py`, sysid
   `g = 0.661`, `γ = 0.178`) to the 6 usable frames before and the 6 after (post frames beyond the
   far wall / side walls or 0.8 s are dropped); the split with the lowest combined residual is the
   collision, provided the pre frames approach the paddle near the robot end and the post frames
   move away;
3. contact time `t_c` = where the two model trajectories cross; `speed_in` / `speed_out` = the
   model velocities at `t_c` (|v|, with the x components and the outgoing angle kept);
4. paddle speed `u_p` = the robot's `speed_x` interpolated at `t_c − lag`, and the lateral offset
   `Δy` between puck and paddle centre at contact.

**Camera lag calibration.** For every moving-paddle trial the lag `L` is the value for which the
paddle pose at `t_c − L` is exactly `r_paddle + r_puck` from the puck at contact; the session lag
is the median (2026-09-10: 102 ± 23 ms over 21 trials, the same for 0.3 / 0.6 / 1.0 m/s paddles).
Only `u_p` depends on it (the puck speeds are camera-only); at the plateau this is a ≤ 5 % effect.

**Validity gates** (a trial that fails any is not a usable head-on collision): outgoing direction
within 40° of head-on; the puck separates from the paddle (`vx_out − u_p ≥ 0.1 m/s`); for
`delta > 0` the hit happened after the paddle reached ≥ 50 % of its plateau speed; the
lag-corrected paddle does not come within 3 cm of re-contacting the post-model puck. **Quality
score** (lower = better): `outgoing angle [deg] + 5 per cm |Δy| + 2 per 10 mm of fit residual +
up to 50 for a hit during the paddle's acceleration`. `select_canonical` keeps the best three
valid trials per condition = the **canonical dataset** (`canonical_dataset.csv`; every file with
its gate and score in `all_trials.csv`, and in `plots/trial_fits.png`).

Sensitivity: `g` / `γ` between (0.55, 0.178), (0.661, 0.178), (0.73, 0.11) and (0.661, 0) move the
mean measured gain by ≤ 0.04.

## Replicating the collision in the sim (`sim_collision.py`)

Per measured collision the env is driven exactly as a policy would: `build_replay_sim_config`
(noise, delays, terminations off; the recording's controller limits) plus a deeper workspace and a
0.4 m per-step move limit (the plant only reaches 0.83 m/s with the robot's 0.26 m lead; the
action → speed map is calibrated by bisection, the plant dynamics are unchanged). The paddle starts
at the real start pose already moving at `u_p`, held there by the PID with the constant action for
that speed; after two steps the puck is placed in its path at `speed_in` so contact happens 20 ms
into the next 50 ms env step; five more steps are run and the puck velocity at the end is the sim's
outgoing speed (gravity and the puck's damping are off so it is read exactly; secondary contacts
of a recoiling paddle are included by construction). Where a single contact occurs the result
equals the closed form above to < 0.02 m/s (tested).

`e` is written to **both** `puck_restitution` and `paddle_restitution` (the listener uses the
max); `r` is realised as `puck_density = m_paddle / (r · π · r_puck²)` with `paddle_density`
fixed at its fitted value (it is the PID plant inertia).

## Objective, search, validation

- objective = RMS over the training collisions of `speed_out_sim − speed_out_real` (m/s);
- one trial per condition held out (`--val-repeat 3` = the third-ranked); canonical (e, r) from
  the base config scored first;
- CMA-ES in the unit square (`e` linear 0–1.5, `r` log 0.25–1000), popsize 12, ≤ 40 iterations,
  IPOP restart; every candidate's validation error is recorded and the selection is checked
  against the candidate set with `fit_validation.percentile_report`;
- a 31 × 31 grid scan draws the landscape (`plots/landscape.png`) with the fitted ridge, and the
  summary lists the ridge points at `r = 1000` (rigid paddle), `4.46` (real weights 58 g / 13 g)
  and the canonical `2.56`.

## The degeneracy — what the fit does and does not determine

Every `(e, r)` with the same gain `(1+e)·r/(r+1)` reproduces the puck speeds identically (the
2026-09-10 fit: train RMS 0.134 m/s at r = 2.56, 4.46, 398 and 1000). The data determine the gain
(≈ 1.63); which point of the ridge to use is a modelling decision:

- large `r` (`e ≈ 0.63`): the paddle does not recoil, matching a position-controlled UR5 whose
  recorded speed is unchanged through the impact; the sim paddle then keeps its speed after a hit;
- `r = 2.56` (`e ≈ 1.26`): today's densities; the paddle loses ~60 % of its speed on every hit and
  a super-elastic `e` compensates;
- breaking the degeneracy would need the paddle's post-impact motion as a second observable
  (the robot's `speed_x` through the contact is recorded and could be added as a term).

## Side-by-side videos (visual check)

`render_collisions.py` writes, for every selected collision (or `--trials all` / names), a GIF and an
MP4 with three panels on the camera clock, one frame per real 20 Hz step (played at `--fps 10`):

1. **camera** — the stored `train_img` frame of that step (it shares the ~0.1 s lag of the puck
   detection, so the arm is seen where it was 0.1 s earlier);
2. **real** — the Box2D table with the env's sprites: the tracked puck (a grey ghost at the fitted
   model's position while occluded), the robot paddle at `t − lag`, the pre / post model trails
   (blue / orange), the contact point (green X), the puck trail;
3. **sim** — `HeadOnCollider.run` for the fitted (or `--params canonical` / `E R`) parameters with
   `puck_launch="free"`, `puck_physics=True` (the puck slides under the sysid gravity / damping,
   launched so it arrives at the measured incoming speed) and the paddle started at constant speed
   so that it reaches the real contact position at the real contact step. The rendered run
   reproduces the fit's outgoing speed to < 0.01 m/s (`render_summary.csv` lists real, rendered-sim
   and fit-run values per trial).

What to look for: the two puck trails should separate from the paddle at the same frame and stay
on top of each other afterwards; a slower / faster sim puck is the residual of that trial. The
sim paddle moves at its plateau speed from the first frame (the real one accelerates over the
first ~0.1 s), so the paddles only coincide from the contact on. Reference set:
`sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision/videos/` (35 selected collisions,
`mosaic_<condition>.png` = last frames per condition).

## Reproduce

```bash
python sysid/paddle_puck_collision/code/fit_collision_cmaes.py \
    --input-dir /data2/air_hockey/robot_data_collection_puck_collision_20260910_1719 \
    --out sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision --val-repeat 3 --workers 8 --restarts 2
```

Outputs: `summary.md`, `canonical_dataset.csv`, `all_trials.csv`, `selection.json`,
`lag_calibration.json`, `split.json`, `fit_result.json`, `candidates.csv`, `evaluations.json`,
`per_trial.csv`, `landscape_grid.json`, `sim_config_fitted.yaml`, `plots/`; then

```bash
python sysid/paddle_puck_collision/code/render_collisions.py \
    --input-dir /data2/air_hockey/robot_data_collection_puck_collision_20260910_1719 \
    --fit-dir sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision        # videos/{gifs,mp4,png}, mosaics, render_summary.csv
```
