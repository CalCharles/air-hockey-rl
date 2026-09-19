# sysid/paddle_puck_collision — paddle–puck collisions → restitution `e` + mass ratio `r`

Fits the paddle–puck **restitution** `e` and the **mass ratio** `r = m_paddle / m_puck` of the Box2D
env to a scripted collision session (`robot_data_collection_puck_collision_<stamp>/`): the puck is
released from three table heights and slides into the paddle, which strikes it head-on at four
action magnitudes (0 / 0.33 / 0.66 / 1.00), three clean repeats per condition intended. Puck speeds
come from damped free-flight fits before / after the contact, the paddle speed from the robot; the
env's own paddle plant replays every collision and CMA-ES matches the outgoing puck speed.

```
code/     dataset.py, speeds.py, sim_collision.py, cmaes_fit.py, report.py, overlay.py   library
          fit_collision_cmaes.py   entrypoint: measure → lag → select best 3 per condition → split → canonical (e, r) → CMA-ES → ridge → validation
          render_collisions.py     side-by-side videos per collision (camera · real on the table · sim replay)
          evaluate_offset_collisions.py   replay an *offset* session with the fitted parameters (no fitting): exit speed / angle errors + videos
          tests/                   synthetic recordings + sim-generated collisions (~5 s)
data/     <session> → the recording session on /data2 (symlink, created by the entrypoint)
results/  <name>/summary.md, fit_result.json, canonical_dataset.csv, all_trials.csv, selection.json, lag_calibration.json,
          split.json, candidates.csv, evaluations.json, per_trial.csv, landscape_grid.json, sim_config_fitted.yaml, plots/,
          videos/ (from render_collisions.py)
          offset_<session>/  evaluate_offset_collisions.py: summary.md, per_trial.csv, all_trials.csv, evaluations.json, plots/, videos/
```

```bash
python sysid/paddle_puck_collision/code/fit_collision_cmaes.py \
    --input-dir /data2/air_hockey/robot_data_collection_puck_collision_20260910_1719 \
    --out sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision --val-repeat 3 --workers 8 --restarts 2
python sysid/paddle_puck_collision/code/render_collisions.py --input-dir <session> --fit-dir sysid/paddle_puck_collision/results/<name>   # → <name>/videos/
python sysid/paddle_puck_collision/code/evaluate_offset_collisions.py \
    --input-dir /data2/air_hockey/robot_data_collection_puck_collision_change_angle_20260910_1818 \
    --fit-dir sysid/paddle_puck_collision/results/cmaes_20260910_puck_collision       # → results/offset_<session>/
pytest sysid/paddle_puck_collision/code/tests -q
```

| File | What |
|---|---|
| `dataset.py` | `load_session` (HDF5 attrs, not the manifest — it only lists the last restart) → `CollisionTrial` with the puck camera track (arm-wait frames + step frames, stale / occluded frames flagged) and the robot paddle track shifted into the puck frame (`pose_x + 1.2`); `split_train_val` (one trial per condition held out) |
| `speeds.py` | `estimate_collision`: change-point search over usable frames, damped free-flight fits (`sysid/common/trajectory_segmentation.fit_damped`) before / after, contact time from the intersection, `speed_in` / `speed_out` at contact, paddle speed `u_p` from the robot at `t_c − camera lag`; validity gates (outgoing angle ≤ 40°, puck separates, hit after the paddle reached speed, no re-contact); quality score; `calibrate_camera_lag`; `select_canonical` (best 3 per condition) |
| `sim_collision.py` | `HeadOnCollider`: the Box2D env with the paddle at the real start pose moving at `u_p` under its own PID (action calibrated per speed), the puck placed in its path at `speed_in`; the outgoing puck speed after the contact step(s) is the sim measurement. `evaluate_measurements` → RMS outgoing-speed error (m/s). `closed_form_out_speed` = the single-impulse formula |
| `cmaes_fit.py` | `ParamBounds` (e linear 0–1.5, r log 0.25–1000), fork-pool `CandidateEvaluator`, `run_cmaes` (IPOP restarts, val error of every candidate), `grid_scan` (landscape) |
| `report.py` | CSVs, `trial_fits.png` (every file with its fits and gates), `gain_vs_speed.png`, `landscape.png` (with the constant-gain ridge), `sim_vs_real.png`, `convergence.png`, `summary.md` |
| `fit_collision_cmaes.py` | entrypoint (percentile validation via `sysid/common/fit_validation.py`) |
| `evaluate_offset_collisions.py` | no fitting: every trial of an offset session replayed with the fitted (e, r), puck launched straight in the real lane (`run(..., dy=)`), exit speed / angle errors per trial and per offset, the offset the sim would need for the real angle, plots, videos |
| `overlay.py`, `render_collisions.py` | side-by-side videos per collision — camera image · real trajectory on the Box2D table (tracker puck, lag-corrected robot paddle, fitted pre / post models, contact point) · sim replay — as GIF + MP4 (`--fps 10` = half speed), last-frame PNG, per-condition mosaics, `render_summary.csv` |

**Read the summary's degeneracy section before using a fitted (e, r).** Head-on speeds only
identify the gain `(1+e)·r/(r+1)`; the CMA-ES point is one representative of a ridge.

**Latest result** (`results/cmaes_20260910_puck_collision/`, 35 of 44 takes, 23 / 12): gain
**1.63** (val outgoing-speed RMS 0.239 → 0.121 m/s) vs 1.50 in the sim (e 1.09, r 2.56) — the sim
launches the puck ~8 % too slowly. `e = 0.63, r = 398` is one ridge point (`r = 2.56` needs
`e ≈ 1.26`); train / val error is identical everywhere on the ridge. Not promoted into `configs/`.
Details: [`notes/docs/environments/real-world/sysid/paddle-puck-collision.md`](../../notes/docs/environments/real-world/sysid/paddle-puck-collision.md);
notes [`2026-09-11_01-21`](../../notes/scratch/experiments/2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md),
background [`2026-09-10_02-49`](../../notes/scratch/experiments/2026-09-10_02-49_paddle-puck-mass-ratio.md).

**Offset session (2026-09-18, no fitting)** — `results/offset_robot_data_collection_puck_collision_change_angle_20260910_1818/`:
the head-on parameters replayed on 23 oblique hits (offset 0.1–7.6 cm). Exit *speed* matches to 7.5 % up to ≈ 5 cm offset, exit
*angle* is 1.9 × too large at every offset (the sim would need 0.57 × the measured offset): the frictionless contact lacks a
tangential term. Grazes ≥ 6.5 cm are unreliable in Box2D (time-of-impact / slop). Note
[`2026-09-18_23-30`](../../notes/scratch/experiments/2026-09-18_23-30_offset-paddle-puck-collisions-replay.md).
