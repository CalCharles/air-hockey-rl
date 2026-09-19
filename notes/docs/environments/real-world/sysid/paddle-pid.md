# Sysid section: paddle plant → `pid_kp`, `pid_ki`, `pid_kd` (CMA-ES)

The paddle in Box2D is a disc of fixed mass (`paddle_density`, `paddle_radius`, linear
`paddle_damping`) driven by a PID force towards the target the action defines. This page is
the complete description of how the PID gains are identified from scripted robot recordings
with CMA-ES, replacing the hand-tuned grid searches of 2026-05 ([`teleop-system-id.md`](../teleop-system-id.md);
those scripts and their data were archived on 2026-09-18, see that page).

| What | Where |
|---|---|
| Folder | `sysid/paddle_pid/` — `code/`, `data/<session>` → the recording session, `results/<name>/` ([`README.md`](../../../../../sysid/paddle_pid/README.md)) |
| Code (all of it) | `sysid/paddle_pid/code/` |
| Entrypoint | `sysid/paddle_pid/code/fit_pid_cmaes.py --input-dir <session> --out sysid/paddle_pid/results/<name>` |
| Scoring only | `sysid/paddle_pid/code/evaluate_gains.py --gains <label> kp ki kd …` |
| Qualitative view | `sysid/paddle_pid/code/render_overlays.py --input-dir <session> --fit-dir <fit out>` — scene overlays (GIF + PNG per trial, mosaics) + per-trial errors |
| Tests | `pytest sysid/paddle_pid/code/tests -q` (sim-generated session, no robot data) |
| Reference run | `sysid/paddle_pid/results/cmaes_20260919_paddle_motion_and_jerk/` (gitignored; the promoted gains) — [`2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md`](../../../../scratch/experiments/2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md); lines-only predecessor `…/cmaes_20260909_paddle_motion/` — [`2026-09-10_03-39_paddle-pid-cmaes-sysid.md`](../../../../scratch/experiments/2026-09-10_03-39_paddle-pid-cmaes-sysid.md) |

## Data: scripted paddle-motion sessions

Recorded on the robot with `configs/robot_data_collection/paddle_motion_config.yaml` (robot
machine, not in this repo). A session directory holds `manifest.json` and one
`traj_<idx>_<condition>_trial<k>.hdf5` per trial. Each trial starts from a fixed pose, waits
`settle_steps` (3) zero actions and then executes `action_steps` (20) scripted actions:

- **lines** `xpos|xneg|ypos|yneg|diagpos|diagneg_delta{0.33,0.66,1.00}` — the same normalised
  action every step (`action_delta` × direction);
- **arcs** `arc_{wide,medium,tight}_v{speed}` — closed-loop tracking of a half-ellipse
  (`curve_tracking`, `curve_velocity_gain` in the manifest).

Every condition is repeated 3 times (81 trials on 2026-09-09).

A second layout, **reversal_jerk** (`/data2/air_hockey/reversal_jerk_20260910_1921`, 63 trials,
2026-09-10), stresses the reversal the lines never contain: `jerk_<idx>_<cond>_delta<d>_out<n>_slow<m>.hdf5`
drives the paddle `out` steps in one direction (`up_down` = x, 10 steps at 0.33 / 0.66 / 1.00 that
saturate at the workspace limit; `right_left` = y, 20–22 steps), reverses it for `back` steps and
scales the last `slow` steps down (`slowdown_scales`). Conditions are the file name after
`jerk_<idx>_`; repeats come from the `repeat` attr or the rank within the condition. Its
`manifest.json` only lists the last restart of the session, so aborted / protective-stop trials
are detected from the file attrs. `load_session` takes one or several session directories and
pools them (`PaddleTrial.session` keeps the origin); the fit reports per-session errors.

The file stores the canonical
35-column `train_vals` row per step (`pose_x/y` = paddle in the **robot frame**, `speed_x/y`,
`desired_pose_x/y`), the normalised `actions`, `is_settle_step`, images, and the controller
settings as attrs (`hist_len`, `move_lims`, `workspace_lims`, `edge_lims`). Row `i` is the
state at the start of step `i` and the action executed during step `i`;
`desired[i] = clip(pose[i] + actions[i] · move_lims)` holds to < 1 mm. The recording's
`block_time` is 0.049 s (measured mean dt 0.0492 s); the sim runs at 0.05 s.

**Split**: `split_train_val` holds out exactly one trial of every condition (drawn per
condition with `--split-seed`, or `--val-repeat k` for a fixed repeat). 27 conditions → 54 train
/ 27 val. Aborted / protective-stop trials in the manifest are skipped.

## Replay

`PaddleReplayer` drives one `AirHockeyEnv` exactly as a policy does: reset to the real pose and
velocity of row 0 (`reset_from_state`), then `env.step(actions[k])` for `k = 0 … N-2` and read
the paddle body position after every step. Config preparation (`build_replay_sim_config`):

- noise, occlusion, observation / action delay, puck delay interpolation, terminations off;
  gravity 0 and the puck parked at the far end (it never touches the paddle);
- `rmax_x/y`, `x_min_lim … y_max`, `top_abs … max_bias_m` and `hist_len` copied from the
  recording so the sim's PID target sequence (`_compute_pid_target_pos` + `_filter_update`)
  equals the robot's (`compute_rect` + `clip_limits` + `robot_control.filter_update`);
  verified: max 0.8 mm difference between the sim target and the recorded `desired_pose`.
- frames: the sim base frame is the robot frame shifted by `center_offset_constant` (1.2 m) in
  x; all outputs are converted back to the robot frame. A reset reproduces the real pose to
  < 0.1 mm.

Gains and `paddle_density` can be changed between replays without rebuilding the env
(`spawn_paddle` re-reads the density at every reset).

## Metric

**Per-step position error**: for one trial the mean over steps `k = 1 … N-1` of
`‖sim_pose[k] − pose[k]‖` in mm (row 0 is identical by construction); for a set of trials the
mean of the per-trial means. Also reported per trial: rms, max, final-step error, the mean
over non-settle steps, and the per-step *displacement* error `‖Δsim − Δreal‖`.

## Search: CMA-ES over (kp, ki, kd), mass fixed

`cma.CMAEvolutionStrategy` in the unit cube (box bounds handled by `cma`), mapped to gains
with `kp = 500·200^z0` (log-uniform in [500, 1e5]), `ki = expm1(z1·log1p(1e5))`,
`kd = expm1(z2·log1p(5e3))` — so the gains, which span orders of magnitude, share one step size
and `ki = 0`, `kd = 0` are exactly representable. Start = the base config's gains
(`configs/new_juggle/sysid_best_params_hist2.yaml`: kp 9000, ki 0, kd 50), `sigma0` 0.3,
popsize 24, ≤ 120 iterations, `tolfun` 0.02 mm, IPOP restarts (population doubled, restart from
the incumbent). Objective = the metric over the **training** trials; every candidate's
validation error is recorded too. Candidates are scored in a `fork` pool (one env per worker):
a whole fit takes ~90 s with 16 workers.

**Validation of the search** (`sysid/common/fit_validation.py`): the selected gains'
validation error is ranked against every candidate evaluated — `selected_beats`, the
validation oracle and the p50 / p75 / p90 candidates. On the reference run the selection beats
91 % of 2472 candidates and equals the oracle (15.21 vs 15.23 mm); the canonical gains beat
16 %.

## Results (2026-09-19: both sessions pooled, split seed 0) — promoted into `sysid_v2_hist2.yaml`

144 trials (81 lines / arcs + 63 reversal jerks), 53 conditions → 99 train / 45 val. Val = mean
per-step error over the held-out trials (mm); the per-session columns are the same gains scored
on each session's held-out trials.

| gains | kp | ki | kd | val pooled | val lines / arcs | val jerks |
|---|---|---|---|---|---|---|
| canonical hist2 (`sysid_best_params_hist2.yaml`) | 9000 | 0 | 50 | 23.23 | 27.02 | 17.55 |
| lines-only CMA-ES (2026-09-10) | 5496 | 5883 | 0 | 24.99 | **15.23** | 39.68 |
| jerks-only CMA-ES | 8339 | 1423 | 0 | 19.86 | 25.91 | **11.14** |
| **pooled CMA-ES (2026-09-19)** | **7532** | **1929** | **0** | **17.45** | 19.66 | 14.14 |

The lines-only optimum's large integral term, which reproduces the ~75 ms onset latency on
straight moves, winds up during a long move and overshoots the reversal by up to 25 cm (worse
than canonical on the jerks). The pooled optimum keeps a moderate integral term (kp 7532, ki
1929, kd 0): −25 % vs canonical on the pooled validation set and better than canonical on both
sessions; the selection beats 78 % of 1656 candidates and reaches 99 % of the validation oracle
(17.33 mm at kp 7463 / ki 1774). No single PID gain set reaches both single-session optima —
the residual is the unmodelled command latency plus, on reversals, the robot's own deceleration
limit. These gains are the `pid_*` values of `configs/new_juggle/sysid_v2_hist2.yaml`.

### Results of the first fit (2026-09-09 session only, split seed 0)

| gains | kp | ki | kd | train (mm) | val (mm) |
|---|---|---|---|---|---|
| canonical hist2 (`sysid_best_params_hist2.yaml`) | 9000 | 0 | 50 | 25.99 | 27.02 |
| hist4 refit gains at density 3000 | 7500 | 0 | 50 | 19.79 | 19.75 |
| **CMA-ES** | **5496** | **5883** | **0** | 15.21 | **15.23** |

Other held-out choices (split seed 1 / 2) give kp 5441 / 5501, ki 5954 / 5699, kd 0, val
15.88 / 16.38 mm vs canonical 26.18 / 25.92. Per-condition and trajectory plots, plus the
latency diagnostic, are in the experiment note linked above.

What the fit does: the robot does not move for ~1.5 steps (≈ 75 ms) after the first non-zero
action, then reaches a higher peak speed than the canonical sim; the canonical PID reacts
within the first step. The optimum trades P for a large I term (a force that ramps with
accumulated error) and drops D entirely, which reproduces the delayed onset. Modelling the
latency explicitly is the next lever (`--action-delay-steps 1` alone brings the canonical gains
to 15.98 mm and the fit to 13.23 mm).

## Qualitative view and per-trial errors

`render_overlays.py` replays every trial with the fit's canonical and CMA-ES gains (plus any
`--gains LABEL KP KI KD`) and draws them on the Box2D table with `AirHockeyRenderer`: the
real paddle is the paddle sprite, each sim a translucent ghost with a trail (orange = first
set, blue = second, aqua = third), the recorded target a grey cross; frames are cropped to the
robot workspace (`--width`, default 360 px; 160 = repo GIF convention) and the GIF plays at
`--fps` 10 (half real time). Under `<fit-dir>/overlays/`: `gifs/<trial>.gif`, `png/<trial>.png`
(full trails), `mosaic_<condition>.png` (the repeats side by side), `mosaic_val.png`,
`mosaic_all.png`, `per_trial_errors.md/.csv` (every trial: split, mean / max / final error per
gain set), `plots/per_trial_errors.png` (dot per trial by condition, validation trials ringed)
and `plots/per_trial_step_errors.png` (per-step error of all trials, one panel per condition).
Findings on the reference run: [`2026-09-10_04-39_paddle-pid-overlays-per-trial.md`](../../../../scratch/experiments/2026-09-10_04-39_paddle-pid-overlays-per-trial.md).

## Reproduce

```bash
# the promoted (pooled) fit
python sysid/paddle_pid/code/fit_pid_cmaes.py \
    --input-dir /data2/air_hockey/vertical_horizontal_diagonal_arc_paddle_motion_20260909_2024 /data2/air_hockey/reversal_jerk_20260910_1921 \
    --out sysid/paddle_pid/results/cmaes_20260919_paddle_motion_and_jerk --popsize 24 --max-iter 120 --restarts 2 --workers 16
python sysid/paddle_pid/code/render_overlays.py --input-dir <both sessions> --fit-dir sysid/paddle_pid/results/cmaes_20260919_paddle_motion_and_jerk --trials val
# any gain set on any session(s)
python sysid/paddle_pid/code/evaluate_gains.py --input-dir <session> [<session> …] --gains canonical 9000 0 50 --gains pooled 7531.5 1928.7 0
# the lines-only fit (2026-09-10)
python sysid/paddle_pid/code/fit_pid_cmaes.py \
    --input-dir /data2/air_hockey/vertical_horizontal_diagonal_arc_paddle_motion_20260909_2024 \
    --out sysid/paddle_pid/results/cmaes_20260909_paddle_motion --popsize 24 --max-iter 120 --restarts 2 --workers 16
# other held-out trials / latency diagnostic
python sysid/paddle_pid/code/fit_pid_cmaes.py --input-dir … --out … --split-seed 1
python sysid/paddle_pid/code/fit_pid_cmaes.py --input-dir … --out … --action-delay-steps 1
```

Outputs under `--out`: `summary.md`, `fit_result.json` (best gains, settings, per-generation
log, percentile validation), `candidates.csv`, `evaluations.json` + `per_trial.csv` (canonical
and fitted, train and val), `split.json`, `sim_config_replay.yaml`, `sim_config_fitted.yaml`
(base config with the fitted gains; `hist_len` stays the base value — the gains describe the
servo, the smoothing window is a separate controller setting), `plots/`.

Options: `--paddle-density` (fixed mass), `--hist-len` (override the recording's smoothing
window), `--x0 KP KI KD`, `--kp-range`, `--ki-max`, `--kd-max`, `--sigma0`, `--popsize`,
`--max-iter`, `--restarts`, `--seed`, `--val-repeat`, `--action-delay-steps`, `--no-plots`.
