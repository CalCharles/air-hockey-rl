# sysid/paddle_pid — paddle (UR5 + PID plant) → `pid_kp`, `pid_ki`, `pid_kd`

Fits the Box2D paddle plant's PID gains to scripted paddle-motion recordings by replaying the
recorded actions in the sim and minimising the per-step paddle position error with CMA-ES (paddle
mass fixed), one trial per condition held out. Two recording layouts are pooled: the
`paddle_motion` lines / arcs session (`traj_*.hdf5`) and the `reversal_jerk` session (`jerk_*.hdf5`:
out, reverse, slow down) — `--input-dir` takes any number of session directories.

```
code/     dataset.py, replay.py, cmaes_fit.py, report.py, overlay.py   library
          fit_pid_cmaes.py      entrypoint: split → canonical baseline → CMA-ES on train → val → percentile validation → sim_config_fitted.yaml
          evaluate_gains.py     score arbitrary gain sets on train / val / all with the same replay
          render_overlays.py    real-vs-sim overlays on the Box2D scene (GIF + PNG per trial, mosaics) + per-trial error table
          tests/                synthetic session generated with the plant itself (~15 s)
data/     <session> → the recording sessions on /data2 (symlinks, created by the entrypoint)
results/  <name>/summary.md, fit_result.json, candidates.csv, per_trial.csv, evaluations.json, split.json,
          sim_config_fitted.yaml, plots/, overlays/ (from render_overlays.py)
```

```bash
python sysid/paddle_pid/code/fit_pid_cmaes.py \
    --input-dir /data2/air_hockey/vertical_horizontal_diagonal_arc_paddle_motion_20260909_2024 /data2/air_hockey/reversal_jerk_20260910_1921 \
    --out sysid/paddle_pid/results/cmaes_20260919_paddle_motion_and_jerk --popsize 24 --max-iter 120 --restarts 2 --workers 16
python sysid/paddle_pid/code/evaluate_gains.py --input-dir <session> [<session> …] \
    --gains canonical 9000 0 50 --gains pooled 7531.5 1928.7 0 --plots
python sysid/paddle_pid/code/render_overlays.py --input-dir <session> [<session> …] --fit-dir sysid/paddle_pid/results/<name> [--trials val]
pytest sysid/paddle_pid/code/tests -q
```

| File | What |
|---|---|
| `dataset.py` | `load_session` (one or several session dirs of `traj_*` / `jerk_*` HDF5s → `PaddleTrial`, aborted trials dropped from the file attrs), `split_train_val` (one trial per condition held out), `session_attrs`, `sessions_of` |
| `replay.py` | `build_replay_sim_config` (noise / delay / terminations off, gravity 0, workspace + edge + move limits and `hist_len` from the recording), `PaddleReplayer` (one env, gains / density switchable), `evaluate_trials` → mean per-step position error (mm) |
| `cmaes_fit.py` | `GainBounds` (unit cube ↔ kp log-uniform, ki / kd `expm1` so 0 is reachable), `CandidateEvaluator` (fork pool), `run_cmaes` (ask / tell, IPOP restarts, val error of every candidate recorded) |
| `report.py` | `summary.md`, `candidates.csv`, `per_trial.csv`, convergence / candidates / per-condition / trajectory / step-error plots |
| `fit_pid_cmaes.py` | entrypoint (percentile validation via `sysid/common/fit_validation.py`) |
| `evaluate_gains.py` | score arbitrary gain sets on train / val / all with the same replay |
| `overlay.py`, `render_overlays.py` | real-vs-sim overlays on the Box2D scene (GIF + PNG per trial, condition / val mosaics) and per-trial error table + plots |

**Latest result** (`results/cmaes_20260919_paddle_motion_and_jerk/`, 144 trials = 81 lines / arcs + 63
reversal jerks, 53 conditions, 99 / 45): **kp 7532, ki 1929, kd 0** vs canonical 9000 / 0 / 50 — validation
per-step error 23.2 → 17.5 mm (lines / arcs 27.0 → 19.7, jerks 17.5 → 14.1); selection beats 78 % of 1656
candidates and reaches 99 % of the validation oracle. The earlier lines-only fit (kp 5496 / ki 5883, 15.2 mm on
its own session) overshoots the reversals by up to 25 cm (39.5 mm on the jerk session, worse than canonical), so
the pooled gains replace it. **Promoted** into `configs/new_juggle/sysid_v2_hist2.yaml` (2026-09-19).
Details: [`notes/docs/environments/real-world/sysid/paddle-pid.md`](../../notes/docs/environments/real-world/sysid/paddle-pid.md);
notes [`2026-09-19_01-35`](../../notes/scratch/experiments/2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md) (pooled refit),
[`2026-09-10_03-39`](../../notes/scratch/experiments/2026-09-10_03-39_paddle-pid-cmaes-sysid.md) (first fit),
[`2026-09-10_04-39`](../../notes/scratch/experiments/2026-09-10_04-39_paddle-pid-overlays-per-trial.md) (overlays).
Reference runs kept: `results/cmaes_20260909_paddle_motion/` (lines / arcs only), `results/cmaes_20260910_reversal_jerk_only/`,
`results/eval_20260919_reference_gains_{pooled,paddle_motion}/`, `results/eval_reversal_jerk_reference_gains/`.
