# sysid — system identification of the Box2D simulator from real recordings

One folder per fit. Each has the **code** that produces it, the **data** it was fitted on and the
**results** of the reference run, nothing else:

```
sysid/
  puck_dynamics/          gravity, puck_damping            free flight of the puck (mouse-teleop recordings)
  wall_collision/         side / end wall restitution      puck–wall bounces (same recordings, replayed in Box2D)
  paddle_pid/             pid_kp, pid_ki, pid_kd           scripted paddle-motion session (Box2D replay, CMA-ES)
  paddle_puck_collision/  paddle–puck restitution + mass   scripted head-on collision session (Box2D replay, CMA-ES)
  common/                 shared library (segmentation, dataset, validation) + the recordings → sections tools
                          and the puck + wall orchestrator; common/runs/<name>/ = an orchestrator run's index + QA GIFs
```

Inside every fit folder: `README.md` (what it fits, how to run it, latest result), `code/` (tracked),
`data/` and `results/` (gitignored, local). `data/` holds the fit's input — for the two puck fits the
harvested sections plus a `recordings` symlink to the raw recordings, for the two scripted-session
fits a symlink to the session on `/data2`. `results/<name>/` is one run: `summary.md`, the
numbers, the plots, `sim_config_fitted.yaml` (the base config with the fitted values replaced;
**nothing is promoted into `configs/` automatically**).

## Compiled result: `configs/new_juggle/sysid_v2_hist2.yaml` (2026-09-19) — the sim config to work off

The four fits below are compiled into **sysid v2**, the canonical sim config from 2026-09-19 on
(policies, DR ranges and the real-robot rollout config derive from it; the seven task configs are
generated into `configs/new_juggle/tasks_v2/` by `scripts/td3/extras/make_sysid_v2_configs.py`).
Masses stay canonical; the paddle–puck fit enters as `puck_restitution` = 1.2626 (its gain at the
canonical mass ratio) with `paddle_restitution` 0 so that one knob controls the contact. The
policy campaign on v2 (sysid / 25 %-low baseline / DR 5-step / DR long-history / RMA, each DR with
the 3-parameter and the full identified set): `runs/td3/sysid_v2_20260919/` and
[`2026-09-19_01-50`](../notes/scratch/experiments/2026-09-19_01-50_sysid-v2-compiled-params-policy-campaign.md).

## Latest fits (2026-09-10 / 11 / 19, all in this layout)

| Fit | Parameter | v1 (`sysid_best_params_hist2.yaml`) | Fitted = **v2** | Validation | In v2? |
|---|---|---|---|---|---|
| `puck_dynamics` | `gravity` / `puck_damping` | −0.661 / 0.178 | **−0.73 / 0.11** | held-out puck prediction 1.82 → 1.51 cm over 0.5 s; selection at the validation oracle, canonical 8–17 % worse | yes |
| `wall_collision` | `side_wall_restitution` | 0.99 | **0.90** | val exit-speed error 20 % → 16 % of real speed; data pin it to 0.85–0.95 | yes |
| `wall_collision` | `end_wall_restitution` | 0.70 | **0.55** | weakly identified (17 / 6 bounces, monotone curve; val 51 % → 45 %) — taken as the best estimate on data, DR covers 0.41–0.69 | yes |
| `paddle_pid` | `pid_kp` / `pid_ki` / `pid_kd` | 9000 / 0 / 50 | **7532 / 1929 / 0** (pooled lines / arcs + reversal jerks, 2026-09-19) | val per-step error 23.2 → 17.5 mm (lines / arcs 27.0 → 19.7, jerks 17.5 → 14.1); the lines-only fit 5496 / 5883 / 0 (15.2 mm there) overshoots reversals by 25 cm and was dropped | yes |
| `paddle_puck_collision` | gain `(1+e)·r/(r+1)` | 1.50 (e 1.09, r 2.56) | **1.63** → `puck_restitution` 1.2626 at r 2.56 | val outgoing-speed RMS 0.239 → 0.121 m/s; e and r exactly degenerate for head-on data. Replayed on the offset session (2026-09-18, no fit): exit speed to 7.5 % up to 5 cm offset, exit angle 1.9 × too large (frictionless contact) | yes |

Experiment notes (source of truth): [`2026-09-10_01-30`](../notes/scratch/experiments/2026-09-10_01-30_sysid-train-val-pipeline.md),
[`2026-09-10_03-20`](../notes/scratch/experiments/2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md) (puck + walls),
[`2026-09-19_01-35`](../notes/scratch/experiments/2026-09-19_01-35_paddle-pid-pooled-refit-reversal-jerk.md) + [`2026-09-10_03-39`](../notes/scratch/experiments/2026-09-10_03-39_paddle-pid-cmaes-sysid.md) (paddle PID),
[`2026-09-11_01-21`](../notes/scratch/experiments/2026-09-11_01-21_paddle-puck-collision-cmaes-sysid.md) (paddle–puck).
Those notes cite the pre-2026-09-18 run directories (`sysid/auto_segments/…`, `sysid/paddle/…`, `sysid/paddle_puck/…`);
the same runs are now under `<fit>/results/` and the originals are in the archive below.

## Running

```bash
# puck free flight + walls, from raw recordings (stages: QA render → sections → puck fit → wall fit)
python sysid/common/run_puck_wall_sysid.py --input-dir <recordings dir> --name <name> --eval-sample 10
#   → sysid/common/runs/<name>/README.md, sysid/{puck_dynamics,wall_collision}/{data,results}/<name>/
# overlays of representative real-vs-Box2D examples for the two puck fits (verification / visualisation)
python sysid/puck_dynamics/code/render_overlays.py --results-dir sysid/puck_dynamics/results/<name>
python sysid/wall_collision/code/render_overlays.py --results-dir sysid/wall_collision/results/<name>
# paddle PID gains (one or several sessions: paddle_motion lines / arcs, reversal_jerk)
python sysid/paddle_pid/code/fit_pid_cmaes.py --input-dir <session> [<session> …] --out sysid/paddle_pid/results/<name>
# paddle–puck collision
python sysid/paddle_puck_collision/code/fit_collision_cmaes.py --input-dir <puck_collision session> --out sysid/paddle_puck_collision/results/<name>
#   … and its fitted parameters replayed on the offset (oblique) session, no fitting: exit speed / angle errors + videos
python sysid/paddle_puck_collision/code/evaluate_offset_collisions.py --input-dir <change_angle session> --fit-dir sysid/paddle_puck_collision/results/<name>
# tests (synthetic data; the pipeline smoke test needs the shared mouse dataset and writes to a temp dir)
pytest sysid/common/tests sysid/paddle_pid/code/tests sysid/paddle_puck_collision/code/tests -q
```

Run the scripts from the repo root (they put the repo on `sys.path` and import as `sysid.<fit>.code.…`).

## Docs

- Pipeline (stages, split, "did the search do anything?", replication checklist): [`notes/docs/environments/real-world/sysid-pipeline.md`](../notes/docs/environments/real-world/sysid-pipeline.md)
- Per fit: [`puck-free-flight.md`](../notes/docs/environments/real-world/sysid/puck-free-flight.md) · [`puck-wall-collision.md`](../notes/docs/environments/real-world/sysid/puck-wall-collision.md) · [`paddle-pid.md`](../notes/docs/environments/real-world/sysid/paddle-pid.md) · [`paddle-puck-collision.md`](../notes/docs/environments/real-world/sysid/paddle-puck-collision.md)
- Segmenter + frame calibration: [`trajectory-auto-segmentation.md`](../notes/docs/environments/real-world/trajectory-auto-segmentation.md)

## Archive (2026-09-18)

Everything that was in this folder before the reorganisation — the 2026-04/05 hand-curated puck, wall
and teleop-paddle fits that produced the canonical config values (`teleop/`, `puck_segments/`,
`wall_collision*/`, `system_id_*`, `paddle_puck_collision/`, their scripts), the tracked 2026-05 paddle
grid searches (`legacy_scripts_from_scripts_sysid/`) and the original September run directories
(`auto_segments/`, `paddle/`, `paddle_puck/`) — was moved, not deleted, to
`/data2/air_hockey/sysid_legacy_20260918/` (2.2 GB). The legacy fits are described in
[`puck-system-id.md`](../notes/docs/environments/real-world/puck-system-id.md),
[`wall-collision-system-id.md`](../notes/docs/environments/real-world/wall-collision-system-id.md) and
[`teleop-system-id.md`](../notes/docs/environments/real-world/teleop-system-id.md).
