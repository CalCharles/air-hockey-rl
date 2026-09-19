# Sysid section: puck–wall collisions → `side_wall_restitution`, `end_wall_restitution`

Part of the [sysid pipeline](../sysid-pipeline.md). This page is the complete description of
the puck–wall section: which bounces it uses, how a bounce is replayed in Box2D, the metrics,
how the restitution is selected and validated, what it produced on the mouse dataset, and how to
reproduce it. It supersedes the 2026-05 hand-curated analysis in
[`wall-collision-system-id.md`](../wall-collision-system-id.md).

| What | Where |
|---|---|
| Folder | `sysid/wall_collision/` — `code/`, `data/<name>/`, `results/<name>/` ([`README.md`](../../../../../sysid/wall_collision/README.md)) |
| Harvest of the bounces | `sysid/common/extract_sysid_sections.py --wall-out sysid/wall_collision/data/<name>` (stage 1: `wall/*.hdf5` + `manifest.*` + `summary.md`; `recordings` → the input dir) |
| Bounce objects, replay, sweeps, plots | `sysid/common/sysid_dataset.py` (`make_wall_bounces`), `sysid/wall_collision/code/wall_restitution_fit.py` |
| Driver | `sysid/wall_collision/code/fit_walls.py --sections-dir <data> --puck-results <puck results.json> --out <results>` (split by recording, Box2D replay sweeps, validation, summary) |
| Outputs | `sysid/wall_collision/results/<name>/`: `summary.md`, `results.json`, `split.json`, `sim_config_fitted.yaml`, `fit_sweeps.png`, `validation_percentiles.png`, key figures |
| Reference run | `sysid/wall_collision/results/mouse_dataset/` (gitignored; regenerated 2026-09-18, identical to the 2026-09-10 numbers) |
| Overlays (verification) | `sysid/wall_collision/code/render_overlays.py --results-dir <results>` → `<results>/overlays/`: 5 validation side-wall bounces at spread exit-speed-error percentiles, real tracker puck vs the fit's own Box2D replay under the fitted restitution, GIF + PNG + mosaic + `overlay_summary.md`; see [`2026-09-18_22-40`](../../../../scratch/experiments/2026-09-18_22-40_puck-wall-real-vs-sim-overlays.md) (post-impact positions are shifted by the wall-line offset; the fit compares exit velocities) |

## Data: wall bounces with determinable pre / post velocities

Stage 1 keeps a `wall_collision` event when

- it is a single event between two `free_fall` segments,
- each side gives ≥ `--min-side-frames` (5) clean samples (at most `--max-side-frames` = 10 are
  used) whose damped-model fit rms is ≤ `--max-side-rms` (2 cm),
- at most `--max-gap` (2) frames are hidden across the impact,
- the paddle is ≥ `--min-paddle-dist` (0.25 m) from the puck (no paddle involvement),
- the approach speed is ≥ `--min-pre-speed` (0.3 m/s), and the normal velocity component
  reverses in the refit.

The slice carries `wall` (x+ / x− end walls, y+ / y− side walls in the sim frame), the last
clean pre-impact frame `a`, the first clean post-impact frame `b`, the calibrated paddle
position at `a`, and pre / post velocities and speed ratios in `manifest.json`.

## Real pre / post state (`bounce_state`)

With the (g, γ) identified by the [puck section](puck-free-flight.md) (or supplied), the
damped model is fitted to the ≤ 10 clean frames before `a` and after `b`. From the pre fit:
position and velocity at `a` (`p_a`, `v_a`) and at the *first* pre-window frame
(`p_start`, `v_start`); from the post fit: the exit velocity at `b` (`v_b`). The furthest
measured puck-centre coordinate along the wall normal near impact is the *apex* (used for the
apparent wall lines).

## Replay in Box2D (`replay_bounce`)

The base `--sim-config` is loaded with noise, occlusion, delays and all terminations off and
`gravity` / `puck_damping` overridden. The puck is placed at (`p_start`, `v_start`) — pulled
5 mm inside the sim's wall line if the real position is already past it (camera scale / table
mismatch) — the paddle is parked at its calibrated real position, and the sim is stepped with a
zero action until the puck's normal velocity reverses (up to `steps_to_b + 6` steps). The exit
velocity is the first sample after the reversal. A bounce that never reverses is "not
reproduced" and excluded from the means (the count is reported).

Frames: bounces are in the sim frame (x long axis, robot at x < 0); `reset_from_state` takes
"base" coordinates (robot at x > 0), i.e. `x_base = −x_sim`.

**How the sim applies restitution** (`CollisionForceListener` in
`airhockey/sims/airhockey_box2d.py`): a puck–wall contact takes the **wall fixture's**
restitution directly — neither Box2D's mixing rule nor `puck_restitution` enters — and PostSolve
adds an impulse so the outgoing normal speed equals `incoming × wall_restitution` when the
incoming normal speed is ≥ `puck_wall_restitution_threshold_speed` (0.25 m/s); below that a fixed
0.1 m/s rebound is enforced. The tangential component is untouched. So the exit-speed error is a
direct function of the swept value, glancing bounces are uninformative, and the exit *angle*
cannot be matched by restitution alone (see limitations).

## Metrics (mean over reproduced bounces, on train and val)

| metric | definition |
|---|---|
| `speed_err` | \|exit speed sim − real\| [m/s] |
| `normal_err` | \|normal component of the exit velocity sim − real\| [m/s] |
| `speed_rel_err` | `speed_err` / real exit speed — **default selection objective** |
| `normal_rel_err` | `normal_err` / real exit speed (normalised by the total speed so glancing bounces do not blow up) |
| `angle_err` | \|direction of the sim exit velocity − direction of the real exit velocity\| [deg] |

## Search, selection, validation

`side_wall_restitution` (y± bounces) and `end_wall_restitution` (x± bounces) are swept
independently over `--restitution-range` (0.4 … 1.0, step 0.025; 25 values), every train and val
bounce replayed at every value. The train minimum of `--objective` is selected; all five
metrics are reported at the selection and at the canonical value.

Validation (general method in the [pipeline doc](../sysid-pipeline.md#stage-2--split-fit-validate)):
for each metric the selection's validation error is ranked against the 25 sweep values' validation
errors. What to look for: the oracle inside the range (an oracle at 0.4 or 1.0 = monotone curve,
parameter not identified), the p90 point not already at ≈ 100 % of the oracle (flat landscape),
train and val curves with a minimum in the same place in `fit_sweeps.png`, and the reproduced
count close to the bounce count.

Plots: `fit_sweeps.png` — per wall kind, top row: absolute speed / normal curves (train, val),
sim-vs-real exit-speed scatter at the selection, exit-angle curve; bottom row: relative curves,
per-bounce relative error vs real exit speed, sim-vs-real exit-angle scatter.
`validation_percentiles.png` — one panel per wall kind × metric (speed, relative speed, relative
normal, angle).

**Key figures** (side walls only; individual PNG + PDF in the results folder, copied to
`--figures-dir`, e.g. `paper/figures/sysid/`):

| file | content |
|---|---|
| `wall_side_exit_speed_sweep` | mean \|exit speed sim − real\| vs `side_wall_restitution`, train and val curves, selection and canonical marked (exit speed only, no normal component) |
| `wall_side_exit_speed_scatter` | sim vs real exit speed per bounce at the selected value, train and val bounces |
| `wall_side_exit_angle_scatter` | sim vs real exit angle [deg] per bounce at the selected value, train and val bounces |

**Apparent wall lines.** Per wall, the p50 / p90 of the measured apex is reported against the
sim's contact line (`half-size − puck_radius`). A mismatch means the puck frame is offset /
scaled relative to the sim table; it shifts *when* a replayed bounce happens, not its exit speed.

## Reproduce

The wall fit replays with a puck model (g, γ): take it from a puck fit's `results.json` or supply it.
The train / val recordings are the same as the puck fit's for the same `--seed` / `--val-fraction`
(both draw the split from the manifest's list of all recordings).

```bash
python sysid/common/run_puck_wall_sysid.py --input-dir <recordings> --name <name>          # all stages
python sysid/wall_collision/code/fit_walls.py --sections-dir sysid/wall_collision/data/<name> \
    --puck-results sysid/puck_dynamics/results/<name>/results.json --out sysid/wall_collision/results/<name>
python sysid/wall_collision/code/fit_walls.py --sections-dir sysid/wall_collision/data/<name> --puck-params -0.73 0.11 --out …
# variants: --objective speed_err (pre-2026-09-10 absolute selection), --restitution-range 0.5 1.0 0.01, --max-side-frames 6, --figures-dir paper/figures/sysid
```

`results/<name>/summary.md` ends with the exact command of the run that produced it.

## Results on the mouse dataset (seed 0, 80 / 20 recordings, g −0.73 / γ 0.11)

| walls | bounces train / val (reproduced) | selected | val exit-speed err | val exit-angle err | canonical | val err @canonical |
|---|---|---|---|---|---|---|
| side (y±) | 74 / 36 (74) | **0.900** | 0.079 m/s = 16.0 % | 13.9° | 0.99 | 0.105 m/s = 20.4 %, 13.0° |
| end (x±) | 17 / 6 (15) | 0.550 | 0.256 m/s = 45 % | 28.8° | 0.70 | 0.276 m/s = 51 %, 24.7° |

Validation, side walls: selection beats 76–92 % of the sweep on the speed metrics, oracle at
0.85 (speed) / 0.925 (normal), p90 point reaches 94–97 % of the oracle → the data pins the value
to the band 0.85–0.95; canonical 0.99 beats only 40–68 % and is clearly too lively. Exit angle:
monotone, improves up to 1.0 (canonical is its oracle). End walls: oracle at the range edge, p90
at 99–100 % → not identified; keep 0.70. Apparent wall lines: x+ 0.865 vs sim 0.933, x− 0.933,
y+ 0.359 vs 0.400, y− 0.442 vs 0.400 (puck frame ≈ −4 cm in y, far end wall 7 cm inside).
On 110 side-wall bounces with fitted velocities the normal-speed retention is ≈ 0.90–0.95, not
the ~0.65 of the 11 hand-picked 2026-05 clips.

Sources: [`2026-09-10_01-30`](../../../../scratch/experiments/2026-09-10_01-30_sysid-train-val-pipeline.md),
[`2026-09-10_03-20`](../../../../scratch/experiments/2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md).

## Knobs

| Knob | Where | Default | When to change |
|---|---|---|---|
| `--min-side-frames`, `--max-side-rms`, `--max-gap` | stage 1 | 5, 2 cm, 2 | noisier tracker / more occlusion → fewer bounces |
| `--min-paddle-dist` | stage 1 | 0.25 m | robot plays near the walls |
| `--min-pre-speed` | stage 1 | 0.3 m/s | keep above the sim's 0.25 m/s restitution gate |
| `--objective` | `fit_walls.py` | `speed_rel_err` | `speed_err` for the absolute criterion; `normal_*` ignores the tangential component |
| `--restitution-range` | `fit_walls.py` | 0.4–1.0 / 0.025 | walls outside the range; finer step once the band is known |
| `--max-side-frames` | `fit_walls.py` | 10 | shorter if the free-flight model drifts over long pre-windows |
| `--puck-results` / `--puck-params` | `fit_walls.py` | from the puck fit's `results.json` | reproduce the wall section alone / test other (g, γ) |

## Limitations

- **No tangential loss in the sim.** The real side-wall bounces lose tangential speed, the sim
  keeps it; the exit-angle error therefore keeps improving towards restitution 1.0 while the
  speed errors bottom out at 0.85–0.925. A wall friction / tangential coefficient is a missing
  parameter; the angle metric is reported for that reason but is not a selection objective.
- **End walls need clean bounces** without a human at the far end; with 17 / 6 bounces the
  validation curve is monotone.
- **Frame offsets** (apparent wall lines) shift replays by 1–2 frames near the y+ / x+ walls;
  fix the homography or add per-wall lines to the sim before position-level comparisons.
- Bounces below the sim's 0.25 m/s normal-speed gate are replayed with the fixed 0.1 m/s rebound
  and carry no information about the restitution.
