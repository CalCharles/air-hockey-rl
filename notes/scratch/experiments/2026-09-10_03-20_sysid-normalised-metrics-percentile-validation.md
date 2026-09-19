# Sysid pipeline: normalised metrics, exit-angle error and percentile validation of the searches

- **Date**: 2026-09-10 03:20 UTC
- **Status**: done
- **Run dir**: `sysid/auto_segments/mouse_dataset/sysid/` (stage 2 rerun on the sections of the 2026-09-10 01:30 run; seed 0, 20-sample windows)
- **Code**: `scripts/sysid/helper/fit_validation.py` (new), `helper/puck_dynamics_fit.py`, `helper/wall_restitution_fit.py`, `scripts/sysid/run_sysid_pipeline.py`, `run_full_sysid.py`
- **Doc**: `notes/docs/environments/real-world/sysid-pipeline.md` ("Did the search do anything?")
- **Predecessor**: [`2026-09-10_01-30_sysid-train-val-pipeline.md`](2026-09-10_01-30_sysid-train-val-pipeline.md)

## Question
1. Report the fit errors as a fraction of what was measured (puck: rms / distance travelled;
   walls: exit-speed error / real exit speed) and select the wall restitution on the relative
   metric instead of the absolute one.
2. Add the absolute exit-angle error for wall bounces.
3. Check whether the grid search / sweep actually resolves anything: rank the validation error
   of the selected parameters against the validation error of *every* candidate and compare with
   the candidates at the 50th / 75th / 90th percentile.

## Setup
Same data and split as the predecessor (80 / 20 recordings; 244 / 93 free-fall windows of 20
samples; 74 / 36 side-wall and 17 / 6 end-wall bounces). The (g, γ) grid (31 × 21 = 651 points)
is now scored on train and val with four metrics; each wall sweep (25 values) with five.
Puck selection objective unchanged (`fit_rms_cm` on train); wall objective changed to
`speed_rel_err`. Percentile report: `beats` = share of candidates with worse validation error,
`of oracle` = best validation error over the candidates / error, p50 / p75 / p90 = the candidate
better than 50 / 75 / 90 % of the others.

## Results

**Puck** (selected g = −0.730, γ = 0.110; unchanged):

| validation metric | selected | beats grid | canonical (−0.661, 0.178) | beats grid | oracle (grid) | p50 / p75 / p90 reach (of oracle) |
|---|---|---|---|---|---|---|
| fit rms | 0.76 cm | 99 % | 0.83 cm | 84 % | 0.75 cm @ (−0.72, 0.08) | 75 / 86 / 95 % |
| pred rms | 1.51 cm | 100 % | 1.82 cm | 87 % | 1.51 cm @ (−0.72, 0.10) | 55 / 70 / 86 % |
| fit rms / distance | 1.4 % | 100 % | 1.5 % | 86 % | 1.4 % @ (−0.72, 0.08) | 74 / 85 / 94 % |
| pred rms / distance | 6.1 % | 99 % | 6.9 % | 91 % | 6.0 % @ (−0.72, 0.10) | 55 / 71 / 87 % |

The train selection is at or next to the validation oracle on every metric, and the p90 grid
point is 5–14 % worse than the oracle, so the held-out data does separate the candidates. The
canonical parameters are a top-15 % grid point but 8–17 % worse than the oracle. The
normalised numbers say the damped model explains the puck to ≈ 1.4 % of the distance travelled
in-sample and predicts the next 0.5 s to ≈ 6 %.

**Side walls** (`side_wall_restitution`, selected on relative speed error → **0.900**, same as
with the absolute objective):

| validation metric | selected @0.900 | beats sweep | canonical 0.99 | beats sweep | oracle | p50 / p75 / p90 reach |
|---|---|---|---|---|---|---|
| exit speed err | 0.079 m/s | 84 % | 0.105 | 48 % | 0.075 @ 0.850 | 71 / 87 / 96 % |
| exit speed err / real speed | 16.0 % | 76 % | 20.4 % | 40 % | 14.9 % @ 0.850 | 79 / 90 / 97 % |
| normal err / real speed | 14.9 % | 92 % | 19.0 % | 64 % | 14.8 % @ 0.925 | 61 / 83 / 95 % |
| exit angle err | 13.9° | 80 % | 13.0° | 96 % | 13.0° @ 1.000 | 75 / 90 / 97 % |

The validation landscape is shallow between 0.85 and 0.95: the p90 sweep point already reaches
94–97 % of the oracle, i.e. any value in that band is within a few percent. The selection is
clearly better than canonical 0.99 on every speed metric (relative error 16 % vs 20 %). The exit
angle behaves differently: it improves monotonically up to restitution 1.0 and the canonical
value is its oracle. The sim's contact only scales the normal component, while the real bounce
also loses tangential speed, so matching the exit direction needs a tangential coefficient the
sim does not have — a higher restitution merely rotates the exit towards the normal at the
price of too much speed.

**End walls** (`end_wall_restitution`, relative objective → **0.550**; the absolute objective
gave 0.650 in the predecessor): val exit-speed error 0.256 m/s = 45 % (canonical 0.70: 0.276 m/s
= 51 %), angle 28.8° (canonical 24.7°). The validation oracle sits at the range edge (0.45) for
the speed metrics and the p90 point reaches 99–100 % of it; on the angle the selection beats only
24 % of the sweep. Still not identified with 17 / 6 bounces; keep 0.70.

Plots: `puck_fit_grid.png` (2 × 4: absolute / relative × fit-train / fit-val / pred-train /
pred-val), `wall_fit_sweeps.png` (per wall kind: absolute, relative and angle curves, sim-vs-real
scatter, per-bounce relative error, angle scatter), `validation_percentiles.png` (every
candidate's error relative to the best of its split against its percentile rank, training and
validation separately, p50 / p75 / p90 and the selected / canonical points marked).

## Conclusion
- The puck grid search is validated: selection at the validation oracle, p90 grid point 5–14 %
  worse, canonical 8–17 % worse. Keep g = −0.73, γ = 0.11.
- Side-wall restitution 0.90 stands under the relative objective; the data only pins it to
  0.85–0.95. End walls remain unidentified; keep 0.70 in any promoted config.
- New limitation on record: no tangential loss at the walls in the sim (exit-angle error is
  minimised by a different restitution than the speed error).
- All of this is now part of stage 2 of the pipeline (`--puck-objective`, `--wall-objective`,
  percentile tables in `summary.md`, three plots) and runs in the smoke test.

## Next
- Add a wall tangential (friction) coefficient to the sim and fit it on the exit angle.
- More clean end-wall bounces (no human at the far end).
- Paddle–puck restitution stage using the same replay + percentile validation template.
