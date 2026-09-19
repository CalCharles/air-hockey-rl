# Sysid section: puck free flight → `gravity`, `puck_damping`

Part of the [sysid pipeline](../sysid-pipeline.md). This page is the complete description of
the free-flight section: which data it uses, the model, the metrics, how the parameters are
selected and validated, what it produced on the mouse dataset, and how to reproduce it.

| What | Where |
|---|---|
| Folder | `sysid/puck_dynamics/` — `code/`, `data/<name>/`, `results/<name>/` ([`README.md`](../../../../../sysid/puck_dynamics/README.md)) |
| Harvest of the clips | `sysid/common/extract_sysid_sections.py --out sysid/puck_dynamics/data/<name>` (stage 1: `free_fall/*.hdf5` + `manifest.*` + `summary.md`; `recordings` → the input dir) |
| Windows + fit + grid + plots | `sysid/common/sysid_dataset.py` (`make_free_fall_windows`), `sysid/puck_dynamics/code/puck_dynamics_fit.py` |
| Driver | `sysid/puck_dynamics/code/fit_puck.py --sections-dir <data> --out <results>` (split by recording, grid, validation, summary) |
| Outputs | `sysid/puck_dynamics/results/<name>/`: `summary.md`, `results.json`, `split.json`, `sim_config_fitted.yaml`, `fit_grid.png`, `validation_percentiles.png`, key figures |
| Reference run | `sysid/puck_dynamics/results/mouse_dataset/` (gitignored; regenerated 2026-09-18, identical to the 2026-09-10 numbers) |
| Overlays (verification) | `sysid/puck_dynamics/code/render_overlays.py --results-dir <results>` → `<results>/overlays/`: 5 validation windows at spread fit-error percentiles, real tracker puck vs the Box2D puck under the fitted (g, γ) stepped with the real sample intervals, GIF + PNG + mosaic + `overlay_summary.md`; see [`2026-09-18_22-40`](../../../../scratch/experiments/2026-09-18_22-40_puck-wall-real-vs-sim-overlays.md) (incl. the ½·g·h integrator bias of Box2D and the wall-offset exclusion) |

## Data: free-fall clips

Stage 1 keeps every `free_fall` segment of the segmenter (puck moving under gravity + damping
only, no paddle / wall / opponent event inside, short occlusion gaps bridged) that has

- ≥ `--min-free-frames` (10) usable samples (occluded and stale-duplicate tracker readings
  are dropped, see the segmenter doc),
- a full-clip damped-model fit rms ≤ `--max-fit-rms` (2.5 cm) with the canonical (g, γ),
- a mean speed ≥ `--min-speed` (0.15 m/s).

Each clip is a split-schema HDF5 slice in the sim frame (robot at x < 0, gravity towards −x)
with the calibration in its attrs and a row in `manifest.json`.

## Datapoints: fixed-length windows

Every clip is chopped into consecutive non-overlapping windows of exactly `--window-frames`
(20) usable samples; the remainder is dropped and a window whose samples span more than
`window_frames + 2` raw frames (it hides an occlusion) is skipped. Every datapoint therefore has
the same length and weight. Windows follow their recording into train or val.

> Use ≥ 20-sample windows. 10-sample (0.5 s) windows cannot separate g from γ: the grid is
> flat to 0.1 cm and fitted vs canonical score identically on validation. 20 and 30 samples give
> the same optimum and a clear validation gap (table below).

## Model and fit

Damped free flight per axis, sim frame:

```
a = g − γ v        g = (gravity_x, 0),  γ = puck_damping
v(t) = (v0 − g/γ) e^{−γt} + g/γ
p(t) = p0 + (v0 − g/γ)(1 − e^{−γt})/γ + (g/γ) t
```

For fixed (g, γ) this is linear in (p0, u = v0 − g/γ), so each window is a 2-parameter-per-axis
least-squares solve (`fit_damped` in `trajectory_segmentation.py`) — the procedure of the
original `sysid/puck_grid_search.py`, which uses the opposite sign convention for g in the raw
puck frame (the two agree on |g|).

## Metrics (full in-window fit; reported on train and val)

Both metrics come from the same fit: (p0, v0) solved by least squares on **all** samples of the
window for the candidate (g, γ). (A forward-prediction variant — fit on the first half, predict
the second — was evaluated on 2026-09-10 and gave the same optimum with a wider spread; it was
dropped in favour of the full fit, validation being the held-out check.)

| metric | definition |
|---|---|
| `fit_rms_cm` | rms position residual over the window's samples, mean over windows (the original grid-search criterion; **selection objective**) |
| `fit_rel` | **final displacement error** — \|model − measured\| at the last sample of the window — divided by the distance the puck travelled in the window (path length of the measured trajectory); mean of per-window ratios |

`fit_rel` weighs fast and slow windows equally and says by what fraction of the motion the model
has drifted at the end of the 1 s window (not an average over the samples — the rms is the
average). In the percentile tables the percentiles run over the (g, γ) candidates; each
candidate's number is its mean over the validation windows.

## Search, selection, validation

The coarse grid g ∈ `--g-range` (−1.0 … −0.4), γ ∈ `--gamma-range` (0 … 0.4), 0.02 steps
(31 × 21 = 651 points) is scored on **both** the train and the val windows with both
metrics. The parameters are the train minimum of `--objective` (default `fit_rms_cm`),
refined on a 0.005-step fine grid ± 0.06 around it. The canonical (g, γ) of `--sim-config` is
scored the same way.

Validation (general method in the [pipeline doc](../sysid-pipeline.md#stage-2--split-fit-validate)):
for each metric the selection's validation error is ranked against the 651 grid points'
validation errors — share beaten, oracle, p50 / p75 / p90 candidates and their share of the
oracle. What to look for: the selection beats ≳ 95 % of the grid, the val panels of
`fit_grid.png` have their own minimum next to the selected star, and the p90 grid point sits
noticeably below the selection on the percentile curves.

Plots in the results folder: `fit_grid.png` — 2 × 2 panels, rows rms / final displacement, columns train /
val, with the selection (red star), canonical (white circle) and each panel's own minimum (cyan
triangle). `validation_percentiles.png` — one panel per metric, every candidate's error relative
to the best of its split against percentile rank, train and val separately, p50 / p75 / p90
marked, selection and canonical at their validation rank.

**Key figures** (individual PNG + PDF in the results folder, copied to `--figures-dir`, e.g.
`paper/figures/sysid/`):

| file | content |
|---|---|
| `puck_final_displacement_vs_percentile` | train and val final displacement error relative to the best candidate of its split vs candidate percentile, p50 / p75 / p90, selection and canonical marked |
| `puck_final_displacement_grid_train`, `puck_final_displacement_grid_val` | the (g, γ) grid of the final displacement error / distance on each split |

## Reproduce

From raw recordings (all stages, also the wall fit) or the puck fit alone on existing sections:

```bash
python sysid/common/run_puck_wall_sysid.py --input-dir <recordings> --name <name> --eval-sample 10
python sysid/puck_dynamics/code/fit_puck.py --sections-dir sysid/puck_dynamics/data/<name> --out sysid/puck_dynamics/results/<name>
# variants: --window-frames 30, --seed 1, --objective fit_rel, --g-range -0.9 -0.5 --gamma-range 0 0.3, --figures-dir paper/figures/sysid
```

`results/<name>/summary.md` ends with the exact command of the run that produced it.

## Results on the mouse dataset (seed 0, 80 / 20 recordings)

Windows: 244 train / 93 val of 20 samples. Selected **g = −0.730 m/s², γ = 0.110 1/s**
(canonical −0.661 / 0.178).

| validation metric | selected | beats grid | canonical | beats grid | oracle (grid point) | p50 / p75 / p90 reach of oracle |
|---|---|---|---|---|---|---|
| rms | 0.76 cm | 99 % | 0.83 cm | 84 % | 0.75 cm @ (−0.72, 0.08) | 75 / 86 / 95 % |
| final displacement / distance | 1.5 % | 99 % | 1.8 % | 86 % | 1.4 % @ (−0.72, 0.06) | 51 / 66 / 84 % |

(The dropped prediction variant, fit on the first half and predict the second, gave val
1.51 vs 1.82 cm canonical and final displacement 8.7 % vs 10.4 % — same optimum; see the
2026-09-10 notes.)

Window-length check (puck only; fit rms / prediction rms in cm, from the run that still had the
prediction variant):

| window | n train / val | fitted (g, γ) | train | val | canonical train | canonical val |
|---|---|---|---|---|---|---|
| 10 samples | 762 / 269 | −0.730, 0.100 | 0.72 / 1.41 | 0.72 / 1.45 | 0.72 / 1.42 | 0.72 / 1.46 |
| **20 samples** | 244 / 93 | **−0.730, 0.110** | 0.79 / 1.55 | 0.76 / 1.51 | 0.85 / 1.88 | 0.83 / 1.82 |
| 30 samples | 104 / 38 | −0.725, 0.090 | 0.92 / 2.10 | 0.94 / 1.90 | 1.15 / 2.99 | 1.12 / 2.74 |

Conclusion: the grid search is validated (selection at or within 2 % of the validation oracle,
p90 grid point 5–18 % worse, canonical 9–21 % worse); gravity is 10 % stronger and damping 40 %
weaker than the 2026-05 in-sample fit on 10 hand-picked clips. In words: over a 1 s window the
fitted model drifts by 1.5 % of the distance travelled at the end of the window (canonical
1.8 %). Sources:
[`2026-09-10_01-30`](../../../../scratch/experiments/2026-09-10_01-30_sysid-train-val-pipeline.md),
[`2026-09-10_03-20`](../../../../scratch/experiments/2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md),
[`2026-09-10_03-55`](../../../../scratch/experiments/2026-09-10_03-55_puck-final-displacement-metric.md).

## Knobs

| Knob | Where | Default | When to change |
|---|---|---|---|
| `--min-free-frames`, `--max-fit-rms`, `--min-speed` | stage 1 | 10, 2.5 cm, 0.15 m/s | trade clip count for cleanliness |
| `--window-frames` | `fit_puck.py` | 20 | keep ≥ 20; longer = fewer windows but sharper (g, γ) |
| `--objective` | `fit_puck.py` | `fit_rms_cm` | `fit_rel` to weigh slow windows like fast ones and select on end-of-window drift |
| `--g-range`, `--gamma-range` | `fit_puck.py` | −1.0 … −0.4, 0 … 0.4 | optimum near an edge |

## Limitations

- The model has no spin, no table tilt in y (`gravity_y = 0`), and one global (g, γ); the
  per-clip quadratic acceleration on the mouse dataset ranges −0.46 … −0.97 m/s², so a
  position-dependent table model would be the next refinement.
- Windows are cut from clips that passed the canonical-parameter rms filter; a very different
  true (g, γ) would need `--max-fit-rms` loosened in stage 1.
