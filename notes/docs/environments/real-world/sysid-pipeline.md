# System-ID pipeline (recordings → sections → train/val → simulator parameters)

Reusable pipeline that goes from raw split-schema HDF5 recordings to identified simulator
parameters with a proper train / validation split and a check that each search actually
resolved something. Built 2026-09-10 on the shared mouse-teleop dataset; designed to be re-run
unchanged on other recording sets.

This page covers the **general** parts: the stages, the split, how a search is validated, the
output tree and the replication checklist. Each fit is its own folder under `sysid/` (code, data,
results — see [`sysid/README.md`](../../../../sysid/README.md)) with its own page describing its data,
model, metrics, plots, results and reproduce command:

| Fit | Doc | Folder |
|---|---|---|
| Puck free flight → `gravity`, `puck_damping` | [`sysid/puck-free-flight.md`](sysid/puck-free-flight.md) | `sysid/puck_dynamics/` (`code/fit_puck.py`, `code/puck_dynamics_fit.py`) |
| Puck–wall collisions → `side_wall_restitution`, `end_wall_restitution` | [`sysid/puck-wall-collision.md`](sysid/puck-wall-collision.md) | `sysid/wall_collision/` (`code/fit_walls.py`, `code/wall_restitution_fit.py`) |

**One command:**

```bash
python sysid/common/run_puck_wall_sysid.py \
    --input-dir <dir with trajectory_*.hdf5> \
    --name <dataset_name> \
    --eval-sample 10 --seed 0
```

It files each stage's output under the fit it belongs to and writes an index `README.md` with the
headline numbers:

```
sysid/common/runs/<name>/               README.md (index + headline numbers), pipeline_run.json (args, git commit,
                                          per-stage command / seconds / exit code / log tail)
  segmentation_eval/                    stage 0: GIF + PNG + segments.json for --eval-sample random recordings (QA)
sysid/puck_dynamics/data/<name>/        stage 1: free_fall/*.hdf5, manifest.{csv,json}, summary.md, recordings → input dir
sysid/wall_collision/data/<name>/       stage 1: wall/*.hdf5,      manifest.{csv,json}, summary.md, recordings → input dir
sysid/puck_dynamics/results/<name>/     stage 2: split.json, summary.md, results.json, sim_config_fitted.yaml, fit_grid.png,
                                          validation_percentiles.png, puck_final_displacement_{vs_percentile,grid_train,grid_val}.{png,pdf}
sysid/wall_collision/results/<name>/    stage 3: split.json, summary.md, results.json, sim_config_fitted.yaml, fit_sweeps.png,
                                          validation_percentiles.png, wall_side_exit_{speed_sweep,speed_scatter,angle_scatter}.{png,pdf}
```

(key figures are also copied to `--figures-dir`, e.g. `paper/figures/sysid`, when the fit is given
`--figures-dir` via `--puck-args` / `--wall-args`). Every `summary.md` ends with the exact command
that reproduces that fit alone. Reference run: `<name>` = `mouse_dataset` (gitignored; the
2026-09-10 mouse-dataset result, regenerated in this layout on 2026-09-18 with identical numbers).
Smoke test: `pytest sysid/common/tests -q` (about 15 s on 8 recordings, writes to a temporary tree).

## Code map

| Stage | Script (each has `--help`) | Library (`sysid/common/`) |
|---|---|---|
| 0 Segment + QA render | `sysid/common/segment_trajectories.py` | `trajectory_segmentation.py` (load, frame calibration, split-fit event detection, classification, segments, HDF5 export), `segment_rendering.py` (GIF / PNG) |
| 1 Harvest sections | `sysid/common/extract_sysid_sections.py` | `trajectory_segmentation.py` |
| 2 Puck fit | `sysid/puck_dynamics/code/fit_puck.py` | `sysid_dataset.py` (manifest, split by recording, windows), `fit_validation.py` (percentile validation + plot); the fit itself: `sysid/puck_dynamics/code/puck_dynamics_fit.py` |
| 3 Wall fit | `sysid/wall_collision/code/fit_walls.py` | `sysid_dataset.py` (bounce objects), `fit_validation.py`; the fit itself: `sysid/wall_collision/code/wall_restitution_fit.py` |
| all | `sysid/common/run_puck_wall_sysid.py` | subprocess-driven orchestrator; extra per-stage flags via `--segment-args / --extract-args / --puck-args / --wall-args`; `--skip-existing` reruns only missing stages and regenerates `README.md`; `--root` redirects the whole tree (tests) |

Segmenter method and the frame-calibration story: [`trajectory-auto-segmentation.md`](trajectory-auto-segmentation.md).
Index of the folder: `sysid/README.md`. Experiment notes:
[`2026-09-10_01-30`](../../../scratch/experiments/2026-09-10_01-30_sysid-train-val-pipeline.md) (first train/val run),
[`2026-09-10_03-20`](../../../scratch/experiments/2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md) (normalised metrics, angle, percentile validation).

## Stage 0 — segmentation + QA render (`segment_trajectories.py`)

Every recording is cut into `free_fall / wall_collision / paddle_collision / opponent_hit /
unknown_impulse / rest / occluded` segments. Before that, the axis conventions are calibrated
from the data (puck mirrored if the measured free-flight acceleration opposes `gravity_x`;
paddle mapping `x = sign·pose_x + offset` fitted from puck–paddle interactions). The stage
prints the calibration on its first line and stores it in `sample_manifest.json` and every
`segments.json`. With `--sample N` it renders N random recordings as GIF (top-down + camera +
timeline) and PNG (x, y, speed, `dv`, paddle distance with label bands). This stage is only
for looking; stage 1 redoes the segmentation on all files.

## Stage 1 — section harvest (`extract_sysid_sections.py`)

Runs the segmenter on all recordings (same calibration, estimated over the whole set) and keeps
the quality-filtered **free-fall clips** and **wall bounces** (criteria in the section docs).
Each kept section is a split-schema HDF5 slice (sim frame, calibration in attrs) and a row in
`manifest.csv` / `manifest.json` with all metrics. The free-fall clips go to `--out`
(`sysid/puck_dynamics/data/<name>/`), the wall bounces to `--wall-out`
(`sysid/wall_collision/data/<name>/`), each with its own manifest and `summary.md` (kept / rejected
counts per reason, per-wall speed ratios — the first place to look when a new dataset yields few
sections). Both manifests list every input recording (`sources`) so the two fits can draw the
same train / val split.

## Stages 2 + 3 — split, fit, validate (`fit_puck.py`, `fit_walls.py`)

**Split (general).** Sections are grouped by source recording; `--val-fraction` (0.2) of the
recordings is held out with `--seed`. The draw is made from the list of *all* input recordings in
the manifest, so the puck fit and the wall fit hold out exactly the same recordings and nothing
from a validation trajectory is seen by either fit. Saved to each fit's `split.json`.

**Fits.** The puck fit runs first (its (g, γ) is the free-flight model the wall fit replays with:
`fit_walls.py --puck-results <puck results.json>`); the wall fit can be run alone with
`--puck-params G GAMMA`. Each fit writes its own results folder and summary; details in the fit
docs. Every metric is reported **absolute and normalised** (puck: rms and final displacement
error / distance travelled of the full in-window fit; walls: error / real exit speed and exit
angle) on train and val, for the selected and the canonical parameters.

**Did the search do anything? (`sysid/common/fit_validation.py`, general).** Every candidate the
search evaluated (651 grid points for the puck, 25 sweep values per wall) has a validation error
for every metric. For each metric the selected parameters are placed in that distribution:

- `beats grid` — share of candidates with a *worse* validation error than the selection (100 % =
  the selection is the validation oracle);
- `oracle` — the best validation error any candidate reaches, and its parameters;
- `of oracle` = oracle error / error (100 % = as good as the oracle);
- the candidates at the 50th / 75th / 90th percentile of the validation ranking (better than
  50 / 75 / 90 % of the candidates) and how much of the oracle each reaches.

Reading it: if the selection beats ≥ 95 % of the candidates and the p90 candidate reaches
clearly less of the oracle than the selection, the training search is finding something the
held-out data confirms. If the p90 candidate already reaches ≈ 100 % of the oracle the
validation landscape is flat there and the parameter is not resolved (any value in that band is
equivalent); if the oracle sits at the edge of the range the curve is monotone and the
parameter is not identified at all. The canonical parameters get the same numbers, so "did we
beat the config we already had" is read off the same row. `validation_percentiles.png` in each
section folder draws every candidate's error divided by the best error of its split against its
percentile rank (100 = best), separately for training and validation, with p50 / p75 / p90 marked
and the selected / canonical points placed at their validation rank.

**Outputs per fit.** `results.json` (everything numeric, the split, the metric definitions and
the command), `summary.md` (headline, fit-quality tables, validation, fitted parameters,
reproduce command), `sim_config_fitted.yaml` = `--sim-config` with that fit's parameters replaced
(`gravity`, `puck_damping` for the puck fit; `side_wall_restitution`, `end_wall_restitution` for the
wall fit, which also carries the puck g / γ it replayed with). Nothing under `configs/` is touched;
promote by hand once trusted.

## Replicating on a new dataset — checklist

1. **Input format.** Split-schema HDF5 per recording with at least `puck` (x, y, occluded),
   `cur_time`, `pose` (paddle x, y in cols 0–1). `image` (camera) is optional and only used by
   the GIFs. Anything else the extractor copies through if present (`SPLIT_DATASETS` in
   `trajectory_segmentation.py`).
2. **Run** `sysid/common/run_puck_wall_sysid.py --input-dir … --name <name>` with `--eval-sample 10`. Read the calibration line: the puck
   sign is decided from the free-flight acceleration and the paddle mapping from
   interactions. If `decided` is false or the paddle mapping explains few impulses, look at
   the GIFs before anything else; pass `--segment-args "--puck-x-sign … --paddle-x-sign …
   --paddle-x-offset …"` and the same via `--extract-args` when the convention is known.
3. **Eyeball the GIFs** in `segmentation_eval/`: wall bounces should carry blue windows at the
   walls, paddle hits red windows at the paddle, free flight green. Pass-through frames
   (puck sliding through the paddle without a velocity change) mean the paddle mapping is
   wrong.
4. **Check the two `data/<name>/summary.md`**: the rejection counts tell you which threshold bites
   (e.g. `paddle_near` for a dataset where the robot plays near the walls, `side_rms` for a
   noisier tracker). Tune via `--extract-args`.
5. **Check the run's `README.md`, then each `results/<name>/summary.md`**: fitted vs canonical must
   differ on *validation*, not just train; in the "Did the search do anything?" tables the puck selection
   should beat ≳ 95 % of the grid with the p90 point noticeably below it, and each wall's oracle
   must lie inside the sweep range with the p90 point not already at ≈ 100 % of the oracle.
   Fit-specific things to check are listed in the fit docs.
6. **Sensitivity**: rerun a fit alone with other seeds / `--window-frames 30` (fast, no
   segmentation) — `sysid/puck_dynamics/code/fit_puck.py --sections-dir sysid/puck_dynamics/data/<name>
   --out sysid/puck_dynamics/results/<name>_seed1 --seed 1` (and `fit_walls.py` likewise). The
   optimum should move little.
7. **Promote** the values from `results/<name>/sim_config_fitted.yaml` into `configs/new_juggle/` by
   hand, and write the dated experiment note (`notes/scratch/experiments/README.md`).

## General knobs

| Knob | Where | Default | When to change |
|---|---|---|---|
| `dv_threshold` | `SegmentationConfig` (`--cfg dv_threshold=…` on stages 0/1) | 0.35 m/s | tracker noisier / cleaner than the ~1-frame camera-loop aliasing seen here (free-flight median `dv` ≈ 0.08) |
| `paddle_contact_slack` | same | 0.06 m | large lag between tracker and pose (puck appears inside the paddle at impact) |
| `max_occlusion_gap` | same | 10 frames | occlusion-heavy data |
| `--val-fraction`, `--seed` | stages 2 + 3 | 0.2, 0 | more recordings → smaller fraction is fine; always try a second seed |
| `--sim-config` | stages 2 + 3 | `sysid_best_params_hist2.yaml` | any base config; only the four parameters are replaced, everything else (incl. canonical values) is read from it |
| `--figures-dir` | stages 2 + 3 (`--puck-args` / `--wall-args "--figures-dir …"` from the orchestrator) | none | copy the key figures (PNG + PDF) into e.g. `paper/figures/sysid/` |

Fit knobs (`--window-frames`, objectives, restitution range, harvest thresholds) are in the fit
docs.

## Known limitations (general)

- The puck frame's y offset and short x+ end (mouse dataset: −4 cm, −7 cm) are reported but not
  corrected; fix the homography or add per-wall lines to the sim before position-level replays.
- Paddle–puck restitution + mass ratio are fitted from the *scripted* head-on collision sessions
  by `sysid/paddle_puck_collision/code/` ([`sysid/paddle-puck-collision.md`](sysid/paddle-puck-collision.md)),
  not from this pipeline's auto-harvested `paddle_collision` segments; head-on data only identify
  the gain `(1+e)·m_pad/(m_pad+m_puck)` (≈ 1.63 on 2026-09-10 vs 1.50 in the sim).
