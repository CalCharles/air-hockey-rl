# Puck relative metric → final displacement error / distance travelled

- **Date**: 2026-09-10 03:55 UTC
- **Status**: done
- **Run dir**: `sysid/auto_segments/mouse_dataset/sysid/puck/` (stage 2 rerun, seed 0, 20-sample windows)
- **Code**: `scripts/sysid/helper/puck_dynamics_fit.py` (`score_windows`), `scripts/sysid/run_sysid_pipeline.py`
- **Doc**: `notes/docs/environments/real-world/sysid/puck-free-flight.md`
- **Predecessor**: [`2026-09-10_03-20`](2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md)

## Change
The two relative puck metrics used the rms over the window's samples divided by the distance
travelled, i.e. an *average* per-sample error. They are now the **final displacement error**:
|model − measured| at the last sample of the window (in-window fit, resp. prediction from the
first half) divided by the distance travelled over the fitted / predicted samples. End-of-horizon
drift as a fraction of the motion. Metric keys (`fit_rel`, `pred_rel`) and the objective
(`fit_rms_cm`) are unchanged, so the fitted parameters are unchanged.

Also in this rerun: stage-2 outputs sectioned into `sysid/puck/` and `sysid/wall/` with general
files at `sysid/`, matching the doc split (`sysid-pipeline.md` general + `sysid/*.md` sections).

## Results (validation, 93 windows; selected g = −0.730, γ = 0.110)

| metric | selected | beats grid | canonical | beats grid | oracle | p50 / p75 / p90 reach |
|---|---|---|---|---|---|---|
| fit final displacement / distance (1 s window) | 1.5 % | 99 % | 1.8 % | 86 % | 1.4 % @ (−0.72, 0.06) | 51 / 66 / 84 % |
| pred final displacement / distance (0.5 s ahead) | 8.7 % | 100 % | 10.4 % | 90 % | 8.6 % @ (−0.72, 0.10) | 48 / 64 / 82 % |
| (old) fit rms / distance | 1.4 % | 100 % | 1.5 % | 86 % | 1.4 % | 74 / 85 / 94 % |
| (old) pred rms / distance | 6.1 % | 99 % | 6.9 % | 91 % | 6.0 % | 55 / 71 / 87 % |

Train values: 1.5 % / 8.1 % (canonical 1.9 % / 9.9 %).

## Conclusion
- The end-of-horizon metric is the sharper one: the p90 grid point reaches only 82–84 % of the
  oracle (vs 87–94 % for rms / distance) and the worst grid point is 4× the oracle. The ranking
  of the selection (99–100 % of the grid beaten) and of the canonical parameters (86–90 %) is the
  same as before.
- Interpretation for the percentile tables: percentiles run over the 651 (g, γ) candidates; each
  candidate's number is the mean over windows of its per-window final-displacement ratio.
