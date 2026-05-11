# Occlusion Analysis (40 Trajectories)

This folder contains tools to analyze occlusion patterns in processed real-data trajectories.

## Target dataset

Default dataset path:

`/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/`

## Scripts

- `analyze_occlusion_patterns.py`
  - Loads `trajectory_data*.hdf5`.
  - Selects exactly `--num-trajectories` (default 40).
  - Computes frequency, spatial, temporal (run/burst), and context-conditioned occlusion metrics.
  - Writes summary artifacts.
- `plot_occlusion_results.py`
  - Reads saved arrays/summary.
  - Produces heatmaps and temporal distribution plots.
- `run_occlusion_analysis.sh`
  - End-to-end runner for analysis + plotting.
  - Uses `.venv/bin/python` if present, otherwise `uv run python`.

## Quick run

From repository root:

```bash
bash scripts/analysis/occlusion_analysis/run_occlusion_analysis.sh \
  /data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/ \
  mouse_state40_run \
  first \
  40
```

Arguments:

1. `data_dir`
2. `run_tag` (optional, pass empty string for timestamp)
3. `sampling_mode` (`first` or `random`)
4. `num_trajectories` (default 40)

## Output artifacts

Under:

`scripts/analysis/occlusion_analysis/output/<run_tag_or_timestamp>/`

- `occlusion_summary.json`
- `per_trajectory_metrics.csv`
- `occlusion_context_bins.csv`
- `occlusion_arrays.npz`
- `occlusion_report.md`
- `puck_visible_vs_occluded_heatmap.png`
- `occlusion_transition_heatmap.png`
- `occlusion_runlength_hist.png`
- `occlusion_window_counts.png`
- `paddle_occluded_heatmap.png`

## Required keys and fallbacks

Expected per file: `puck` and preferably `cur_time`, `paddle`, `speed`, `pose`.

- Hard fail: missing/malformed `puck`.
- Fallback: if `paddle` missing, use `pose`.
- Fallback: if `cur_time` missing/invalid, use 20 Hz synthetic timestamps.

