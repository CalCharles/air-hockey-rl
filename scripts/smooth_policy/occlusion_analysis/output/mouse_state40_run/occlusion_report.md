# Occlusion Analysis Report

- Trajectories processed: 40 (requested 40)
- Total frames: 8312
- Occluded frames: 577 (6.94%)
- Trajectories with any occlusion: 36/40 (90.00%)

## Headline Answers

- Where occlusions generally occur: mean occluded puck position is (0.611, 0.036). Compare with the visible-vs-occluded heatmap for spatial concentration.
- How often occlusions occur: 6.94% of all frames are occluded (median per-trajectory occlusion rate 3.25%).
- Are they isolated or bursty: overall windowed burstiness is bursty (1s Fano=7.196, 5s Fano=11.178).

## Temporal Structure

- Number of occlusion runs: 172
- Mean run length: 3.35 frames
- Max run length: 58 frames
- Run classes: isolated=76, short=70, medium=22, long=4

## Artifacts

- `occlusion_summary.json`
- `per_trajectory_metrics.csv`
- `occlusion_context_bins.csv`
- `occlusion_arrays.npz`

Generate plots with `plot_occlusion_results.py` in the same output directory.