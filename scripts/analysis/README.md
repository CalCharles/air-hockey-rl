# Analysis tools

Standalone analysis tools that don't depend on the training stack and don't get imported by it.

| Subdir | What |
|--------|------|
| [`occlusion_analysis/`](occlusion_analysis/) | Analyze occlusion patterns in processed real-data trajectories. See its README. |

| Script | What |
|--------|------|
| [`aggregate_online_tb.py`](aggregate_online_tb.py) | Collapse sim2sim online fine-tuning TensorBoard runs (`<online>/env*/*seed*/online_tb`) into one run per env (+ `all_envs`) with mean ± SE over seeds; Custom Scalars tab shows margin (shaded ±SE) charts. |
