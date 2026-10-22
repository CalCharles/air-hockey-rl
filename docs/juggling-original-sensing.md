# Sustained juggling with original sensing

The user's requirement is to retain the original sensing during training. Clean-sensing trials are diagnostic only; they are not the deployment recipe. Do not disable or simplify sensing for new juggling training.

`configs/new_juggle/sustained/original_sensing.yaml` copies **every simulator parameter unchanged** from `configs/new_juggle/sysid_v2_hist2.yaml`: puck noise enabled (0.01 standard deviation), random occlusions enabled (0.05 rate), observation delay enabled (0.025 seconds), interpolation enabled, and action delay disabled. It retains the original history sampling default, command smoothing, identified PID gains, real workspace, geometry and contact dynamics. No extra puck-observation lag or camera-frame grid is introduced. The launcher checks the entire environment config against this recipe before training.

The episode cap increases from 250 to 10,000. The established ±25% physics randomization on paddle density, puck damping and gravity is retained from `tasks_v2/sim_dr3_juggle.yaml`. Sensing is not randomized beyond the original noise/occlusion processes. Existing canonical configs and the earlier experimental configs are unchanged.

## Experiments

Three trials, each 1M steps, seed 0, same original sensing throughout:

| Trial | Actor initialization | Discount |
|---|---|---:|
| `source_g0995` | Original canonical hist2-v2 policy | 0.995 |
| `clean_warm_g0995` | Best clean-sim juggling policy | 0.995 |
| `source_g099` | Original canonical hist2-v2 policy | 0.99 |

Fresh critics; actor warmup 50k steps; fixed 32-step training cadence; small action-change penalty; original upper-half reward. The clean-trained actor is only an initialization in the middle trial: all its new training observations use the original corrupted sensing, starting at step zero. No clean-sensing curriculum or privileged observations.

The new evaluator scenario `training` preserves the entire training distribution, including physics randomization and sensing. It never invokes `clean_sim_config`. Before training, evaluate each initial actor on 16 shared starts with this configuration. Checkpoint evaluations use eight starts and a 5,000-step cap; final validation uses 32 fresh starts and a 10,000-step cap. Start/end GIFs and raw trajectories are saved for all trials. Distinct juggle counts filter stationary contact artifacts. Validation chooses the best tested trial; it is not an independent post-selection test.

```bash
.venv/bin/python -m scripts.td3.juggle_original_sensing --prepare
.venv/bin/python -m scripts.td3.juggle_original_sensing --launch
cat runs/td3/original_sensing/status.json
```

Artifacts: `runs/td3/original_sensing/`, with `<trial>.log`, `<trial>_status.json`, `<trial>/validation/training.gif`, and `results.md`. Uses the three existing allocations, requests no new allocation, and refuses to overwrite an existing campaign. After completion it commits a new result note if still on `codex/sustained-juggling`; it does not automatically push. Model weights and original GIFs remain local.
