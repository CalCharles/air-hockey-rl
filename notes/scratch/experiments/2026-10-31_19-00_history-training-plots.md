# History trials improve juggle counts during training but fluctuate late in training

- **Date**: 2026-10-31 19:00 UTC analysis/writeup; historical training was not rerun.
- **Status**: done
- **Run dirs**: `runs/td3/noisy_history_20261029_seed{0,1}`, `runs/td3/noisy_history_long_discount_20261029_seed0`, `runs/td3/noisy_history_low_actor_lr_20261029_seed0`
- **Configs**: [history](../../../configs/td3/sustained/noisy_history_scratch.yaml), [seed 1](../../../configs/td3/sustained/noisy_history_scratch_seed1.yaml), [longer discount](../../../configs/td3/sustained/noisy_history_long_discount_scratch.yaml), [lower actor LR](../../../configs/td3/sustained/noisy_history_low_actor_lr_scratch.yaml)

## Measurements across training

These historical figures were generated on 2026-10-31 from saved checkpoint evaluation JSON and TensorBoard scalar events. All four scratch trials completed 6M environment steps. No new learning, rollout, simulator, sensing, control or randomization changes were made. Each curve is one training seed; the two history seeds are shown separately, not pooled as a replicated estimate of all variants.

![Juggles, episode steps and reward across training](../../../docs/assets/juggling/history-checkpoint-curves.png)

Each checkpoint point is the unsmoothed mean of eight reused validation starts with a 5,000-step recording limit, using online actor weights. Physical juggles use the simulator true-state trajectory heuristic; sensed juggles use the noisy-position direction-reversal heuristic. Neither is contact timesteps. Episode lengths and juggle totals stop at the recording limit; these are observed bounded quantities, not full eventual episode durations. Reward is the undiscounted sum per episode, not the TD3 value estimate. This repeated validation series is distinct from the 32-start comparisons below.

![Exploratory training reward and episode steps](../../../docs/assets/juggling/history-training-reward-curves.png)

Training curves average completed episodes in fixed 100,000-environment-step bins, assigned by their logged completion step; bins have different episode counts. Exploratory behavior and randomized resets differ from checkpoint evaluation. TensorBoard has no per-training-episode juggle counts: juggle curves therefore use saved checkpoint evaluations, rather than inferring counts from reward or contacts. Raw episode data are provided to inspect variation hidden by means.

## Completed final comparisons: dated status correction

This updates the queued/running statuses in [E041](../../../docs/experiment_logbook.md#e041) and [E042](../../../docs/experiment_logbook.md#e042), preserving their historical entries. Saved status files show all original three trials and the conditional lower actor-LR trial completed. The earlier queued processes are no longer visible locally. Completion timestamps in saved status files are recording milestones; original rollout start/end times were not reconstructed.

Final 6M comparisons below reuse the same 32 selection starts (700000–700031), with a 20,000-step recording limit. These are descriptive measurements, without binary acceptance gates.

| Final policy | Mean physical juggles | Mean sensed juggles | Mean episode steps | Median episode steps |
|---|---:|---:|---:|---:|
| History seed 0 online | 77.53 | 68.03 | 5,447.00 | 2,419.5 |
| History seed 0 target | 138.41 | 121.72 | 9,859.81 | 7,798.5 |
| History seed 1 online | 137.81 | 120.38 | 9,861.56 | 9,633.5 |
| History seed 1 target | 145.16 | 128.63 | 10,558.88 | 8,706 |
| Longer discount online | 107.13 | 92.41 | 7,518.75 | 3,847.5 |
| Longer discount target | 198.63 | 175.53 | 14,141.53 | 19,725.5 |

For context, the historical 3.6M target actor averaged 225.06 physical juggles and 15,427.63 steps on these reused starts. Its separate 32 independent starts averaged 200.34 physical juggles and 13,761.03 steps. These selection comparisons do not establish a population ranking.

The lower actor-LR final benchmark used 32 starts from seed 200000 and a shorter 10,000-step limit: mean 97.06 physical juggles, 85.97 sensed juggles, 6,905.38 steps; median 10,000 steps. Different starts and recording limits prevent direct comparison with the table. Its checkpoint validation is comparable in protocol to the other curves.

## Sources and reproduction

- [Raw checkpoint episode metrics](../../../docs/assets/juggling/history-checkpoint-episodes.csv)
- [Raw training episode reward and steps](../../../docs/assets/juggling/history-training-episodes.csv)
- [Final comparison data and source paths](../../../docs/assets/juggling/history-final-comparison.csv)
- [Source SHA256 manifest and generation timestamp](../../../docs/assets/juggling/history-plot-provenance.json)
- [Plot generator](../../../scripts/analysis/juggle_training_plots.py)
- Local ignored raw sources: each run's `train/checkpoint_*/survival_eval/results.json`, `train/survival_eval/results.json`, and `train/events.out.tfevents.*`. Checkpoints remain under each run's `train/`; CSVs and provenance are tracked for remote review without ignored artifacts.

Reproduce from the repository root: `.venv/bin/python -m scripts.analysis.juggle_training_plots`. Requires the original local ignored run artifacts, matplotlib, NumPy and TensorBoard. The two figures and CSV exports are standalone tracked artifacts. Validation checks episode/scalar alignment, exported means, figure readability, documentation links, anchors/TOCs and whitespace. No expensive training checks were run for this retrospective analysis.
