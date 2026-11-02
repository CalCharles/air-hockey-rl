# Longer-discount smoothness reduces motion but has weaker early juggling

- **Date**: 2026-11-02 00:27 UTC writeup; historical baseline evaluation submitted 00:22:25 UTC and completed by 00:23:21 UTC.
- **Status**: fixed 1M comparison done; eight scratch training runs and independent long-reference evaluation in-flight.
- **Run dirs**: `runs/td3/juggle_history_longdiscount_baseline_1m_20261102`, `runs/td3/noisy_history_smooth_0005_g0999_20261101_seed{0,1}`.
- **Configs**: [seed0](../../../configs/td3/sustained/noisy_history_smooth_0005_g0999_seed0.yaml), [seed1](../../../configs/td3/sustained/noisy_history_smooth_0005_g0999_seed1.yaml).

## Question and setup

Does weak action-change regularization retain performance with gamma .999? Compare frozen online/target actors at exactly 1M training steps on 32 reused starts700000–700031, 20,000-step recording limit. Historical unregularized gamma .999 seed0 is the discount-matched control. CPU-only job115884 completed, exit0, in56seconds; no policy training or hardware operation in this control evaluation. New policies started from scratch; their complete6M budgets remain unchanged. Simulator, sensing, control and established DR remain unchanged. This extends [the original-discount 1M comparison](2026-11-01_23-54_first-1m-smoothness-comparison.md), without rewriting its historical pending status.

Effective saved simulator YAML matches exactly. Shared effective training arguments differ only in the intended penalty, checkpoint cadence and run/file locations; seed1 additionally changes the replication seed. Live source hash guards pass. Validation checked151 summary rows against1688 episode records, unique ordered logbook anchors/TOCs, relative links, figure existence, Python compilation and whitespace; no new implementation tests are warranted for report-only changes.

## Results

| Variant | Physical juggles | Sensed juggles | Steps | Reward | Action-change RMS | Median episode-p95 acceleration (m/s²) | Jerk (m/s³) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Penalty .0005 seed0 online | 0.62500 | 0.65625 | 80.46875 | 48.90625 | 0.14492 | 13.17595 | 239.66753 |
| Penalty .0005 seed0 target | 0.84375 | 0.84375 | 94.28125 | 56.03125 | 0.12859 | 12.03452 | 230.90532 |
| Penalty .0005 seed1 online | 0.46875 | 0.43750 | 74.78125 | 43.81250 | 0.12686 | 12.49095 | 239.76707 |
| Penalty .0005 seed1 target | 0.53125 | 0.43750 | 79.65625 | 47.40625 | 0.12339 | 12.07333 | 218.91961 |
| Baseline seed0 online | 2.46875 | 2.00000 | 210.06250 | 129.53125 | 0.65211 | 25.07715 | 483.45565 |
| Baseline seed0 target | 2.25000 | 2.00000 | 221.00000 | 143.87500 | 0.68327 | 27.39117 | 553.13860 |

Each value for juggles, duration, reward and command change is an episode mean. Motion columns are medians of episode-wise p95 position-difference estimates at the control rate. Different durations and visited states confound motion comparisons. These metrics do not measure motor energy, joint torque or robot protective-stop risk. Online/target actors share training; seeds remain separate rather than presenting episodes as independent training replications. One historical baseline seed limits inference. Reused evaluation starts are descriptive selection evidence.

![Fixed longer-discount comparison](../../../docs/assets/juggling/smooth-longdiscount-1m-comparison-20261102.png)

Figure generated2026-11-02 from the six frozen summaries/192 saved episodes; every bar keeps one seed/weight variant separate. [Summary](../../../docs/assets/juggling/smooth-longdiscount-1m-summary-20261102.csv), [raw episodes](../../../docs/assets/juggling/smooth-longdiscount-1m-episodes-20261102.csv), [result-source hashes](../../../docs/assets/juggling/smooth-longdiscount-1m-provenance-20261102.json), [launch/weights/source provenance and exact commands](../../../docs/juggling-longdiscount-baseline-1m-launch.json). Original trajectories/checkpoints remain in the ignored run directories named in those records. The moving [training/reward/motion plots](../../../docs/juggling-smooth-sweep.md) now show a separate historical control for each gamma, clipped to available same-gamma training progress; 8-start5k curves remain separate from these32-start20k results.

## Conclusion and next

At1M, the regularized longer-discount pair is quieter but juggles less than the matched historical control. This is not a final-budget conclusion or an acceptance gate. Continue all eight full6M jobs and scheduled frozen3M/6M tests before deciding the next algorithm/reward change. Eleven total user GPUs are allocated including three preexisting jobs, below16.

The independent retained-target100k-horizon CPU evaluation115864 remains live; its afterany continuation115869 stays pending. Two completed cases reached100000 steps with1403 and1539 physical juggles; another ended after57980 steps with948. These partial observations do not establish the complete32-case distribution or indefinite reliability. No new policy is established as robot-ready.

Reproduce the figures without rollouts:

```sh
.venv/bin/python -m scripts.analysis.juggle_smooth_sweep
.venv/bin/python -m scripts.analysis.plot_juggle_fixed_comparison --summary docs/assets/juggling/smooth-longdiscount-1m-summary-20261102.csv --out /tmp/longdiscount-1m.png --title 'Longer-discount smoothness at 1M steps'
```
