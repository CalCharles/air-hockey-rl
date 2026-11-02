# Complete 3M controls retain a large online/target juggling difference

- **Date**: 2026-11-02 01:06 UTC collection/writeup; workers submitted2026-11-02 00:46UTC. Gamma .995 worker reports completion01:04:46UTC; authoritative Slurm durations below identify execution rather than note timestamps.
- **Status**: control evaluations done; eight fresh smoothness training runs and independent long-reference test in-flight.
- **Run dirs**: `runs/td3/juggle_history_baseline_3m_20261102`, `runs/td3/juggle_history_longdiscount_baseline_3m_20261102`.
- **Configs**: frozen saved3M evaluation bundles and exact configuration/source hashes in [gamma .995 manifest](../../../docs/juggling-baseline-3m-comparison-launch.json) and [gamma .999 manifest](../../../docs/juggling-longdiscount-baseline-3m-launch.json).

## Setup and completion evidence

Follow-up to [the launch record](2026-11-02_00-46_matched-3m-controls.md), preserving that note's historical running status. Two CPU-only jobs evaluated unregularized historical seed0 online and target actors at exactly3M training steps. Both use32 reused selection starts700000–700031 and20k recording limits, matching the scheduled new-policy3M tests. Job115886 (gamma .995) completed in18m44s, job115887 (gamma .999) in9m16s; both Slurm COMPLETED/exit0 and both worker statuses complete. No new learner, pretrained warm start, sensing/dynamics/control/DR change or GPU allocation. All eight new policies remain from scratch with unchanged6M budgets.

## Results

| Discount | Weights | Physical juggles | Sensed juggles | Steps | Reward | Action-change RMS | Median episode-p95 acceleration (m/s²) | Jerk (m/s³) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| .995 | online | 134.25000 | 118.37500 | 9049.28125 | 6775.84375 | 0.35854 | 17.18944 | 336.60024 |
| .995 | target | 202.62500 | 178.81250 | 13708.56250 | 10279.34375 | 0.35004 | 16.86994 | 325.45011 |
| .999 | online | 34.12500 | 30.40625 | 2391.53125 | 1683.28125 | 0.57400 | 18.44648 | 377.72600 |
| .999 | target | 142.37500 | 129.50000 | 9376.62500 | 6915.43750 | 0.49599 | 18.10044 | 370.09000 |


Juggles, steps, rewards and command-change values are episode means over32 complete records per actor,128 total. Motion columns are medians of per-episode p95 true-position finite differences at20Hz, excluding an initial reset-position sample. Longer episodes visit different states, so these do not isolate a causal smoothness change. Online and target weights come from the same learner; the four rows are not independent training replications. One historical seed per discount limits inference. Reused selection starts are not an independent generalization cohort. Finite recording limits censor counts/durations; no infinite-juggling or robot protective-stop claim.

[Fixed raw summaries](../../../docs/assets/juggling/historical-3m-controls-summary-20261102.csv), [all128 episode records](../../../docs/assets/juggling/historical-3m-controls-episodes-20261102.csv), [result-source hashes](../../../docs/assets/juggling/historical-3m-controls-provenance-20261102.json). Frozen actor hashes and ignored local source trajectories/configs/commands remain linked from the launch manifests. No equivalent result from the new regularized3M actors is available yet; compare identical training budgets when those tests finish. Historical3.6M/6M policies are different budgets and are not substituted here.

## Training and motion reporting

The [moving report](../../../docs/juggling-smooth-sweep.md) now includes all four complete3M control rows separately from8-start5k checkpoint validation. The32-start controls are not plotted as if they were8-start curve points.

![Checkpoint training, rewards, juggles, steps and commands](../../../docs/juggling-smooth-sweep.png)

![Whole-episode sampled physical motion](../../../docs/juggling-smooth-sweep-motion.png)

Figures regenerated2026-11-02 from saved checkpoint artifacts. Compare the same training-step x-coordinate; latest points differ across runs. These moving curves are validation evidence rather than new long-duration3M control plots. Whole-episode motion remains confounded by duration/visited states; see [the equal-prefix diagnostic](2026-11-02_00-33_paired-prefix-motion.md) for a different window at1.4M. The four-row control comparison is reported directly in the table; no separate historical control figure is generated.

## Checks and reproduction

Verified Slurm terminal states/exit codes and worker status, all four32-episode result protocols, frozen bundle/model/source hashes, summary means and motion medians against raw records, CSV counts, links/images, ordered unique logbook anchors/TOCs and whitespace. Report refresh uses saved files only; no training was rerun for documentation. No implementation changes or additional implementation tests for this collection.

Rebuild the moving report with `.venv/bin/python -m scripts.analysis.juggle_smooth_sweep`. The dated CSVs select `phase == 32_reused_starts_historical` and `step == 3000000`; the dated provenance identifies exact source JSON hashes. Original inference commands and submitted worker/batch sources are preserved in each launch manifest.

## Next

Continue the eight6M learners and scheduled frozen3M/6M online/target comparisons. The large historical online/target gap reinforces measuring both rather than interpreting one checkpoint actor alone. The retained3.6M target100k-horizon CPU reference115864 is still live with29 saved episodes at inspection; continuation115869 remains pending. Partial reference files do not establish a full-cohort result. Total user GPUs remain11 including three preexisting allocations, below16. Useful smooth long-duration performance and hardware stop behavior remain unproven.
