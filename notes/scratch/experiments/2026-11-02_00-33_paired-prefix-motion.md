# Equal-prefix diagnostics retain a physical-motion reduction for weak regularization

- **Date**: 2026-11-02 00:33 UTC analysis/writeup; saved1.4M checkpoint evaluations generated earlier by live training jobs.
- **Status**: offline analysis done; all eight full-budget scratch learners and the long-reference evaluation remain in-flight.
- **Run dirs**: `runs/td3/noisy_history_{20261029_seed0,smooth_0005_20261101_seed0,smooth_0005_20261101_seed1}/train/checkpoint_1400000/survival_eval` (expand the names individually).
- **Configs**: [weak seed0](../../../configs/td3/sustained/noisy_history_smooth_0005_seed0.yaml), [weak seed1](../../../configs/td3/sustained/noisy_history_smooth_0005_seed1.yaml).

## Question and method

Do lower command changes correspond to lower sampled physical motion when compared over equal episode prefixes? Compare original-discount gamma .995 baseline and coefficient .0005 at1.4M on the identical eight reused starts10000–10007 with5000-step recording limits. For every reset seed use the first min(250,baseline episode length,candidate episode length) recorded true paddle positions. Compute finite differences at20Hz without prepending the reset position. Both members of a pair use exactly the same number of samples. Different visited states remain; cropping depends on outcomes, so this is a descriptive diagnostic, not a causal or hardware-safety claim. No new environment steps, policy inference or training.

Added [paired analysis command](../../../scripts/analysis/juggle_paired_motion.py), which refuses different effective simulator configs, horizons, scenario sets, duplicate seeds and different seed sets, and checks trajectory lengths and finite positions. It retains full-episode juggling/duration context, per-pair raw motion distributions, source/trajectory/code hashes and missing higher derivatives explicitly. Undefined ratios for zero baseline motion stay missing. It does not modify hash-protected live inputs.

## Results

| Regularized training seed | Metric | Baseline median episode-p95 | Regularized median episode-p95 | Median per-episode regularized/baseline ratio |
|---|---|---:|---:|---:|
| 0 | Speed (m/s) | 2.092886 | 1.805705 | 0.858074 |
| 0 | Acceleration (m/s²) | 24.117063 | 19.398697 | 0.850750 |
| 0 | Jerk (m/s³) | 525.875661 | 384.627326 | 0.752062 |
| 1 | Speed (m/s) | 2.090143 | 1.833593 | 0.892928 |
| 1 | Acceleration (m/s²) | 24.104683 | 21.434353 | 0.933743 |
| 1 | Jerk (m/s³) | 517.914672 | 380.882916 | 0.825276 |

The median ratio is computed per matched episode before aggregation; it is not the ratio of the displayed medians. Seed0 paired prefix lengths are239,250,250,250,250,250,201,131; seed1 lengths133,204,211,141,250,141,122,125. Each comparison uses8 pairs,16total; one historical baseline training seed is reused for both. The two regularized training seeds are reported separately.

Full-episode validation means give context: baseline6.75 physical/7 sensed juggles and612.875 steps; seed0 regularized10.5/9.75 and742 steps; seed1 regularized1.875/1.75 and167.375 steps. Action-change RMS baseline .581364, regularized seeds0/1 .368255/.324812. These are eight-start5k validation results, not the32-start20k milestone comparisons. The seed variation precludes a reliable-performance conclusion.

[Seed0 raw paired metrics and provenance](../../../docs/assets/juggling/smooth-paired-prefix-1m4-seed0-20261102.json); [seed1](../../../docs/assets/juggling/smooth-paired-prefix-1m4-seed1-20261102.json). No new figure is generated for this small comparison; the table is the direct result. Existing whole-episode [training/reward/motion plots](../../../docs/juggling-smooth-sweep.md) use a different aggregation window and remain labeled accordingly. Related: [first fixed1M comparison](2026-11-01_23-54_first-1m-smoothness-comparison.md), [actual PID-target audit](2026-11-01_22-59_pid-target-replay-audit.md).

## Verification and reproduction

Six tests pass (three new paired-diagnostic tests plus three existing kinematic tests). Analytic cubic motion with a large spike outside the shared window verifies equal cropping and physical jerk units; mismatched seeds/rates are rejected; short trajectories and zero denominators stay missing. All97 reviewed hash entries across live learners/queued continuation match. No simulator/sensing/control/DR change, learner restart or new allocation; eleven total user GPUs remain allocated.

```sh
.venv/bin/python -m unittest scripts.td3.tests.test_juggle_paired_motion scripts.td3.tests.test_juggle_motion_metrics -q
.venv/bin/python -m scripts.analysis.juggle_paired_motion --baseline runs/td3/noisy_history_20261029_seed0/train/checkpoint_1400000/survival_eval/results.json --candidate runs/td3/noisy_history_smooth_0005_20261101_seed0/train/checkpoint_1400000/survival_eval/results.json --out /tmp/paired-seed0.json
# Repeat with seed1 candidate for the second comparison.
```

## Next

Keep all scratch runs at their existing full6M budget, then compare the scheduled frozen3M/6M tests and independent long-reference distribution. Equal-prefix evidence supports a sampled motion reduction, but useful smooth long-duration juggling remains unproven; no real robot is connected and no protective-stop prediction is made.
