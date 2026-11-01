# Early physical-motion curves show a tradeoff before useful juggling emerges

- **Date**: 2026-11-01 23:18 UTC retrospective analysis of saved active-run checkpoints.
- **Status**: done analysis; underlying training remains in-flight.
- **Run dirs**: the [six-trial sweep](2026-11-01_22-48_weaker-smoothness-sweep.md), [longer-discount pair](2026-11-01_23-09_smooth-long-discount-pair.md), and historical `runs/td3/noisy_history_20261029_seed0` baseline.

## Measurement and protocol

Extended the [progress generator](../../../scripts/analysis/juggle_smooth_sweep.py) to report physical motion, not just normalized command changes, and added the historical unregularized seed-0 control at matching saved training steps. All plotted points use the same eight reset starts from seed 10000, 5k-step recording limit and online actor weights. The control is truncated to the newest available sweep checkpoint. Missing checkpoints are not interpolated. Latest available points may have different training budgets, so compare equal x-coordinates rather than the ends of different curves.

The generator reads original NPZ true body positions and caches the existing [read-only finite-difference diagnostics](../../../scripts/analysis/juggle_motion_metrics.py) alongside ignored result files. Physical speed/acceleration/jerk curves show the **median across episodes of each episode's p95**, not a pooled-timestep percentile, maximum, or training-seed average. Raw episode juggles, steps and reward remain means. The new per-episode CSV preserves the cases behind all aggregates. Protocol guards reject mismatched checkpoint seeds/horizons/counts; motion sample counts must match episode lengths.

No training, policy inference, simulator stepping, sensing, action smoothing, PID, physics or randomization changes were needed for this analysis. Reviewed hashes of all live trainer/evaluator/simulator sources remain unchanged. Position-difference displacement speed is not instantaneous cached body velocity; workspace/solver position corrections and sampling can contribute. It must not be compared directly to the cached velocity cap as if both were the same measurement. No joint torque, arm geometry or UR5 stop-risk estimate is available here.

## Early matched 500k results

Each row is one training seed, averaged over eight episodes for juggles/steps. Motion columns are medians of episode-p95 values. The two seeds per coefficient remain separate; no across-seed mean or confidence estimate is asserted.

| Coefficient | Seed | Mean physical juggles | Mean episode steps | Median episode p95 acceleration (m/s²) | Median episode p95 jerk (m/s³) |
|---|---:|---:|---:|---:|---:|
| 0, historical baseline | 0 | 0.375 | 85.000 | 27.069 | 564.484 |
| 0.002 | 0 | 0.500 | 73.625 | 14.429 | 273.041 |
| 0.002 | 1 | 0.375 | 72.375 | 15.861 | 324.069 |
| 0.008 | 0 | 0.625 | 83.250 | 9.927 | 212.037 |
| 0.008 | 1 | 0.125 | 56.875 | 12.358 | 206.713 |

Lower sampled motion is visible in these early records, but none represents useful high-performance juggling. The gamma-0.995 coefficient-0.0005 pair had not saved a 500k evaluation at this snapshot, and the new gamma-0.999 pair had not reached its first 500k checkpoint. These observations do not determine full-budget outcomes or justify stopping the learners. No binary acceptance gate is used.

![Juggling, episode steps, reward and command changes](../../../docs/assets/juggling/sweep-early-juggling-20261101.png)

![Sampled physical motion distributions](../../../docs/assets/juggling/sweep-early-motion-20261101.png)

Figures generated 2026-11-01 from saved JSON/NPZ artifacts, with the historical control dashed black. These dated figure copies preserve this analysis snapshot; the separate moving report/figures may later include additional checkpoints. New curves are separate training seeds, not replications for every episode. Whole-episode motion distributions still depend on episode duration and visited states; quieter motion can reflect little useful movement. Neither plot is a robot-safety certificate or proof of infinite juggling.

## Sources, checks and continuation

- [Dated summary snapshot](../../../docs/assets/juggling/sweep-early-summary-20261101.csv), [dated per-episode snapshot](../../../docs/assets/juggling/sweep-early-episodes-20261101.csv), and [source/analysis hashes](../../../docs/assets/juggling/sweep-early-provenance-20261101.json) preserve the 33-point/264-episode analysis below.

- [Moving report and protocol explanation](../../../docs/juggling-smooth-sweep.md).
- [Raw checkpoint/milestone summary CSV](../../../docs/juggling-smooth-sweep.csv), now including physical motion and seed base.
- [Raw per-episode CSV](../../../docs/juggling-smooth-sweep-episodes.csv), including juggles, steps, reward, motion, seed and original trajectory path.
- Each source's local `motion_metrics.json` contains full episode distributions and sample counts. Original source JSON/NPZ/checkpoints remain in the ignored run roots.
- Reproduce: `.venv/bin/python -m scripts.analysis.juggle_smooth_sweep` from repository root with the original run artifacts available.

Validated 33 exported summary rows against 264 episode rows for means and medians; all source protocol and motion-sample-count guards passed. Reviewed figures, relative links/images, unique ordered TOCs/anchors and whitespace. These are fresh analysis checks, not rerun training checks.

All eight research learners remained live, with 11 total user GPU allocations including three preexisting jobs. CPU-only long reference job 115864 remained live. Its first saved trajectory recorded 100,000 steps and 1,403 physical juggles without terminal/truncation flags; the other 31 cases were still pending and no aggregate claim is made. The research goal remains active: continue full budgets, compare matched milestones, then test retained policies on longer independent starts jointly with motion.
