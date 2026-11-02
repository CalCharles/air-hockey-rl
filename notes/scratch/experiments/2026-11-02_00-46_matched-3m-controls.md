# Discount-matched frozen 3M controls are running ahead of the smoothness milestone

- **Date**: 2026-11-02 00:46 UTC submission; this note records the launch, not completed outcomes.
- **Status**: in-flight
- **Run dirs**: `runs/td3/juggle_history_baseline_3m_20261102`, `runs/td3/juggle_history_longdiscount_baseline_3m_20261102`.
- **Configs**: archived `args.yaml`/`config.yaml`/`rma_meta.json` copied from each original `train/checkpoint_3000000` into separate online/target evaluation bundles; source paths/hashes are in the manifests below.

## Question and setup

Prepare fixed-budget controls for the eight ongoing scratch smoothness trials' scheduled3M comparisons. Existing original3M evaluations use seeds300000/600000 and/or10k horizons, so cannot replace these exact32 reused starts700000–700031 with20k recording limits. Historical3.6M weights are also a different training budget. No existing result with this full3M budget/weights/starts/horizon was reused incorrectly.

Two CPU-only jobs evaluate historical unregularized gamma .995 and .999 actors, each online then target. Model files are frozen in new bundles; this is inference, with no learner, checkpoint warm start or residual policy. The active eight new policies remain from scratch and keep their6M budgets. Each control requests2CPUs/8GiB/4hours, explicitly no GPUs and empty CUDA_VISIBLE_DEVICES. GIF recording is disabled to avoid redundant rendering; true trajectories and raw metrics are retained. Existing performance GIFs remain [available](2026-11-02_00-38_weak-policy-performance-gifs.md).

| Control | Job | Submission UTC | Status at inspection |
|---|---|---|---|
| gamma .995 historical seed0 | 115886 | 2026-11-02 00:46:02.915757 | running, online role |
| gamma .999 historical seed0 | 115887 | 2026-11-02 00:46:03.052888 | running, online role |

The saved effective simulator YAMLs match the active canonical original-sensing YAML exactly. Shared saved training arguments differ only in intended regularization and run/file locations, plus the reduced checkpoint cadence for the .999 pair; seed1 changes only its replication seed. Reviewed simulator/source hashes match the established recipe. The workers verify frozen actor and evaluator/simulator/config hashes before each role; manifests additionally preserve bundle file hashes and exact submitted worker/batch sources. No sensing, dynamics, PID, action filter, geometry, contact, limit or DR change.

[Gamma .995 manifest and commands](../../../docs/juggling-baseline-3m-comparison-launch.json); [gamma .999 manifest and commands](../../../docs/juggling-longdiscount-baseline-3m-launch.json). Frozen weight hashes identify online and target separately. No result, mean, binary gate or hardware-readiness claim is available from these new jobs yet.

## Reporting and verification

Updated the existing [report generator](../../../scripts/analysis/juggle_smooth_sweep.py) to collect each available3M control role separately, with the fixed protocol/budget label. Missing roles remain absent. The [moving training/reward/motion report](../../../docs/juggling-smooth-sweep.md) still plots8-start5k curves separately from32-start20k comparisons. Historical curves stop at available same-gamma sweep progress. Refreshed plots use saved completed evaluations only, with no extra rollout.

![Training reward, juggling, duration and commands](../../../docs/juggling-smooth-sweep.png)

![Sampled physical motion](../../../docs/juggling-smooth-sweep-motion.png)

Figures regenerated2026-11-02 from saved artifacts. These are descriptive validation curves across unequal latest budgets; compare matching x-coordinates and keep seeds separate. Motion is whole-episode median per-episode p95 of position differences at20Hz, with duration/visited-state confounding and no UR5 load/stop model. These figures are refreshed by the generator, while this launch note preserves the pending status at writing. [Summary CSV](../../../docs/juggling-smooth-sweep.csv), [per-episode raw data](../../../docs/juggling-smooth-sweep-episodes.csv).

The generator ran successfully; source compilation and per-summary episode mean/count checks, manifests/bundle/source hashes, relative links/images, unique ordered logbook anchors/TOCs and whitespace are validated before push. No implementation training code changed; no new implementation test is needed for adding control artifact locations to the report.

Reproduce inference with the exact worker/batch commands stored in each manifest; do not resubmit duplicate jobs while115886/115887 remain live. Rebuild reports with `.venv/bin/python -m scripts.analysis.juggle_smooth_sweep` after results arrive.

## Next

Collect complete control results, then compare to the scheduled frozen regularized3M actors on the exact same32 starts and20k horizon. Retain full6M budgets and later independent testing; no early acceptance gate. Eight learners and the long100k reference remain active, with CPU continuation115869 queued after the original reference115864. Total user GPUs remain11 including three preexisting allocations, below16. One historical baseline training seed per discount limits statistical inference.
