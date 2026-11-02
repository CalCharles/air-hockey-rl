# Complete original 1M comparison shows mixed weak-penalty performance and lower motion

- **Date**: 2026-11-01 23:54 UTC retrospective analysis/writeup; rollout start/end times for the in-job comparisons were not reconstructed.
- **Status**: done analysis; all eight 6M learners remain running.
- **Run dirs**: original [sweep](2026-11-01_22-48_weaker-smoothness-sweep.md), [longer-discount pair](2026-11-01_23-09_smooth-long-discount-pair.md), and CPU-only `runs/td3/juggle_history_baseline_1m_20261101` historical comparison.

## Fixed-budget comparison

The original six-trial 1M tests are complete: coefficients 0.0005, 0.002 and 0.008, seeds 0/1, each with frozen online and target weights. The analysis began at 23:54 UTC and incorporated the last original-pair results before publication; exact snapshot generation time is in provenance. Added an archived unregularized 1M online/target comparison on exactly the same 32 reused starts 700000–700031, 20k-step recording limit, canonical noisy sensing/control and established DR. Effective environment configs match after normalizing reset-seed keys that the evaluator overrides per case. The historical actor was trained from scratch; this new comparison loads it for inference only and does not initialize any learner from weights.

CPU-only baseline job **115874** completed with exit 0, elapsed 48 seconds. Its worker recorded completion at **2026-11-01 23:44:01.971121 UTC / 17:44:01.971121 -0600**; that is a worker milestone, not an exact rollout end time. [Exact launch/commands/model hashes](../../../docs/juggling-baseline-1m-comparison-launch.json).

Each row below is one training seed/weight variant over 32 episodes. Online/target variants from one run are correlated, not independent seed replications. Motion columns use episode distributions; different durations/visited states remain a confounder. All recorded episodes in this snapshot ended before 20k. These are early descriptive comparisons on reused starts, with no binary acceptance gate.

| Policy at 1M | Mean physical juggles | Mean episode steps | Mean reward | Action-change RMS | Median episode p95 position-derived jerk (m/s³) |
|---|---:|---:|---:|---:|---:|
| Unregularized seed 0 online | 1.250 | 147.250 | 89.469 | 0.62428 | 478.759 |
| Unregularized seed 0 target | 1.094 | 154.688 | 98.344 | 0.58125 | 471.590 |
| 0.0005 seed 0 online | 1.719 | 155.344 | 95.312 | 0.33966 | 383.956 |
| 0.0005 seed 0 target | 2.250 | 185.250 | 118.750 | 0.33559 | 409.068 |
| 0.0005 seed 1 online | 0.781 | 78.000 | 47.938 | 0.26640 | 305.404 |
| 0.0005 seed 1 target | 0.938 | 85.312 | 50.844 | 0.26408 | 300.828 |
| 0.002 seed 0 online | 0.688 | 80.375 | 47.469 | 0.15001 | 266.419 |
| 0.002 seed 0 target | 0.562 | 75.844 | 44.969 | 0.14585 | 262.351 |
| 0.002 seed 1 online | 0.188 | 72.000 | 37.656 | 0.17277 | 256.103 |
| 0.002 seed 1 target | 0.188 | 73.938 | 42.594 | 0.17734 | 300.750 |
| 0.008 seed 0 online | 0.375 | 75.812 | 43.094 | 0.09961 | 249.735 |
| 0.008 seed 0 target | 0.312 | 68.812 | 39.000 | 0.09738 | 221.976 |
| 0.008 seed 1 online | 0.594 | 75.750 | 41.219 | 0.08287 | 199.998 |
| 0.008 seed 1 target | 0.719 | 78.875 | 43.188 | 0.08370 | 203.197 |

![First matched 1M comparison](../../../docs/assets/juggling/smooth-first-1m-comparison-20261101.png)

Figure generated on America/Chicago date 2026-11-01 from saved results (exact UTC timestamp in provenance): each bar is one seed/weights variant, not an across-seed average. Coefficient 0.0005 seed 0 averages 1.71875/2.25 physical juggles and 155.34375/185.25 steps online/target; seed 1 averages only 0.78125/0.9375 juggles and 78/85.3125 steps. The weak-penalty results therefore do not establish a consistent improvement. Stronger coefficients 0.002/0.008 have lower command changes and sampled jerk but mean episode lengths only 68.8–80.4 steps, versus 147.3–154.7 for the historical baseline. None is useful high-performance juggling yet at this early budget. All learners continue unchanged toward 6M; the longer-discount pair has not completed its 1M tests at the snapshot.

[Fixed summary snapshot](../../../docs/assets/juggling/smooth-first-1m-summary-20261101.csv); [448 per-episode records](../../../docs/assets/juggling/smooth-first-1m-episodes-20261101.csv); [result/model hashes and generation time](../../../docs/assets/juggling/smooth-first-1m-provenance-20261101.json). The [moving report](../../../docs/juggling-smooth-sweep.md) keeps 8-start checkpoint curves distinct from these 32-start tests, and now includes the archived 1M comparison only in its separately labeled CSV rows.

## Offline deployment-adapter check

The retained 3.6M target actor (SHA256 `c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b`) produced **9,146 exactly matching normalized commands across 32 saved episodes** through the real-eval factory and history adapter. Inputs were only the saved noisy observations and known previous actions. Each episode reset adapter memory; up to 300 actions per episode covered repeated 50-step history rollovers. The actual robot environment was never created or connected, and no physics was stepped.

[Parity records/source hashes](../../../docs/assets/juggling/history-deployment-parity-20261101.json); [standalone verifier](../../../scripts/analysis/juggle_deployment_parity.py). Reproduce with `.venv/bin/python -m scripts.analysis.juggle_deployment_parity --bundle BUNDLE --results RESULTS_JSON --out OUTPUT_JSON --steps 300` using the source paths in the parity record. This check applies to the retained target model and normal adapter calls, not automatically to every future checkpoint or to full real-world rollouts.

**Unresolved transfer behavior:** the adapter source documents that a post-policy held/overridden command can leave its private history recording the intended action instead of the final command sent to the environment. The normal-path parity check does not cover or resolve that path, e-stop recovery, live camera timing, joint dynamics or stop risk. The next implementation check should address command-history bookkeeping with hold/override tests while preserving PID, workspace, action filters and existing hold behavior. No real-robot readiness claim is made.

## Checks and continuation

Regenerate the dated figure with `.venv/bin/python -m scripts.analysis.plot_juggle_fixed_comparison --summary docs/assets/juggling/smooth-first-1m-summary-20261101.csv --out docs/assets/juggling/smooth-first-1m-comparison-20261101.png --title "Original sweep: fixed 1M comparison"`. The [plot source](../../../scripts/analysis/plot_juggle_fixed_comparison.py) rejects mixed training budgets or protocols.

Checked exact matched configs, model/result hashes, source protocols, raw means/motion aggregates, figure readability, deployment-action equality, relative links/images, ordered unique logbook anchors/TOCs and whitespace. The parity verifier was rerun after adding its final source-hash metadata; this is an offline inference check, not new training. All eight live trainer/evaluator/simulator manifests remain unchanged. No new dynamics/sensing/control/DR changes or policy warm starts.

The original CPU reference test remains live; its completed cases include 100k/1,403 juggles, 100k/1,539 juggles and 57,980 steps/948 juggles. The other cases were pending at inspection, so these are individual observations rather than a complete-cohort estimate. The dependency-guarded CPU continuation remains queued. User GPU allocation remains 11. Continue training, then compare later fixed milestones and joint motion/performance before selecting new longer independent tests. The goal remains active and unsolved.
