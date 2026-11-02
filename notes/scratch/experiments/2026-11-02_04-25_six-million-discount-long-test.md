# The late longer-discount target is frozen for a paired100k-step endurance test.

- **Date**: 2026-11-02 04:25:20 UTC documentation; actual submission2026-11-02T04:18:56.105327+00:00.
- **Status**: in-flight, CPU115924 confirmed Slurm RUNNING.
- **Run dirs**: `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/juggle_history_g0999_target6m_long_20261102`.
- **Configs**: frozen6M history actor metadata/args and canonical original noisy sensing/DR; [complete manifest](../../../docs/juggling-longdiscount-6m-long-reference-launch.json).

## Question / selection

The [completed6M controls](2026-11-02_04-25_six-million-strong-and-controls.md) identify the unregularized gamma .999 seed0 target as a useful endurance/command-motion tradeoff:198.625 physical/175.53125 sensed juggles and14141.53125steps on32 reused starts/20k; action-change RMS .3124846113 and median episode p95 position jerk287.9918377. It has fewer juggles than the retained3.6M target on that development protocol but warrants a longer paired check. Mean command changes and sampled jerk do not prove absence of large actual controller-target tails.

## Exact setup

CPU job **115924**, zero GPUs,2 CPUs,8GB,12h maximum. Private target model SHA256 **204f6af6731cafd1dfa60d8466776908e85122f79722e2a92797033f3f05880e**. Source is the completed6M archived gamma .999 control's frozen target, not a learner warm start. Args/config/history metadata and all simulator/loader/evaluator sources are hashed. No training, source edits or new sensing/control changes.

32 fresh candidate episodes use the **globally reused paired-development reference starts9000000–9000031**,100000-step recording limit, ordinary raw reward. This cohort is not independent and is not an acceptance gate. Comparator is retained gamma .995 seed0 3.6M target SHAc8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b, whose completed reference has mean546.25 physical/482.9375 sensed juggles and37131.03125steps,29 puck-loss endings and three100k-censored records. Different training budgets/discounts mean this is a policy comparison rather than a causal hyperparameter estimate. Source reference results are linked/hashed in the manifest; no old reference NPZ is reused as candidate behavior.

Expected outputs: `paired_reference_100k_eval/results.json`, one raw NPZ per reset, `motion_metrics.json`, existing first-episode renderer GIF and durable worker status. Worker checks every frozen source/model hash before and after evaluation. Missing or partial episodes remain unreported, never counted as zero or treated as completed. Observe actual Slurm state before any terminal handling; a timeout or temporary missing path alone is not a restart reason. No continuation job is queued.

## Evidence / next

Both source control roles completed; this longer candidate outcome is still pending. No candidate endurance figure/GIF exists yet. The [retained reference distribution and performance clip](2026-11-02_01-16_long-reference-results.md) provide historical context. After completion, verify trajectories, publish every reset's physical/sensed juggle counts, steps and reward, and compare equal-prefix position motion and actual filtered-controller target tails. Finite recordings cannot establish infinite juggling or real-robot stop safety.

```sh
squeue -j 115924 -o '%.18i %.9T %.10M %b %j'
# Exact worker/batch and reproduction command are embedded in the launch manifest.
```

This test runs alongside the new [reward-cost learners](2026-11-02_04-25_target-cost-scratch-launch.md); total GPU usage is unaffected. No other jobs were restarted or canceled.
