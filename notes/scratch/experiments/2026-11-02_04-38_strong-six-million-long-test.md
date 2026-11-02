# The stronger-penalty seed0 target is frozen for a paired100k-step endurance test.

- **Date**: 2026-11-02 04:38:31 UTC documentation; actual submission2026-11-02T04:34:24.848079+00:00.
- **Status**: in-flight; CPU job115927 confirmed Slurm RUNNING, worker evaluating.
- **Run dirs**: `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/juggle_smooth_008_seed0_target6m_long_20261102`.
- **Configs**: completed6M gamma .995/.008 seed0 target bundle and canonical noisy sensing/DR; [full launch/worker manifest](../../../docs/juggling-strong-6m-long-reference-launch.json).

## Question / selection

The [completed6M actual-command audit](2026-11-02_04-38_medium-six-million-command-tails.md) distinguishes a smoother policy from a merely smaller action-delta proxy. On32 reused20k starts, this target averages100.6875 physical/89.46875 sensed juggles and7270.6875steps. Versus the late gamma .999 target on equal per-start prefixes it reduces mean cached PID-target change RMS .0302402 vs.0491409m, largest measured change .279315 vs.415912m and median cached-velocity jerk p99 574.069 vs689.313m/s³. The endurance loss is real; long testing will measure its extent rather than applying a binary acceptance gate.

## Exact test

CPU **115927**, zero GPUs,2 CPUs,8GB,12h maximum. Private model SHA256 **3098fd104583404e69a577fd903f44099a57dd0a40b29de9bc8d5f2e14f32241**. Its original training completed from random initialization; this job is frozen inference, not training or a checkpoint warm start. Args/config/history metadata and current source fingerprints are checked before and after evaluation. No sensing/control/physics changes or further action filter.

32 **new candidate episodes on globally reused paired-development starts9000000–9000031**,100k recording horizon. This cohort is not independent. Comparator is retained gamma .995 seed0 3.6M target SHAc8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b, completed mean546.25 physical/482.9375 sensed juggles and37131.03125steps,29 puck-loss endings and three100k-censored records. Same discount/seed but different training budgets/penalty make this a policy comparison; no causal or infinite-horizon claim. No old reference NPZ is reused as new candidate behavior.

Expected artifacts: `paired_reference_100k_eval/results.json`, every reset's NPZ, position-motion JSON, durable worker status and existing first-predeclared-episode renderer GIF. All outcomes remain pending. No long performance figure/GIF exists yet; the [retained reference](2026-11-02_01-16_long-reference-results.md) is historical context only. After completion, verify all trajectories, publish raw physical/sensed juggles, episode steps, reward and motion, and compare the same prefix exposure with the retained reference.

This runs alongside CPU115924's [late longer-discount target test](2026-11-02_04-25_six-million-discount-long-test.md). Both use zero GPUs; user total remains9 at this turn's live check, including three unrelated allocations. Existing learners are untouched. No restart or continuation job is queued. A timeout or temporarily missing output is not terminal evidence; consult authoritative Slurm state.

```sh
squeue -j 115924,115927 -o '%.18i %.9T %.10M %b %j'
```

Neither this selection nor the lower measured command tails establish that the robot would avoid an e-stop. No real hardware has been accessed.
