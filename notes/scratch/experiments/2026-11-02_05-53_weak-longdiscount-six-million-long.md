# The weak-penalty longer-discount seed0 target receives a paired100k-step endurance test.

- **Date**: 2026-11-02 05:53:47 UTC documentation; actual submission2026-11-02T05:45:30.643820+00:00.
- **Status**: in-flight at authoritative check2026-11-02T05:48:02.228250+00:00, CPU115950 RUNNING, worker evaluating. No outcome analyzed in this launch note.
- **Run dirs**: `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/juggle_smooth_0005_g0999_seed0_target6m_long_20261102`.
- **Configs**: completed6M gamma .999/.0005 seed0 target and canonical noisy sensing/control/DR; [full source/worker/batch manifest](../../../docs/juggling-weak-longdiscount-6m-long-reference-launch.json).

## Selection and counterevidence

[The completed6M comparison](2026-11-02_05-53_weak-longdiscount-full-six-million.md) has target131.75 physical/114.9375 sensed juggles and9347.3125steps on32 reused20k starts. Compared with the same-discount unregularized target, equal-prefix command replay lowers mean PID-target change RMS .0342456 vs.0496633m, target p99 .202848 vs.242101m, and median p99 body jerk569.786 vs685.940m/s³. Its endurance is lower than control198.625/14141.53125. This is an intermediate tradeoff worth measuring longer, not a success gate.

**Subsequent full-table inspection finds worse rare maxima:** candidate acceleration56.893813 vs42.189274m/s² and jerk1531.071296 vs1294.700735m/s³. The manifest explicitly records this post-submission observation. The policy is not promoted or described as hardware-ready; lower average action/target changes do not eliminate motion spikes.

## Exact test and provenance

CPU **115950**, zero GPUs,2 CPUs,8GB,12h maximum. Actor SHA **0f4e78c40eafd4b3bbd89f92b5199752a01ac67172cb5a790347e56c2ef7e68c**. Private model/args/config/history bundle and source hashes are checked before/after inference. This is fixed-policy evaluation of a completed from-scratch learner, not new training or a pretrained warm start.

32 new candidate episodes on **globally reused paired-development starts9000000–9000031**,100k recording limit, original raw reward. Reference remains gamma .995 seed0 3.6M targetC8, mean546.25 physical/482.9375 sensed juggles and37131.03125steps,29 puck losses and three100k-censored records. Different discounts/budgets and repeated cohort use limit causal/generalization inference. No reference NPZ is reused as candidate behavior.

Expected outputs are all32 NPZ trajectories, `paired_reference_100k_eval/results.json`, motion metrics, first-predeclared-episode GIF and durable worker status. Missing/partial cases remain pending, not zero results. Consult actual Slurm state before terminal handling; an observation timeout does not authorize restarting a live job. No continuation is queued. No candidate long outcome figure/GIF exists yet.

```sh
squeue -j 115950 -o '%.18i %.9T %.10M %b %j'
```

This test adds no GPUs; user total is9 at submission/check, including three unrelated allocations. Original learners continue unchanged, as do the four reward-cost learners. No additional RL variant, cancellation, simulator modification or real-robot action. Finite simulated measurements establish neither infinite juggling nor UR5 stop avoidance.
