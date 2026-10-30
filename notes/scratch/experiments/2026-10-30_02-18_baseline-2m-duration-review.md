# The fixed 2M baseline sustains repeated juggling for about 51 seconds on average

- **Date**: 2026-10-30 02:18 UTC (retrospective analysis time, not rollout time)
- **Status**: done
- **Run dirs**: `runs/td3/noisy_iteration2_20261029_long/milestone_2m_benchmark`
- **Configs**: [noisy_long.yaml](../../../configs/td3/sustained/noisy_long.yaml), saved evaluation config in the run directory

## Question
Does the fixed 2M baseline show useful sustained juggling independently of the arbitrary 10,000-step evaluation horizon?

## Setup
Read the existing 32 episodes, seeds300000–300031, without rerunning the simulator or training. This is one trained policy seed and a reused benchmark, not 32 independent training seeds. Preserve the original noisy sensing, dynamics and randomization. Physical-trajectory juggle counts are diagnostic heuristics. Convert steps using the 0.05-second control interval. Related historical result: [E015](../../../docs/experiment_logbook.md#e015).

## Results
Mean duration is **1017.34375 steps (50.87 seconds)**, sample standard deviation **1056.40 steps**; median633 (31.65 seconds). The middle half is370.25–1391.25 steps (18.51–69.56 seconds). At least500/1000/2000 steps:20/32,11/32,3/32. All episodes fail before10k, so this horizon does not censor the measured durations.

Physical juggles average9.4375, sample standard deviation **9.870**, median6. At least5/10/20 juggles:20/32,12/32,4/32. Longest episode seed300017:4722 steps (236.1 seconds),40 juggles; next seed300006:3755 steps,32 juggles. Removing the longest episode leaves mean897.84 steps. Duration–juggle correlation is0.978, descriptive rather than causal.

End labels:19 passed-paddle only,9 bottom-contact only,4 both. These describe outcomes and do not establish the underlying perception/control cause. Episode-bootstrap95% interval for the mean is689.96–1409.70 steps (20,000 resamples, NumPy default_rng seed0); this captures finite evaluation-sample variability, not training-seed uncertainty.

[All32 episode rows](../../../docs/juggling-baseline-2m-duration.csv); [computed summary](../../../docs/juggling-baseline-2m-duration.json); [existing paired checkpoint data](../../../docs/juggling-2m-benchmark.csv). Raw source: `runs/td3/noisy_iteration2_20261029_long/milestone_2m_benchmark/results.json`; trajectories are saved alongside it. No new figure or GIF generated; existing [1.9M selected-episode GIF](../../../docs/assets/juggling/noisy-first-cap-start.gif) belongs to another checkpoint and is not evidence for this2M distribution.

## Conclusion
The baseline demonstrates meaningful repeated juggling, with substantial variation across episodes. Failure to reach an arbitrary horizon should not obscure the50.9-second average and repeated returns. Newer main-thread results remain separate; this retrospectively characterizes a fixed historical checkpoint.

## Next
Compare final approaches in typical and long episodes to distinguish positioning, timing and tracking failures. This is proposed analysis, not executed work.
