# The noisy scratch baseline first reaches 5k on one validation start, then fails at 6,859 steps

- **Date**: 2026-10-29 18:49 UTC
- **Status**: done (milestone evaluation); baseline training remains running toward 6M
- **Run dirs**: `runs/td3/noisy_iteration2_20261029_long/{first_cap_benchmark,first_cap_stretch,milestone_2m_benchmark}`, `runs/td3/noisy_iteration2_20261029_low_explore/milestone_2m_benchmark`
- **Configs**: [baseline](../../../configs/td3/sustained/noisy_long.yaml), [reduced exploration](../../../configs/td3/sustained/noisy_low_explore.yaml)

## Question
Has the canonical noisy scratch campaign begun to sustain repeated juggling, and how do the two small-network recipes compare at the fixed 2M milestone?

## First capped validation rollout
At the baseline 1.9M checkpoint, one of eight validation starts (10005) completed 5,000 steps, with 59 physical-trajectory juggles (52 from noisy observations). Seven starts failed; mean survival 1,236.875. Both observed and physical genuine-cap fractions are 1/8. Earlier checkpoints in these new noisy scratch runs had no capped completions. This is bounded survival on a reused validation start, not an independent reliability result.

Extended that specifically selected successful seed to a 20k horizon: failure at step **6,859** (`puck_passed_paddle`), 75 physical-trajectory juggles, 65 observed juggles. At 20 Hz this is 342.95 seconds, about 5 min 43 s. Every one of the eight saved NPZ fields exactly matches the original trajectory prefix through all 5,000 steps: observations, next observations, actions, rewards, terminated, truncated, actual paddle xy and actual puck xy. The cap success was not created by changing sensing or dynamics when extending the budget.

![Beginning of the selected validation-seed demonstration](../../../docs/assets/juggling/noisy-first-cap-start.gif)

![Last 20 seconds, ending in puck loss](../../../docs/assets/juggling/noisy-first-cap-end.gif)

New simulation clips generated 2026-10-29. The first shows 10 seconds from the start; the second shows the final 20 seconds of the same 6,859-step episode. Between them is an intentional time gap. Compacted to 160×357, 10 fps and 64 colors with each clip's playback duration preserved. This seed was chosen because it had already survived validation; these GIFs do not estimate aggregate reliability. Original 20 fps clips and full NPZ remain local under `first_cap_stretch/`.

## Shared benchmark and fixed 2M comparison
All actors evaluated on the same 32 shared benchmark starts (300000–300031), cap 10k. This benchmark has been reused across recipes/checkpoints; it is not untouched acceptance data.

| Actor | Mean steps | Median | Physical-trajectory juggles | Genuine 10k completions |
|---|---:|---:|---:|---:|
| Previous selected scratch incumbent, 1.8M | 816.9375 | 612 | 7.75 | 0/32 |
| New baseline, selected 1.9M | 827.9375 | 743.5 | 7.53125 | 0/32 |
| New baseline, fixed 2M | 1017.34375 | 633 | 9.4375 | 0/32 |
| Reduced exploration, fixed 2M | 664.4375 | 402.5 | 6.0625 | 0/32 |

Baseline 2M minus reduced exploration 2M: paired mean +352.90625 steps, wins 24/32, episode-bootstrap 95% interval [−128.93, 832.44]. Baseline 2M minus prior incumbent: +200.40625, wins 18/32, interval [−270.88, 701.06]. Selected 1.9M minus prior incumbent: +11.0, interval [−286.52, 295.50]. Each interval uses 10,000 paired resamples, default_rng seed 42. None establishes a reliable mean advantage; all use one training seed, and benchmark reuse adds selection bias. Baseline 2M has the best measured mean here, not a proven robust superiority.

Ongoing later validation at 2.2M reaches mean 1681.125 with 1/8 capped starts: a different seed, 10004, with 39 physical-trajectory juggles. These are still the reused eight validation starts, not new independent evidence.

At the fixed 2M eight-start validation, baseline/reduced exploration means are 1077.25/392.0, both 0/8 caps. Reduced exploration's best saved validation mean remains its 1M checkpoint (894.125).

[Per-start raw comparison CSV](../../../docs/juggling-2m-benchmark.csv), [updated curves and cap fraction](../../../docs/juggling-noisy-progress.md). Local raw JSON: [1.9M benchmark](../../../runs/td3/noisy_iteration2_20261029_long/first_cap_benchmark/results.json), [stretch](../../../runs/td3/noisy_iteration2_20261029_long/first_cap_stretch/results.json), [baseline 2M](../../../runs/td3/noisy_iteration2_20261029_long/milestone_2m_benchmark/results.json), [low exploration 2M](../../../runs/td3/noisy_iteration2_20261029_low_explore/milestone_2m_benchmark/results.json).

## Reproduce

```sh
.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_iteration2_20261029_long/train/checkpoint_1900000 --scenario-set training --episodes 1 --horizon 20000 --seed-base 10005 --gif-steps 400 --out-dir <new-stretch-output>
.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_iteration2_20261029_long/train/checkpoint_2000000 --scenario-set training --episodes 32 --horizon 10000 --seed-base 300000 --gif-steps 0 --out-dir <new-baseline-output>
.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_iteration2_20261029_low_explore/train/checkpoint_2000000 --scenario-set training --episodes 32 --horizon 10000 --seed-base 300000 --gif-steps 0 --out-dir <new-low-output>
```

## Conclusion and next
A new noisy scratch policy can repeatedly juggle for several minutes on one favorable start, but fails every 10k shared-benchmark episode. Keep the strong saved candidates; continue baseline toward 6M and evaluate future selected policies on new untouched starts if sustained-cap performance emerges. Do not call this solved. The reduced-exploration pilot is subsequently interrupted and its GPU reassigned; that separate status update is in the companion launch record.
