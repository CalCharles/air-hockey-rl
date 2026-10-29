# Reduced exploration improves 1M validation but does not beat the scratch incumbent on the shared benchmark

- **Date**: 2026-10-29 18:03 UTC
- **Status**: done (1M milestone benchmark); production training remains running
- **Run dirs**: `runs/td3/noisy_iteration2_20261029_low_explore/milestone_1m_benchmark/`
- **Configs**: [low exploration](../../../configs/td3/sustained/noisy_low_explore.yaml), [baseline](../../../configs/td3/sustained/noisy_long.yaml)

## Question
Does reduced exploration improve the frozen scratch policy after one million environment steps, and does it replace the stronger E008 incumbent? The 1M checkpoint is a fixed milestone, not selected on the benchmark results.

## Setup
Existing E005 trials: canonical noisy simulator, unchanged dynamics and three-parameter DR, all actors/critics trained from scratch. Eight reused validation starts 10000–10007, cap 5k. Additional fixed 1M actor benchmark on 32 starts 300000–300031, cap 10k. Those starts are the shared E008 benchmark already used for other actors; they are not untouched acceptance starts. All current training continues unchanged toward the planned 6M budgets.

## Results

| 1M validation actor | Mean survival | Physical-trajectory juggle heuristic | Cap completions |
|---|---:|---:|---:|
| Baseline | 266.25 | 2.125 | 0/8 |
| Reduced exploration | 894.125 | 6.625 | 0/8 |

At 800k/900k, reduced-exploration versus baseline survival was 591.875/316.375 and 478.5/226.375. At 600k they were nearly tied (308.375/311.5). This suggests a useful recent trend after exploration annealing, not a multi-seed result. The larger network has reached only 700k and averages 165 steps there; compare at matched budgets before interpreting capacity.

| Shared 32-start benchmark actor | Mean survival | Median | Physical-trajectory juggles | Cap completions |
|---|---:|---:|---:|---:|
| New reduced exploration, 1M | 695.375 | 553 | 6.09375 | 0/32 |
| Existing selected scratch incumbent, 1.8M | 816.9375 | 612 | 7.75 | 0/32 |

New-minus-incumbent paired mean difference −121.5625 steps; episode-bootstrap 95% interval [−469.88, 210.23], 10,000 resamples, default_rng seed 42. New policy wins 16/32 pairs. This does not establish a reliable difference between these two actors. Training budgets also differ, so it is not a matched recipe comparison. Both are far from solving sustained juggling.

[Per-start benchmark CSV](../../../docs/juggling-exploration-1m-benchmark.csv), [new raw benchmark](../../../runs/td3/noisy_iteration2_20261029_low_explore/milestone_1m_benchmark/results.json), [incumbent raw benchmark](../../../runs/td3/noisy_scratch_20261028_seed0/selected_paired_eval/results.json), [checkpoint curves](../../../docs/juggling-noisy-progress.md).

![Reduced-exploration 1M benchmark rollout](../../../docs/assets/juggling/noisy-low-explore-1m.gif)

New simulation GIF generated 2026-10-29, showing the last 20 seconds of the first benchmark episode. Compacted to 160×357, 10 fps, fixed 64-color palette with playback duration preserved. It illustrates one episode, not aggregate reliability; original 20 fps source remains in the raw benchmark directory.

## Reproduce

```sh
.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_iteration2_20261029_low_explore/train/checkpoint_1000000 --scenario-set training --episodes 32 --horizon 10000 --seed-base 300000 --gif-steps 400 --out-dir <new-output>
```

## Conclusion
Retain the E008 1.8M checkpoint as incumbent. Reduced exploration is promising on recent matched validation checkpoints but has not yet produced an independently superior policy. Never promote from the eight-start checkpoint mean alone.

## Next
Continue the declared training comparison. Validation-selected checkpoint evaluations remain queued on disjoint seed-400000 benchmark starts after training. Final acceptance must use new untouched starts and longer horizons; neither current benchmark qualifies as a solved result.
