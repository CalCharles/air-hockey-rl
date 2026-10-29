# Noisy juggling progress

Generated 2026-10-29T20:12:18.125723+00:00 from saved evaluations; no new rollouts.

Validation curves use repeatedly evaluated starts, not independent acceptance tests. Each point is an episode mean for one training seed. Check the CSV for episode counts, seeds and caps. Observed juggle counts use noisy camera positions and a direction-reversal filter. Contact timesteps are not distinct juggles. Cap fractions use the physical-trajectory heuristic when available, otherwise the legacy observed-count heuristic.

![Checkpoint validation](juggling-noisy-progress.png)

[Raw validation table](juggling-noisy-progress.csv)

| Run | Runner status | Latest validation step | Mean survival | Mean juggles | Genuine cap fraction |
|---|---|---:|---:|---:|---:|
| noisy_scratch_20261028_seed0 | complete | 1,900,000 | 663.4 | 5.75 | 0.0% |
| noisy_iteration2_20261029_long | training | 4,300,000 | 1450.6 | 15.38 | 0.0% |
| noisy_iteration2_20261029_low_explore | interrupted | 2,000,000 | 392.0 | 3.50 | 0.0% |
| noisy_iteration2_20261029_large_low_explore | interrupted | 1,000,000 | 152.6 | 1.00 | 0.0% |
| noisy_history_20261029_seed0 | training | 2,100,000 | 2992.8 | 42.38 | 50.0% |
| noisy_survival_20261029_seed0 | training | 2,200,000 | 176.1 | 1.38 | 0.0% |

Separate-start evaluations (selection and benchmark phases labeled):

- [noisy_scratch_20261028_seed0, final actor benchmark](../runs/td3/noisy_scratch_20261028_seed0/final_eval/results.json): 416.1 mean steps, 3.78 observed juggles, 0.0% cap completion; 32 starts from seed 200000, cap 10000.
- [noisy_scratch_20261028_seed0, paired selected checkpoint](../runs/td3/noisy_scratch_20261028_seed0/selected_paired_eval/results.json): 816.9 mean steps, 7.41 observed juggles, 7.75 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_scratch_20261028_seed0, paired final actor](../runs/td3/noisy_scratch_20261028_seed0/final_paired_eval/results.json): 415.3 mean steps, 3.28 observed juggles, 3.84 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_long, first cap benchmark](../runs/td3/noisy_iteration2_20261029_long/first_cap_benchmark/results.json): 827.9 mean steps, 7.16 observed juggles, 7.53 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_long, milestone 2m benchmark](../runs/td3/noisy_iteration2_20261029_long/milestone_2m_benchmark/results.json): 1017.3 mean steps, 8.03 observed juggles, 9.44 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_long, through3m selected benchmark](../runs/td3/noisy_iteration2_20261029_long/through3m_selected_benchmark/results.json): 920.3 mean steps, 7.81 observed juggles, 8.78 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_low_explore, validation-selected checkpoint benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/selected_eval/results.json): 824.4 mean steps, 7.12 observed juggles, 7.91 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 400000, cap 10000.
- [noisy_iteration2_20261029_low_explore, matched history 1m6 benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/matched_history_1m6_benchmark/results.json): 580.2 mean steps, 4.62 observed juggles, 4.88 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_low_explore, milestone 1m benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/milestone_1m_benchmark/results.json): 695.4 mean steps, 5.44 observed juggles, 6.09 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_low_explore, milestone 2m benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/milestone_2m_benchmark/results.json): 664.4 mean steps, 5.25 observed juggles, 6.06 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_large_low_explore, validation-selected checkpoint benchmark](../runs/td3/noisy_iteration2_20261029_large_low_explore/selected_eval/results.json): 225.8 mean steps, 1.62 observed juggles, 1.94 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 400000, cap 10000.
- [noisy_history_20261029_seed0, independent 1m6 20k eval](../runs/td3/noisy_history_20261029_seed0/independent_1m6_20k_eval/results.json): 3606.7 mean steps, 46.50 observed juggles, 51.66 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 500000, cap 20000.
- [noisy_history_20261029_seed0, milestone 1m6 benchmark](../runs/td3/noisy_history_20261029_seed0/milestone_1m6_benchmark/results.json): 3519.4 mean steps, 45.16 observed juggles, 51.31 physical-trajectory juggles, 9.4% cap completion; 32 starts from seed 300000, cap 10000.

The final-start benchmark is reused across recipes. A selected policy requires untouched seeds and longer horizons before acceptance. Bounded survival does not prove infinite survival.
