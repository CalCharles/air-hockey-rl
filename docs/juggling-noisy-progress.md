# Noisy juggling progress

Generated 2026-10-29T22:06:07.466733+00:00 from saved evaluations; no new rollouts.

Validation curves use repeatedly evaluated starts, not independent acceptance tests. Each point is an episode mean for one training seed. Check the CSV for episode counts, seeds and caps. Observed juggle counts use noisy camera positions and a direction-reversal filter. Contact timesteps are not distinct juggles. Cap fractions use the physical-trajectory heuristic when available, otherwise the legacy observed-count heuristic.

![Checkpoint validation](juggling-noisy-progress.png)

[Raw validation table](juggling-noisy-progress.csv)

| Run | Runner status | Latest validation step | Mean survival | Mean juggles | Genuine cap fraction |
|---|---|---:|---:|---:|---:|
| noisy_scratch_20261028_seed0 | complete | 1,900,000 | 663.4 | 5.75 | 0.0% |
| noisy_iteration2_20261029_long | complete | 5,900,000 | 834.4 | 7.88 | 0.0% |
| noisy_iteration2_20261029_low_explore | interrupted | 2,000,000 | 392.0 | 3.50 | 0.0% |
| noisy_iteration2_20261029_large_low_explore | interrupted | 1,000,000 | 152.6 | 1.00 | 0.0% |
| noisy_history_20261029_seed0 | training | 4,200,000 | 4378.1 | 57.38 | 37.5% |
| noisy_history_20261029_seed1 | training | 900,000 | 151.0 | 1.38 | 0.0% |
| noisy_survival_20261029_seed0 | interrupted | 4,000,000 | 461.0 | 2.75 | 0.0% |
| noisy_history_long_discount_20261029_seed0 | training | 800,000 | 170.4 | 1.12 | 0.0% |

Separate-start evaluations (selection and benchmark phases labeled):

- [noisy_scratch_20261028_seed0, final actor benchmark](../runs/td3/noisy_scratch_20261028_seed0/final_eval/results.json): 416.1 mean steps, 3.78 observed juggles, 0.0% cap completion; 32 starts from seed 200000, cap 10000.
- [noisy_scratch_20261028_seed0, paired selected checkpoint](../runs/td3/noisy_scratch_20261028_seed0/selected_paired_eval/results.json): 816.9 mean steps, 7.41 observed juggles, 7.75 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_scratch_20261028_seed0, paired final actor](../runs/td3/noisy_scratch_20261028_seed0/final_paired_eval/results.json): 415.3 mean steps, 3.28 observed juggles, 3.84 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_long, final actor benchmark](../runs/td3/noisy_iteration2_20261029_long/final_eval/results.json): 890.8 mean steps, 8.56 observed juggles, 9.53 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 200000, cap 10000.
- [noisy_iteration2_20261029_long, validation-selected checkpoint benchmark](../runs/td3/noisy_iteration2_20261029_long/selected_eval/results.json): 1091.5 mean steps, 9.34 observed juggles, 9.88 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 400000, cap 10000.
- [noisy_iteration2_20261029_long, first cap benchmark](../runs/td3/noisy_iteration2_20261029_long/first_cap_benchmark/results.json): 827.9 mean steps, 7.16 observed juggles, 7.53 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_long, milestone 2m benchmark](../runs/td3/noisy_iteration2_20261029_long/milestone_2m_benchmark/results.json): 1017.3 mean steps, 8.03 observed juggles, 9.44 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_long, through3m selected benchmark](../runs/td3/noisy_iteration2_20261029_long/through3m_selected_benchmark/results.json): 920.3 mean steps, 7.81 observed juggles, 8.78 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_low_explore, validation-selected checkpoint benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/selected_eval/results.json): 824.4 mean steps, 7.12 observed juggles, 7.91 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 400000, cap 10000.
- [noisy_iteration2_20261029_low_explore, matched history 1m6 benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/matched_history_1m6_benchmark/results.json): 580.2 mean steps, 4.62 observed juggles, 4.88 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_low_explore, milestone 1m benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/milestone_1m_benchmark/results.json): 695.4 mean steps, 5.44 observed juggles, 6.09 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_low_explore, milestone 2m benchmark](../runs/td3/noisy_iteration2_20261029_low_explore/milestone_2m_benchmark/results.json): 664.4 mean steps, 5.25 observed juggles, 6.06 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_iteration2_20261029_large_low_explore, validation-selected checkpoint benchmark](../runs/td3/noisy_iteration2_20261029_large_low_explore/selected_eval/results.json): 225.8 mean steps, 1.62 observed juggles, 1.94 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 400000, cap 10000.
- [noisy_history_20261029_seed0, independent 1m6 20k eval](../runs/td3/noisy_history_20261029_seed0/independent_1m6_20k_eval/results.json): 3606.7 mean steps, 46.50 observed juggles, 51.66 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 500000, cap 20000.
- [noisy_history_20261029_seed0, independent 3m6 20k eval](../runs/td3/noisy_history_20261029_seed0/independent_3m6_20k_eval/results.json): 13099.8 mean steps, 176.25 observed juggles, 197.44 physical-trajectory juggles, 43.8% cap completion; 32 starts from seed 700000, cap 20000.
- [noisy_history_20261029_seed0, independent 3m 20k eval](../runs/td3/noisy_history_20261029_seed0/independent_3m_20k_eval/results.json): 9116.6 mean steps, 120.44 observed juggles, 133.88 physical-trajectory juggles, 15.6% cap completion; 32 starts from seed 600000, cap 20000.
- [noisy_history_20261029_seed0, milestone 1m6 benchmark](../runs/td3/noisy_history_20261029_seed0/milestone_1m6_benchmark/results.json): 3519.4 mean steps, 45.16 observed juggles, 51.31 physical-trajectory juggles, 9.4% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_history_20261029_seed0, milestone 2m1 benchmark](../runs/td3/noisy_history_20261029_seed0/milestone_2m1_benchmark/results.json): 4885.3 mean steps, 66.69 observed juggles, 74.34 physical-trajectory juggles, 28.1% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_history_20261029_seed0, milestone 3m benchmark](../runs/td3/noisy_history_20261029_seed0/milestone_3m_benchmark/results.json): 6662.7 mean steps, 88.41 observed juggles, 99.19 physical-trajectory juggles, 43.8% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_history_20261029_seed0, milestone 4m benchmark](../runs/td3/noisy_history_20261029_seed0/milestone_4m_benchmark/results.json): 7335.6 mean steps, 95.12 observed juggles, 107.22 physical-trajectory juggles, 50.0% cap completion; 32 starts from seed 300000, cap 10000.
- [noisy_history_20261029_seed0, paired 2m1 20k eval](../runs/td3/noisy_history_20261029_seed0/paired_2m1_20k_eval/results.json): 7066.3 mean steps, 97.81 observed juggles, 108.72 physical-trajectory juggles, 12.5% cap completion; 32 starts from seed 500000, cap 20000.
- [noisy_history_20261029_seed0, target actor 3m6 paired 20k eval](../runs/td3/noisy_history_20261029_seed0/target_actor_3m6_paired_20k_eval/results.json): 15427.6 mean steps, 200.62 observed juggles, 225.06 physical-trajectory juggles, 62.5% cap completion; 32 starts from seed 700000, cap 20000.
- [noisy_survival_20261029_seed0, validation-selected checkpoint benchmark](../runs/td3/noisy_survival_20261029_seed0/selected_eval/results.json): 811.2 mean steps, 6.78 observed juggles, 7.59 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 400000, cap 10000.
- [noisy_survival_20261029_seed0, milestone 3m benchmark](../runs/td3/noisy_survival_20261029_seed0/milestone_3m_benchmark/results.json): 717.7 mean steps, 6.25 observed juggles, 5.81 physical-trajectory juggles, 0.0% cap completion; 32 starts from seed 300000, cap 10000.

The final-start benchmark is reused across recipes. A selected policy requires untouched seeds and longer horizons before acceptance. Bounded survival does not prove infinite survival.
