# New sim: the best recipe trained from scratch matches fine-tuning once it is trained to convergence with a 7000-step training cap

- **Date**: 2026-11-03 08:19 UTC start (02:19 CST)
- **Status**: done
- **Follows**: [2026-10-31_13-50](2026-10-31_13-50_new-sim-best-source-5-seed-check.md) (fine-tuned best recipe, 87.4 ± 4.5 % reach 10k over 5 seeds), [2026-10-31_02-00](2026-10-31_02-00_new-sim-retraining-and-finetuning.md) (earlier 3M-step from-scratch screen)
- **Run dirs**: `runs/ceiling/scratch_newsim/` (runs, `summary.md`, `orchestrator.log`, `orchestrator_state.json`, `tools/`), SWA policies `runs/analysis/swa/scratch_newsim__*`, evals `runs/analysis/new_sim_eval/scratch_newsim__*`, GIF `runs/ceiling/scratch_newsim/gifs/cap7000_swa_seeds_side_by_side.gif`
- **Configs**: `configs/ceiling/scratch_newsim.yaml` → `configs/ceiling/generated/scratch_newsim/`

## Question

Can the best recipe be trained directly on the new sim from scratch, without warm-starting from an old-sim checkpoint? The hypothesis was that the earlier from-scratch runs failed because they were not trained to convergence: they ran 3M steps, and only 2 of 3 runs juggled, the best at 55 % reach 10k. The plan was to raise the training episode cap by +5k until performance matches.

## Setup

- **Recipe**: `realplant_screen2/term03_nopretanh` with no warm start. That is TD3 via `scripts.rma.train_base_policy`, `rma_mode: history_privileged` (50-step history actor, true-state critic), `term03` shaping with edge termination, `actor_pre_tanh_penalty: 0`, `action_latency_steps: 2`, on the unchanged new sim `sysid_v2_hist4_realplant.yaml`.
  - The generated args are identical to the earlier from-scratch run except for `seed`, `run_name`, the paths and `total_timesteps`.
  - The cap-2000 and cap-7000 sim configs differ only in `max_timesteps`.
  - Compared with the fine-tuned best policy, the only differences are the warm start and its fine-tune-only exploration settings.
- **Training caps**: 2000 (as before) and 7000 (+5k). Both ran in parallel, seeds 0–4 each, 5 jobs per GPU on GPUs 0 and 1. The +5k cap was started early so it could finish in one night.
- **Training length**: `total_timesteps: 15M` was an upper bound only. No schedule depends on it, so a run stopped at M is identical to a run configured for M.
- **Convergence**: checked at 5M, then every +2M. A run counts as converged when `losses/q_loss`, `losses/actor_loss` and `charts/avg_episodic_return` each change by less than 10 % between the two halves of the last 1M steps, and the run has taken off (return > 50).
  - The take-off condition is needed because a run stuck at the return floor (~17) also has flat losses.
  - A run still stuck at 9M was stopped, and one extra seed of the same cap was started.
- **Post-processing**: actor SWA over the checkpoints in the last 40 % of training, [0.6 M, M]. The best recipe averaged 1.2–2.0M of a 2M run.
- **Eval**: `scripts.analysis.new_sim_eval`, 100 episodes, 10k cap, seeds 1000000+. This is the same final test as the fine-tuned policies and was never used for selection. Both the SWA policy and the raw checkpoint at M were evaluated.
- **Match criterion**: the first-5-seed mean reach 10k with SWA ≥ 83 %, i.e. within one std err of 87.4 ± 4.5 %.

## Results

Per policy, ± is the std err over 100 episodes; seed means are ± std err over seeds. All policies have 0 % edge hits and ≤ 1 % of time near the edge.

**Cap 7000**

| Seed | Stopped (converged) | Reach 5k (SWA) | Reach 10k (SWA) | Return (SWA) | Reach 10k (raw ckpt) |
|---|---|---|---|---|---|
| 0 | 5M | 96 ± 2 % | 96 ± 2 % | 7079 ± 145 | 68 ± 5 % |
| 1 | 7M | 92 ± 3 % | 88 ± 3 % | 6731 ± 179 | 83 ± 4 % |
| 2 | 13M (stuck at the floor until ~6M) | 95 ± 2 % | 94 ± 2 % | 7076 ± 154 | 89 ± 3 % |
| 3 | 9M | 96 ± 2 % | 96 ± 2 % | 6805 ± 140 | 87 ± 3 % |
| 4 | 7M | 96 ± 2 % | 96 ± 2 % | 7046 ± 145 | 84 ± 4 % |
| **Mean** | | **95.0 ± 0.8 %** | **94.0 ± 1.5 %** | | 82.2 ± 3.7 % |

**Cap 2000**

| Seed | Stopped | Reach 5k (SWA) | Reach 10k (SWA) | Return (SWA) | Reach 10k (raw ckpt) |
|---|---|---|---|---|---|
| 0 | 14.9M (converged only at the 15M bound) | 94 ± 2 % | 94 ± 2 % | 6872 ± 171 | 45 ± 5 % |
| 1 | 9M (never took off) | 0 % | 0 % | 16 ± 1 | 0 % |
| 2 | 5M | 61 ± 5 % | 42 ± 5 % | 4646 ± 274 | 46 ± 5 % |
| 3 | 7M | 96 ± 2 % | 95 ± 2 % | 7090 ± 147 | 92 ± 3 % |
| 4 | 5M | 92 ± 3 % | 86 ± 3 % | 6414 ± 176 | 64 ± 5 % |
| 11 (extra, for s1) | 5M | 95 ± 2 % | 94 ± 2 % | 7033 ± 162 | 67 ± 5 % |
| **Mean, seeds 0–4** | | 68.6 ± 18.3 % | **63.4 ± 18.6 %** | | 49.4 ± 15.0 % |
| Mean with s11 in place of s1 | | 87.6 ± 6.7 % | 82.2 ± 10.2 % | | 62.8 ± 8.6 % |

The per-check loss and return changes for every run are in `runs/ceiling/scratch_newsim/summary.md`.

## Conclusion

- **From scratch now matches fine-tuning.** At cap 7000, the 5-seed mean of 94.0 ± 1.5 % reach 10k is above the fine-tuned recipe (87.4 ± 4.5 %) and equal to the best fine-tuned single policy (94 ± 2 %), with a tighter seed spread (88–96 % vs 70–94 %). No old-sim checkpoint is needed.
- **Training length was the main missing piece.** Runs needed 5–13M steps to converge (cap 2000 seed 0 took until 15M). The earlier screen stopped at 3M. Cap 7000 seed 2 sat at the return floor until ~6M and still ended at 94 %.
- **The longer training cap also matters.** At cap 2000, one seed never took off by 9M and one converged to a weaker policy (42 %). The mean is 63.4 ± 18.6 %, or 82.2 ± 10.2 % with the extra seed in place of the stuck one. At cap 7000 all 5 seeds learned. This is one comparison with 5 seeds per cap, so the cap effect is less certain than the convergence effect.
- **SWA matters a lot for these runs.** At cap 7000, SWA raised the reach-10k mean from 82.2 ± 3.7 % to 94.0 ± 1.5 %, by 5–28 points per seed. At cap 2000 the gap reached 49 points (seed 0). The raw final checkpoint alone would not have matched the fine-tuned recipe.
- **Caveats.**
  - The convergence test (< 10 % change over the last 1M steps) is a threshold I chose.
  - SWA windows differ in length between runs because the stopping points differ.
  - Cap 12000 was never run, because cap 7000 met the target.

## Next

- Real-robot test of the cap-7000 SWA policies (the new sim reproduces the robot; see [2026-10-30_17-40](2026-10-30_17-40_ceiling-sim2real-plant-gap.md)).
- Whether a higher cap (12000) or more seeds push the mean toward 99 %, the original goal of the juggling search.
- Ablate SWA's window (last 40 % vs a fixed 1M) on these runs.
