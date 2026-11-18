# Ceiling of model-free RL juggling on the sysid-v2 hist4 target sim: memory removes the gap

- **Date**: 2026-10-29 00:09 UTC start (2026-10-28 19:09 CDT)
- **Status**: in-flight (first round done; 4M-step, 100-step-window and extra-seed runs still training)
- **Run dirs**: `runs/ceiling/` (sweeps `phase1_screen`, `phase1_confirm`, `phase1_confirm_noper`, `phase2_gpu0`, `combo1`, `teacher`, `hist50_confirm`, `hist50_privcritic_confirm`); evaluations `runs/ceiling/evals/{selection,selection_long,final}/`
- **Configs**: `configs/ceiling/*.yaml` (sweep specs) → `configs/ceiling/generated/` (args + sim configs); base `configs/ceiling/base_td3_hist4.yaml`
- **Code**: `scripts/ceiling/` (`eval_target.py`, `leaderboard.py`, `sweep.py`); `rma_mode: history_privileged` in `scripts/rma/train_base_policy.py`; `teacher_true_state` in `scripts/td3/td3_training.py`

## Question

Following the meeting after the SimDist results: what is the best juggling performance a model-free RL policy can reach on the target sim, and is sustained (indefinite) juggling possible there? The simulator itself must not change.

## Setup

- **Target sim**: `configs/new_juggle/sysid_v2_hist4.yaml`, unchanged since 94acd81 (hist_len 4, puck noise σ 0.01, 5 % random occlusions, 25 ms observation delay with interpolation, no physics DR). Training configs may differ only in the episode cap, training-only reward shaping (`enable_survival_bonus`, `survival_bonus_per_step`, `puck_loss_penalty`) and the RMA trainer's inert `random_variables` bookkeeping (`domain_random` stays off). `scripts/ceiling/sweep.py` refuses any other difference, and every evaluation runs on the unmodified file.
- **Baseline**: `runs/td3/sysid_v2_real_workspace/hist4_seed{0..4}` (TD3, 5-frame obs, γ 0.975, episode cap 250, 25 critic + 6 actor updates per episode, 2M steps).
- **Evaluation protocols** (`scripts/ceiling/eval_target.py`):
  - selection: 32 episodes, 2,000-step cap, seeds 500000+.
  - selection_long: the same seeds at a 10,000-step cap.
  - final: 100 fresh episodes, 10,000-step cap, seeds 900000+, never used for selection.

  A drop is `puck_passed_paddle` or `puck_hit_bottom`. Juggles use the existing direction-flip counter.
- **Seeds**: each variant was screened with 1 training seed. Promising recipes got 3 seeds. Values are mean ± std err over training seeds; per-policy values are over episodes.
- Runs that changed the update cadence use fixed-step updates (8 critic + 2 actor updates every 32 env steps), so the update rate does not depend on episode length. Under the baseline's per-episode updates, a longer cap would starve learning.
- **Hardware**: GPUs 0–1 only (Quadro RTX 6000), 4 jobs per GPU, about 470 env steps/s per job.

## Results

### Without memory (selection protocol; mean episode steps, 2,000-step cap)

| Recipe | Seeds | Mean steps |
|---|---:|---:|
| Baseline (hist4, γ 0.975, cap 250) | 5 | 252.6 ± 13.7 |
| Fixed-step updates, cap 250, γ 0.975 | 1 | 231.2 |
| Cap 2000, γ 0.975 | 1 | 174.7 |
| Cap 2000, γ 0.995 | 1 | 242.7 |
| Cap 2000, γ 0.99 | 3 | 290.9 ± 36.3 |
| Cap 2000, γ 0.99, PER off | 3 | **358.3 ± 26.7** |
| Cap 2000, γ 0.99, survival bonus 0.25 + drop penalty 5 | 1 | 362.3 |
| Cap 2000, γ 0.99, 4× updates | 1 | 328.8 |
| Cap 2000, γ 0.99, 256-wide networks | 1 | 238.5 |
| Cap 2000, γ 0.995, low exploration (primitives 0.01, noise 0.05) | 1 | 350.3 |
| Cap 2000, γ 0.995, lr ÷3 | 1 | 188.3 |
| Cap 2000, γ 0.99, PER off, low exploration | 2 | 318.9 |

No policy without memory reached 2,000 steps in any episode. Survival to 1,000 steps was at most 9 %.

### With memory or privileged information (final protocol, 100 fresh episodes, 10,000-step cap)

| Policy | Survived 10,000 steps | Mean steps | Drops | Juggles / episode |
|---|---:|---:|---:|---:|
| Teacher: actor + critic see the true puck/paddle state (γ 0.99, PER off), seed 0 | **100 %** | 10000.0 ± 0.0 | 0 | 144.7 |
| 50-step memory + true-state critic, seed 0, checkpoint 1.7M | 97 % | 9782.6 | 3 | 132.8 |
| 50-step memory + true-state critic, seed 2, checkpoint 1.6M | 95 % | 9782.3 | 5 | 125.6 |
| 50-step memory, seed 0, final 2.0M | 88 % | 9370.5 ± 199.8 | 12 | 120.1 |
| 50-step memory, seed 1, checkpoint 1.9M | 90 % | 9343.9 | 10 | 124.2 |
| 50-step memory, seed 2, checkpoint 1.9M | 63 % | 8212.9 | 37 | 106.7 |
| **50-step memory, recipe (3 seeds, selected checkpoints)** | **80.3 ± 8.7 %** | **8975.8 ± 381.5** | | |
| 50-step memory, recipe (3 seeds, final 2M models) | 64.3 ± 21.2 % | 7866.8 ± 1359.7 | | |

- **Memory recipe**: `rma_mode: history` in `scripts/rma/train_base_policy.py`, the long-history TD3 control from the RMA baseline. A 50-step (x, a) window feeds a conv encoder to an 8-dim latent, which conditions a 64×2 actor. It uses γ 0.995, a 2,000-step cap and the same fixed-step cadence, trained for 2M steps.
- **True-state critic**: the new `rma_mode: history_privileged`. The critic additionally sees the true simulator state (puck and paddle position + velocity, noise-free and undelayed). The actor skips that block and saves as a plain `HistoryActor`, so it remains deployable.
- **Checkpoint selection**: each seed's checkpoints from 1.0M to 2.0M were scored on selection_long. The best was then final-tested on fresh seeds. The gain survived the fresh seeds (seed 2: 25 % → 56 % on selection, 22 % → 63 % on final), so it is not selection luck.
- **Training was still improving at 2M steps.** On selection_long, plain memory first reaches the 10,000-step cap at 1.6–1.9M steps. The true-state critic gets there at 1.3–1.5M steps and is steadier across later checkpoints.
- **Juggling quality**: every long episode juggles at a steady 11–13 juggles per 1,000 steps (about every 4 s), and the puck travels the full table. No episode stayed alive by pinning the puck. Frames at steps 1,600–2,000 of `runs/ceiling/evals/selection/phase2_gpu0__hist50/*/target.gif` show repeated hits into the far half.
- **Failure**: memory + true-state critic seed 1 never learned (41.7 steps). Its actor saturated at the constant corner action (+1, −1) at every step, and all 19 checkpoints are identical. This is the known TD3 tanh-saturation failure. Seed 3 is training to estimate how often it happens.

## Conclusion

- **This sim permits effectively indefinite juggling.** A policy that sees the true puck/paddle state never dropped the puck in 100 × 10,000 steps. The 95 % upper bound on the per-episode drop probability is about 3 %.
- **The baseline's limit is state estimation from noisy, occluded, delayed 5-frame observations, not dynamics or hyperparameters.** Tuning plain TD3 across cap, discount, PER, exploration, network size, update rate and reward shaping topped out at about 360 steps. Adding a 50-step observation/action memory, with no privileged input at deployment, took the same TD3 to 80–97 % survival over 10,000 steps.
- **Remaining gap**, from 63–97 % to the teacher's 100 %:
  - seed-to-seed variance;
  - policies still improving at 2M steps;
  - checkpoint-to-checkpoint swings;
  - a small residual drop rate: about one drop per 80k–250k steps for the better policies.

  The true-state critic looks steadier when it trains, but one of three seeds collapsed. With 2 working seeds, it is not yet established as better than plain memory.
- **Not tested or not established:**
  - real-robot transfer;
  - smoothness: action-change RMS is about 0.6, high and likely jerky on hardware;
  - survival beyond 10,000 steps;
  - more than 3 seeds per recipe.

## Next

- Running: 4M-step training of both memory recipes; a 100-step window; memory + true-state critic with γ 0.99 and PER off; seed 3 of memory + true-state critic.
- If the collapse recurs, add an actor saturation guard (pre-tanh penalty or LayerNorm).
- Final recipe: 5 seeds of the best memory variant, checkpoint selection on selection_long, final test on fresh seeds.
- Separate question for the team: whether, and how, the memory policy's jerkiness matters for hardware.
