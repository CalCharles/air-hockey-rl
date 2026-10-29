# Juggling ceiling on the target sim, final recipe: memory + true-state critic + pre-tanh penalty survives 10,000 steps in 99 % of episodes

- **Date**: 2026-10-29 07:49 UTC writeup (training started 2026-10-29 05:10 UTC)
- **Status**: done
- **Follows**: [`2026-10-29_00-09_juggle-ceiling-target-sim.md`](2026-10-29_00-09_juggle-ceiling-target-sim.md), whose in-flight runs are reported here.
- **Run dirs**:
  - `runs/ceiling/final_privcritic_pretanh/` (final recipe, 5 seeds)
  - `runs/ceiling/{hist50_privcritic_4m,hist50_4m,hist100_privcritic,hist50_privcritic_g099_noper,hist50_privcritic_confirm2}/`
  - evaluations in `runs/ceiling/evals/{selection_long,final}/`
- **Configs**: `configs/ceiling/final_privcritic_pretanh.yaml`, `hist50_privcritic_4m.yaml`, `hist50_4m.yaml`, `hist100_privcritic.yaml`, `hist50_privcritic_g099_noper.yaml`, `hist50_privcritic_confirm2.yaml`
- **Code**:
  - new `actor_pre_tanh_penalty` arg (default 0): `scripts/td3/td3_training.py` → `scripts/td3/helper/td3_graphed_update.py`, plus `get_pre_tanh()` on the history actors in `scripts/rma/networks.py`
  - selection + final helper `runs/ceiling/select_and_final.sh`

## Question

How reliably can the best recipe from the first round reach indefinite juggling on the target sim (`configs/new_juggle/sysid_v2_hist4.yaml`, unchanged)? That recipe was 50-step memory + a critic that sees the true state, and 1 of its 3 seeds had collapsed. Do longer training, a longer window or other settings help?

## Setup

Same target sim, protocols and selection rule as the [first note](2026-10-29_00-09_juggle-ceiling-target-sim.md):
- selection_long: 32 episodes, 10,000-step cap, seeds 500000+.
- Each seed's checkpoints from 1.0M steps to the end are scored on selection_long. The best (fraction reaching the cap, then mean steps) goes to the final test: 100 fresh episodes, 10,000-step cap, seeds 900000+.

**Final recipe** (`rma_mode: history_privileged`, 2,000-step training cap, γ 0.995, PER on, 8 critic + 2 actor updates every 32 steps):
- **Actor**: 50-step (x, a) window → conv encoder → 64×2 MLP. It sees only the standard noisy, delayed observations, so it is deployable as a plain `HistoryActor`.
- **Critic**: additionally sees the true puck/paddle position and velocity.
- **New**: a pre-tanh actor penalty `actor_pre_tanh_penalty: 0.001`, which adds λ·mean(u²) on the action head's pre-tanh output u to the actor loss.
- 3M steps, seeds 0–4.

## Results

### Final recipe (final protocol, 100 fresh episodes per seed)

| Seed | Selected checkpoint | Survived 10,000 steps | Mean steps | Drops | Juggles / episode | Action-change RMS |
|---|---|---:|---:|---:|---:|---:|
| 0 | 2.8M | 98 % | 9846.6 | 2 | 133.9 | 0.168 |
| 1 | 2.9M | **100 %** | 10000.0 | 0 | 126.0 | 0.130 |
| 2 | 2.8M | **100 %** | 10000.0 | 0 | 127.1 | 0.151 |
| 3 | 2.8M | 98 % | 9902.9 | 2 | 126.2 | 0.135 |
| 4 | 2.5M | 99 % | 9961.4 | 1 | 129.4 | 0.116 |
| **Recipe (5 seeds)** | | **99.0 ± 0.4 %** | **9942.2 ± 29.8** | 5 / 500 | 128.5 ± 1.5 | |

Pooled over the 5 seeds: 5 drops in about 4.97M steps, about one drop per 1M control steps (≈ 14 h of juggling at 20 Hz).

- **All 5 seeds trained successfully.** Seed 1 collapsed under the previous recipe without the penalty: its actor was saturated at the constant (+1, −1), with 100 % of actions at the limits from 300k steps on. With the penalty, seed 1 had 0 % saturated actions at 200k/300k steps and became one of the two drop-free seeds. TD3 runs are not bit-reproducible, so this is strong evidence for the fix but not a controlled same-trajectory comparison.
- **The penalty also made the policies much smoother.** Action-change RMS is 0.12–0.17 under the penalty, against 0.48–0.81 for every unpenalized policy, including the true-state teacher (0.48). Peak paddle speed still reaches the sim's 2 m/s `max_paddle_vel` in every recipe.
- **Learning speed was unchanged**: training-episode length at 300k steps is 41–76 vs 57–69 before.

### Other runs (one seed each, final protocol on the selected checkpoint)

| Run | Selected | Survived 10,000 steps | Mean steps |
|---|---|---:|---:|
| Memory + true-state critic, 4M steps, no penalty | 3.9M | 99 % | 9953.7 |
| Memory only, 4M steps | 2.7M | 86 % | 9297.4 |
| Memory + true-state critic, 100-step window, 2M steps | 2.0M (final) | 96 % | 9744.0 |
| Memory + true-state critic, seed 3 (no penalty), 2M steps | 1.6M | 100 % | 10000.0 |
| Memory + true-state critic, γ 0.99, PER off | not final-tested | 81 % at the 2,000-step cap (selection) | 1837.3 |

- **Longer training**:
  - With the true-state critic, nearly all checkpoints after 2.3M reach the 10,000-step cap on selection. Isolated dips remain (2.1M: 25 %, 2.7M: 59 %, 3.4M: 62 %), so checkpoint selection stays necessary.
  - Memory-only plateaus at 70–95 % on selection from 2.3M to 3.9M. Its selected checkpoint scored 100 % on selection but 86 % on the final test, an example of best-of-31 selection optimism.
- A 100-step window performs like the 50-step window. γ 0.99 / PER off (the best no-memory setting) is worse with memory.
- With seed 3 included, the unpenalized true-state-critic recipe had 3 of 4 seeds succeed (97 %, 95 %, 100 % final) and 1 collapsed.

## Conclusion

- **Ceiling on the target sim:** a deployable, observation-only TD3 policy juggles effectively indefinitely. It survives 10,000 steps (8.3 sim-minutes) in 99.0 ± 0.4 % of fresh episodes across 5 training seeds, with about one drop per million steps. The baseline policies survive 252.6 ± 13.7 steps. A privileged teacher seeing the true state survives 100 %, so remaining sensing loss is at most about 1 %.
- **Final recipe**:
  - 50-step memory: the decisive ingredient.
  - True-state critic during training: steadier and better with long training than memory alone.
  - Pre-tanh actor penalty: removes the seed collapses and gives about 4× smoother actions.
  - 3M training steps, with checkpoint selection on a 10,000-step test.
- **Confidence**: 5 seeds × 100 fresh episodes on the final protocol, with selection and test seeds disjoint. Survival beyond 10,000 steps and the drop rate per million steps rest on only 5 observed drops.
- **Not tested**:
  - real-robot transfer;
  - the effect of the true-state critic and the penalty separately in a 5-seed design (they were introduced in sequence);
  - whether the penalty alone, without the true-state critic, gives memory-only the same reliability.

## Next

- Decide with the team whether to take this policy family toward sim2real: the actor's inputs are the standard observations, and the smoother actions help.
- Optional ablation: memory-only + pre-tanh penalty, 5 seeds, 3M steps. This separates the penalty's contribution from the true-state critic's.
