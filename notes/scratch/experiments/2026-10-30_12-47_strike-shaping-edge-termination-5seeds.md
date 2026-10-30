# Strike shaping with edge-contact termination, 5 seeds (Task 1 follow-up)

- **Date**: 2026-10-30 12:47 UTC start
- **Status**: done
- **Follows**: [`2026-10-30_04-22_natural-strike-shaping.md`](2026-10-30_04-22_natural-strike-shaping.md) (recipe `edgehit3_weak3_pout2`, 73.0 ± 4.9 % at 10k)
- **Run dirs**: `runs/ceiling/shape_screen3/term03_weak3_pout2_s0/` and `runs/ceiling/shape_seeds_term03/term03_weak3_pout2_s{1..4}/`
  - evals: `runs/ceiling/evals/{selection_long/sellong__shape_seeds_term03, final/final__term03_5seeds}/`
  - hit metrics: `runs/ceiling/hit_motion/term03_5seeds/`
- **Configs**: `configs/ceiling/shape_screen3.yaml` (seed 0), `configs/ceiling/shape_seeds_term03.yaml` (seeds 1–4)
- **Policies**: `latest_models/ceiling/strike_term03/seed{0..4}/`
- **GIFs**: `eval_gifs/strike_term03/seed{0..4}_ep900000_600steps.gif`

## Question

In round 3, the single seed of `term03_weak3_pout2` was the steadiest strike variant across checkpoints. Does it hold up over 5 seeds, and does it beat `edgehit3_weak3_pout2`?

`term03_weak3_pout2` swaps the edge-contact penalty for ending the (training) episode on any contact within 3 cm of the top edge. It keeps the weak-contact penalty −3 below 0.3 m/s, the puck-exit bonus +2, and the edge time penalty 0.5.

## Setup

Same as the parent note. Selection on `selection_long`, checkpoints 1.5M–3.0M plus the final model. Final test: 100 fresh episodes at a 10,000-step cap on the unmodified target sim; the 5k numbers are the same episodes stopped at 5,000 steps.

## Results

| seed | selected ckpt | reached 10k | mean steps | reached 5k |
|---:|---|---:|---:|---:|
| 0 | 2.6M | 80 % | 8957 | 92 % |
| 1 | 2.7M | 75 % | 8802 | 89 % |
| 2 | 2.9M | 88 % | 9299 | 93 % |
| 3 | 2.2M | 64 % | 8013 | 80 % |
| 4 | final (3M) | 91 % | 9553 | 95 % |
| **5 seeds** | | **79.6 ± 4.8 %** | **8925 ± 263** | **89.8 ± 2.6 %** (mean 4723 ± 82 steps) |
| `edgehit3_weak3_pout2` (parent note) | | 73.0 ± 4.9 % | 8514 ± 278 | 84.8 ± 3.0 % |
| dribble recipe | | 99.0 ± 0.4 % | 9942 ± 30 | 99.4 ± 0.4 % |

Hit motion: 20 episodes, 2000-step cap, mean ± std err over 5 seeds.

| | term03_weak3_pout2 | edgehit3_weak3_pout2 | original hist4 (sim) | hist4 real robot |
|---|---:|---:|---:|---:|
| contacts within 2 cm of the edge | **0.0 ± 0.0 %** | 0.6 ± 0.2 % | 8.1 ± 2.2 % | 0 % |
| distance from edge at contact (median) | 11.9 ± 1.5 cm | 14.1 ± 2.0 cm | 10.3 ± 0.6 cm | 20.9 cm |
| forward paddle speed into contact | 0.72 ± 0.03 m/s | 0.72 ± 0.04 | 0.94 ± 0.03 | 0.81 ± 0.03 |
| swing / backswing (12 steps) | 22.8 ± 0.8 / **13.5 ± 3.1 cm** | 23.1 ± 1.7 / 5.6 ± 3.5 | 26.9 ± 0.6 / 15.2 ± 2.6 | 15.1 ± 0.7 / 4.3 ± 0.7 |
| puck speed after the hit | 4.15 ± 0.12 m/s | 4.20 ± 0.09 | 3.78 ± 0.11 | 0.96 ± 0.06 |
| time within 2 cm of the edge | 1.9 ± 1.2 % | 0.2 ± 0.2 % | 10.4 ± 2.4 % | 4.1 % |

## Conclusion

- **On these numbers, `term03_weak3_pout2` is slightly the better Task 1 recipe.**
  - Survival is a little higher: 79.6 ± 4.8 % vs 73.0 ± 4.9 % at 10k, and 89.8 vs 84.8 % at 5k. The gap is about one std err, so it is not significant.
  - Strike quality is the same: no edge contacts, 0.72 m/s, puck out 4.15 m/s.
  - It cocks back more: 13.5 cm, close to hist4's 15 cm in sim.
- **Both recipes still trail the dribble recipe** (99 %) on sustained survival.

## Next

- `edgehit3_weak3v05_pout2` (weak-contact threshold 0.5 m/s) reached 91 % on its single seed and is unconfirmed.
- Combining it with edge termination is the obvious next 5-seed candidate.
