# Sustained juggling with hist4-like strikes instead of top-edge dribbling (Task 1)

- **Date**: 2026-10-30 04:22 UTC start (written 13:00 UTC)
- **Status**: done (5-seed final test of the chosen recipe). A second 5-seed recipe (`term03_weak3_pout2`) is still training and will be written up in a separate file.
- **Builds on**: [`2026-10-29_07-49_juggle-ceiling-final-recipe.md`](2026-10-29_07-49_juggle-ceiling-final-recipe.md) (the "dribble" recipe, `latest_models/ceiling/final_privcritic_pretanh/`)
- **Run dirs**: `runs/ceiling/shape_screen1/`, `shape_screen2/`, `shape_screen3/`, `shape_screen4/`, `shape_seeds_term03/` (in flight); hit metrics `runs/ceiling/hit_motion/<label>/`; survival evals `runs/ceiling/evals/{selection_long,final}/`
- **Configs**: `configs/ceiling/shape_screen{1,2,3,4}.yaml`, `configs/ceiling/shape_seeds_term03.yaml` (expanded in `configs/ceiling/generated/<sweep>/`)
- **Policies**: `latest_models/ceiling/strike_shaped/seed{0..4}/` (`SOURCE` names the checkpoint each was copied from)
- **GIFs**: `eval_gifs/strike_shaped/seed{0,2,4}_400steps.gif`, plus the dribble policy for comparison: `baseline_final_privcritic_seed0_400steps.gif` (unmodified target sim, seed 900000)

## Question

The final ceiling recipe survives 10,000 steps in 99 % of episodes, but it does so by parking the paddle at the top edge of its workspace. There it blocks or dribbles the puck: 92 % of contacts are within 2 cm of the edge, at ~0 m/s forward paddle speed. This relies on the sim's paddle–puck restitution of 1.26 > 1 (and on the edge-clip velocity bug).

Can the same recipe instead learn to strike like the original hist4 policies, while keeping long sustained juggling? Striking means: track the puck, swing forward fast, and hit inside the workspace rather than at the edge. **The simulator was not modified.** Only the way the policy learns changed: training-only reward shaping and termination, read from the true sim state. As in the earlier ceiling work, every evaluation below uses the unmodified target sim `configs/new_juggle/sysid_v2_hist4.yaml` and the unshaped task reward.

## Setup

- **Recipe:** unchanged from the final ceiling recipe (`configs/ceiling/final_privcritic_pretanh.yaml`): 50-step history actor, true-state critic, `actor_pre_tanh_penalty` 0.001, gamma 0.995, 3M steps, 2000-step training cap, ±25 % DR on paddle_density / puck_damping / gravity.
- **Added:** training-only shaping, `scripts/rma/motion_shaping.py`, wired into `scripts/rma/train_base_policy.py` as `shape_*` args. All terms default to off, so existing recipes are bit-identical. The variables:
  - d = paddle distance to the top edge of its workspace (robot-frame x − `x_min_lim`)
  - v_fwd = paddle speed toward that edge over the control step
  - contact step = `paddle_puck_collision_count` > 0

  | term | when | effect |
  |---|---|---|
  | `shape_edge_coef` (margin 5 cm) | every step | −c · (fraction of the margin the paddle is inside) |
  | `shape_edge_hit_coef` (margin 8 cm) | contact | −c · (same, at the contact) |
  | `shape_weak_coef` (vmin 0.3 m/s) | contact | −c if v_fwd < vmin |
  | `shape_hit_coef` (vref 1 m/s) | contact | +c · clip(v_fwd / vref, 0, 1) |
  | `shape_pout_coef` (1.5 → 3.5 m/s) | contact | +c · clip((puck speed − 1.5) / 2, 0, 1) |
  | `shape_edge_term_margin` | contact | end the episode if d < margin |
  | `shape_cock_coef` (ref 10 cm) | contact | +c · backswing in the last 12 steps / ref |

- **Metrics:** new `scripts/ceiling/hit_motion.py` reports survival plus per-hit motion on the target sim:
  - edge hits = contacts within 2 cm of the edge
  - dtop = distance from the edge at contact
  - vfwd = forward paddle speed into the contact
  - swing / cock = forward travel / backswing in the 12 steps before the hit
  - pout = puck speed after the hit
  - near-edge time = fraction of steps the paddle is within 2 cm of the edge
- **Selection and final test:** same protocol as the ceiling work.
  - Selection: score checkpoints 1.5M–3.0M (every 100k, plus the final model) on `selection_long` (32 episodes, 10,000-step cap) and keep the best.
  - Final test: 100 fresh episodes, 10,000-step cap, seeds 900000+.
- **Hardware:** GPUs 0 and 1 only.

## Results

### What we are aiming for

hit_motion: 10 episodes, 2000-step cap, per-seed values averaged over 5 seeds, mean ± std err. Real robot: 69 hits from the 2026-10-20 hist4 eval logs (`data/sysid_v2_real_workspace_original_hist4_ckpt1975000_newest_frame_maxr12_no_estop_count`).

| policy | edge hits | dtop median | vfwd (m/s) | swing | backswing | puck out (m/s) | near-edge time | steps per hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original hist4 (sim) | 8.1 ± 2.2 % | 10.3 ± 0.6 cm | 0.94 ± 0.03 | 26.9 ± 0.6 cm | 15.2 ± 2.6 cm | 3.78 ± 0.11 | 10.4 ± 2.4 % | 24 |
| original hist4 (**real robot**) | 0 % | 20.9 cm | 0.81 ± 0.03 | 15.1 ± 0.7 cm | 4.3 ± 0.7 cm | 0.96 ± 0.06 | 4.1 % | — |
| dribble recipe (`final_privcritic_pretanh`) | 91.6 ± 3.0 % | 0.9 ± 0.0 cm | −0.03 ± 0.02 | 32.9 ± 1.9 cm | 6.5 ± 3.4 cm | 1.46 ± 0.06 | 42.8 ± 11.4 % | 43 |

### Round 1: per-step edge penalty and small contact bonuses (failed)

8 variants, 1 seed each (`shape_screen1`): `shape_edge_coef` 0.25–1 with margin 5–10 cm, `shape_hit_coef` 1–2, `shape_weak_coef` 0.5–1.
- The near-edge time penalty works: 43 % → 2–5 % of steps.
- But the policy still **meets the puck at the edge**: 33–88 % of contacts at the final model, vfwd ≈ 0. It swings up and arrives exactly when the puck does, so it pays almost no time penalty and still blocks at the edge.
- Small per-contact bonuses (≤ 2 per contact, with ~40 steps between contacts) were too weak against the +1-per-step survival reward.

### Round 2: strong contact-location terms, exit-speed bonus, edge termination

10 variants, 1 seed each (`shape_screen2`), all keeping `edge_coef` 0.5. hit_motion at 8 episodes and a 2000-step cap; the 2.5M checkpoint and the final model:

| variant | cap reached | edge hits | dtop median | vfwd | puck out | kind |
|---|---:|---:|---:|---:|---:|---|
| edgehit5 | 100 % | 0–1 % | 12–15 cm | 0.08–0.10 | 1.9 | stationary block, off the edge |
| edgehit10m12 | 100 % | 0 % | 28–32 cm | 0.17 | 2.0 | stationary block |
| term03 (end episode on contact < 3 cm) | 88–100 % | 0 % | 9–10 cm | 0.20 | 2.2 | block |
| term06 | 88 % | 0 % | 8–16 cm | 0.05–0.08 | 1.8 | block |
| term06_hit1_weak2 | 100 % | 0 % | 13–16 cm | 0.12 | 2.0 | block |
| edgehit5_weak2 | 100 % | 0–19 % | 7–17 cm | 0.07–0.12 | 2.0 | block |
| pout3 | 100 % | 18–23 % | 3–4 cm | 0.27–0.29 | 2.8–3.0 | mostly edge |
| hit2_weak5 | 62–75 % | 1–4 % | 8–13 cm | 0.71–0.82 | 3.9–4.3 | **strike** |
| **edgehit3_weak3_pout2** | 62–100 % | 2–7 % | 5–12 cm | 0.39–0.75 | 3.0–3.8 | **strike** |

- **Moving the contact off the edge is easy, but it does not create a strike.** A penalty on contact location, or ending the episode on an edge contact, gives up to 0 % edge hits. The policy then simply blocks a few centimetres further in, still at ~0.1 m/s. Restitution > 1 lets a stationary paddle keep juggling anywhere.
- **A strike needs a penalty on weak contacts that is large compared with any bonus, plus a reward on puck exit speed.** `edgehit3_weak3_pout2` uses −3 for a contact at < 0.3 m/s, −3 · edge proximity, and +2 · exit speed.

### Round 3: 5 seeds of the strike recipe, plus variants

`shape_screen3` (and `shape_screen4` for the backswing bonus). Selection was on `selection_long`: the best checkpoint's share of 32 episodes that reached the 10,000-step cap. Final test is 100 fresh episodes at a 10,000-step cap.

| variant | seed | selected ckpt | selection | **final: reached 10k** | final mean steps |
|---|---:|---|---:|---:|---:|
| edgehit3_weak3_pout2 | 0 | 2.6M | 81 % | 61 % | 7810 |
| edgehit3_weak3_pout2 | 1 | 2.8M | 72 % | 78 % | 8837 |
| edgehit3_weak3_pout2 | 2 | 2.4M | 72 % | 74 % | 8686 |
| edgehit3_weak3_pout2 | 3 | 1.6M | 75 % | 64 % | 7951 |
| edgehit3_weak3_pout2 | 4 | 2.9M | 81 % | 88 % | 9287 |
| **edgehit3_weak3_pout2, 5 seeds** | | | | **73.0 ± 4.9 %** | **8514 ± 278** |
| term03_weak3_pout2 (end episode on contact < 3 cm) | 0 | 2.6M | 84 % | 80 % | 8957 |
| edgehit3_weak3v05_pout2 (weak threshold 0.5 m/s) | 0 | 2.7M | 97 % | 91 % | 9544 |
| edgehit3_weak3_pout2_hit1 (+ speed bonus) | 0 | final | 78 % | 79 % | 8708 |
| edgehit5_weak3_pout2 | 0 | 2.9M | 66 % | — | — |
| edgehit3_weak3_pout2_cock3 (+ backswing bonus) | 0 | not selected | — | — | — |

Two round-3 variants were stopped at 1M steps to free GPU slots for the backswing variants: gamma 0.99, and weak penalty 2. Neither was ahead.

**Hit motion of the 5 selected strike policies** (`runs/ceiling/hit_motion/final__shape_strike/`; 20 episodes, 2000-step cap; mean ± std err over the 5 seeds):

| | strike recipe (5 seeds) | original hist4 (sim) | hist4 real robot | dribble recipe |
|---|---:|---:|---:|---:|
| contacts within 2 cm of the edge | **0.6 ± 0.2 %** | 8.1 ± 2.2 % | 0 % | 91.6 ± 3.0 % |
| distance from edge at contact (median) | **14.1 ± 2.0 cm** | 10.3 ± 0.6 cm | 20.9 cm | 0.9 cm |
| forward paddle speed into contact | **0.72 ± 0.04 m/s** | 0.94 ± 0.03 | 0.81 ± 0.03 | −0.03 ± 0.02 |
| forward swing (12 steps before the hit) | 23.1 ± 1.7 cm | 26.9 ± 0.6 | 15.1 ± 0.7 | 32.9 ± 1.9 |
| backswing (12 steps before the hit) | 5.6 ± 3.5 cm | 15.2 ± 2.6 | 4.3 ± 0.7 | 6.5 ± 3.4 |
| puck speed after the hit | **4.20 ± 0.09 m/s** | 3.78 ± 0.11 | 0.96 ± 0.06 | 1.46 ± 0.06 |
| time within 2 cm of the edge | **0.2 ± 0.2 %** | 10.4 ± 2.4 % | 4.1 % | 42.8 ± 11.4 % |
| steps between hits | 16.5 ± 0.3 | 23.9 ± 1.0 | — | 42.6 ± 0.8 |
| reached the 10k cap (final protocol) | **73.0 ± 4.9 %** | ~0 % (mean ~250 steps) | — | 99.0 ± 0.4 % |

In the GIFs (`eval_gifs/strike_shaped/`), the strike policy rests at the back of its workspace. When the puck falls, it swings ~25 cm toward the edge, meets the puck ~12 cm short of it at ~0.7 m/s, and retracts. The dribble policy instead waits and meets the puck at the edge.

**Backswing bonus (`shape_cock_coef` 1 and 3):** little effect; backswing stays at 2–7 cm with similar survival. hist4 backswings 15 cm in sim, but on the real robot it is only 4.3 cm, so the strike policies already match the real robot here. Dropped.

## Conclusion

- **The strikes look like hist4's in every motion metric we measure.**
  - Contacts are off the edge (0.6 % vs 92 % for the dribble policy).
  - They happen ~14 cm into the workspace, at 0.72 m/s forward.
  - The puck leaves at 4.2 m/s, a bit faster than hist4's 3.8.
  - The paddle is almost never parked at the edge (0.2 % of steps).
  - This holds on all 5 seeds.
- **Most of the long survival is kept.** 73.0 ± 4.9 % of episodes reach 10,000 steps (mean 8514 ± 278 steps). That is ~35× the original hist4 policies' ~250 steps, but below the dribble recipe's 99.0 ± 0.4 %. Per-checkpoint survival is noisier than for the dribble recipe, so selection matters more.
- **Striking costs some reward per step:** 0.64 vs 0.74 for dribbling. The puck spends more time travelling between hits, and every hit is a risk.
- **Which terms do what:**
  - The weak-contact penalty plus the exit-speed bonus is what produces the strike.
  - The edge penalties alone only move the block off the edge.
  - Per-step time penalties alone do nothing: the policy just times its arrival at the edge.
- **Confidence:** the 5-seed numbers are solid for this recipe. The other variants are single seeds; `edgehit3_weak3v05_pout2` (91 %) and `term03_weak3_pout2` (80 %, the steadiest across checkpoints) might be better recipes but are not yet confirmed.
- **Not tested:** whether these strikes survive the real-robot gap better than dribbling. They should not depend on the edge clamp, but they do still use the sim's 1.26 restitution, which makes every hit faster.

## Next

- `term03_weak3_pout2`, seeds 1–4, is training (`runs/ceiling/shape_seeds_term03/`). Its 5-seed final test goes in a separate note.
- Confirm `edgehit3_weak3v05_pout2` (single seed, 91 %) with 5 seeds.
- Task 2 (training on the sim with the edge-velocity fix, no shaping): separate note, in progress.
- Real-robot test of `latest_models/ceiling/strike_shaped/seed4` (88 %) against the dribble policy.
