# Retraining the sustained-juggling recipe on a sim without the top-edge velocity bug (Task 2)

- **Date**: 2026-10-30 12:54 UTC start
- **Status**: done
- **Builds on**: [`2026-10-29_07-49_juggle-ceiling-final-recipe.md`](2026-10-29_07-49_juggle-ceiling-final-recipe.md) (the "dribble" recipe); companion to [`2026-10-30_04-22_natural-strike-shaping.md`](2026-10-30_04-22_natural-strike-shaping.md) (Task 1)
- **Run dirs**: `runs/ceiling/task2_edgefix/hist50_privcritic_pretanh_edgefix_s{0..4}/`
  - survival evals: `runs/ceiling/evals/{selection_long/sellong__task2_edgefix__fixsim, final/final__task2_edgefix__{fixsim,origsim}, final/final__dribble_baseline__fixsim}/`
  - hit metrics: `runs/ceiling/hit_motion/task2_edgefix__{fixsim,origsim}/`
- **Configs**: `configs/ceiling/task2_edgefix.yaml` (expanded: `configs/ceiling/generated/task2_edgefix/`)
- **Policies**: `latest_models/ceiling/edgefix_privcritic_pretanh/seed{0..4}/` (`SOURCE` names the checkpoint)
- **GIFs**: `eval_gifs/edgefix_privcritic_pretanh/seed{0..4}_fixsim_ep900000_600steps.gif` (fixed sim)

## Question

After each Box2D sub-step, the simulator clips the paddle back into its workspace (`airhockey/sims/airhockey_box2d.py`, the `_clip_pid_target_to_workspace(pos)` call after `world.Step`). It keeps the paddle body's velocity, so a paddle pinned at the top edge still hands its outward velocity to the paddle–puck restitution model.

If the recipe is retrained on a sim where a pinned paddle loses that velocity, does it stop dribbling at the edge and learn strong hits inside the workspace?

## Setup

- **The fix:** `scripts/rma/edge_velocity_fix.py`. Whenever the post-step clip moves the paddle, it removes the outward component of the paddle body's velocity.
  - It is applied as an instance patch to each env's simulator after every `reset`, because the simulator is rebuilt under DR.
  - **`airhockey/` is not modified.** To undo, drop the flag.
  - Training: `edge_velocity_fix: true` in `scripts/rma/train_base_policy.py` (default false).
  - Evaluation: `--edge-velocity-fix` on `scripts/ceiling/eval_target.py` and `scripts/ceiling/hit_motion.py`.
  - Checks:
    - The patch reproduces the earlier scratch-script counterfactual: on dribble seed 0, reward per step goes 0.67 → 0.52.
    - In the GIF rollouts it removes outward velocity 400–670 times per 600 control steps, so the trained policies are pressing against the clamp.
- **Recipe:** `final_privcritic_pretanh` unchanged except for the fix. No reward shaping. 5 seeds, 3M steps.
- **Selection:** checkpoints 1.5M–3.0M on `selection_long` (32 episodes, 10,000-step cap) on the fixed sim, the policy's own training sim.
- **Final test:** 100 fresh episodes at a 10,000-step cap (seeds 900000+), on both the fixed sim and the unmodified sim. The 5,000-step numbers are the same episodes stopped at 5,000 steps.
- **Hardware:** GPUs 0 and 1 only.

## Results

### Survival, final test (5 seeds, mean ± std err)

| policies | eval sim | reached 10k | mean steps (10k cap) | reached 5k | mean steps (5k cap) |
|---|---|---:|---:|---:|---:|
| **trained with the fix** | fixed | **94.6 ± 2.8 %** | 9692 ± 155 | 96.6 ± 1.7 % | 4918 ± 41 |
| trained with the fix | unmodified | 95.0 ± 2.7 % | 9745 ± 137 | 97.2 ± 1.5 % | 4938 ± 34 |
| dribble recipe (trained unmodified) | fixed | 93.6 ± 2.4 % | 9673 ± 123 | 96.6 ± 1.2 % | 4926 ± 28 |
| dribble recipe (trained unmodified) | unmodified | 99.0 ± 0.4 % | 9942 ± 30 | 99.4 ± 0.4 % | 4986 ± 10 |
| Task 1 strike recipe `edgehit3_weak3_pout2` | unmodified | 73.0 ± 4.9 % | 8514 ± 278 | 84.8 ± 3.0 % | 4596 ± 78 |

Per seed for the fixed-sim policies on the fixed sim, reached 10k: 96 / 100 / 98 / 95 / 84 % (checkpoints 2.1M / 2.6M / 2.7M / 2.7M / 2.4M).

### How it hits

hit_motion on the fixed sim, 20 episodes, 2000-step cap, mean ± std err over 5 seeds:

| | trained with fix | dribble recipe | Task 1 strike | original hist4 (unmodified sim) |
|---|---:|---:|---:|---:|
| contacts within 2 cm of the top edge | **51 ± 12 %** (8 / 41 / 63 / 65 / 78 % per seed) | 79 ± 2 % | 0.6 ± 0.2 % | 8.1 ± 2.2 % |
| distance from edge at contact (median) | 2.3 ± 0.6 cm | 1.2 ± 0.1 cm | 14.1 ± 2.0 cm | 10.3 ± 0.6 cm |
| forward paddle speed into contact | **0.11 ± 0.01 m/s** | 0.03 ± 0.02 | 0.72 ± 0.04 | 0.94 ± 0.03 |
| puck speed after the hit | 1.85 ± 0.03 m/s | 1.34 ± 0.05 | 4.20 ± 0.09 | 3.78 ± 0.11 |
| time within 2 cm of the edge | 48 ± 11 % | 36 ± 10 % | 0.2 ± 0.2 % | 10.4 ± 2.4 % |
| reward per step | 0.73 ± 0.01 | 0.57 ± 0.02 | 0.64 ± 0.01 | 0.62 ± 0.01 |
| steps between hits | 35 | 35 | 16.5 | 24 |

The Task 1 strike and original hist4 columns are from the unmodified sim; they never pin the paddle at the edge, so the fix hardly applies to them. On the unmodified sim, the fixed-sim policies hit the same way: 51 % edge contacts, 0.11 m/s.

## Conclusion

- **No: fixing the edge velocity does not, by itself, make the policy learn strong hits away from the edge.**
  - The retrained policies still spend about half their time parked at the top edge.
  - Half their contacts are at the edge, and the rest are only a few cm in, at ~0.1 m/s.
  - The puck leaves at 1.9 m/s, against 3.8–4.2 m/s for a real strike.
  - Seed 1 is the exception, with 8 % edge contacts, but it still blocks (0.13 m/s, 4.3 cm from the edge).
- **The fix makes little difference to survival either way.**
  - Old dribble policies on the fixed sim: 93.6 % reach 10k, against 99 % on the unmodified sim.
  - Policies retrained with the fix: 94.6 % on the fixed sim and 95.0 % on the unmodified sim. The difference from 99 % is about 1–2 std errs, i.e. weak evidence.
  - Reward per step, which measures how much of the time the puck is in the upper half, recovers to 0.73 after retraining (dribble policies on the fixed sim: 0.57). The retrained policies have learned to dribble well without the extra edge velocity.
- **Why:**
  - The extra velocity the bug provides is small (~0.14 m/s, measured earlier).
  - What makes dribbling work is the sim's paddle–puck restitution of 1.26 > 1: a stationary paddle sends the puck back faster than it arrived, wherever the paddle is.
  - The top edge is simply the safest place to wait: highest, and with nothing behind the paddle.
  - Getting hist4-like strikes needs the training-side shaping of Task 1. The weak-contact penalty plus the exit-speed bonus are what produced strikes there; edge penalties only moved the block.
- **Confidence:** high for the qualitative result. All 5 seeds still block slowly and mostly at the edge. The small survival gap to the unmodified-sim recipe rests on a single 100-episode final test per seed.

## Next

- To make sim dribbling less attractive without shaping, the lever is the paddle–puck restitution of 1.26 (sysid gain 1.63 at the canonical mass ratio), not the edge clip. That would be a sim change, which needs the user's go-ahead.
- If the fix is wanted permanently, put it in `airhockey/` behind a config flag and regenerate the target config. Until then, `edge_velocity_fix: true` and `--edge-velocity-fix` keep it opt-in.
