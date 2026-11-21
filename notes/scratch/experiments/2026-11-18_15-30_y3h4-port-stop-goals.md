# Y3H4 (PyTorch port of y3h4.py) never learns to touch the puck: after 2M steps it is indistinguishable from random actions

- **Date**: launched 2026-11-17 14:37 CST; runs done 2026-11-18 (s3/s4 after 15.9 h, s0–s2 after 24.7 h); eval 2026-11-18 15:30 CST
- **Status**: done (negative result)
- **Follows**: [`2026-11-16_21-28_crl-stationary-goals-to-5M.md`](2026-11-16_21-28_crl-stationary-goals-to-5M.md) (same goals,
  same env, same eval)
- **Run dirs**: `runs/crl/y3h4_s{0..4}` (`final.pt`, `ckpt_*.pt` every 250k, `train.log`, TensorBoard events); evals
  `runs/crl/y3h4_eval_{prev,apex}/summary.json`, random baseline `runs/crl/y3h4_eval_random_{prev,apex}/`
- **Code**: `scripts/goal_conditioned_rl/{y3h4_agent.py, train_y3h4.py, eval_y3h4.py, test_y3h4.py, parity_y3h4.py}`,
  config `configs/crl/juggle_y3h4.yaml` (`--total-steps 2000000`). `c0` / `crl_agent.py` / `train_crl.py` untouched
  apart from the import of `converged`.

## Question
Does the y3h4 recipe (z = ψ(commanded goal) fed into φ(s, a, z), L2 energy with a projection head, 4-head ensemble with
a voted action, rest coordinates zeroed in the commanded goal) stop the puck on (x, y, 0, 0) targets better than c0 / paper?

## Setup
- Port parity against the JAX y3h4.py: losses, grads, accuracies, sampled and voted actions all within 3e-6
  (`parity_y3h4.py`); 9 unit tests pass.
- New sim (`sysid_v2_hist4_realplant.yaml` + 2-step action latency), 8 envs, 1 update per env step, batch 256,
  replay 1M, 10k random warm-up steps, 2M env steps. y3h4 agent values: 4 heads, width 256, depth 8, repr 64, lr 3e-4,
  logsumexp penalty 0.1, goal jitter 0.02, target entropy −1, commit fraction 0.25.
- 5 seeds: s0–s2 on GPU 0 (22.4 steps/s each), s3/s4 on GPU 1 (34.7 steps/s each).
- Eval: the same paired stop-goal eval as the earlier notes (100 goals per set, velocity zeroed, paired control goal,
  horizon 7000, deterministic voted action).

## Results
Mean ± std err over 5 seeds. "random" = uniform random actions (one run of the 100 goals).

| Set B (juggler states) | stop-on-target | stop lift | hit 10 cm | min dist (m) | episode steps |
|---|---|---|---|---|---|
| **y3h4 2M** | 0.0 ± 0.0 % | 0.0 ± 0.0 pp | 14.2 ± 0.4 % | 0.401 ± 0.004 | 45 ± 1 |
| random | 0.0 % | 0.0 | 13.0 % | 0.413 | 44 |
| c0 stationary 5M | 2.2 ± 0.6 % | +1.4 ± 0.7 | 28.2 ± 1.8 % | 0.301 ± 0.007 | 172 ± 15 |
| paper stationary 5M | 4.0 ± 0.9 % | +3.4 ± 1.2 | 34.6 ± 3.7 % | 0.269 ± 0.027 | 151 ± 21 |
| c0 4-D goals 2M | 12.4 ± 1.1 % | +7.6 ± 1.1 | 56.6 ± 2.1 % | 0.161 ± 0.012 | 554 ± 48 |

| Set A (slow states) | stop-on-target | stop lift | hit 10 cm | min dist (m) | episode steps |
|---|---|---|---|---|---|
| **y3h4 2M** | 1.0 ± 0.0 % | 0.0 ± 0.0 pp | 11.2 ± 0.2 % | 0.539 ± 0.001 | 45 ± 1 |
| random | 1.0 % | 0.0 | 12.0 % | 0.545 | 44 |
| c0 stationary 5M | 2.2 ± 0.4 % | +0.8 ± 0.7 | 14.8 ± 0.9 % | 0.446 ± 0.012 | 160 ± 18 |
| paper stationary 5M | 1.6 ± 0.2 % | 0.0 ± 0.3 | 20.0 ± 2.9 % | 0.420 ± 0.032 | 124 ± 15 |
| c0 4-D goals 2M | 16.0 ± 1.2 % | +5.4 ± 2.7 | 49.2 ± 4.3 % | 0.217 ± 0.019 | 587 ± 73 |

- **The policy never learned contact.** The training episode length stayed at 41–46 steps for all 2M steps in every
  seed, the same as during the random warm-up (42). c0 / paper grew from 42 to 140–160 steps by 1.6M. The
  in-training eval (10 goals) gave identical numbers at every checkpoint and every seed (min dist 0.398, hit 0.1):
  with no paddle contact, the puck's fall does not depend on the policy.
- **The critic does learn.** InfoNCE accuracy 0.82 at 200k, drifting to 0.73 by 2M; rolled-code accuracy (does φ use
  z?) rose 0.05 → 0.10; no non-finite updates. α fell from 0.37 to 0.02 within 200k and entropy sat at its target
  (−1) from then on.
- **Actions are not degenerate.** The greedy action std is ~0.7 per axis with no saturation, and the paddle moves around
  (std 10–20 cm). The 4 heads propose very different actions for the same state (e.g. (−0.12, 0.87) vs (0.02, −0.98)),
  so the vote switches between unrelated proposals from step to step.

## Interpretation
- The port is numerically faithful, so this is the recipe failing on this task, not a translation bug. Most
  likely cause: y3h4 only has a learning signal once some hindsight futures depend on the action. Here a
  random paddle almost never touches the puck within the ~40-step episode, so the critic learns to predict the puck's free
  fall (high accuracy, without using the action). The actor then has no gradient towards hitting.
- c0 / paper escape this on the same data. A plausible difference is that they score the actor against goals
  without a z-dependent φ and an ensemble vote. The independent heads plus the vote add jitter instead of
  sustained strokes. This is not tested.
- Cost: 2.3× slower per step than c0 and 15.9–24.7 h for 2M steps; it never got off the ground.

## Next steps (not run)
- Diagnose on 1 seed rather than scale up: H=1 (no vote), or z fed only into the actor, to see which y3h4 ingredient
  blocks learning contact. Alternatively, warm-start the replay with c0 data or a c0 policy so that contact
  data exist from the start.
