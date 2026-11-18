# Contrastive RL (PyTorch port of the official recipe, and with the colleague's c0 switches) learns goal-conditioned juggling on the new sim only weakly in 2M steps

- **Date**: 2026-11-15 04:43 UTC start (2026-11-14 22:43 CST)
- **Status**: done (all 10 runs reached 2M; c0 2M eval added 2026-11-15 16:20 UTC)
- **Run dirs**: `runs/crl/crl_{paper,c0}_s{0..4}`; evals `runs/crl/paired_eval_{paper,c0,c0_2M}/summary.json`, `runs/crl/curve_eval_paper/`, `runs/crl/final_eval_paper/`
- **Configs**: `configs/crl/juggle_crl_paper.yaml`, `configs/crl/juggle_crl_c0.yaml`
- **Code**: `scripts/goal_conditioned_rl/{crl_agent,crl_replay,crl_env,train_crl,eval_crl,parity_c0,test_crl}.py`

## Question
Does reward-free contrastive RL (Eysenbach et al. 2022, arXiv 2206.07568) learn to steer the puck to a commanded
state on the new sim? Two variants: (1) as close to the official `google-research/contrastive_rl` state-env recipe
as possible, and (2) the same loop with the colleague's `c0.py` switches.

## Setup
- **Env**: new sim unchanged (`configs/new_juggle/sysid_v2_hist4_realplant.yaml`: puck noise 1 cm, occlusion 0.05,
  25 ms puck lag) + 2 control steps of action latency; no DR; 7000-step episode cap; 4 envs.
- **Goal**: true puck (x, y, vx, vy) read from the Box2D body (not `env.current_state`, see Findings).
  Training goals = hindsight future states of the same episode, offset ∝ γ^k (γ 0.99). Collection goal = goal view
  of a random replay state. Actor goals `random_goals 0.5` (batch doubled with `roll(goal, 1)`), both variants.
- **paper**: binary NCE over the 256×256 matrix, 2×256 ReLU encoders, repr 64, tanh-Gaussian actor (min std 1e-6),
  no entropy term, Adam 3e-4 / eps 1e-7, batch 256, no target nets, 10k random steps, 1 update per env step.
- **c0**: InfoNCE + 0.1·logsumexp², 8-layer residual LayerNorm/swish nets, c0 actor head, auto entropy.
  Numerical parity with the JAX `c0.py` checked on one batch (logits, losses, grads, action ≤ 1e-6, `parity_c0.py`).
- 5 seeds each, 2M env steps. c0 updates cost 5.4 ms vs 0.9 ms (Quadro RTX 6000), so c0 ran ~2–3× slower.
- **Eval** (`eval_crl.py`): 100 goals from held-out rollouts of the reference juggler
  (`latest_models/ceiling/newsim_scratch_cap7000_swa/seed0`), one deterministic episode per goal, horizon 7000.
  Main metric is the **paired gap**: same env seed run twice, once commanding goal A and once commanding the control
  goal B; gap = min distance to A when B is commanded − min distance to A when A is commanded. A goal-agnostic
  policy scores exactly 0 (verified: juggler +0.000 ± 0.000). Hit = puck within 10 cm of the goal position;
  hit lift = hit(A commanded) − hit(B commanded). Distances use position only.

## Results
Mean ± std err over 5 seeds (each seed's value is itself a mean over 100 goals).

| variant | step | paired gap (cm) | hit lift (pp) | min dist (cm) | hit10 (%) | episode steps |
|---|---|---|---|---|---|---|
| paper | 500k | 0.2 ± 0.5 | 3.4 ± 1.9 | 25.7 ± 0.6 | 37.8 ± 1.4 | 113 ± 2 |
| paper | 1M | 0.7 ± 0.5 | 1.8 ± 1.0 | 21.0 ± 1.3 | 46.2 ± 2.2 | 153 ± 5 |
| paper | 1.5M | 2.7 ± 0.5 | 2.2 ± 1.0 | 18.1 ± 1.2 | 51.0 ± 2.7 | 235 ± 20 |
| paper | 2M | 3.0 ± 1.2 | 5.2 ± 3.5 | 15.2 ± 1.6 | 59.8 ± 4.9 | 373 ± 52 |
| c0 | 500k | −0.6 ± 0.9 | −5.2 ± 2.4 | 25.1 ± 0.9 | 35.0 ± 1.6 | 125 ± 3 |
| c0 | 1M | 0.7 ± 1.0 | 2.4 ± 3.7 | 18.1 ± 0.5 | 53.6 ± 2.1 | 267 ± 13 |
| c0 | 1.5M | 1.6 ± 0.5 | 5.0 ± 1.1 | 15.6 ± 0.5 | 64.0 ± 1.2 | 447 ± 26 |
| c0 | 2M | 0.7 ± 0.7 | 5.4 ± 2.4 | 15.7 ± 0.5 | 64.0 ± 1.7 | 448 ± 42 |

Baselines (single policy, mean ± se over 100 goals): juggler min dist 3.7 ± 1.5 cm, hit 94 ± 2 %, 6651 steps,
paired gap 0; random actions min dist 41.3 ± 2.8 cm, hit 13 ± 3 %, 44 steps.

Training diagnostics (TensorBoard, s0): paper critic accuracy 0.23 → 0.55 (500k) → 0.38 (2M), chance 1/256;
c0 0.67 → 0.94 (260k) → 0.74 (1M); accuracy falls as episodes lengthen and the replay gets more varied. Paper actor
entropy settles at about −50 (min std 1e-6, no entropy term, so collection is nearly deterministic). Rollout episode
length rises steadily for both (paper 43 → 213 at 2M; c0 43 → 211 at 1M, 230–340 at 1.5–2M).

## Conclusion
- Both variants learn something reward-free: the puck stays in play 3–4× longer than with random actions and gets
  within 10 cm of a juggler-state goal 60–64 % of the time, but that is mostly a *goal-agnostic* skill (keeping the
  puck alive); the juggler, which ignores goals, hits 94 %.
- Goal conditioning is real but small and still growing: commanding the goal brings the puck 2–3 cm closer to it
  than commanding another goal (paper 3.0 ± 1.2 cm at 2M, c0 1.6 ± 0.5 cm at 1.5M), on a 15 cm typical miss.
- At matched steps c0 survives longer and gets closer (1.5M: 447 vs 235 steps, 15.6 vs 18.1 cm) but its paired gap
  is not larger (1.6 ± 0.5 vs 2.7 ± 0.5 cm). At 2M c0 plateaus (448 steps, 15.7 cm, hit 64 %) and its paired gap
  falls to 0.7 ± 0.7 cm (two seeds slightly negative), while paper reaches 3.0 ± 1.2 cm: the paper recipe is the
  better goal-conditioner here, c0 the better goal-agnostic survivor. Hit lift is similar (5.2 vs 5.4 pp).
- 2M steps is short for this env (TD3 juggling needed 5–13M steps), so this is an early-learning check, not a ceiling.

## Findings outside the experiment
- `env.current_state`, and so `scripts/rma/env_wrapper.py:true_state_features` (documented as noise-free,
  undelayed), carries the 25 ms puck lag and the puck noise: mean |diff| vs the Box2D body 1.8 / 0.9 cm in position,
  4.3 / 1.0 cm/s in velocity. The privileged TD3 critic recipes use it. Not changed.

## Next
- Longer runs (5–10M) of the better-performing variant; the paired gap was still rising at the end.
- The "puck in goal radius with vx = 0" eval, and position+velocity distance in the metric.
