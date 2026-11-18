# Commanding stationary-puck goals (x, y, 0, 0) during CRL training makes both CRL variants worse at stopping the puck on target than the earlier 4-D-goal policies

- **Date**: 2026-11-15 23:47 UTC start (17:47 CST)
- **Status**: done
- **Follows**: [`2026-11-15_04-43_crl-paper-vs-c0-new-sim.md`](2026-11-15_04-43_crl-paper-vs-c0-new-sim.md) (same code, same sim, 4-D goals)
- **Run dirs**: `runs/crl/crl_{paper,c0}_still_s{0..4}`; evals `runs/crl/still_eval_{apex,prev}/`, `runs/crl/still_eval_c0_{apex,prev}/` (each `summary.json` + per-episode JSON)
- **Configs**: `configs/crl/juggle_crl_{paper,c0}_still.yaml`; launcher `scripts/goal_conditioned_rl/launch_crl_still_5seeds.sh` (waits until 2 GPUs are idle ≤ 5 % for 10 min)

## Question
Can CRL learn to *stop* the puck at a commanded position, i.e. reach goal (x, y, vx = 0, vy = 0)?

## Setup
- Identical to the 4-D-goal runs (new sim unchanged, 2-step action latency, no DR, cap 7000, 4 envs, 1 update / env
  step, `random_goals 0.5`) except:
  - every commanded goal has its velocity set to 0: collection goals = a random replay state's position with
    v = 0 (`command_zero_velocity`), and the actor-loss goals likewise (`agent.actor_goal_zero_velocity`);
  - the critic still trains on true hindsight future states (pure CRL);
  - convergence stop: after ≥ 2M steps, checked every 250k, stop once critic loss, actor loss and rollout episode
    length each changed < 10 % between the last two 500k windows; cap 10M.
- Slow puck states are rare in the data: with the reference juggler |v| < 0.1 m/s on 1.4 % of steps (mostly arc
  apexes, x ≈ −0.7), with the 4-D CRL policy 2.1 %.
- **Success ("stopped on target")**: puck within 10 cm of the goal *and* |v| < 0.1 m/s at the same step
  (`hit_still`, `eval_crl.py`). Paired versions as before: same seed with the goal vs the control goal commanded;
  `stop lift` = hit_still(goal) − hit_still(control goal), exactly 0 for a goal-agnostic policy.
- **Goal sets** (both with velocity zeroed, `--zero-velocity`):
  - **A (apex)** `runs/crl/eval_goals_apex.npz`: 100 states where the juggler's puck was < 0.1 m/s
    (`eval_crl.py goals --max-speed 0.1`; 5 episodes had none and use their slowest state, ≤ 0.5 m/s);
    x = −0.69 ± 0.21 (far end), y −0.38..0.39.
  - **B (prev)** `runs/crl/eval_goals.npz`: the 100 positions of the 4-D eval.
- Controls: the 4-D-goal policies of the previous experiment (2M `final.pt`), commanded the same (x, y, 0, 0) goals.

## Training
| variant | stopped at | wall clock |
|---|---|---|
| paper, s0–s4 | 2.25 / 2.5 / 2.0 / 2.5 / 2.0M | 4.5–6.4 h |
| c0, s0–s4 | 2.75 / 3.25 / 2.75 / 3.5 / 2.5M | 10.9–16.9 h |

Paper on GPU 0 (s0–1) and GPU 2 (s2–4), c0 on GPU 0 (s0–2) and GPU 2 (s3–4). All converged by the rule; rollout
episode length plateaued at ~110–170 steps (paper) and ~150–250 (c0), well below the 4-D runs (still rising at 2M).

## Results
Mean ± std err over 5 seeds (each a mean over 100 goals, horizon 7000). Baselines are single policies.

**Set B (previous positions, v = 0)**

| policy | stopped on target (%) | stop lift (pp) | within 10 cm, any speed (%) | paired gap (cm) | min dist (cm) | episode steps |
|---|---|---|---|---|---|---|
| paper, stationary goals | 5.4 ± 0.9 | +5.0 ± 1.2 | 33.6 ± 2.5 | 4.0 ± 0.5 | 27.1 ± 0.7 | 153 ± 4 |
| paper, 4-D goals (control) | 9.4 ± 1.0 | +6.4 ± 1.2 | 47.0 ± 3.1 | 6.1 ± 0.6 | 18.4 ± 0.9 | 398 ± 27 |
| c0, stationary goals | 2.2 ± 0.9 | +0.8 ± 1.0 | 30.8 ± 1.1 | 0.9 ± 0.7 | 29.6 ± 0.7 | 179 ± 12 |
| c0, 4-D goals (control) | **12.4 ± 1.1** | **+7.6 ± 1.1** | 56.6 ± 2.1 | **6.6 ± 1.2** | 16.1 ± 1.2 | 554 ± 48 |
| juggler (goal-agnostic) | 31 | 0 | 94 | 0 | 3.7 | 6651 |
| random | 0 | 0 | 13 | 0.2 | 41.3 | 44 |

**Set A (juggler apex positions, v = 0)**

| policy | stopped on target (%) | stop lift (pp) | within 10 cm (%) | paired gap (cm) | episode steps |
|---|---|---|---|---|---|
| paper, stationary goals | 2.4 ± 0.5 | +0.2 ± 0.6 | 24.6 ± 2.6 | 1.2 ± 1.3 | 142 ± 4 |
| paper, 4-D goals | 8.4 ± 1.9 | −0.2 ± 2.0 | 48.2 ± 6.1 | 0.7 ± 0.5 | 535 ± 76 |
| c0, stationary goals | 1.8 ± 0.4 | +0.4 ± 0.5 | 15.6 ± 0.9 | 0.5 ± 0.3 | 169 ± 16 |
| c0, 4-D goals | **16.0 ± 1.2** | +5.4 ± 2.7 | 49.2 ± 4.3 | 2.6 ± 1.5 | 587 ± 73 |
| juggler | 83 | 0 | 95 | 0 | 6651 |
| random | 1 | 0 | 11 | 0 | 44 |

## Conclusion
- Training with stationary commanded goals **hurt**: both variants keep the puck in play for far shorter
  (142–179 vs 398–587 steps), get less close, and stop it on target less often than the 4-D-goal policies given
  the same (x, y, 0, 0) commands. c0 with stationary goals shows no goal conditioning at all (lift and gap ≈ 0).
- The best stop-on-target policy is the **4-D-goal c0** (12.4 ± 1.1 % on set B with a +7.6 ± 1.1 pp lift over the
  control goal; 16 % on set A). On 4-D goals its paired gap was only 0.7 ± 0.7 cm at 2M (previous note), so
  zero-velocity commands read as a clearer instruction to it than full 4-D juggler states.
- Likely cause: commanding (x, y, 0, 0) during collection asks for states that barely occur (1–2 % of steps), so
  collection and the actor objective chase goals the critic has rarely trained on; the policy learns less of the
  goal-agnostic skill (keeping the puck alive) that everything else rests on, and the convergence rule then fires
  at a lower plateau.
- Set A is dominated by survival: the juggler stops on target 83 % of the time there just by juggling.

## Next
- Oversample slow future states as critic training goals (e.g. 50 % of positives drawn from |v| < 0.2 m/s), so the
  critic actually learns ψ for stationary goals; the option deferred when this run was set up.
- Or keep 4-D training (it learns more) and only command (x, y, 0, 0) at eval, with longer training of c0.
