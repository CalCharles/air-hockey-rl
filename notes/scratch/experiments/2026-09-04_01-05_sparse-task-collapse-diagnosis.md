# Why `paddle_reach_position`, `paddle_reach_position_velocity` and `puck_velocity` collapse to a constant action under the canonical TD3 recipe, and what fixes it

- **Date**: 2026-09-04 01:05 UTC start
- **Status**: done (2 of 18 runs, `reach_x10flat` and `reach_vel_x10flat_nowd_s1`, were at 815k / 850k steps and 100 % success when this was written; they finish unattended)
- **Run dirs**: baseline `runs/td3/full_nodr_20260903_2252/{reach,reach_vel,puck_vel,touch,juggle}_nodr`; diagnostics `runs/td3/diag_sparse_20260904/round1{a,b,c}/*`, `runs/td3/diag_sparse_20260904/round2/*`, `runs/td3/diag_sparse_20260904/round3/*`
- **Configs**: archived next to this note in `2026-09-04_01-05_sparse-task-collapse-diagnosis-configs/{td3,sim}/` (originally `configs/td3/diag_sparse/`, `configs/new_juggle/diag_sparse/`; each run dir also keeps its `args.yaml` / `config.yaml`). The recipe they led to lives in `configs/td3/tasks/` and the reward classes.
- **Related**: `2026-09-03_04-20_training-throughput-optimization.md` (same trainer, same 5 tasks), commit `d1e75af` (sparse rewards + spawn randomization these runs use)

## Question

With the sparse rewards introduced on 2026-09-03 (+1 on goal / touch, upward puck displacement for `puck_velocity`) the canonical no-DR recipe (`configs/td3/throughput_bench/full/*_nodr.yaml`, 1M steps) learns `touch` (85–90 % success) and `juggle` (return 55–75) but `reach`, `reach_vel` and `puck_vel` end at or below a random policy, with the actor emitting one saturated action for every observation. Why, and what is the smallest change that gives a curve that goes up and stays up?

## Setup

Baseline = the 5 runs of `full_nodr_20260903_2252` (seed 0, 1M steps, one job per GPU). Diagnostics are the same TD3 args with one or two knobs changed, run through `scripts/td3/run_experiments.py --mode nodr`, two to four jobs per GPU. No training code was modified; every variant is a config. Probes (`/tmp/.../probe_*.py`, not kept) load `model.pth` / `qf1.pth` from checkpoints and roll out the deterministic actor in the run's own env config.

## Baseline symptoms (from TensorBoard + checkpoint probes)

| Task | Random policy success | Best training success | Final success | Final actor | Final critic |
|---|---:|---:|---:|---|---|
| reach | 0.44 | **0.79 @ 90k** (probe: 0.84 @ 75k, y-tracking slope 1.02) | 0.01 | constant (−1, −1), 100 % of outputs saturated, mean |pre-tanh| 9.7 | Q(s,a) = 0.012 for every (s, a); mean |∂Q/∂a| = 0.0002 |
| reach_vel | 0.17 | 0.05 (warm-up only) | 0.00 | constant, 100 % saturated | Q ≡ 0.0005 |
| puck_vel | 0.10 (return 0.14) | ≈ random | return 0.04 | constant (+1, +1) then (−1, −1) | Q ≈ 0.02 flat |
| touch (works) | 0.27 | 0.90 | 0.85 | 77 % saturated | |∂Q/∂a| = 0.065, Q range 0..0.9 |
| juggle (works) | – | – | return 55–75 | 80 % saturated | |∂Q/∂a| = 0.35, Q range 0..21 |

Three things stand out in the reach baseline trajectory:

1. **Negative Q on a task with rewards in {0, 1}.** From 25k to 75k the critic's mean Q is −0.01…−0.02 and the Bellman target is negative. Clipped double-Q takes `min(Q1, Q2)`; with two near-identical critics that is `mean − 0.56 σ`, and bootstrapping with γ = 0.975 amplifies that bias by ≈ 1/(1−γ) = 40. The critic settles at a small negative constant whenever the reward signal is weak. The actor's objective is then noise.
2. **The actor saturates within 5k steps of `learning_starts`.** At 25k steps 98 % of actions are already at ±1 (|pre-tanh| ≈ 4). Adam is scale-invariant: a critic with |∂Q/∂a| ≈ 1e-3 still moves the actor at full `policy_lr` × 6 updates per env step, only in an arbitrary direction. Once the tanh is saturated the actor gradient vanishes and the policy is stuck at a corner. This is the "goes to a single location" behaviour.
3. **The success/failure replay split freezes.** `finalize_episode_if_done` routes an episode to the success buffer when its return ≥ the `1 − success_top_fraction` quantile of the last 500 returns. With binary returns the threshold is 1 when success > 50 % and 0 otherwise. Below 50 % *every* episode goes to the 30k-transition success FIFO and the 70k failure buffer stops receiving data, yet `critic_failure_sample_fraction = 0.7` keeps drawing 70 % of every critic **and actor** batch from it. In the baseline reach run the failure buffer froze at 25,480 transitions at step 113k (policy from 77k–113k) and never changed again; success fell from 79 % to 5 % over the following 100k steps. Before 77k the failure buffer held the 104 transitions of the first two episodes and supplied 70 % of every batch — the policy still reached 74 % on the remaining 30 %.

Two smaller env findings, not the cause but worth fixing:

- The first observation of every episode reports the paddle at (−0.8, 0) with `valid = 1` (placeholder in `AirHockeyBox2D.paddle_history`, `airhockey/sims/airhockey_box2d.py:1108`), not at its spawn point. The transition from that state is stored in replay.
- `rewards/sampled_reward_mean` is 1–2 % even when success is ≪ 1 %: PER (α = 0.6) resamples the few rewarded transitions heavily, so the critic "sees" them but has ~35 unique examples to generalise from (reach_vel).

## Results (all 1M steps, seed 0 unless noted; success = rolling 5k-step training success rate, return = unscaled episodic return over the last 20 %)

`x10`/`x100` = `base_reward_scaling` in the sim config. `flat` = one replay buffer (`recent_episode_window_size: 1`, `success_buffer_size: 1e6`; puck_vel additionally `failure_buffer_size: 1e6` + 50/50 sample fractions because float32 rounding of a continuous return sends ~half the episodes to "failure"). `nowd` = `q_weight_decay: 0`. `slowactor` = `actor_updates_per_iteration: 1`, `policy_lr: 1e-4`. `explore` = `exploration_noise: 0.3` + `learning_starts: 100k`.

| Run | success 0–200k | 200–500k | 500k–1M | min 500k–1M | return, last 20 % |
|---|---:|---:|---:|---:|---:|
| **reach** baseline | 0.28 | 0.08 | 0.09 | 0.00 | 0.08 |
| reach_x10 | 0.91 | **1.00** | **1.00** | 0.96 | 0.997 |
| reach_flat | 0.86 | 0.99 | 0.92 | 0.10 (dip @ ~300k, recovered) | 1.00 |
| reach_x10flat | 0.92 | **1.00** | **1.00** | 1.00 | 1.00 |
| reach_x10flat_nowd | 0.89 | **1.00** | **1.00** | 0.98 | 1.00 |
| **reach_vel** baseline | 0.03 | 0.01 | 0.01 | 0.00 | 0.01 |
| reach_vel_x10 / _flat / _x10flat | 0.03–0.04 | 0.01 | 0.01–0.02 | 0.00 | ≤ 0.013 |
| reach_vel_x10flat_slowactor / _explore / _ls100k | 0.02–0.07 | 0.00–0.01 | 0.01–0.02 | 0.00 | ≤ 0.017 |
| reach_vel_nowd (weight decay off only) | 0.02 | 0.01 | 0.02 | 0.00 | 0.01 |
| **reach_vel_x10flat_nowd** | 0.50 | **1.00** | **1.00** | 0.97 | 0.998 |
| **reach_vel_x10flat_nowd** seed 1 | 0.30 | **1.00** | **1.00** | 0.96 | 0.999 |
| **puck_vel** baseline | 0.06 | 0.05 | 0.04 | 0.00 | 0.06 (random policy: 0.14) |
| puck_vel_x10 | 0.07 | 0.05 | 0.05 | 0.00 | 0.07 |
| puck_vel_flat | 0.20 | 0.27 | 0.12 | 0.00 | 0.25 |
| puck_vel_x10flat | 0.48 | 0.76 | 0.78 | 0.52 | 1.45 |
| puck_vel_x10flat_slowactor | 0.45 | 0.71 | 0.80 | 0.61 | 1.47 |
| puck_vel_x10flat_nowd | 0.53 | 0.84 | 0.89 | 0.75 | 2.16 |
| **puck_vel_x100flat** | 0.51 | 0.86 | **0.91** | 0.75 | **2.21** |
| puck_vel_x100flat_slowactor | 0.42 | 0.78 | 0.85 | 0.69 | 1.57 |
| **puck_vel_x100flat** seed 1 | 0.53 | 0.86 | **0.90** | 0.71 | **2.15** |

Probes of the final policies (100 deterministic episodes each, fresh seeds):

- `reach_x10`: success 1.00, mean length 7.8 steps, final position = 0.95·goal_x / 1.01·goal_y (perfect tracking), mean |final − goal| 0.05.
- `reach_vel_x10flat_nowd` @750k: success 1.00, mean length 6.1, median final position error 0.028 m, median velocity error 0.20 m/s (radius 0.5), 36 % of actions saturated (vs 100 % in the collapsed baseline).
- `puck_vel_x100flat`: return 3.06 unscaled (scripted intercept: 0.59, random: 0.14), 4.4 paddle–puck hits per episode, every episode has ≥ 1 hit, 79 % of episodes run to the 100-step limit (i.e. it juggles), paddle position std (0.17, 0.22) — no corner-sitting. Critic |∂Q/∂a| = 2.4 (baseline 5e-4), Q range 0..126.

## Conclusion

The collapse is a critic problem that the actor turns into a permanent one. Under the canonical recipe three things conspire whenever the reward signal is sparse and O(1) or smaller:

1. **Coupled L2 on the critic (`q_weight_decay: 1e-4` inside `optim.Adam`)** — with sparse TD gradients the L2 term dominates the Adam-normalized update and shrinks Q toward a constant; |∂Q/∂a| falls to 1e-4–1e-3. This alone is fatal for reach_vel: nothing else (scale, buffer, slower actor, exploration, longer warm-up) rescues it, and turning it off (with scale + flat buffer) gives 100 % on two seeds. It also gives the best puck_vel run at ×10.
2. **The success/failure replay split** — on binary or near-binary returns the threshold snaps between 0 and 1, the failure buffer freezes, and 70 % of every critic/actor batch comes from stale data. A flat buffer fixes reach on its own (with one dip) and is required for puck_vel (×10 alone stays at random; ×10 + flat gives 78 %).
3. **Reward scale** — juggle's dense ~0.5/step reward gives Q ≈ 2–9; the sparse tasks give Q ≈ 0.01–0.1, comparable to the clipped-double-Q bias (negative Q on non-negative rewards) and to the L2 pull. ×10 fixes reach alone; puck_vel needs ×100 to reach juggle's scale (Q ≈ 6 mean, 126 max).

With a flat critic, Adam still moves the actor at full `policy_lr` × 6 updates/step in an arbitrary direction; 98 % of actions are saturated within 5k steps of `learning_starts`, after which the tanh gradient is zero and the paddle parks in a corner forever. Slowing the actor down does not help because the critic never recovers.

**Recommended configs (all config-only, no trainer change):**

- reach → `reach_x10` (or `reach_x10flat_nowd` for a recipe shared with reach_vel).
- reach_vel → `reach_vel_x10flat_nowd`.
- puck_vel → `puck_vel_x100flat` (`puck_vel_x10flat_nowd` is close behind at 0.89 / 2.16).
- **Adopted 2026-09-04 (user decision): ×10 baked into all three reward classes, `q_weight_decay: 0` + `single_replay_buffer: true` in every canonical config → `configs/td3/tasks/`.**

Untested: whether `q_weight_decay: 0` and the flat buffer change juggle / touch (both fine under the old recipe, not re-run here); more than two seeds; the DR (`td3_training_dr`) versions of these tasks.

## Next

- Decide whether `q_weight_decay: 0` + flat buffer should become the default for all five tasks (needs juggle / touch re-runs) or stay task-specific.
- Re-run the three winners under the DR recipe (`*_dr.yaml`) for the sim2real path.
- Fix the paddle-history placeholder on reset (`airhockey/sims/airhockey_box2d.py:1108`): first obs of every episode has the paddle at (−0.8, 0) with `valid = 1`.
- Consider Adam → AdamW (decoupled) for the critic if any weight decay is kept.
