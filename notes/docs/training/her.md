# TD3 + hindsight experience replay (goal-conditioned puck tasks)

**Trainer**: `scripts/td3/td3_training_her.py` · **HER core**: `scripts/td3/helper/td3_her.py` ·
**Eval**: `scripts/td3/helper/her_eval.py` · **Tasks**: `airhockey/airhockey_tasks/puck_goal_sparse.py` ·
**Configs**: `configs/td3/her/*.yaml` (sim configs `configs/new_juggle/tasks/sim_sysid_puck_goal{,_vel}.yaml`) ·
**Tests**: `scripts/td3/tests/test_her_relabel.py`

Goal-conditioned TD3 with hindsight experience replay (HER, Andrychowicz et
al. 2017) for the two sparse puck-goal tasks.  The trainer is the canonical
recipe (`configs/td3/tasks/*_sysid.yaml`: TD3 with transformed Bellman targets,
PER, single flat 1M replay buffer, `q_weight_decay: 0`, q=25 / actor=6 updates
per episode, primitive exploration, CUDA-graph updates, CPU rollout, async
per-checkpoint eval) with exactly two changes:

1. the desired goal is appended to the canonical 30-dim history observation
   (actor input = 30 + goal_dim + 2 last-action dims; critic input adds the
   2-dim action), and
2. every finished episode is written to replay twice: the original
   transitions, then `her_k` hindsight copies per transition whose goal is
   an achieved goal from a later step of the same episode.

The networks are scaled up to 128 (actor) / 256 (critic) wide, 2 residual
blocks (the five-task recipe uses 64 / 64).

## Tasks

| Task | Goal | Achieved goal | Success (+10, episode ends) |
|---|---|---|---|
| `puck_goal_position_sparse` | `(x, y)` uniform over the upper half of the table (inset `goal_position_margin` + puck radius) | puck `(x, y, vx, vy, contacted)` | puck centre within `base_goal_radius` (0.10 m) **and the paddle has touched the puck in this episode** |
| `puck_goal_position_velocity_sparse` | `(x, y, vx, vy)`: a state of a simulated **shot** from the paddle workspace (`goal_velocity_sampling: shot`, default: launch speed `goal_shot_speed_range` [1, 3] m/s, angle within `goal_shot_max_angle_deg` 45° of straight up, free flight with the table's gravity / damping and side-wall bounces, a random upper-half state still moving up at ≥ `goal_shot_min_upward_speed` 0.3 m/s). `intercept_shot` = the same shot but launched from where *this episode's* puck will cross the paddle workspace (its spawn state integrated forward), resampled right after the world is spawned, so the goal direction is reachable from the puck the agent actually gets. `box` = position as above with an independent uniform velocity from `goal_puck_vx_range` × `goal_puck_vy_range` — mostly infeasible, kept for the rounds 1–3 runs | puck `(x, y, vx, vy, contacted)` | position within `base_goal_radius` **and** velocity within `base_goal_velocity_radius` (0.5 m/s, Euclidean) on the same step, **and contact made** |
| `puck_goal_position_speed_sparse` | `(x, y, speed)`: a shot-sampled state's position and puck **speed** (scalar); direction left to the shot geometry | puck `(x, y, vx, vy, contacted)` | position within `base_goal_radius` **and** speed within `base_goal_speed_radius` (0.5 m/s), **and contact made** |

Everything else is the canonical juggle setup (`sim_sysid_juggle.yaml`):
sysid physics, canonical puck spawn (uniform over the top 2/3 of the table at
0–0.5 m/s), random paddle spawn, terminate when the puck hits the bottom or
passes the paddle, `max_timesteps: 250`, observation delay / puck noise /
occlusion on.  The contact condition makes both tasks "hit the puck to the
goal": the achieved goal carries a flag that latches on the first
paddle-puck collision of the episode (`paddle_puck_collision_count` from the
simulator) and clears on reset.  Without it 16 % of random-action episodes
score because the puck spawns on the goal or drifts through it, and — worse
for HER — the puck's own spawn drift (random heading, up to 0.5 m/s) yields
"achieved" goals the policy did nothing to reach.  Round 1 used the repo's
older puck-goal convention (puck moving up the table, `vx < 0`) instead and
did not learn (see the experiment note).  The goal velocity ranges were
measured with scripted hits (upper-half upward puck speed 0.6–2.9 m/s, p5–p95).
The `box` velocity sampler turned out to be the wrong task: the puck's
direction at the goal is dictated by where it was hit from, so an
independently drawn velocity is unreachable for most goals (a policy that
reached the goal position saw a 2 m/s velocity error there with zero
correlation to the goal velocity).  The `shot` sampler makes every goal
consistent with some straight shot from the paddle region.

The dense `puck_goal_position` / `puck_goal_position_velocity` tasks are
untouched (the SGCRL real-robot baseline uses them).

## What HER does here (`td3_her.py`)

`GoalEnvVector` wraps the env (`return_goal_obs: true` → gym `GoalEnv`
dict), flattens the obs to `[observation, desired_goal]` and forwards the
achieved goal of the next state.  `HEREpisodeTrajectory` stages the episode
with those achieved goals plus a *hard-terminal* flag (the step ended the
episode for a non-goal reason: puck hit bottom / passed paddle).
`HERRelabeler.relabel` runs at episode end, in numpy:

- **candidate goals** = achieved goals of the episode that pass the task's
  own success test against themselves (`env.reward.goal_met(ag, ag[:goal_dim])`)
  — for these tasks, states after the paddle has touched the puck; a
  pre-contact puck position is never proposed as a goal because no
  transition could be rewarded for it — and, with `her_goal_filter: true`
  (default), that lie in the task's goal-sampling region
  (`env.goal_in_distribution`: upper half; for the velocity task also the
  goal velocity box widened by the tolerance).  Without the filter (classic
  HER) replay fills with goals the task never asks for, e.g. post-contact
  puck states in the lower half.
- **`future` strategy** (default): for transition *t*, `her_k` goals drawn
  uniformly from the candidates at steps ≥ *t* (`final` / `episode` also
  available via `her_strategy`).
- **achieved → goal projection**: candidate goals are `env.achieved_to_desired(ag)`
  (default: the first `goal_dim` achieved components; the speed task maps
  `(x, y, vx, vy, c)` to `(x, y, |v|)`).
- **reward** recomputed with `env.compute_reward` (× `base_reward_scaling`);
  **done** = hard-terminal ∨ goal-met (`her_done_on_success`), because the env
  ends an episode on goal arrival and the copy must not bootstrap past it.
- The stored `next_obs` is the *final* observation on **every** episode end
  (the canonical trainer only does this for truncations): a transition that
  terminated by reaching its goal is non-terminal under a relabelled goal and
  then bootstraps from its `next_obs`.

`her_k: 0` turns relabelling off (plain goal-conditioned TD3 — the ablation
configs `*_k0.yaml`).  Replay therefore holds up to `(1 + her_k)` × the env
steps; with the 1M flat buffer and k=4 that is ~200k env steps of history.

TensorBoard adds `her/relabeled_per_episode`, `her/valid_fraction` (share of
env steps whose achieved state is a valid goal) and `her/relabel_ratio`
(relabelled / original transitions, ≤ `her_k`).

## Evaluation

`scripts/td3/helper/her_eval.py` rolls the deterministic actor on fresh
goals, writes `eval_0.gif` (goal circle drawn by `AirHockeyRenderer`) and
`goal_eval.json` (success rate, mean return, length, contacts, end reasons,
per-episode records).  Per checkpoint it runs as a CPU subprocess
(`eval_n_eps`, default 20); the final in-process eval uses
`eval_n_eps_final` (100).  Success = the task's `info["success"]`, i.e. the
sparse goal test, so **mean return = 10 × success rate**.

## Running

```bash
# one run
python -m scripts.td3.td3_training_her --args-file configs/td3/her/puck_goal_sysid.yaml --device cuda:0
# batch, one job per GPU (auto mode picks the HER trainer for YAMLs that set her_k)
python -m scripts.td3.run_experiments --mode her --configs configs/td3/her/*.yaml --gpus 0 1 2 3 --out-root runs/her/<name>
```

Results and the recipe iterations live in
`notes/scratch/experiments/2026-09-10_06-30_her-puck-goal-tasks.md`.
