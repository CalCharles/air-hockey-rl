# Real-world frozen-policy eval pipeline

Loads a frozen policy, resets the env between episodes with a
task-appropriate reset strategy, and writes a fixed-size kept-episode
batch to JSON / JSONL / HDF5. No learner, no replay, no checkpointing, no
exploration.

The pipeline has two independent extension points so non-juggle tasks
and non-TD3 agents drop in without touching the orchestrator:

* **Agent dispatch** — `--agent <kind>` selects how the actor is built
  and loaded (`td3`, `sgcrl`, …).
* **Task hooks** — the env config's `task:` string selects (a) the
  **between-episode reset strategy** (puck-sweep FSM for tasks where the
  puck falls back down the table; paddle-reposition for puck-less tasks
  such as reach, which need no reset policy at all), (b) which extra
  per-episode metrics get computed and surfaced in the summary, plus
  (c) the `min_timesteps` floor and per-field console precision.

All **five canonical tasks** (`juggle`, `touch`, `puck_vel`, `reach`,
`reach_vel`) have registered hooks and ready-to-run real-robot configs —
see [Five canonical tasks](#five-canonical-tasks).

**Entrypoint:**
[`scripts/td3/extras/async_td3_real_eval.py`](../../../scripts/td3/extras/async_td3_real_eval.py)

---

## Quick reference

| Surface | File |
|---|---|
| Orchestrator | [`scripts/td3/extras/async_td3_real_eval.py`](../../../scripts/td3/extras/async_td3_real_eval.py) |
| Agent dispatch + builders | [`scripts/td3/helper/real_eval_agents.py`](../../../scripts/td3/helper/real_eval_agents.py) |
| Task hooks + registry (metrics **and** reset strategy) | [`scripts/td3/helper/real_task_eval_hooks.py`](../../../scripts/td3/helper/real_task_eval_hooks.py) |
| Paddle-only reset controller | [`scripts/td3/helper/real_paddle_reposition_fsm.py`](../../../scripts/td3/helper/real_paddle_reposition_fsm.py) |
| Puck reset FSM | [`scripts/real/rollout_reset_policy_real.py`](../../../scripts/real/rollout_reset_policy_real.py) (`ResetPolicyFSM`) |
| Real-robot task configs | [`configs/real_configs/tasks/`](../../../configs/real_configs/tasks/) (`rollout_td3_<task>_hist{2,4}.yaml`; juggle uses `configs/real_configs/rollout_td3_config_hist{2,4}.yaml`) |
| Aggregate stats + console formatter | [`scripts/td3/helper/real_eval_stats.py`](../../../scripts/td3/helper/real_eval_stats.py) |
| Per-episode rollout loop | [`scripts/td3/helper/real_policy_runner.py`](../../../scripts/td3/helper/real_policy_runner.py) |
| Reset FSM driver | [`scripts/td3/helper/real_reset_runner.py`](../../../scripts/td3/helper/real_reset_runner.py) |

---

## Commands

Both commands assume the RTDE control program is already running on the
UR5; otherwise the env construction will fail after ~60 s with
`RuntimeError: RTDE control program is not running on controller`.

### TD3 (canonical juggle)

```bash
python -m scripts.td3.extras.async_td3_real_eval \
  --agent td3 \
  --config configs/real_configs/rollout_config_residual.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --train-args <path_to_args.yaml> \
  --model-path <path_to_training_state.pth> \
  --collector-device cpu \
  --eval-episodes 20
```

`--agent td3` is the default, so the flag is optional. `--train-args`
(architecture) and `--args-file` (online-behavior defaults) are both
required on the TD3 path so the rebuilt actor matches the saved
checkpoint exactly.

### Five canonical tasks

Same command shape for every task — only `--config` (which carries the
`task:` string), `--train-args` and `--model-path` change. The task name
selects the reset strategy and metrics automatically; there is no
per-task flag.

```bash
# <task> ∈ {touch, reach, reach_vel, puck_vel}; juggle uses rollout_td3_config_hist2.yaml
python -m scripts.td3.extras.async_td3_real_eval \
  --agent td3 \
  --config configs/real_configs/tasks/rollout_td3_<task>_hist2.yaml \
  --args-file configs/td3_real_world/td3_online.yaml \
  --train-args <run_dir>/args.yaml \
  --model-path <run_dir>/training_state.pth \
  --collector-device cpu \
  --eval-episodes 20
```

| task | `task:` | real config | reset between episodes | kept-episode metrics (beyond return / length / success) |
|---|---|---|---|---|
| juggle | `puck_juggle_upper_half_reward` | `rollout_td3_config_hist2.yaml` | puck FSM | `episode_juggles`, `episode_contacts`, `episode_juggle_success` |
| touch | `puck_touch` | `tasks/rollout_td3_touch_hist2.yaml` | puck FSM | `episode_touched`, `episode_contacts`, `episode_steps_to_touch` |
| puck_vel | `puck_velocity` | `tasks/rollout_td3_puck_vel_hist2.yaml` | puck FSM | `episode_upward_displacement_m`, `episode_max_upward_step_m`, `episode_contacts`, `episode_hit_puck` |
| reach | `paddle_reach_position` | `tasks/rollout_td3_reach_hist2.yaml` | **paddle reposition** (no puck, no reset policy) | `episode_goal_reached`, `episode_final_goal_dist_m`, `episode_min_goal_dist_m`, `episode_steps_to_goal`, `goal_x/y` |
| reach_vel | `paddle_reach_position_velocity` | `tasks/rollout_td3_reach_vel_hist2.yaml` | **paddle reposition** | reach metrics + `episode_final_goal_vel_dist_mps`, `episode_min_goal_vel_dist_mps`, `goal_vx/vy` (steps_to_goal uses the joint position+velocity criterion) |

The `tasks/` configs are the real simulator block of
`rollout_td3_config_hist{2,4}.yaml` plus the task keys (budget, puck
count, termination flags, goal params) of the matching
`configs/new_juggle/tasks/sim_sysid_<task>.yaml`, so the real episode
budget matches the sim one (50 / 100 / 100 / 100 steps at ~20 Hz).
`_hist4` siblings exist for hist4 checkpoints.

The whole loop also runs in Box2D (point `--config` at the sim task
config) — handy for smoke-testing a checkpoint / the reset path before
going to the robot:

```bash
python -m scripts.td3.extras.async_td3_real_eval --agent td3 \
  --config configs/new_juggle/tasks/sim_sysid_reach.yaml \
  --args-file configs/td3_real_world/td3_online.yaml \
  --train-args runs/td3/tasks_20260904/sysid/reach_sysid/args.yaml \
  --model-path runs/td3/tasks_20260904/sysid/reach_sysid/training_state.pth \
  --collector-device cpu --eval-episodes 6 --data-root-dir /tmp/smoke_reach \
  --no-enable-episode-gif --no-enable-episode-camera-video
```

### SGCRL on `puck_goal_position`

```bash
python -m scripts.td3.extras.async_td3_real_eval \
  --agent sgcrl \
  --config configs/gcrl/gcrl.yaml \
  --model-path gcrl/03500032_sgcrl_AirHockeyPuckGoalPosition-v0.pkl \
  --collector-device cpu \
  --eval-episodes 20
```

`--train-args` and `--args-file` are *not* required: the architecture
lives inside the `.pkl` (state_dim / goal_dim / hidden_dims are read
out at load time), and the policy-state contract is synthesized via
[`synthesize_eval_train_args`](../../../scripts/td3/helper/real_eval_agents.py)
with `use_last_action_in_policy_state=False` (SGCRL doesn't augment
obs with the last action).

The `puck_goal_position` task is not in the registry, so the task hooks
fall through to `GenericEvalHooks` automatically — no juggle / contacts
columns in the summary, `min_timesteps=10` floor, puck-FSM reset.

---

## Agent dispatch

`--agent <kind>` is parsed in
[`_parse_eval_specific_args`](../../../scripts/td3/extras/async_td3_real_eval.py)
and dispatched to
[`EVAL_AGENT_BUILDERS`](../../../scripts/td3/helper/real_eval_agents.py) in
`real_eval_agents.py`:

```python
EVAL_AGENT_BUILDERS = {
    "td3":   build_td3_eval_agent,
    "sgcrl": build_sgcrl_eval_agent,
}
```

Each builder returns an `EvalAgent` bundle:

| Field | Contract |
|---|---|
| `actor` | exposes `.get_action(policy_obs_tensor) -> action_tensor` and `.eval()` |
| `train_args` | only `use_last_action_in_policy_state` is read on the eval path; architecture fields can be filler |
| `metadata` | surfaced in `eval_summary.json` / `episode_summaries.jsonl` (TD3 fills `q_updates` / `actor_updates` from the checkpoint; SGCRL leaves them 0 and stashes the model path) |

The runner queries the actor through
[`deterministic_actor_action`](../../../scripts/td3/helper/real_td3_runtime.py)
which forwards to `actor.get_action(policy_obs)`. The SGCRL adapter
(`_SGCRLActorAdapter`) bridges numpy↔tensor IO; TD3 actors expose
`get_action` natively.

### Adding a new agent

1. Write a builder `build_<kind>_eval_agent(*, args, train_args, obs_dim, act_dim, action_low_np, action_high_np, device) -> EvalAgent` in `real_eval_agents.py`.
2. Wrap the policy in an object exposing `.get_action(tensor) -> tensor`
   and `.eval()` if its native interface differs.
3. Add `"<kind>": build_<kind>_eval_agent` to `EVAL_AGENT_BUILDERS`.
4. If the agent doesn't need `--train-args` / `--args-file`, the CLI
   guard in `async_td3_real_eval.py` already routes non-`td3` kinds
   to the synthesized-TrainArgs branch.

---

## Task hooks

The env config's `task:` field selects a hooks class via
[`TASK_EVAL_HOOKS`](../../../scripts/td3/helper/real_task_eval_hooks.py):

```python
TASK_EVAL_HOOKS = {task: JuggleEvalHooks for task in _JUGGLE_TASKS}
TASK_EVAL_HOOKS.update({
    "puck_touch":                     PuckTouchEvalHooks,
    "puck_velocity":                  PuckVelocityEvalHooks,
    "paddle_reach_position":          PaddleReachEvalHooks,
    "paddle_reach_position_neg":      PaddleReachEvalHooks,
    "paddle_reach_position_velocity": PaddleReachVelocityEvalHooks,
})
# Unknown tasks → GenericEvalHooks.
```

Every class derives from `BaseTaskEvalHooks`, whose defaults reproduce
the historical juggle behaviour (puck FSM reset, bare metrics).

**Metrics side**

| Hook | Purpose |
|---|---|
| `numeric_series_fields` / `rate_fields` | which per-episode fields land in `eval_summary.json` `series` / `rates` |
| `field_format_overrides` | per-field console precision (juggle uses `.2f` / `.0f` / `.1f` / `.2f` for `episode_juggles` / `episode_contacts`) |
| `min_timesteps` | floor passed to `clean_episode_hdf5` (juggle 50, generic 10, touch / puck_vel 5, reach 1 — a reach from a nearby start can legitimately succeed in a handful of steps) |
| `on_episode_start(env)` | snapshot anything the metrics need (goal tasks record the goal + radii) |
| `compute_episode_metrics(result, rows, env)` | task-specific fields splatted into the eval record + episode summary; `None` values are omitted from the series (e.g. `episode_steps_to_goal` on a miss) |
| `format_kept_console_extras(metrics)` | the fragment appended after `return=…` in the per-episode log line |

**Reset side**

| Hook | Purpose |
|---|---|
| `reset_strategy` | label (`puck_reset_fsm` / `paddle_reposition`), surfaced in logs + `run_meta` |
| `make_reset_fsm_cls()` | the `(env, rng)` controller class `ResetRunner` drives between episodes |
| `on_soft_reset(env)` | runs after `env.soft_reset()` and **before** the paddle-history priming; goal tasks resample the goal here (`soft_reset` alone keeps the old goal) |
| `force_fsm_after_hard_reset` | always run the FSM after a physical `env.reset()` instead of consulting the puck bottom/occluded heuristic (paddle-only tasks: `True`) |
| `periodic_hard_reset_every` | cadence of the periodic physical reset (default 3, as in training; `0` disables) |
| `post_reset_transition_hold_steps` | override of the post-reset zero-action hold (`None` → `args.transition_hold_steps_post_reset`; paddle-only tasks `0`) |

| Class | `task:` values | Reset | Notes |
|---|---|---|---|
| `JuggleEvalHooks` | `puck_juggle*`, `multipuck_juggle*` (12 variants) | puck FSM | Historical default; `episode_juggles` / `episode_contacts` / `episode_juggle_success` via [`juggle_counter`](../../../scripts/td3/helper/juggle_counter.py); `min_timesteps=50` |
| `PuckTouchEvalHooks` | `puck_touch` | puck FSM | `episode_touched` (env success or a contact event), `episode_contacts`, `episode_steps_to_touch`; `min_timesteps=5` |
| `PuckVelocityEvalHooks` | `puck_velocity` | puck FSM | upward displacement summed over consecutive visible frames (same measurement as the reward), `episode_max_upward_step_m`, `episode_contacts`, `episode_hit_puck`; `min_timesteps=5` |
| `PaddleReachEvalHooks` | `paddle_reach_position`, `paddle_reach_position_neg` | paddle reposition | goal snapshot at episode start; `episode_goal_reached`, final / min goal distance, `episode_steps_to_goal`, `goal_x/y`; `min_timesteps=1`, post-reset hold 0 |
| `PaddleReachVelocityEvalHooks` | `paddle_reach_position_velocity` | paddle reposition | reach metrics + velocity distance to the goal velocity; `episode_steps_to_goal` requires both tolerances on the same row |
| `GenericEvalHooks` | everything else (`puck_goal_position`, `puck_strike`, …) | puck FSM | Bare runner metrics only; `min_timesteps=10` |

### Adding a task

1. Write a `<Task>EvalHooks(BaseTaskEvalHooks)` class: override the field
   lists + the two metric methods, and — if the task has no puck — set
   `reset_strategy = RESET_STRATEGY_PADDLE_REPOSITION`,
   `force_fsm_after_hard_reset = True`, `post_reset_transition_hold_steps = 0`
   (copy `PaddleReachEvalHooks`). Goal tasks also override `on_soft_reset`
   to resample the goal and `on_episode_start` to snapshot it.
2. Register it in `TASK_EVAL_HOOKS` against the relevant `task:` strings.
3. Add a real config under `configs/real_configs/tasks/` (real simulator
   block + the task keys of the sim config).
4. The eval orchestrator picks it up automatically — no edits to the
   orchestrator, `ResetRunner`, the stats module, or the save helper.

If you only need to lower the `min_timesteps` floor on a puck task,
that's also a single-class registration.

---

## Reset strategies

`ResetRunner` (shared with the training collector) owns the four reset
kinds — `STARTUP`, `SOFT`, `HARD_WITH_FSM`, `HARD_SKIP_FSM` — and blocks
until the reset succeeds. The eval passes it three task-supplied plug-ins
(`reset_policy_fsm_cls`, `post_soft_reset_hook`,
`force_fsm_after_hard_reset`); the training collector passes none, so its
behaviour is unchanged.

### Puck tasks (`puck_reset_fsm`)

`ResetPolicyFSM` sweeps the paddle along the bottom edge to collect the
puck, flicks it up the table, waits for it to fall back and strikes it up
again; the policy takes over once the FSM reports success. Every third
episode (and after any stop) the runner first does a physical
`env.reset()` (arm `moveL` to the home pose), then runs the FSM only if
the puck-position heuristic says the puck is stuck at the bottom /
occluded. This is the historical juggle path and is used verbatim for
`puck_touch` and `puck_velocity` — both start with the puck coming back
down the table toward the paddle, exactly like a juggle episode.

### Paddle-only tasks (`paddle_reposition`)

There is no puck, so there is nothing for a reset policy to do.
[`PaddleRepositionFSM`](../../../scripts/td3/helper/real_paddle_reposition_fsm.py)
drives the paddle to a fresh start pose and lets it settle:

1. **Start pose** — uniform over the reachable workspace
   (`env.get_paddle_workspace_bounds`, inset 2 cm) when the task uses
   `random_paddle_spawn` (the reach tasks do, matching their Box2D
   training distribution); otherwise the task's fixed default spawn.
   Drawn from the reset RNG (`--seed`), independent of the env RNG that
   samples the goal.
2. **`goto_start`** — move toward the target at ≤ 10 cm per step (same
   projection as the puck FSM's `_toward_target`).
3. **`settle`** — zero action until the paddle has stayed within 2 cm of
   the target for 5 steps and its reported speed is < 5 cm/s. Drifting
   out (PID overshoot) goes back to `goto_start`.
4. **Done** — `success`; or `hard_reset_required` after 200 steps, which
   makes `ResetRunner` fall back to a physical `env.reset()` exactly like
   the puck FSM's give-up path.

After the FSM the runner calls `env.soft_reset()`, then the hooks'
`on_soft_reset` → `env.set_goals(...)` + goal-marker sync (so the camera
overlay shows the new goal), then the paddle-history priming; the primed
observation therefore already carries the new goal via
`_append_goal_if_goal_env`. Hard resets (every third episode / after a
stop) always run the reposition FSM (`force_fsm_after_hard_reset`) — the
puck heuristic is meaningless here and the FSM is what randomises the
start pose. The post-reset zero-action hold is 0 for these tasks: the
paddle is already stationary, and the hold would eat into the 50-step
reach budget.

Reset trajectories land in `reset_hdf5/<success|failure>/…` with
`reset_stage_id = 0` for both phases, and `reset_summaries.jsonl` rows
look exactly like puck-FSM ones (`done_reason`, `step_count`, …).

### Real-robot caveats for puck-less tasks

* The real simulator always reports a `pucks[0]` entry (camera
  detection); with no puck on the table it is an occluded placeholder,
  and the puck slots of the 30-dim `history` obs carry that placeholder.
  Box2D reports no `pucks` entry at all for `num_pucks: 0`; the row
  builder writes the same occluded placeholder in that case so the HDF5
  schema is identical. Whether the placeholder the policy saw in sim
  matches the one the camera stack emits is a sim-to-real question for
  the reach policies, not for the eval pipeline.
* `episode_gif_require_puck` defaults to `False`, so GIF generation does
  not depend on a puck.

---

## Why juggle eval is bit-identical

The hooks refactor was deliberately non-breaking for juggle eval:

* `JuggleEvalHooks.numeric_series_fields` ≡ the historical
  `real_eval_stats.NUMERIC_SERIES_FIELDS` global.
* `JuggleEvalHooks.rate_fields` ≡ historical `RATE_FIELDS`.
* `JuggleEvalHooks.field_format_overrides` ≡ the per-field precision
  rules that used to be hardcoded in
  `format_eval_summary_console`.
* `JuggleEvalHooks.min_timesteps == 50` (was `EPISODE_MIN_TIMESTEPS`
  in `async_td3_real.py`).
* `_save_episode_artifacts_and_pending_reset(min_timesteps=…)` defaults
  to `EPISODE_MIN_TIMESTEPS`, so the training-loop call site is
  unchanged.

The legacy `compute_eval_aggregate(records)` / `format_eval_summary_console(...)`
calls (no field-list kwargs) still reproduce the juggle output via a
module-level `_LEGACY_FIELD_FORMAT_OVERRIDES` default — verified with a
batch parity test against the hooks-driven path.

---

## Reset path and goal-env priming gotcha

`AirHockeyGoalEnv` (parent of `puck_goal_position`,
`paddle_reach_position`, etc.) concatenates `get_desired_goal()` onto
the base obs in `step` / `reset` when `return_goal_obs=False`. The
soft-reset priming path (`_prime_paddle_history_stand_still_non_occluded`
in `real_reset_runner.py`) calls `env.get_observation(...)` directly
and bypasses that wrapper, so the *very first* obs the policy sees
after every soft reset would be one slot short of every subsequent
step's obs.

`_append_goal_if_goal_env` re-applies the append when the env is a
goal env with `return_goal_obs=False`. Non-goal envs (juggle) lack
`get_desired_goal` and the helper is a no-op — juggle priming stays
bit-identical.

---

## Outputs

```
<run_data_dir>/
    eval_summary.json          ← run_meta + aggregate + per_episode
    eval_per_episode.jsonl     ← one row per kept episode (incremental)
    episode_summaries.jsonl    ← every attempt (kept + discarded)
    reset_summaries.jsonl      ← every reset event
    run_events.jsonl           ← run_start / eval_done
    episode_hdf5/<bucket>/trajectory_data*.hdf5   ← per-step trajectories
```

`run_meta` carries:

* `agent` (the `--agent` kind),
* `task`, `task_hooks`, `reset_strategy`, `reset_fsm`,
  `post_reset_transition_hold_steps`, `min_timesteps` (what the hooks
  resolved to for this run),
* `agent_metadata` (builder-returned dict — checkpoint counters for
  TD3, model path for SGCRL),
* `model_path`, `config`, `train_args_file`, `args_file`,
* `n_target_episodes`, `n_attempts`, `n_kept`, `n_discarded`,
  `started_iso` / `finished_iso` / `elapsed_s`.

`aggregate.series` / `aggregate.rates` keys vary by task hook — load
the JSON and read whatever's there rather than hardcoding column
names.

---

## Useful CLI knobs

* `--eval-episodes N` — target kept episodes (default 20).
* `--eval-max-attempts M` — safety cap on total attempts (kept +
  discarded). Default 0 (unlimited). Set slightly higher than
  `--eval-episodes` when running a long real-robot batch so a string
  of validator-rejects doesn't run forever.
* `--eval-summary-filename` / `--eval-per-episode-filename` — rename
  the two top-level eval outputs (the episode HDF5 / JSONL streams
  always use their canonical names).
* `--verbose` — restore noisy per-step / per-reset debug prints (the
  eval entrypoint installs a quiet filter by default).
