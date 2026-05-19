# Real-world frozen-policy eval pipeline

Loads a frozen policy, resets the env between episodes via the standard
reset FSM, and writes a fixed-size kept-episode batch to JSON / JSONL /
HDF5. No learner, no replay, no checkpointing, no exploration.

The pipeline has two independent extension points so non-juggle tasks
and non-TD3 agents drop in without touching the orchestrator:

* **Agent dispatch** — `--agent <kind>` selects how the actor is built
  and loaded (`td3`, `sgcrl`, …).
* **Task hooks** — the env config's `task:` string selects which extra
  per-episode metrics get computed and surfaced in the summary, plus
  the `min_timesteps` floor and per-field console precision.

**Entrypoint:**
[`scripts/td3/extras/async_td3_real_eval.py`](../../../scripts/td3/extras/async_td3_real_eval.py)

---

## Quick reference

| Surface | File |
|---|---|
| Orchestrator | [`scripts/td3/extras/async_td3_real_eval.py`](../../../scripts/td3/extras/async_td3_real_eval.py) |
| Agent dispatch + builders | [`scripts/td3/helper/real_eval_agents.py`](../../../scripts/td3/helper/real_eval_agents.py) |
| Task hooks + registry | [`scripts/td3/helper/real_task_eval_hooks.py`](../../../scripts/td3/helper/real_task_eval_hooks.py) |
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

The `puck_goal_position` task is not in the juggle registry, so the
task hooks fall through to `GenericEvalHooks` automatically — no
juggle / contacts columns in the summary, `min_timesteps=10` floor.

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
# Unknown tasks → GenericEvalHooks.
```

| Hook | Purpose |
|---|---|
| `numeric_series_fields` / `rate_fields` | which per-episode fields land in `eval_summary.json` `series` / `rates` |
| `field_format_overrides` | per-field console precision (juggle uses `.2f` / `.0f` / `.1f` / `.2f` for `episode_juggles` / `episode_contacts`) |
| `min_timesteps` | floor passed to `clean_episode_hdf5` (juggle 50, generic 10) |
| `compute_episode_metrics(result, rows)` | task-specific fields splatted into the eval record + episode summary |
| `format_kept_console_extras(metrics)` | the fragment appended after `return=…` in the per-episode log line |

| Class | `task:` values | Notes |
|---|---|---|
| `JuggleEvalHooks` | `puck_juggle*`, `multipuck_juggle*` (12 variants) | Historical default; computes `episode_juggles` / `episode_contacts` / `episode_juggle_success` via [`juggle_counter`](../../../scripts/td3/helper/juggle_counter.py); `min_timesteps=50` |
| `GenericEvalHooks` | everything else (`puck_goal_position`, `puck_strike`, `paddle_reach_position`, …) | Bare runner metrics only; `min_timesteps=10` |

### Adding task-specific metrics

1. Write a `<Task>EvalHooks` class with the four hook fields + two
   methods.
2. Register it in `TASK_EVAL_HOOKS` against the relevant `task:` strings.
3. The eval orchestrator picks it up automatically — no edits to the
   orchestrator, the stats module, or the save helper.

If you only need to lower the `min_timesteps` floor on a non-juggle
task, that's also a single-class registration.

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
