# `async_td3_real.py` modularization plan

Handoff doc. If prompted with "continue cleaning up real-world training" or
"continue async_td3_real modularization" (or similar), start here.

## Status

**Design only — nothing implemented yet.** This plan captures the chosen
split, the shared-state contracts between modules, and the order to land
the changes. First step on resume is implementation (§4), not more design.

## 1. Goal

Take `scripts/td3/extras/async_td3_real.py`
(3156 lines) and split the **online real-world training loop** into three
self-contained components with explicit contracts:

1. **PolicyRunner** — runs the policy episode until termination, collecting
   data. No knowledge of reset logic.
2. **ResetRunner** — runs the reset FSM until it succeeds; on failure (stop
   event, FSM error) retries until success. No knowledge of training.
3. **Orchestrator** (the file itself, post-refactor) — owns the outer
   `while True:` loop, decides which component runs next, bridges shared
   state (replay, learner, artifact ids, periodic logging).

The orchestrator file should end up being mostly init/teardown + ~150
lines of step-loop glue. PolicyRunner ≤ ~400 lines, ResetRunner ≤ ~300
lines.

## 2. Why this split (and what we're not changing)

- The user diagnosed correctly: the "global logic that decides what runs
  now" is where the worst entanglement lives. The single closure
  `begin_transition_hold` (L1405–1436) mutates six nonlocals — that's the
  smell that points at where the state machine actually lives.
- We are **not** rewriting the helpers under `td3/helper/`. They're already
  small, single-purpose, and well-factored (motion rewards, episode
  buffers, replay sampling, metrics, checkpointing, stop state). The
  refactor only touches the orchestration surface.
- We are **not** changing any algorithmic behavior. Every numeric output —
  replay contents, reward values, transition-hold timing, learner gradient
  steps — must be byte-identical pre/post refactor. The split is purely
  structural.
- We are **not** introducing a new entrypoint. `async_td3_real.py` keeps
  its `collector_process()` / `_run_sync_learner_iteration()` /
  `__main__` shape externally. Configs, CLI flags, and run-dir layout are
  unchanged.

## 3. Module breakdown

All three new modules live in
`scripts/td3/helper/` (next to the
existing helpers; this is already the home for extracted real-world
logic).

### 3.1 `real_policy_runner.py` — PolicyRunner

**Owns** (instance state):
- `actor` reference (rollout actor; orchestrator does
  `actor.load_state_dict(...)` from learner_state between episodes —
  PolicyRunner sees updated weights on next call)
- `env` reference
- `device`
- `exploration_noise`, `primitive_selector`, `_extract_primitive_state_tensors`
- References to two shared dataclasses (see §3.4):
  - `transition_hold: TransitionHoldState` — runner consumes `tick()`
    in its hot loop; orchestrator triggers `begin(...)` between calls
  - `ctx: RolloutContext` — owns `last_action_for_policy`,
    `last_executed_action`, `previous_puck_position_for_primitive`.
    These cross the runner/orchestrator boundary because
    `transition_hold.begin()` (called by the orchestrator on actor-sync
    and after reset) mutates them, and the runner reads/writes them
    every step.
- Genuinely private per-episode mutable state:
  `motion_reward_state`, `episode_rows`, `episode_trajectory`,
  `episode_images`, latency lists, motion-metric sums, readiness-fail
  counters, stop-event flags. None of these are touched by the
  orchestrator or `transition_hold.begin`.

  **Full per-episode reset list** (cleared inside
  `seed_after_reset()` to match current source ordering at
  L2265–2292, which runs *after* the reset has produced new env
  state; this preserves byte-equivalence on the `motion_reward_state`
  re-anchor that depends on the post-reset paddle/puck positions):

  - stop/end flags: `stop_penalty_applied_this_episode`,
    `episode_had_stop`, `episode_had_protective_stop`,
    `episode_had_controller_disconnect`,
    `episode_had_readiness_fail_estop`
  - readiness-fail per-episode trackers:
    `episode_readiness_first_fail_step_idx`,
    `episode_readiness_first_fail_reason`
  - readiness-fail streak/run state (these persist *across* steps
    inside an episode at L1591–1597, so they're per-step state
    seeded at episode start): `readiness_fail_streak`,
    `readiness_fail_first_episode_step_idx`,
    `readiness_fail_first_total_step`, `readiness_fail_prev`,
    `readiness_fail_prev_reason`
  - motion metric accumulators: `episode_motion_metric_sums`
    (dict re-init from `motion_metric_names`),
    `episode_motion_metric_count`
  - per-episode trajectory buffers: `episode_rows`,
    `episode_trajectory`, `episode_images`, latency lists (the
    L2147–2153 clears, which today happen at HDF5-save time —
    move into `seed_after_reset` to consolidate; replay push has
    already consumed them via the returned `PolicyEpisodeResult`)
  - re-anchor `motion_reward_state` from current post-reset env
    paddle/puck positions (L2279–2292) — must run last, after the
    flags are cleared

  All other state on `PolicyRunner` is either constructor-time
  immutable (`actor`, `env`, `device`, etc.) or owned by
  `RolloutContext` / `TransitionHoldState` (§3.4).

**Public surface** (deliberately tiny):
```python
class PolicyRunner:
    def __init__(self, env, actor, *, device, args, train_args,
                 primitive_selector, transition_hold,
                 _extract_primitive_state_tensors,
                 motion_reward_anchor_fn): ...

    def seed_after_reset(self, obs, previous_puck_position) -> None:
        """Called by orchestrator after a successful reset.
        Initializes paddle/puck history + last-action state for the
        upcoming episode."""

    def run_episode(self) -> PolicyEpisodeResult:
        """Runs one policy episode until terminal. Returns trajectory,
        rows, images, metrics, terminal info. Does not push to replay
        and does not invoke the learner — those are orchestrator
        concerns."""
```

**`PolicyEpisodeResult`** (frozen dataclass):
- `trajectory: EpisodeTrajectory` — already truncated for readiness-fail
  if applicable. The orchestrator pushes this to replay unconditionally;
  PolicyRunner is responsible for trimming via
  `truncate_collector_episode_for_readiness_fail` before returning.
- `rows: list[dict]` — also pre-trimmed
- `images: list[np.ndarray]` — also pre-trimmed
- `total_env_steps: int`
- `terminal: TerminalInfo` — flags + the four `step_info`-derived
  fields the orchestrator needs for logging:
  - flags: `dones`, `truncated`, `success`, `protective_stop`,
    `controller_disconnect`, `readiness_fail_estop`,
    `first_readiness_fail_step_idx`, `first_readiness_fail_reason`
  - propagated from the terminal `step_info` (currently read at
    L1831–1851): `episode_success: bool`, `episode_end_type: str | None`,
    `episode_end_reasons: list[str]`, `episode_end_reason: str | None`
  - the readiness-fail dropped step count: `readiness_fail_dropped_steps: int`
    (used at L1808 for the cumulative counter)
  - `stop_state` snapshot (`reason`, `artifact_label`, `episode_end_type`,
    `episode_end_reason`) so the orchestrator can compose the same log
    line without reaching into env state again
- `metrics: EpisodeMetrics` — return, length, task/motion totals,
  per-step latency lists, motion-metric sums, plus **delta counters
  the orchestrator accumulates** (these are bare locals in current
  `collector_process`):
  - `delta_total_steps`, `delta_protective_stop_steps`,
    `delta_controller_disconnect_steps`,
    `delta_readiness_fail_steps`,
    `delta_readiness_fail_estop_dropped_steps`,
    `delta_transition_hold_steps`
  - `delta_interval_primitive_env_steps`,
    `delta_interval_primitive_horizontal_env_steps`,
    `delta_interval_target_position_directional_env_steps`
  - `episode_estop_flag: float` (1.0 if protective_stop or
    readiness_fail_estop)
  - `had_protective_stop: bool`, `had_controller_disconnect: bool`
    (so the orchestrator can increment its episode-level counters)

**Lift from**: L1607–1891 of current file (per-step body + episode-end
metric accumulation, up to but not including the
`_add_episode_to_shared_replay` call). Everything inside the per-step
branch becomes private methods on `PolicyRunner`. The terminal-detection
condition (`if dones:` block, L1778) becomes the loop exit; the metric
finalization that today lives between the dones-check and the artifact
save (L1816–1873) moves into `_finalize_metrics()`. The
readiness-fail truncation (L1786–1815) moves into `_finalize_metrics()`
too — orchestrator never sees the un-truncated trajectory.

**Dependencies it does NOT have**:
- No `replay`, no `learner_state`, no `_run_sync_learner_iteration`
- No `pending_reset_artifact`, no episode-id minting (those are
  orchestrator concerns; PolicyRunner returns a result, doesn't write)
- No tensorboard writer (orchestrator logs)

### 3.2 `real_reset_runner.py` — ResetRunner

**Owns**:
- `ResetPolicyFSM` factory (so it can build a fresh FSM per attempt)
- `env` reference
- `reset_rng`, `device`
- The "soft vs hard" reset decision helpers
  (`_should_run_reset_policy_at_episode_start`, `_hard_reset_with_pause`,
  `_prime_paddle_history_stand_still_non_occluded`)
- Per-attempt failure counter for retry budget

**Public surface**:
```python
class ResetRunner:
    def __init__(self, env, *, device, args, reset_rng,
                 episode_start_reset_counters,
                 episode_start_reset_bottom_margin,
                 episode_start_reset_bottom_fail_count,
                 episode_start_reset_occluded_fail_count,
                 _extract_primitive_state_tensors,
                 _prime_paddle_history): ...

    def run(self, *, kind: ResetKind, artifact_episode_id: int,
            episode_had_stop_flags: StopFlags,
            episode_end_wall_time: float) -> ResetResult:
        """Block until a reset succeeds. Internally retries the FSM on
        protective-stop / FSM failure. Returns the seeded observation
        and the artifact rows for the orchestrator to flush.

        `episode_end_wall_time` is the orchestrator's `time.time()`
        snapshot taken when the previous policy episode terminated
        (set at L1779 in the current source). ResetRunner uses it to
        compute the soft-path `artificial_delay_s = max(0,
        min_reset_delay_s - processing_elapsed_s)` (current source
        L2156–2169). On hard paths the delay is enforced inside
        `_hard_reset_with_pause(pause_s=min_reset_delay_s)` and
        `episode_end_wall_time` is unused — but the orchestrator
        always passes it; ResetRunner picks per-kind whether to apply
        it."""
```

**`ResetKind` enum — full table** (all four cases the current code handles,
with the exact reason string and `merge_reset_fsm_artifact_into_pending`
flag that must be preserved):

| Kind | When orchestrator picks it | Runs FSM? | Hard reset first? | `transition_reason` | `startup_buffered_message` |
|---|---|---|---|---|---|
| `startup` | Once before the main loop (L1373) | yes | no | `"startup_reset_to_policy"` | `True` |
| `soft` | After a normal episode (no stop, not periodic-3) | yes | no | `"reset_fsm_to_policy"` | `False` |
| `hard_with_fsm` | After a stop-driven OR periodic-3 hard reset, **and** `_should_run_reset_policy_at_episode_start` returns True | yes (after hard reset) | yes | `"hard_reset_reset_fsm_to_policy"` | `False` |
| `hard_skip_fsm` | Same as `hard_with_fsm` but the gate returns False | no (just primitive state extract) | yes | `"hard_reset_to_policy"` | `False` |

The `_should_run_reset_policy_at_episode_start` decision is currently
made in the orchestrator (L2213), but it depends only on env state +
counters. Pull it inside `ResetRunner.run` when `kind=='hard_with_fsm'`
is requested — if the gate returns False, internally degrade to
`hard_skip_fsm` and report that downgrade in `ResetResult.kind_actual`.
Orchestrator picks `hard_with_fsm` (not `hard_skip_fsm`) whenever it
chose a hard-reset path; the runner decides whether the FSM actually
fires.

**`ResetResult`** (frozen dataclass):
- `obs: np.ndarray` — the seeded observation the policy will start from.
  Source per kind: `soft_reset_prime` for `soft` and `hard_with_fsm`
  (L2184, L2230); `_hard_reset_with_pause` for `hard_skip_fsm` (L2207);
  `soft_reset_prime` for `startup` (L1395). ResetRunner.run is
  responsible for picking the right one and returning it here.
- *Not* `previous_puck_position_for_primitive` — the orchestrator's
  post-reset `transition_hold.begin(sim_hold=True)` re-extracts this
  from env on every kind (see §3.4 `begin` method body), so any value
  ResetRunner returned would be discarded immediately. Don't propagate
  it through the dataclass.
- `artifact: PendingResetArtifact | None`
- `total_fsm_steps: int`
- `transition_reason: str` — one of the four reason strings in the table
  above. Orchestrator passes this to `transition_hold.begin(reason=...)`.
- `startup_buffered_message: bool` — passed by the orchestrator to
  `merge_reset_fsm_artifact_into_pending`; True only for the startup
  kind, False for everything else.
- `kind_actual: ResetKind` — what actually ran (may differ from `kind`
  if the gate downgraded `hard_with_fsm` to `hard_skip_fsm`). For
  logging; orchestrator otherwise doesn't branch on it.
- `attempts: int` — for orchestrator logging
- `next_reset_file_id: int` — incremented id returned by
  `merge_reset_fsm_artifact_into_pending`. Orchestrator threads this
  through unchanged; do not reinvent the increment rule.

**Failure semantics**: matches the user's spec. If
`_classify_stop_event` fires during FSM, ResetRunner waits for the stop
to clear, then re-runs the FSM. Same if FSM raises. Loop exits only on
clean FSM `done=True` with no active stop. Caller never sees a "reset
failed" outcome — it always blocks until success. Optionally a
`max_attempts` arg with a hard exception if exceeded; current code has
no such cap, so default to unbounded with logging on each retry.

**Lift from**: L661–737 (`run_reset_fsm`), L1158–1188
(`_should_run_reset_policy_at_episode_start`), L1191–1224
(`_hard_reset_with_pause`), L2156–2264 (the orchestrator's
periodic-vs-stop-vs-soft branching). The branching collapses into one
ResetRunner method that is called once per episode boundary, with a
`kind` argument the orchestrator picks based on episode counter and
stop flags.

**Dependencies it does NOT have**:
- No actor, no policy obs, no replay, no learner
- No transition_hold mutation — it returns the *reason* string and
  hold-steps target; orchestrator calls `transition_hold.begin(...)`

### 3.3 `async_td3_real.py` (orchestrator) — what stays

After the lift, the file contains:

- All `dataclass Args` / `dataclass TrainArgs` (config — stays here so
  the entrypoint owns its config schema)
- `_load_train_args`, `_build_args_file_defaults` (CLI + YAML loading)
- `_setup_run_data_dir`, `_prompt_optional_run_note`,
  `_init_sync_learner_state`, `_finalize_sync_learner_state`,
  `_run_sync_learner_iteration`, `_save_async_checkpoint` (init,
  teardown, learner — these are not duplicated in any other entrypoint)
- The shrunk `collector_process()` — see §3.5 for the target shape
- `__main__`

Everything else moves out.

### 3.4 `real_transition_hold.py` — TransitionHoldState + RolloutContext

**Currently** the transition-hold logic lives as five mutable locals
(`transition_hold_steps_remaining`, `transition_hold_reason`,
`transition_hold_events_total`, `transition_hold_reason_counts`,
`transition_last_action_mode`) plus a closure `begin_transition_hold`
that mutates the locals + `last_action_for_policy` +
`previous_puck_position_for_primitive`. Both PolicyRunner and the
orchestrator (via the actor-sync trigger at L1886) need to mutate this
state mid-episode — and `previous_puck_position_for_primitive` is
*also* updated every step inside the per-step body (L1761) — so it
needs to live where both can reach it.

Two sibling dataclasses, both in this module:

```python
@dataclass
class RolloutContext:
    """Mutable per-rollout state that crosses the PolicyRunner /
    orchestrator boundary. PolicyRunner mutates these every step;
    TransitionHoldState.begin() also mutates them when the
    orchestrator triggers a hold mid-episode (actor-sync case)."""
    last_action_for_policy: torch.Tensor
    last_executed_action: torch.Tensor
    previous_puck_position_for_primitive: torch.Tensor

@dataclass
class TransitionHoldState:
    steps_remaining: int = 0
    reason: str = "none"
    events_total: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    last_action_mode: str = "zero"   # "zero" | "executed" | "keep"
    steps_total: int = 0             # for stats

    def active(self) -> bool: ...
    def tick(self) -> None:           # called once per env step from PolicyRunner
        if self.steps_remaining > 0:
            self.steps_remaining -= 1
            self.steps_total += 1

    def begin(self, *, reason: str, hold_steps: int, sim_hold: bool,
              env, ctx: RolloutContext, primitive_selector,
              extract_primitive_state_tensors,
              use_last_action_in_policy_state: bool,
              device) -> None:
        """All six nonlocal mutations from the current closure live here.

        - bumps steps_remaining / events_total / reason_counts
        - resets primitive_selector rollout state
        - re-extracts ctx.previous_puck_position_for_primitive from env
        - if use_last_action_in_policy_state: updates
          ctx.last_action_for_policy per `last_action_mode`
        - if sim_hold: calls _request_sim_transition_hold(env, ...)
        """
```

`last_action_mode` has three legal values per
`_normalize_transition_last_action_mode` (currently in `async_td3_real.py`):
`"zero"` (zero out), `"executed"` (clone last_executed_action), `"keep"`
(don't touch). The original docstring on this dataclass said two values;
that was wrong.

Owners:
- `PolicyRunner` reads `active()`, calls `tick()`, reads/writes the
  three `RolloutContext` fields in its hot loop (L1761, L1764–1766).
- Orchestrator calls `begin(...)` exactly twice per episode boundary,
  with these flag values (matches current code; **don't flip
  `sim_hold`**):

  | Call site | reason | sim_hold |
  |---|---|---|
  | After actor-sync (was L1886, fires only if `actor_updated`) | `"actor_sync_update"` | `False` |
  | After ResetRunner returns | `reset_result.transition_reason` | `True` |

This module is ≤120 lines.

### 3.5 Target orchestrator main-loop shape

After the refactor, `collector_process()` should compress the current
L1553–L2293 (~740 lines) into roughly this skeleton (~120 lines):

```python
while True:
    if smoke_test_done(): break

    # 1. Run one policy episode (PolicyRunner ticks transition_hold
    #    internally, truncates trajectory on readiness-fail, returns
    #    result with all delta counters and pre-trimmed buffers).
    result = policy_runner.run_episode()
    episode_end_wall_time = time.time()    # ← snapshot for soft-path
                                           #   artificial_delay (L1779)
    total_steps += result.metrics.delta_total_steps
    total_episodes += 1
    accumulate_orchestrator_counters(result.metrics)  # protective_stop,
        # controller_disconnect, readiness_fail, transition_hold_steps,
        # interval_primitive_*, episode_estop_flag, rolling50 deques

    # 2. Push to replay UNCONDITIONALLY (trajectory is already
    #    truncated for readiness-fail). Then run the learner.
    partition, ep_return, threshold, n_inserted = \
        _add_episode_to_shared_replay(replay, result.trajectory,
                                      recent_returns,
                                      args.success_top_fraction)
    learner_changed_actor = _run_sync_learner_iteration(
        args=args, train_args=train_args, replay=replay,
        stats=stats, state=learner_state)
    if learner_changed_actor:
        actor.load_state_dict(  # rollout actor from learner state
            {k: v.detach().cpu()
             for k, v in learner_state.actor.state_dict().items()},
            strict=False)
        transition_hold.begin(
            reason="actor_sync_update",
            hold_steps=int(args.transition_hold_steps_post_actor_sync),
            sim_hold=False,                      # ← actor-sync: NO sim hold
            env=env, ctx=ctx,
            primitive_selector=primitive_selector,
            extract_primitive_state_tensors=_extract_primitive_state_tensors,
            use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
            device=device)

    # 3. Save HDF5 / GIF / camera video for the episode.
    _save_episode_artifacts(result, episode_id=next_episode_file_id,
                            stats_writer=writer, ...)
    if pending_reset_artifact is not None:
        _flush_pending_reset_artifact(pending_reset_artifact, ...)
        pending_reset_artifact = None
    next_episode_file_id += 1

    # 4. Pick reset kind and run the reset to completion.
    kind = pick_reset_kind(total_episodes, result.terminal)
    reset_result = reset_runner.run(
        kind=kind,
        artifact_episode_id=next_reset_file_id,
        episode_had_stop_flags=result.terminal.stop_flags,
        episode_end_wall_time=episode_end_wall_time)
    pending_reset_artifact, next_reset_file_id = \
        merge_reset_fsm_artifact_into_pending(   # ← keep existing helper;
            reset_result.artifact,                 #   it owns the id rule
            pending_reset_artifact,
            next_reset_file_id,
            startup_buffered_message=reset_result.startup_buffered_message)
    transition_hold.begin(
        reason=reset_result.transition_reason,
        hold_steps=int(args.transition_hold_steps_post_reset),
        sim_hold=True,                           # ← reset path: sim hold ON
        env=env, ctx=ctx,
        primitive_selector=primitive_selector,
        extract_primitive_state_tensors=_extract_primitive_state_tensors,
        use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
        device=device)
    policy_runner.seed_after_reset(reset_result.obs)
    # NOTE: ctx.previous_puck_position_for_primitive was just refreshed
    # by transition_hold.begin() above (re-extracted from env, same on
    # every ResetKind); PolicyRunner reads it from ctx. ResetRunner
    # never returns this value — see §3.2.

    # 5. Periodic logging.
    #    NOTE: this runs only between episodes after the refactor.
    #    See §6 for the cadence-change tradeoff vs. mid-episode logging.
    if time.time() - last_log_time >= args.collector_log_interval_sec:
        _periodic_log(...)
        last_log_time = time.time()
```

`pick_reset_kind` is a 5-line helper (periodic_every_3 / stop /
soft) — it stays in the orchestrator because the cadence logic
(`total_episodes % 3 == 0`) is a training-policy decision, not a reset
internal. The `hard_with_fsm` vs. `hard_skip_fsm` decision happens
inside `ResetRunner.run`, **not** here — that gate depends on env state
+ counters that ResetRunner already owns.

## 4. Implementation order

Each step is independently testable. Land them in this order so we
never have a half-broken file.

### 4.0. Validation oracle: structural invariants, not byte-identical

The original plan said "byte-identical diff after each step." That
won't hold against the real env (latency arrays vary every run, camera
frames are wall-clock-bound) and is fragile against per-step RNG
(`torch.randn_like` for exploration noise at L1628; any new
torch-allocating call inside an extracted class shifts the sequence
and every downstream action diverges). Replace the oracle with:

- **Run target**: smoke test in **sim mode** (`AirHockeyEnv` with the
  Box2D backend, NOT the real-robot backend), with fixed `--seed` and
  `numpy`/`torch` seeded explicitly. `--smoke-test-seconds 90` is
  enough to cover ≥3 episodes and hit at least one actor-sync hold and
  one each of soft/hard reset paths.
- **Compared invariants** (must match exactly across pre- and
  post-refactor runs):
  - `replay.state_snapshot()` size and partition counts
  - `total_steps`, `total_episodes`
  - `transition_hold_events_total`, `transition_hold_reason_counts`
    dict, `transition_hold_steps_total`
  - `reset_fsm_steps_total`, `protective_stop_steps`,
    `controller_disconnect_steps`, `readiness_fail_steps_total`,
    `readiness_fail_estop_episodes`,
    `readiness_fail_estop_dropped_steps_total`
  - sequence of `episode_id` → (steps, end_type, end_reason,
    success, `episode_estop_flag`)
  - HDF5 file count + per-episode row count
- **Banned in extracted modules**: any `__init__` that allocates a
  torch tensor with non-deterministic init (`torch.empty`, anything
  random). Use `torch.zeros` only. Reuse the orchestrator's existing
  zero-tensors by passing them in via `RolloutContext`, don't
  reallocate.
- **Excluded from comparison** (legitimately non-deterministic):
  per-step latency arrays, wall-clock timestamps, camera frame
  contents.

This is weaker than byte-identical but strong enough that any
behavioral drift in the state machine, reset path, or replay
accounting will fail the diff.

### 4.1. Steps

1. **Cut a baseline.** Run `python -m py_compile` on the current
   file. Capture the structural-invariant trace from §4.0 against
   the **current** `async_td3_real.py`. This is the oracle; check it
   into `notes/scratch/async_td3_real_baseline_<date>.json`.
2. **Extract `RolloutContext` + `TransitionHoldState`** (§3.4). No
   behavior change — move the five locals + closure into the two
   dataclasses. Re-run §4.0; invariants must match the baseline.
3. **Extract `ResetRunner`** (§3.2). The four FSM call-sites collapse
   to `reset_runner.run(kind=...)`. Verify all four `ResetKind` cases
   are exercised in the smoke trace (startup + at least one of
   soft/hard_with_fsm/hard_skip_fsm). Re-run §4.0.
4. **Extract `PolicyRunner`** (§3.1). The per-step block becomes
   `runner.run_episode()`. Verify `EpisodeMetrics.delta_*` counters
   accumulate to the same orchestrator totals as before. Re-run §4.0.
5. **Inline the orchestrator skeleton** (§3.5). At this point
   `collector_process()` should fit on a screen. Final §4.0 run.
6. **Update CLAUDE.md** "Active code paths" row for real-world
   rollout to also list the three new modules.

Steps 2–4 each touch ~one extraction. If an invariant diverges
between consecutive steps, bisect within that single extraction —
don't move on.

## 5. Test surface

There is a `tests/` dir under
`scripts/td3/`. Add unit-style
checks for the new classes where it's cheap:
- `TransitionHoldState`: `tick`/`active`/`begin` arithmetic, no env.
- `PolicyEpisodeResult` / `ResetResult`: round-trip dataclass equality.
- `pick_reset_kind`: table-driven test for the periodic / stop / soft
  branches.

`PolicyRunner.run_episode` and `ResetRunner.run` are too coupled to the
real env to unit-test cheaply; rely on the smoke-test diff for those.

## 6. Tradeoffs and open questions

### 6.1 Accepted behavioral changes (none, by intent)

The refactor is supposed to be behavior-preserving on the structural
invariants in §4.0. One subtle change is unavoidable in the proposed
skeleton; flagging it here so it's an explicit accept, not a silent
regression:

- **Periodic logging cadence.** Currently the periodic log block at
  L2294+ is in the outer `while True:`, so on a long episode it can
  fire mid-episode. After the refactor, `policy_runner.run_episode()`
  is one synchronous call, so the periodic log only fires *between*
  episodes. For typical episode lengths (≪ `collector_log_interval_sec`)
  this is invisible. For a stuck or unusually long episode, the
  operator stops seeing TB updates until it ends. **Decision: accept
  this change** — the transition_hold_active scalar and similar
  per-step state aren't load-bearing for monitoring; they're already
  end-of-episode summaries. Note in the post-refactor commit message.
  If we later need mid-episode logging back, pass a
  `log_interval_callback` into `PolicyRunner` that the orchestrator
  closure-captures.

### 6.2 Open questions (resolve at implementation time)

- **Should `PolicyRunner` own the actor or borrow it?** It must see
  weight updates the learner makes between episodes. Easiest: the
  orchestrator holds the `actor` reference and passes it into
  `PolicyRunner.__init__`; the orchestrator does the
  `actor.load_state_dict(...)` after the learner step, so the runner
  observes new weights on the next call. (Note: the *learner*'s actor
  is a separate object from the *rollout* actor — see L1881–1885;
  don't conflate them.)
- **What does `seed_after_reset` actually take?** Just `(obs)` — the
  primitive puck seed already lives on `RolloutContext` and is
  refreshed by `transition_hold.begin(...)` immediately before this
  call. `seed_after_reset` clears the full per-episode state
  enumerated in §3.1 (stop/end flags, readiness-fail trackers,
  motion-metric accumulators, trajectory buffers) and re-anchors
  `motion_reward_state` from current post-reset env paddle/puck
  (L2287–2292). The current source clears trajectory buffers
  (`episode_rows = []` etc.) at HDF5-save time (L2147–2153) and the
  rest at L2265–2278; consolidating both into `seed_after_reset`
  preserves byte-equivalence because nothing reads those buffers
  between the replay push (which `PolicyRunner` already returned its
  copy of) and the next episode's first step.
- **Where does `_add_episode_to_shared_replay` live?** Stays in
  `async_td3_real.py` (L764–778). It's tiny and uses
  `args.success_top_fraction`. Don't move replay logic into
  PolicyRunner — that pulls training concerns into the data-collection
  module.
- **Can the same modules be reused by
  `async_td3_real_reset_policy.py`?** Probably yes for `ResetRunner`.
  The reset-only script's main loop (L459–617) does almost exactly
  the PolicyRunner-then-replay-then-learner shape, with a single-head
  Q and a simpler reward. Out of scope for the first refactor;
  revisit after the three modules are stable.
- **Does `min_reset_delay_s = 3.0` (L2156) need to become an arg?**
  No — keep hardcoded inside `ResetRunner` for behavior preservation.
  Optional follow-up: lift to `args.min_reset_delay_s`.
- **`hard_with_fsm` → `hard_skip_fsm` downgrade observable from the
  outside?** Current code logs the decision at L2221–2228. The
  refactored ResetRunner needs to emit the same log line (use
  `result.kind_actual` from §3.2). Verify the smoke trace contains
  the `hard_reset_start_decision=` line in both pre- and post-refactor
  runs.

## 7. Files / locations (quick reference)

- Current entrypoint: `scripts/td3/extras/async_td3_real.py`
- Reset-only entrypoint (out of scope but informative):
  `scripts/td3/extras/async_td3_real_reset_policy.py`
- ResetPolicyFSM: `scripts/real/rollout_reset_policy_real.py` (imported by both)
- Existing helpers (do not modify): `scripts/td3/helper/real_*.py`, `td3_*.py`
- New modules (to be created):
  - `…/helper/real_transition_hold.py`
  - `…/helper/real_reset_runner.py`
  - `…/helper/real_policy_runner.py`
- Tests: `scripts/td3/tests/`
- Smoke-test invocation: as currently used to validate
  `async_td3_real.py` in sim — reuse the same command.

## 8. What to do when resuming this work

1. Re-read §3 (module contracts) and §4 (order).
2. Capture a baseline smoke-test trace (§4 step 1).
3. Land step 2 (`TransitionHoldState`). Diff. If clean, commit.
4. Land step 3 (`ResetRunner`). Diff. If clean, commit.
5. Land step 4 (`PolicyRunner`). Diff. If clean, commit.
6. Land step 5 (orchestrator skeleton). Diff. If clean, commit.
7. Update CLAUDE.md (§4 step 6).
8. Promote stable parts of this doc to
   `notes/docs/training/real-world-training-modules.md` once the split
   is in place; leave this scratch file as the change log.
