"""Fixed-policy evaluation entrypoint for real-world TD3.

Loads a frozen policy from a ``training_state.pth`` checkpoint and runs
N *kept* (validator-passed) policy episodes against the real (or sim)
env, with:

  * no learner updates,
  * no replay pushes,
  * no periodic checkpointing,
  * no exploration noise,
  * no exploration primitive (chance forced to 0).

Outputs land under the standard run-data dir created by
``_setup_run_data_dir`` (same layout as a real-world training run, so
the analyst's existing tooling works unchanged):

  ``eval_per_episode.jsonl``  — one row per *kept* episode (the eval set).
  ``eval_summary.json``       — aggregate stats + run metadata + per-episode.
  ``episode_summaries.jsonl`` — every episode (kept *and* discarded), shape
                                identical to training-time output.
  ``reset_summaries.jsonl``   — every reset event (success/failure).
  ``run_events.jsonl``        — ``run_start`` / ``eval_done`` events.
  ``episode_hdf5/<bucket>/trajectory_data*.hdf5`` — per-step trajectories
                                so the eval batch is fully re-enactable.

Aggregate captures (see ``helper/real_eval_stats.py``):

  * ``series.<field>``: count / mean / std / min / max / median / p25 / p75
    over the numeric fields supplied by the active ``TaskEvalHooks``.
    For juggle tasks (``JuggleEvalHooks``): ``episode_return``,
    ``episode_juggles``, ``episode_contacts``, ``episode_reward``,
    ``episode_length``. For unregistered tasks (``GenericEvalHooks``):
    the runner-emitted subset minus the juggle columns.
  * ``rates.<field>``: count / total / rate over the boolean fields the
    hooks list. Juggle includes ``episode_juggle_success``; every task
    includes ``episode_success``, the e-stop class flags, and
    ``readiness_fail_estop``.
  * ``estop_total``: collapsed e-stop count (any class).

The task hooks (``helper/real_task_eval_hooks.py``) also drive the
``min_timesteps`` floor passed to ``clean_episode_hdf5`` and per-field
console precision, so plugging a new task into the eval pipeline only
requires (optionally) registering a hooks class for it.

The eval-loop body lives inside ``run_eval`` and depends only on the
env, actor, and runner factories — future training-loop integration
can call it directly without spawning a separate process.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import tyro
import yaml

from airhockey import AirHockeyEnv
from scripts.td3.helper.episode_artifacts import (
    clean_episode_hdf5,
    save_split_episode_hdf5,
)
from scripts.td3.helper.juggle_counter import count_juggles_from_rows
from scripts.td3.helper.real_collector_factories import (
    build_primitive_exploration_selector_for_real_collector,
)
from scripts.td3.helper.real_eval_stats import (
    compute_eval_aggregate,
    format_eval_summary_console,
    write_eval_summary_json,
)
from scripts.td3.helper.real_policy_runner import (
    PolicyRunner,
)
from scripts.td3.helper.real_reset_runner import (
    ResetKind,
    ResetRunner,
    StopFlags,
    _rewind_goal_sequence_if_available,
    pick_reset_kind,
)
from scripts.td3.helper.real_task_eval_hooks import (
    get_task_eval_hooks,
)
from scripts.td3.helper.real_transition_hold import (
    RolloutContext,
    TransitionHoldState,
    normalize_transition_last_action_mode,
)
from scripts.td3.helper.run_event_log import (
    append_episode_summary,
    append_reset_summary,
    append_run_event,
    episode_summaries_path,
    reset_summaries_path,
    run_data_dir_from_args,
    run_events_path,
)
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM

from scripts.td3.helper.real_td3_runtime import (
    Args,
    TrainArgs,
    _build_args_file_defaults,
    _env_timing_info,
    _latest_camera_frame,
    _load_train_args,
    _next_available_episode_id,
    _prepare_air_hockey_config,
    _reset_primitive_rollout_state,
    _safe_nonnegative_ms,
    _setup_run_data_dir,
    _build_split_episode_row,
    _simulator_step_readiness,
    augment_policy_observation,
    deterministic_actor_action,
    install_quiet_print_filter,
    primitive_exploration_chance_for_step,
)
from scripts.td3.helper.eval_goal_grid import (
    GOAL_GRID_COLS,
    GOAL_GRID_ROWS,
    GOAL_GRID_TASK_NAMES,
    build_eval_goal_grid_from_env,
)
from scripts.td3.helper.real_eval_agents import (
    EvalAgent,
    build_eval_agent,
    synthesize_eval_train_args,
)
from scripts.td3.extras.async_td3_real import (
    _save_episode_artifacts_and_pending_reset,
)


# ---------------------------------------------------------------------------
# Eval-specific args. Kept on a separate small dataclass (not inherited from
# Args) because subclassing a dataclass that uses ``str | None`` annotations
# under ``from __future__ import annotations`` breaks tyro's type resolution
# on Python 3.9. Parse argparse-style off ``sys.argv`` before handing the
# remaining flags to tyro for the (unmodified) Args class.
# ---------------------------------------------------------------------------


@dataclass
class EvalSpecificArgs:
    """Eval-only knobs. All training-side flags live on ``Args``."""

    # Number of *kept* (validator-passed) episodes to evaluate. The loop
    # keeps running episodes until this many pass validation (or the
    # attempt cap is hit, see below).
    eval_episodes: int = 20

    # Optional safety cap on total episode attempts (kept + discarded).
    # 0 disables the cap. Useful on the real robot to bound total
    # wall-clock time when the validator rejects many trajectories
    # back-to-back.
    eval_max_attempts: int = 0

    # Filenames inside the run-data dir. Defaults to plain names so the
    # JSONL/JSON sit next to the standard train-style logs and are
    # easy to discover (``ls <run_data_dir>``).
    eval_summary_filename: str = "eval_summary.json"
    eval_per_episode_filename: str = "eval_per_episode.jsonl"

    # When True (default), suppress noisy per-step / per-reset debug
    # prints that come from the training-side helpers we reuse here.
    # See ``_install_quiet_print_filter`` for the exact prefix list.
    quiet: bool = True

    # Agent kind dispatched to ``real_eval_agents.build_eval_agent``.
    # Default ``td3`` preserves the historical CLI contract (``--train-args``
    # / ``--args-file`` still required). Non-TD3 kinds (e.g. ``sgcrl``)
    # synthesize a minimal ``TrainArgs`` and skip the args-file step.
    agent: str = "td3"

    # Multiply the actor's y output by this factor before env clamp/step.
    # Default 1.0 = off (omit --action-y-scaling to disable).
    action_y_scaling: float = 1.0

    # 1-based eval episode to start collecting from (default 1 = full run).
    # Episode N uses the Nth scripted goal in the eval grid; only episodes
    # N..eval_episodes are collected into this run's eval set.
    restart_eval_from_episode: int = 1

    # When set, protective-stop / readiness-fail e-stop episodes rewind the
    # scripted goal so the next attempt retries the same target instead of
    # advancing the eval grid. Episodes with more than one paddle-puck hit
    # are kept and advance normally even if they end on an e-stop.
    rewind_goal_on_estop: bool = False


def _parse_eval_specific_args() -> EvalSpecificArgs:
    """Strip eval-specific flags from ``sys.argv`` before tyro sees it.

    Returns a populated ``EvalSpecificArgs``; mutates ``sys.argv`` to
    leave only the training-side flags so the existing tyro flow on
    ``Args`` is unmodified.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-attempts", type=int, default=0)
    parser.add_argument("--eval-summary-filename", type=str, default="eval_summary.json")
    parser.add_argument(
        "--eval-per-episode-filename", type=str, default="eval_per_episode.jsonl"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Restore noisy per-step/per-reset debug prints from training-side helpers.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="td3",
        help=(
            "Agent kind. 'td3' (default) requires --train-args and --args-file. "
            "'sgcrl' loads a pickled SGCRL actor and synthesizes the policy-state "
            "contract; 'iwr' loads an interaction-weighted-sampling checkpoint "
            "with the same actor contract. Goal-conditioned variants "
            "'crtr', 'sac-gcrl', 'sac-her', 'sac-weighted-her', and 'ppo-gcrl' use the same "
            "contract (underscore/hyphen spellings are equivalent). "
            "--train-args / --args-file may be omitted for all non-TD3 agents. "
            "See scripts/td3/helper/real_eval_agents.EVAL_AGENT_BUILDERS."
        ),
    )
    parser.add_argument(
        "--action-y-scaling",
        type=float,
        default=1.0,
        help=(
            "Multiply the actor's y output by this factor before env clamp/step. "
            "Default 1.0 = no boost."
        ),
    )
    parser.add_argument(
        "--restart-eval-from-episode",
        type=int,
        default=1,
        help=(
            "1-based eval episode to start from (default 1 = full run). "
            "Episode 4 uses the 4th goal in the eval grid and collects "
            "episodes 4..eval-episodes only."
        ),
    )
    parser.add_argument(
        "--rewind-goal-on-estop",
        action="store_true",
        help=(
            "After a protective-stop or readiness-fail e-stop, rewind the "
            "scripted goal so the next attempt retries the same target when "
            "the episode had at most one paddle-puck hit. Multi-hit e-stop "
            "episodes are kept and advance the eval grid normally."
        ),
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return EvalSpecificArgs(
        eval_episodes=int(parsed.eval_episodes),
        eval_max_attempts=int(parsed.eval_max_attempts),
        eval_summary_filename=str(parsed.eval_summary_filename),
        eval_per_episode_filename=str(parsed.eval_per_episode_filename),
        quiet=not bool(parsed.verbose),
        agent=str(parsed.agent),
        action_y_scaling=float(parsed.action_y_scaling),
        restart_eval_from_episode=int(parsed.restart_eval_from_episode),
        rewind_goal_on_estop=bool(parsed.rewind_goal_on_estop),
    )


# ---------------------------------------------------------------------------
# Eval-mode arg overrides.
# ---------------------------------------------------------------------------


def _force_eval_mode(args: Args) -> None:
    """Zero out everything that makes the policy non-deterministic and
    everything that mutates training state.

    Mutates ``args`` in place. Called once at the top of ``main()`` so
    every downstream consumer (PolicyRunner, primitive selector,
    finalize hooks) sees the eval-mode values without further
    branching.
    """
    args.exploration_noise = 0.0
    args.exploration_primitive_chance = 0.0
    args.exploration_primitive_chance_start = 0.0
    args.collector_policy_stand_still = False
    # Keep checkpointing OFF — eval must never overwrite or write next
    # to the source checkpoint.
    args.enable_periodic_checkpointing = False
    # Eval mode never touches replay; loud disable in case someone reads
    # the value off args later.
    args.load_replay_from_checkpoint = False


def _make_eval_actor_action_fn(
    action_y_scaling: float,
) -> Callable[..., torch.Tensor]:
    """Return ``deterministic_actor_action``, optionally scaling y on output."""
    scale = float(action_y_scaling)
    if scale == 1.0:
        return deterministic_actor_action

    def _scaled_deterministic_actor_action(actor: Any, policy_obs: torch.Tensor) -> torch.Tensor:
        action = deterministic_actor_action(actor, policy_obs)
        action = action.clone()
        action[..., 1] = action[..., 1] * scale
        return action

    return _scaled_deterministic_actor_action


# ---------------------------------------------------------------------------
# Deterministic goal grid for goal-tasks. Shared with
# ``scripts/visualization/render_eval_goal_grid_homography.py``.
# ---------------------------------------------------------------------------


def _maybe_install_eval_goal_grid(env, task_name: str) -> None:
    """If ``task_name`` is a goal-position task, install a fixed eval grid."""
    if task_name not in GOAL_GRID_TASK_NAMES:
        return
    set_seq = getattr(env, "set_goal_sequence", None)
    if not callable(set_seq):
        print(
            f"[eval_goal_grid] task={task_name!r} expected to expose "
            "`set_goal_sequence` but does not; skipping deterministic grid."
        )
        return
    grid = build_eval_goal_grid_from_env(env)
    set_seq(grid)
    print(
        f"[eval_goal_grid] task={task_name!r} installed "
        f"{GOAL_GRID_ROWS}x{GOAL_GRID_COLS}={len(grid)} row-major grid. "
        f"x_range=[{grid[0][0]:+.3f},{grid[-1][0]:+.3f}] "
        f"y_range=[{grid[0][1]:+.3f},{grid[-1][1]:+.3f}] "
        f"(sequence wraps modulo {len(grid)} for longer eval runs)"
    )


def _episode_had_estop(result) -> bool:
    return bool(
        result.metrics.had_protective_stop or result.terminal.readiness_fail_estop
    )


def _count_episode_paddle_puck_hits(rows) -> int:
    return int(count_juggles_from_rows(rows).n_contacts)


def _estop_triggers_goal_retry(
    *,
    rewind_on_estop: bool,
    had_estop: bool,
    n_paddle_puck_hits: int,
) -> bool:
    return bool(
        rewind_on_estop and had_estop and int(n_paddle_puck_hits) <= 1
    )


def _should_rewind_goal_after_episode(
    *,
    episode_kept: bool,
    rewind_on_estop: bool,
    had_estop: bool,
    n_paddle_puck_hits: int,
) -> bool:
    if not episode_kept:
        return True
    return _estop_triggers_goal_retry(
        rewind_on_estop=rewind_on_estop,
        had_estop=had_estop,
        n_paddle_puck_hits=n_paddle_puck_hits,
    )


def _counts_toward_eval_set(
    *,
    episode_kept: bool,
    rewind_on_estop: bool,
    had_estop: bool,
    n_paddle_puck_hits: int,
) -> bool:
    if not episode_kept:
        return False
    if _estop_triggers_goal_retry(
        rewind_on_estop=rewind_on_estop,
        had_estop=had_estop,
        n_paddle_puck_hits=n_paddle_puck_hits,
    ):
        return False
    return True


def _maybe_rewind_goal_sequence_after_episode(
    env,
    *,
    episode_kept: bool,
    rewind_on_estop: bool,
    had_estop: bool,
    n_paddle_puck_hits: int,
) -> bool:
    """Rewind scripted goal before inter-episode reset when retrying.

    Returns True when a rewind was applied (caller may need to compensate
    again on hard-reset + soft-reset paths).
    """
    if not _should_rewind_goal_after_episode(
        episode_kept=episode_kept,
        rewind_on_estop=rewind_on_estop,
        had_estop=had_estop,
        n_paddle_puck_hits=n_paddle_puck_hits,
    ):
        return False
    reason = "after estop" if episode_kept and had_estop else "after discard"
    _rewind_goal_sequence_if_available(env, reason=reason)
    return True


def _log_goal_sequence_state(env, *, label: str) -> None:
    seq = getattr(env, "_goal_sequence", None)
    if not seq:
        return
    idx = int(getattr(env, "_goal_sequence_idx", 0))
    goal_pos = getattr(env, "goal_pos", None)
    goal_str = np.array2string(np.asarray(goal_pos, dtype=float), precision=3)
    print(f"[eval_goal_grid] {label} next_idx={idx} goal_pos={goal_str}")


# ---------------------------------------------------------------------------
# Restart-eval helpers.
# ---------------------------------------------------------------------------


def _validate_restart_eval_args(eval_args: EvalSpecificArgs) -> None:
    restart_from = int(eval_args.restart_eval_from_episode)
    if restart_from < 1:
        raise SystemExit("--restart-eval-from-episode must be >= 1.")
    if restart_from > int(eval_args.eval_episodes):
        raise SystemExit(
            f"--restart-eval-from-episode={restart_from} exceeds "
            f"--eval-episodes={eval_args.eval_episodes}."
        )


def _align_goal_sequence_for_restart(env, *, restart_from_episode: int) -> None:
    """Point the scripted goal grid at the Nth goal before the first policy episode."""
    if int(restart_from_episode) <= 1:
        return
    prepare = getattr(env, "prepare_goal_sequence_for_kept_index", None)
    if not callable(prepare):
        return
    prepare(int(restart_from_episode) - 1)
    idx = int(getattr(env, "_goal_sequence_idx", 0))
    print(
        f"[eval_restart] goal sequence aligned for episode "
        f"{int(restart_from_episode)} (next_idx={idx})"
    )


# ---------------------------------------------------------------------------
# Eval loop. Mirrors the structure of `collector_process_modular` but with
# every learning / replay / checkpoint hook stripped out.
# ---------------------------------------------------------------------------


def run_eval(
    args: Args,
    train_args: TrainArgs,
    eval_args: EvalSpecificArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
) -> Dict[str, Any]:
    """Run the fixed-policy eval loop end-to-end.

    Returns the JSON payload that gets written to ``eval_summary.json``
    so a future in-process caller (e.g. training-loop integration) can
    consume the result without round-tripping through disk.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.collector_device)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    collector_config = _prepare_air_hockey_config(config, seed=args.seed)
    sim_params = collector_config.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
        sim_params["transition_hold_steps_on_estop_enter"] = int(
            args.transition_hold_steps_post_estop_enter
        )
        sim_params["transition_hold_steps_on_estop_clear"] = int(
            args.transition_hold_steps_post_estop_clear
        )
        sim_params["transition_hold_steps_on_safety_rearm"] = int(
            args.transition_hold_steps_post_safety_rearm
        )
    env = AirHockeyEnv(collector_config)
    action_y_scaling = float(eval_args.action_y_scaling)
    if action_y_scaling != 1.0:
        print(f"[eval_run] action_y_scaling={action_y_scaling} (actor output)")
    # Task-specific eval hooks supply: which extra per-episode metrics to
    # compute, which fields land in eval_summary.json, the min-episode-length
    # floor for clean_episode_hdf5, and per-field console precision. Falls
    # back to GenericEvalHooks for any task not in the registry.
    task_name = str(collector_config.get("task", ""))
    task_hooks = get_task_eval_hooks(task_name)
    print(
        f"[eval_run] task={task_name!r} "
        f"hooks={type(task_hooks).__name__} "
        f"min_timesteps={int(task_hooks.min_timesteps)}"
    )

    # Deterministic goal grid for goal-tasks. Without this the goal would
    # (a) be frozen across the ~2 of every 3 episodes that take the SOFT
    # reset path, and (b) vary run-to-run on the HARD path. We inject a
    # fixed grid in row-major order so every eval visits the same goals
    # in the same sequence.
    _maybe_install_eval_goal_grid(env, task_name)

    restart_from_episode = int(eval_args.restart_eval_from_episode)
    per_episode_records: List[Dict[str, Any]] = []
    total_attempts = 0

    # Agent + metadata. Dispatched on ``eval_args.agent``: 'td3' rebuilds the
    # historical DeterministicAgent (or ResidualActor) and loads
    # training_state.pth; 'sgcrl' wraps the pickled SGCRL actor behind a
    # tensor-IO adapter. Both expose the same ``.get_action(tensor)``
    # contract the runner uses.
    agent_bundle: EvalAgent = build_eval_agent(
        eval_args.agent,
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
    )
    # Append a dedicated row capturing what the builder actually produced —
    # the checkpoint's own ``algorithm_name`` (when it differs from the CLI
    # ``--agent`` kind), update counts for TD3, source path, etc. Pairs with
    # the ``run_start`` event but fires only after the actor is in memory.
    append_run_event(
        args,
        "agent_loaded",
        agent=str(eval_args.agent),
        model_path=str(args.model_path) if args.model_path is not None else None,
        metadata=dict(agent_bundle.metadata),
    )
    actor = agent_bundle.actor
    action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0)
    action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0)
    primitive_selector = build_primitive_exploration_selector_for_real_collector(
        args, device, initial_total_steps=0
    )
    # We rely solely on ``exploration_primitive_chance == 0`` (forced by
    # ``_force_eval_mode``) to disable the primitive. ``set_primitive_weights``
    # rejects all-zero weights; the weights are irrelevant when chance=0
    # since ``primitive_selector.apply`` is gated by chance every step.

    ctx = RolloutContext(
        last_action_for_policy=torch.zeros((1, act_dim), dtype=torch.float32, device=device),
        last_executed_action=torch.zeros((1, act_dim), dtype=torch.float32, device=device),
    )
    transition_hold = TransitionHoldState(
        last_action_mode=normalize_transition_last_action_mode(args.transition_last_action_mode),
        log_every_step=bool(args.transition_hold_log_every_step),
    )

    # Reset runner. Even in eval we want resets between episodes (matches
    # training-time setup), so the policy starts each episode from a
    # comparable state.
    next_reset_file_id = _next_available_episode_id(args.reset_artifact_dir)
    reset_rng = np.random.default_rng(args.seed)
    reset_runner = ResetRunner(
        env,
        device=device,
        reset_rng=reset_rng,
        reset_policy_fsm_cls=ResetPolicyFSM,
        build_split_episode_row=_build_split_episode_row,
        latest_camera_frame=_latest_camera_frame,
    )
    pending_reset_artifact = None

    # Startup reset.
    startup_result = reset_runner.run(
        kind=ResetKind.STARTUP,
        artifact_episode_id=next_reset_file_id,
        episode_had_stop_flags=StopFlags(),
        episode_end_wall_time=time.time(),
        pending_reset_artifact=pending_reset_artifact,
        next_reset_file_id=next_reset_file_id,
    )
    pending_reset_artifact = startup_result.pending_reset_artifact
    next_reset_file_id = startup_result.next_reset_file_id

    _align_goal_sequence_for_restart(
        env, restart_from_episode=restart_from_episode
    )

    counters: dict = {
        "reset_fsm_steps_total": int(startup_result.total_fsm_steps),
        "protective_stop_episodes": 0,
        "protective_stop_steps": 0,
        "controller_disconnect_episodes": 0,
        "controller_disconnect_steps": 0,
        "readiness_fail_steps_total": 0,
        "readiness_fail_estop_episodes": 0,
        "readiness_fail_estop_dropped_steps_total": 0,
        "episodes_saved": 0,
        "episodes_removed_short": 0,
        "episodes_removed_invalid": 0,
        "episodes_gif_generated": 0,
        "episodes_gif_failed": 0,
        "episodes_homography_gif_generated": 0,
        "episodes_homography_gif_failed": 0,
        "episodes_camera_video_generated": 0,
        "episodes_camera_video_failed": 0,
        "successful_online_episodes_kept": 0,
    }

    policy_runner = PolicyRunner(
        env,
        actor,
        device=device,
        args=args,
        train_args=train_args,
        action_low=action_low,
        action_high=action_high,
        primitive_selector=primitive_selector,
        transition_hold=transition_hold,
        ctx=ctx,
        reset_primitive_rollout_state=_reset_primitive_rollout_state,
        deterministic_actor_action=_make_eval_actor_action_fn(action_y_scaling),
        augment_policy_observation=augment_policy_observation,
        primitive_exploration_chance_for_step=primitive_exploration_chance_for_step,
        latest_camera_frame=_latest_camera_frame,
        env_timing_info=_env_timing_info,
        safe_nonnegative_ms=_safe_nonnegative_ms,
        build_split_episode_row=_build_split_episode_row,
        readiness_fn=_simulator_step_readiness,
    )
    policy_runner.seed_initial(startup_result.obs)
    transition_hold.begin(
        reason=startup_result.transition_reason,
        hold_steps=int(args.transition_hold_steps_post_reset),
        sim_hold=True,
        env=env,
        ctx=ctx,
        primitive_selector=primitive_selector,
        reset_primitive_rollout_state=_reset_primitive_rollout_state,
        use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
        device=device,
    )

    next_episode_file_id = _next_available_episode_id(args.episode_artifact_dir)
    target_kept = int(eval_args.eval_episodes) - restart_from_episode + 1
    max_attempts = (
        int(eval_args.eval_max_attempts) if int(eval_args.eval_max_attempts) > 0 else None
    )
    if restart_from_episode > 1:
        print(
            f"[eval_restart] collecting episodes {restart_from_episode}.."
            f"{eval_args.eval_episodes} ({target_kept} kept target)"
        )

    eval_start_time = time.time()
    eval_started_iso = datetime.fromtimestamp(
        eval_start_time, tz=timezone.utc
    ).isoformat()

    while len(per_episode_records) < target_kept:
        if max_attempts is not None and total_attempts >= max_attempts:
            print(
                f"[eval] attempt cap reached (eval_max_attempts={max_attempts}) "
                f"with kept={len(per_episode_records)}/{target_kept}; stopping early."
            )
            break

        # 1. Run one policy episode.
        policy_runner.set_artifact_episode_id(next_episode_file_id)
        result = policy_runner.run_episode()
        episode_end_wall_time = time.time()
        total_attempts += 1
        # Operational counters reflect physical events (e-stops, disconnects,
        # readiness faults). Always update — they're independent of whether
        # the trajectory passed validation.
        counters["protective_stop_steps"] += result.metrics.delta_protective_stop_steps
        counters["controller_disconnect_steps"] += result.metrics.delta_controller_disconnect_steps
        counters["readiness_fail_steps_total"] += result.metrics.delta_readiness_fail_steps
        counters["readiness_fail_estop_dropped_steps_total"] += (
            result.metrics.delta_readiness_fail_estop_dropped_steps
        )
        if result.metrics.had_protective_stop:
            counters["protective_stop_episodes"] += 1
        if result.metrics.had_controller_disconnect:
            counters["controller_disconnect_episodes"] += 1
        if result.terminal.readiness_fail_estop:
            counters["readiness_fail_estop_episodes"] += 1

        # 2. Save artifacts (HDF5 + GIF + camera video) and flush pending reset.
        # This is the same helper the training orchestrator uses, so the on-
        # disk layout is identical and the existing training-side analysis
        # tools work without changes. ``min_timesteps`` is task-controlled:
        # the juggle default (50) keeps long episodes; short-success tasks
        # (e.g. puck_strike) override via GenericEvalHooks / a task hook.
        saved_episode_id = next_episode_file_id
        (
            next_episode_file_id,
            episode_kept,
            clean_reason,
            artifact_path,
        ) = _save_episode_artifacts_and_pending_reset(
            args=args,
            result=result,
            next_episode_file_id=next_episode_file_id,
            pending_reset_artifact=pending_reset_artifact,
            latency_output_dir=None,
            counters=counters,
            min_timesteps=int(task_hooks.min_timesteps),
        )
        pending_reset_artifact = None

        task_metrics = task_hooks.compute_episode_metrics(
            result=result, rows=result.rows
        )
        had_estop = _episode_had_estop(result)
        rewind_on_estop = bool(eval_args.rewind_goal_on_estop)
        n_paddle_puck_hits = _count_episode_paddle_puck_hits(result.rows)
        count_toward_eval = _counts_toward_eval_set(
            episode_kept=episode_kept,
            rewind_on_estop=rewind_on_estop,
            had_estop=had_estop,
            n_paddle_puck_hits=n_paddle_puck_hits,
        )
        if count_toward_eval:
            # Note: ``counters["successful_online_episodes_kept"]`` is already
            # incremented inside ``_save_episode_artifacts_and_pending_reset``
            # — do NOT increment it again here or the count is doubled.
            record = {
                "episode_id": int(saved_episode_id),
                "kept_index": int(len(per_episode_records) + restart_from_episode),
                "wall_time_s": float(episode_end_wall_time),
                "timestamp_iso": datetime.fromtimestamp(
                    episode_end_wall_time, tz=timezone.utc
                ).isoformat(),
                "n_steps": int(len(result.rows)),
                "episode_length": float(result.metrics.episode_length),
                "episode_return": float(result.metrics.episode_return),
                "episode_reward": float(result.metrics.episode_reward),
                "episode_success": bool(result.terminal.episode_success),
                "episode_estop_flag": float(result.metrics.episode_estop_flag),
                "had_protective_stop": bool(result.metrics.had_protective_stop),
                "had_controller_disconnect": bool(result.metrics.had_controller_disconnect),
                "readiness_fail_estop": bool(result.terminal.readiness_fail_estop),
                "episode_end_type": result.terminal.episode_end_type,
                "episode_end_reason": result.terminal.episode_end_reason,
                "stop_state_artifact_label": result.terminal.stop_state_artifact_label,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
                **task_metrics,
            }
            per_episode_records.append(record)
            # Mirror this row into eval_per_episode.jsonl so a partial-run
            # crash still leaves the eval set queryable on disk.
            _append_eval_per_episode_row(args, eval_args, record)
            parts = [
                f"[eval] kept {len(per_episode_records)}/{target_kept}",
                f"episode_id={saved_episode_id}",
                f"return={result.metrics.episode_return:.3f}",
            ]
            extras = task_hooks.format_kept_console_extras(task_metrics)
            if extras:
                parts.append(extras)
            parts.append(f"estop={int(result.metrics.episode_estop_flag)}")
            parts.append(f"len={result.metrics.episode_length:.0f}")
            print(" ".join(parts))
        elif _estop_triggers_goal_retry(
            rewind_on_estop=rewind_on_estop,
            had_estop=had_estop,
            n_paddle_puck_hits=n_paddle_puck_hits,
        ):
            print(
                f"[eval] estop episode_id={saved_episode_id}; rewinding goal for retry "
                f"(hits={n_paddle_puck_hits}, kept "
                f"{len(per_episode_records)}/{target_kept}, attempts={total_attempts})"
            )
        else:
            print(
                f"[eval] discarded episode_id={saved_episode_id} reason={clean_reason} "
                f"(kept {len(per_episode_records)}/{target_kept}, attempts={total_attempts})"
            )

        # 3. Always emit one row to the standard episode_summaries.jsonl so
        # the eval directory has the same train-time analysis surface
        # (kept *and* discarded episodes are visible). Task-specific keys
        # come from the hooks; juggle tasks keep the historical
        # juggles/contacts/juggle_success triple.
        append_episode_summary(
            args,
            {
                "episode_id": int(saved_episode_id),
                "run_episode_index": int(total_attempts),
                "wall_time_s": float(episode_end_wall_time),
                "timestamp_iso": datetime.fromtimestamp(
                    episode_end_wall_time, tz=timezone.utc
                ).isoformat(),
                "kept": bool(episode_kept),
                "clean_reason": clean_reason,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
                "n_steps": int(len(result.rows)),
                "episode_length": float(result.metrics.episode_length),
                "episode_return": float(result.metrics.episode_return),
                "episode_reward": float(result.metrics.episode_reward),
                "episode_success": bool(result.terminal.episode_success),
                "episode_estop_flag": float(result.metrics.episode_estop_flag),
                "had_protective_stop": bool(result.metrics.had_protective_stop),
                "had_controller_disconnect": bool(result.metrics.had_controller_disconnect),
                "readiness_fail_estop": bool(result.terminal.readiness_fail_estop),
                "episode_end_type": result.terminal.episode_end_type,
                "episode_end_reason": result.terminal.episode_end_reason,
                "stop_state_artifact_label": result.terminal.stop_state_artifact_label,
                "replay_partition": None,
                "episode_return_success_threshold": None,
                "total_steps": int(policy_runner.total_steps),
                "actor_version": int(agent_bundle.metadata.get("actor_updates", 0)),
                "run_elapsed_total_s": float(time.time() - eval_start_time),
                "exploration_primitive_chance_runtime": float(primitive_selector.chance),
                **task_metrics,
            },
        )

        # 4. Reset between episodes. Rewind the scripted goal first when this
        # attempt was discarded or hit an e-stop with --rewind-goal-on-estop.
        should_rewind_goal = _maybe_rewind_goal_sequence_after_episode(
            env,
            episode_kept=episode_kept,
            rewind_on_estop=rewind_on_estop,
            had_estop=had_estop,
            n_paddle_puck_hits=n_paddle_puck_hits,
        )
        kind = pick_reset_kind(
            total_attempts,
            StopFlags(
                had_stop=result.terminal.stop_flags.had_stop,
                had_protective_stop=result.terminal.stop_flags.had_protective_stop,
                had_controller_disconnect=result.terminal.stop_flags.had_controller_disconnect,
            ),
        )
        reset_result = reset_runner.run(
            kind=kind,
            artifact_episode_id=next_reset_file_id,
            episode_had_stop_flags=StopFlags(
                had_stop=result.terminal.stop_flags.had_stop,
                had_protective_stop=result.terminal.stop_flags.had_protective_stop,
                had_controller_disconnect=result.terminal.stop_flags.had_controller_disconnect,
            ),
            episode_end_wall_time=episode_end_wall_time,
            pending_reset_artifact=pending_reset_artifact,
            next_reset_file_id=next_reset_file_id,
            compensate_goal_sequence_before_soft_prime=(
                should_rewind_goal and kind != ResetKind.SOFT
            ),
        )
        _log_goal_sequence_state(env, label="after inter-episode reset")
        counters["reset_fsm_steps_total"] += reset_result.total_fsm_steps
        pending_reset_artifact = reset_result.pending_reset_artifact
        next_reset_file_id = reset_result.next_reset_file_id

        transition_hold.begin(
            reason=reset_result.transition_reason,
            hold_steps=int(args.transition_hold_steps_post_reset),
            sim_hold=True,
            env=env,
            ctx=ctx,
            primitive_selector=primitive_selector,
            reset_primitive_rollout_state=_reset_primitive_rollout_state,
            use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
            device=device,
        )
        policy_runner.seed_after_reset(reset_result.obs)

    eval_finished_time = time.time()
    eval_finished_iso = datetime.fromtimestamp(eval_finished_time, tz=timezone.utc).isoformat()
    aggregate = compute_eval_aggregate(
        per_episode_records,
        numeric_fields=task_hooks.numeric_series_fields,
        rate_fields=task_hooks.rate_fields,
    )

    run_meta: Dict[str, Any] = {
        "agent": str(eval_args.agent),
        "model_path": str(args.model_path) if args.model_path is not None else None,
        "config": str(args.config),
        "args_file": str(args.args_file) if args.args_file is not None else None,
        "train_args_file": str(args.train_args) if args.train_args is not None else None,
        "run_data_dir": str(run_data_dir_from_args(args)),
        "seed": int(args.seed),
        "eval_episodes": int(eval_args.eval_episodes),
        "restart_eval_from_episode": int(restart_from_episode),
        "rewind_goal_on_estop": bool(eval_args.rewind_goal_on_estop),
        "n_target_episodes": int(target_kept),
        "n_attempts": int(total_attempts),
        "n_kept": int(len(per_episode_records)),
        "n_discarded": int(total_attempts - len(per_episode_records)),
        "started_iso": eval_started_iso,
        "finished_iso": eval_finished_iso,
        "elapsed_s": float(eval_finished_time - eval_start_time),
        "checkpoint_q_updates": int(agent_bundle.metadata.get("q_updates", 0)),
        "checkpoint_actor_updates": int(agent_bundle.metadata.get("actor_updates", 0)),
        "agent_metadata": dict(agent_bundle.metadata),
        "full_checkpoint_load": str(args.full_checkpoint_load),
        "residual_scale": float(args.residual_scale),
        "policy_obs_dim": int(obs_dim),
        "policy_act_dim": int(act_dim),
        "action_y_scaling": action_y_scaling,
        "counters_at_finish": dict(counters),
    }
    summary_path = Path(run_data_dir_from_args(args)) / eval_args.eval_summary_filename
    write_eval_summary_json(
        summary_path,
        run_meta=run_meta,
        aggregate=aggregate,
        per_episode=per_episode_records,
    )
    print(f"[eval] wrote summary: {summary_path}")
    print(format_eval_summary_console(
        aggregate,
        n_target=target_kept,
        n_attempts=total_attempts,
        n_discarded=total_attempts - len(per_episode_records),
        numeric_fields=task_hooks.numeric_series_fields,
        rate_fields=task_hooks.rate_fields,
        field_format_overrides=task_hooks.field_format_overrides,
    ))

    env.close()
    return {"run_meta": run_meta, "aggregate": aggregate, "per_episode": per_episode_records}


# ---------------------------------------------------------------------------
# JSONL helper local to this entrypoint — mirrors run_event_log's pattern but
# writes the eval-specific filename so analysts can find the eval set with
# one ls. (We still fan out to episode_summaries.jsonl too, in run_eval.)
# ---------------------------------------------------------------------------


def _append_eval_per_episode_row(
    args: Args, eval_args: EvalSpecificArgs, record: Dict[str, Any]
) -> None:
    import json

    path = Path(run_data_dir_from_args(args)) / eval_args.eval_per_episode_filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, default=str)
            f.write("\n")
    except Exception:
        print(
            f"[eval] failed to append eval per-episode row to {path}:\n"
            f"{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# main + __main__
# ---------------------------------------------------------------------------


def main(args: Args, train_args: TrainArgs, eval_args: EvalSpecificArgs) -> None:
    # Line-buffer stdout so early diagnostic prints reach the log before env
    # construction blocks (e.g. real-robot RTDE wait can sit indefinitely on
    # the controller-start prompt — without flushing the user can't see what
    # config / model / args were resolved).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if eval_args.quiet:
        prefixes, substrs = install_quiet_print_filter()
        print(
            "[eval_quiet] suppressing per-step/per-reset debug prints "
            "(pass --verbose to restore). "
            f"prefixes={list(prefixes)} substrings={list(substrs)}"
        )
    _force_eval_mode(args)
    if eval_args.eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_args.eval_episodes}")
    _validate_restart_eval_args(eval_args)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    probe_config = _prepare_air_hockey_config(config, seed=args.seed, return_goal_obs=False)
    probe_config["simulator"] = "box2d"
    probe_sim_params = dict(probe_config.get("simulator_params", {}))
    for key in ("control_mode", "wait_for_space_to_start", "save_path", "debug_control", "debug_control_every"):
        probe_sim_params.pop(key, None)
    probe_config["simulator_params"] = probe_sim_params
    probe_env = AirHockeyEnv(probe_config)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    action_low_np = np.asarray(probe_env.action_space.low, dtype=np.float32)
    action_high_np = np.asarray(probe_env.action_space.high, dtype=np.float32)
    probe_env.close()

    print(
        "[eval_run] "
        f"model_path={args.model_path} "
        f"eval_episodes={eval_args.eval_episodes} "
        f"restart_eval_from_episode={eval_args.restart_eval_from_episode} "
        f"rewind_goal_on_estop={eval_args.rewind_goal_on_estop} "
        f"eval_max_attempts={eval_args.eval_max_attempts} "
        f"config={args.config} "
        f"seed={args.seed} "
        f"obs_dim={obs_dim} act_dim={act_dim}"
    )
    print(
        "[run_event_log] writing per-run JSONL streams to:\n"
        f"    episodes : {episode_summaries_path(args)}\n"
        f"    resets   : {reset_summaries_path(args)}\n"
        f"    events   : {run_events_path(args)}"
    )
    append_run_event(
        args,
        "run_start",
        run_data_dir=str(run_data_dir_from_args(args)),
        run_name=str(getattr(args, "run_name", "")),
        seed=int(getattr(args, "seed", 0)),
        args_file=str(getattr(args, "args_file", "")) if getattr(args, "args_file", None) else None,
        train_args=str(getattr(args, "train_args", "")) if getattr(args, "train_args", None) else None,
        config=str(getattr(args, "config", "")) if getattr(args, "config", None) else None,
        model_path=str(getattr(args, "model_path", "")) if getattr(args, "model_path", None) else None,
        # Agent kind from --agent (e.g. "td3", "sgcrl", "iwr"). Pinned at the
        # run_start row so a JSONL-only reader knows exactly which actor class
        # was loaded for this run without having to open eval_summary.json.
        agent=str(eval_args.agent),
        mode="eval",
        eval_episodes=int(eval_args.eval_episodes),
        eval_max_attempts=int(eval_args.eval_max_attempts),
        restart_eval_from_episode=int(eval_args.restart_eval_from_episode),
        rewind_goal_on_estop=bool(eval_args.rewind_goal_on_estop),
    )

    eval_outcome_reason = "completed"
    payload: Dict[str, Any] = {}
    try:
        payload = run_eval(
            args=args,
            train_args=train_args,
            eval_args=eval_args,
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low_np=action_low_np,
            action_high_np=action_high_np,
        )
    except KeyboardInterrupt:
        print("[eval] interrupted by user; partial summary may be on disk.")
        eval_outcome_reason = "keyboard_interrupt"
    except BaseException:
        eval_outcome_reason = "exception"
        raise
    finally:
        append_run_event(
            args,
            "eval_done",
            reason=eval_outcome_reason,
            n_target_episodes=int(eval_args.eval_episodes),
            n_kept=int(len(payload.get("per_episode", []))) if payload else 0,
            n_attempts=int(payload.get("run_meta", {}).get("n_attempts", 0)) if payload else 0,
            elapsed_s=float(payload.get("run_meta", {}).get("elapsed_s", 0.0)) if payload else 0.0,
        )


if __name__ == "__main__":
    # Strip eval-specific flags from sys.argv before tyro sees it. EvalArgs
    # cannot inherit from Args (Python 3.9 + ``str | None`` annotations
    # under PEP 563 break tyro's get_type_hints walk on subclasses), so we
    # parse the small extra group with argparse and let tyro work on the
    # unmodified Args.
    eval_args = _parse_eval_specific_args()

    temp_args = tyro.cli(Args)
    if eval_args.agent == "td3":
        # TD3 path: --train-args (architecture) + --args-file (online-behavior
        # defaults) are both required. Same contract as before the agent
        # refactor — a TD3 checkpoint can't be rebuilt without them.
        if temp_args.train_args is None:
            raise SystemExit(
                "--agent td3 requires --train-args pointing to the "
                "training run's args.yaml (the same file the training run used). "
                "Architecture must match the saved checkpoint."
            )
        if temp_args.args_file is None:
            raise SystemExit(
                "--agent td3 requires --args-file pointing to an online-behavior "
                "YAML (e.g. td3_residual.yaml). Same args-file the training run "
                "used is fine — exploration knobs are forced to zero regardless "
                "of what the file specifies."
            )
        train_args = _load_train_args(temp_args.train_args)
        mapped_defaults, applied_keys, ignored_keys = _build_args_file_defaults(
            temp_args.args_file
        )
        mapped_defaults["args_file"] = temp_args.args_file
        mapped_defaults["train_args"] = temp_args.train_args
        default_args = Args(**mapped_defaults)

        args = tyro.cli(Args, default=default_args)
        print(f"[train_args] loaded architecture from: {args.train_args}")
        print(f"[args_file] loaded defaults from: {args.args_file}")
        if applied_keys:
            print("[args_file] applied keys:", ", ".join(applied_keys))
        if ignored_keys:
            print("[args_file] ignored unsupported keys:", ", ".join(ignored_keys))
    else:
        # Non-TD3 path (e.g. --agent sgcrl): the architecture lives inside the
        # agent's own checkpoint, so --train-args / --args-file are not
        # required. We still want all other CLI flags (config, model-path,
        # collector-device, …) to flow through tyro normally; ``temp_args``
        # already captured them, so use it directly as the resolved Args
        # without re-applying any YAML defaults.
        args = temp_args
        if args.model_path is None:
            raise SystemExit(
                f"--agent {eval_args.agent} requires --model-path pointing to a "
                f"checkpoint file."
            )
        train_args = synthesize_eval_train_args(use_last_action=False)
        print(
            f"[train_args] synthesized for --agent {eval_args.agent} "
            f"(use_last_action_in_policy_state=False)"
        )

    # No interactive run-note prompt — eval is meant to run unattended.
    _validate_restart_eval_args(eval_args)
    _setup_run_data_dir(args, run_note="")
    main(args, train_args, eval_args)
