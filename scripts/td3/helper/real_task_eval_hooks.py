"""Task-specific eval hooks for the real-world fixed-policy eval pipeline.

The eval orchestrator (``extras/async_td3_real_eval.py``) is task-agnostic
in its episode loop — termination is driven by the env's ``terminations`` /
``truncations`` / ``step_info``. The hooks plugged in here supply the
task-specific bits the orchestrator can't know:

**Metrics side**

  * which extra per-episode metrics to compute (juggles / goal distance / …);
  * which numeric and rate fields to summarize in ``eval_summary.json``;
  * the per-task minimum episode length for ``clean_episode_hdf5``;
  * any per-field precision overrides for the printed console summary.

**Reset side** (how the robot gets from one episode to the next)

  * ``make_reset_fsm_cls`` — the between-episode controller. Puck tasks use
    ``ResetPolicyFSM`` (sweep the puck up the table, hand over when it
    falls back). Paddle-only tasks use ``PaddleRepositionFSM`` (drive the
    paddle to a fresh start pose and settle — no puck, no reset policy).
  * ``on_soft_reset(env)`` — runs after every ``env.soft_reset()`` and
    before the paddle-history priming. Goal tasks resample their goal here
    (``soft_reset`` alone keeps the previous goal).
  * ``force_fsm_after_hard_reset`` — paddle-only tasks always run the
    reposition FSM after a physical ``env.reset()``; the puck-position
    heuristic that gates the FSM on the juggle path has no meaning there.
  * ``periodic_hard_reset_every`` — cadence of the periodic physical reset
    (``0`` disables it; stop-driven hard resets are unaffected).
  * ``post_reset_transition_hold_steps`` — override of the post-reset
    zero-action hold (``None`` → ``args.transition_hold_steps_post_reset``).
    Paddle-only tasks use ``0``: the FSM already ends with the paddle at
    rest, and a hold would eat into the short reach budget.
  * ``on_episode_start(env)`` — snapshot whatever the metrics need at
    episode start (goal tasks record the goal).

Five canonical tasks ship with a registered hooks class:

  ================================= ================================ ======================
  ``task:``                         hooks                            reset
  ================================= ================================ ======================
  ``puck_juggle_upper_half_reward`` ``JuggleEvalHooks``              ``ResetPolicyFSM``
  (and the rest of the juggle family)
  ``puck_touch``                    ``PuckTouchEvalHooks``           ``ResetPolicyFSM``
  ``puck_velocity``                 ``PuckVelocityEvalHooks``        ``ResetPolicyFSM``
  ``paddle_reach_position``         ``PaddleReachEvalHooks``         ``PaddleRepositionFSM``
  ``paddle_reach_position_velocity`` ``PaddleReachVelocityEvalHooks`` ``PaddleRepositionFSM``
  ================================= ================================ ======================

Unknown tasks fall through to ``GenericEvalHooks`` (puck FSM reset, bare
runner metrics), so plugging a new task into the eval pipeline only
requires registering a hooks class when the task needs something beyond
``episode_return`` / ``episode_success`` or a different reset.

``JuggleEvalHooks`` / ``GenericEvalHooks`` output is bit-identical to the
pre-refactor pipeline (same record keys, summary fields, console precision,
``min_timesteps``, and reset behaviour).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from .juggle_counter import CONTACT_THRESH, count_juggles_from_rows


# ---------------------------------------------------------------------------
# Base fields produced by the runner itself; every hooks impl includes them.
# ---------------------------------------------------------------------------


BASE_NUMERIC_SERIES_FIELDS: Tuple[str, ...] = (
    "episode_return",
    "episode_reward",
    "episode_length",
)
BASE_RATE_FIELDS: Tuple[str, ...] = (
    "episode_success",
    "had_protective_stop",
    "had_controller_disconnect",
    "readiness_fail_estop",
)


# Reset-strategy labels surfaced in logs / ``eval_summary.json`` run_meta.
RESET_STRATEGY_PUCK_FSM = "puck_reset_fsm"
RESET_STRATEGY_PADDLE_REPOSITION = "paddle_reposition"


# ---------------------------------------------------------------------------
# Protocol the eval entrypoint consumes.
# ---------------------------------------------------------------------------


class TaskEvalHooks(Protocol):
    """Plug-in surface for task-specific eval behavior."""

    # Fields summarized in ``compute_eval_aggregate.series`` / ``.rates``.
    # Must be a superset of (a) any task-specific keys returned by
    # ``compute_episode_metrics`` and (b) the runner-emitted fields the
    # task wants surfaced.
    numeric_series_fields: Tuple[str, ...]
    rate_fields: Tuple[str, ...]

    # Per-field ``(avg, lim, median, std)`` precision overrides for the
    # console summary. Empty dict = use the formatter's defaults.
    field_format_overrides: Dict[str, Tuple[str, str, str, str]]

    # Minimum episode length passed to ``clean_episode_hdf5`` before an
    # episode is kept. Juggle uses 50 (long-direction-flip window needs
    # ≥ 50 frames); shorter tasks can lower this.
    min_timesteps: int

    # --- reset side -------------------------------------------------------
    reset_strategy: str
    force_fsm_after_hard_reset: bool
    periodic_hard_reset_every: int
    post_reset_transition_hold_steps: Optional[int]

    def make_reset_fsm_cls(self) -> Any:
        """Return the ``(env, rng)`` factory ``ResetRunner`` drives between
        episodes."""

    def on_soft_reset(self, env: Any) -> None:
        """Called after ``env.soft_reset()`` and before paddle-history
        priming on every soft / FSM reset path."""

    def on_episode_start(self, env: Any) -> None:
        """Called right before each policy episode starts."""

    # --- metrics side -----------------------------------------------------
    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        """Return task-specific fields to splat into the per-episode record
        and the ``episode_summaries.jsonl`` row. Keys must include every
        task-specific entry in ``numeric_series_fields`` + ``rate_fields``."""

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        """Per-episode console fragment appended after ``return=…`` in the
        ``[eval] kept …`` line. Return ``""`` to add nothing."""


# ---------------------------------------------------------------------------
# Shared defaults.
# ---------------------------------------------------------------------------


class BaseTaskEvalHooks:
    """Defaults every hooks class inherits: puck-FSM reset, bare metrics.

    Subclasses override class attributes and the two metric methods; the
    reset-side defaults reproduce the historical juggle eval behaviour.
    """

    numeric_series_fields: Tuple[str, ...] = BASE_NUMERIC_SERIES_FIELDS
    rate_fields: Tuple[str, ...] = BASE_RATE_FIELDS
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {}
    min_timesteps: int = 10

    reset_strategy: str = RESET_STRATEGY_PUCK_FSM
    force_fsm_after_hard_reset: bool = False
    periodic_hard_reset_every: int = 3
    post_reset_transition_hold_steps: Optional[int] = None

    def make_reset_fsm_cls(self) -> Any:
        if self.reset_strategy == RESET_STRATEGY_PADDLE_REPOSITION:
            from .real_paddle_reposition_fsm import PaddleRepositionFSM

            return PaddleRepositionFSM
        # Lazy: the puck FSM module pulls in cv2 / the real-robot stack, and
        # tests + non-juggle callers shouldn't pay for that at import time.
        from scripts.real.rollout_reset_policy_real import ResetPolicyFSM

        return ResetPolicyFSM

    def on_soft_reset(self, env: Any) -> None:
        return None

    def on_episode_start(self, env: Any) -> None:
        return None

    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        return {}

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        return ""


# ---------------------------------------------------------------------------
# Row helpers (split-schema rows from ``_build_split_episode_row``).
# ---------------------------------------------------------------------------


def _rows_xy(rows: list, key: str) -> np.ndarray:
    """(T, 2) float array of ``rows[i][key][:2]``; empty (0, 2) when no rows."""
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    out = np.zeros((len(rows), 2), dtype=np.float64)
    for i, row in enumerate(rows):
        vec = np.asarray(row.get(key, [np.nan, np.nan]), dtype=np.float64).reshape(-1)
        out[i, 0] = vec[0] if vec.size > 0 else np.nan
        out[i, 1] = vec[1] if vec.size > 1 else np.nan
    return out


def _rows_puck_occluded(rows: list) -> np.ndarray:
    """(T,) bool array from the third ``puck`` column (1 = occluded)."""
    if not rows:
        return np.zeros((0,), dtype=bool)
    out = np.zeros((len(rows),), dtype=bool)
    for i, row in enumerate(rows):
        vec = np.asarray(row.get("puck", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
        out[i] = bool(vec[2] > 0.5) if vec.size > 2 else False
    return out


def _terminal_reasons(result: Any) -> list:
    terminal = getattr(result, "terminal", None)
    reasons = getattr(terminal, "episode_end_reasons", None)
    return list(reasons) if isinstance(reasons, (list, tuple)) else []


def _terminal_success(result: Any) -> bool:
    terminal = getattr(result, "terminal", None)
    return bool(getattr(terminal, "episode_success", False))


def _finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# Juggle (historical default).
# ---------------------------------------------------------------------------


class JuggleEvalHooks(BaseTaskEvalHooks):
    """Hooks for the puck-juggle family of tasks.

    Computes paddle-puck contacts + long-direction-flip juggles using
    ``helper.juggle_counter`` and exposes the same fields the eval pipeline
    has tracked since the juggle-only era. Juggle eval output is bit-identical
    before and after the refactor: same record keys, same summary fields,
    same console precision, same 50-step ``min_timesteps`` floor, same
    ``ResetPolicyFSM`` reset with the puck-position hard-reset heuristic.
    """

    numeric_series_fields: Tuple[str, ...] = (
        "episode_return",
        "episode_juggles",
        "episode_contacts",
        "episode_reward",
        "episode_length",
    )
    rate_fields: Tuple[str, ...] = (
        "episode_juggle_success",
        "episode_success",
        "had_protective_stop",
        "had_controller_disconnect",
        "readiness_fail_estop",
    )
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {
        "episode_juggles":  (".2f", ".0f", ".1f", ".2f"),
        "episode_contacts": (".2f", ".0f", ".1f", ".2f"),
    }
    min_timesteps: int = 50

    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        counts = count_juggles_from_rows(rows)
        return {
            "episode_juggles":        int(counts.n_juggles),
            "episode_contacts":       int(counts.n_contacts),
            "episode_juggle_success": bool(counts.juggle_success),
        }

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        return (
            f"juggles={int(metrics['episode_juggles'])} "
            f"contacts={int(metrics['episode_contacts'])}"
        )


# ---------------------------------------------------------------------------
# Task-agnostic default.
# ---------------------------------------------------------------------------


class GenericEvalHooks(BaseTaskEvalHooks):
    """Default for any task not in the registry.

    Emits no task-specific fields; the eval summary reduces to
    ``episode_return`` / ``episode_*_reward`` / ``episode_length`` plus the
    standard rate fields (``episode_success``, e-stop flags). Reset is the
    puck FSM. Plenty of signal for a first-pass eval on a new puck task;
    register a richer hooks class once you know what to measure (or when
    the task has no puck and needs ``PaddleRepositionFSM``).
    """

    # 10 is a permissive floor — short success-terminating tasks
    # (puck_strike, …) routinely end well before juggle's 50.
    min_timesteps: int = 10


# ---------------------------------------------------------------------------
# Puck tasks that reuse the juggle reset (puck falls from the top).
# ---------------------------------------------------------------------------


class PuckTouchEvalHooks(BaseTaskEvalHooks):
    """``puck_touch``: +1 on the step the paddle touches the puck.

    Reset: ``ResetPolicyFSM`` (identical to juggle — the puck has to come
    back down the table for the policy to intercept). Episodes end on the
    touch, so they are short; ``min_timesteps`` is relaxed to 5.

    Metrics:
      * ``episode_touched``      — env success OR any paddle-puck contact
                                   in the rows (juggle counter threshold).
      * ``episode_contacts``     — contact events (should be 0 or 1).
      * ``episode_steps_to_touch`` — 1-based row index of the first contact
                                   frame; ``None`` (omitted from series)
                                   when there was no touch.
    """

    numeric_series_fields: Tuple[str, ...] = (
        "episode_return",
        "episode_contacts",
        "episode_steps_to_touch",
        "episode_reward",
        "episode_length",
    )
    rate_fields: Tuple[str, ...] = (
        "episode_touched",
        "episode_success",
        "had_protective_stop",
        "had_controller_disconnect",
        "readiness_fail_estop",
    )
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {
        "episode_contacts":       (".2f", ".0f", ".1f", ".2f"),
        "episode_steps_to_touch": (".1f", ".0f", ".1f", ".1f"),
    }
    min_timesteps: int = 5

    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        counts = count_juggles_from_rows(rows)
        paddle = _rows_xy(rows, "pose")
        puck = _rows_xy(rows, "puck")
        occluded = _rows_puck_occluded(rows)
        steps_to_touch: Optional[int] = None
        if paddle.shape[0] > 0:
            dist = np.linalg.norm(paddle - puck, axis=1)
            hit = np.where(np.isfinite(dist) & (dist < CONTACT_THRESH) & (~occluded))[0]
            if hit.size > 0:
                steps_to_touch = int(hit[0]) + 1
        touched = bool(_terminal_success(result) or counts.n_contacts > 0)
        return {
            "episode_touched":        touched,
            "episode_contacts":       int(counts.n_contacts),
            "episode_steps_to_touch": steps_to_touch,
        }

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        steps = metrics.get("episode_steps_to_touch")
        return (
            f"touched={int(bool(metrics['episode_touched']))} "
            f"contacts={int(metrics['episode_contacts'])} "
            f"steps_to_touch={'-' if steps is None else int(steps)}"
        )


class PuckVelocityEvalHooks(BaseTaskEvalHooks):
    """``puck_velocity``: reward ∝ upward puck displacement per step.

    Reset: ``ResetPolicyFSM`` (same as juggle). Metrics are measured the
    way the reward is — from consecutive non-occluded puck positions in
    the rows (table x grows toward the robot, so upward = decreasing x):

      * ``episode_upward_displacement_m`` — Σ max(0, x_{t-1} − x_t) over
                                            consecutive visible frames.
      * ``episode_max_upward_step_m``     — largest single-step upward move
                                            (a proxy for peak puck speed
                                            right after the hit).
      * ``episode_contacts``              — paddle-puck contact events.
      * ``episode_hit_puck``              — contacts > 0.
    """

    numeric_series_fields: Tuple[str, ...] = (
        "episode_return",
        "episode_upward_displacement_m",
        "episode_max_upward_step_m",
        "episode_contacts",
        "episode_reward",
        "episode_length",
    )
    rate_fields: Tuple[str, ...] = (
        "episode_hit_puck",
        "episode_success",
        "had_protective_stop",
        "had_controller_disconnect",
        "readiness_fail_estop",
    )
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {
        "episode_upward_displacement_m": (".3f", ".3f", ".3f", ".3f"),
        "episode_max_upward_step_m":     (".3f", ".3f", ".3f", ".3f"),
        "episode_contacts":              (".2f", ".0f", ".1f", ".2f"),
    }
    min_timesteps: int = 5

    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        counts = count_juggles_from_rows(rows)
        puck = _rows_xy(rows, "puck")
        occluded = _rows_puck_occluded(rows)
        upward_total = 0.0
        upward_max = 0.0
        for t in range(1, puck.shape[0]):
            if occluded[t] or occluded[t - 1]:
                continue
            dx = float(puck[t - 1, 0] - puck[t, 0])
            if not np.isfinite(dx) or dx <= 0.0:
                continue
            upward_total += dx
            upward_max = max(upward_max, dx)
        return {
            "episode_upward_displacement_m": float(upward_total),
            "episode_max_upward_step_m":     float(upward_max),
            "episode_contacts":              int(counts.n_contacts),
            "episode_hit_puck":              bool(counts.n_contacts > 0),
        }

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        return (
            f"upward={float(metrics['episode_upward_displacement_m']):.3f}m "
            f"max_step={float(metrics['episode_max_upward_step_m']):.3f}m "
            f"contacts={int(metrics['episode_contacts'])}"
        )


# ---------------------------------------------------------------------------
# Paddle-only goal tasks (no puck → paddle reposition reset).
# ---------------------------------------------------------------------------


def _resample_goal(env: Any) -> None:
    """Draw a fresh goal on a goal-conditioned env and push it to the
    on-screen marker. No-op on envs without ``set_goals``."""
    set_goals = getattr(env, "set_goals", None)
    if not callable(set_goals):
        return
    set_goals(getattr(env, "goal_radius_type", None))
    sync = getattr(env, "_sync_goal_marker_to_simulator", None)
    if callable(sync):
        sync()


def _goal_reached_from_result(result: Any) -> bool:
    return bool(_terminal_success(result) or ("goal_reached" in _terminal_reasons(result)))


class PaddleReachEvalHooks(BaseTaskEvalHooks):
    """``paddle_reach_position``: +10 when the paddle centre enters the
    goal radius; the episode ends there.

    Reset: ``PaddleRepositionFSM`` — no puck, no reset policy. The paddle
    is driven to a uniformly random reachable start pose and settled, then
    ``on_soft_reset`` draws a fresh goal (``set_goals``) so the primed obs
    already carries it. Hard (physical) resets always run the reposition
    FSM too, and the post-reset zero-action hold is disabled: the paddle
    is already at rest and a hold would eat into the 50-step budget.

    Metrics (goal snapshotted in ``on_episode_start``):
      * ``episode_goal_reached``       — env success or ``goal_reached``
                                         termination.
      * ``episode_final_goal_dist_m``  — paddle-goal distance on the last row.
      * ``episode_min_goal_dist_m``    — closest approach over the episode.
      * ``episode_steps_to_goal``      — 1-based row index of first entry
                                         into the goal radius (``None``,
                                         omitted from series, if never).
      * ``goal_x`` / ``goal_y``        — the goal itself (record only).
    """

    numeric_series_fields: Tuple[str, ...] = (
        "episode_return",
        "episode_final_goal_dist_m",
        "episode_min_goal_dist_m",
        "episode_steps_to_goal",
        "episode_reward",
        "episode_length",
    )
    rate_fields: Tuple[str, ...] = (
        "episode_goal_reached",
        "episode_success",
        "had_protective_stop",
        "had_controller_disconnect",
        "readiness_fail_estop",
    )
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {
        "episode_final_goal_dist_m": (".3f", ".3f", ".3f", ".3f"),
        "episode_min_goal_dist_m":   (".3f", ".3f", ".3f", ".3f"),
        "episode_steps_to_goal":     (".1f", ".0f", ".1f", ".1f"),
    }
    # Success can land within a handful of steps from a nearby start.
    min_timesteps: int = 1

    reset_strategy: str = RESET_STRATEGY_PADDLE_REPOSITION
    force_fsm_after_hard_reset: bool = True
    periodic_hard_reset_every: int = 3
    post_reset_transition_hold_steps: Optional[int] = 0

    def __init__(self) -> None:
        self._goal: Optional[np.ndarray] = None
        self._goal_radius: Optional[float] = None

    # --- reset side ---
    def on_soft_reset(self, env: Any) -> None:
        _resample_goal(env)

    def on_episode_start(self, env: Any) -> None:
        self._goal = self._read_goal(env)
        radius = getattr(env, "goal_radius", None)
        self._goal_radius = float(radius) if radius is not None else None

    @staticmethod
    def _read_goal(env: Any) -> Optional[np.ndarray]:
        getter = getattr(env, "get_desired_goal", None)
        if not callable(getter):
            return None
        try:
            return np.asarray(getter(), dtype=np.float64).reshape(-1)
        except Exception:
            return None

    # --- metrics side ---
    def _position_metrics(self, rows: list, goal: Optional[np.ndarray]) -> Dict[str, Any]:
        paddle = _rows_xy(rows, "pose")
        out: Dict[str, Any] = {
            "episode_final_goal_dist_m": None,
            "episode_min_goal_dist_m": None,
            "episode_steps_to_goal": None,
        }
        if goal is None or goal.size < 2 or paddle.shape[0] == 0:
            return out
        dist = np.linalg.norm(paddle - goal[:2][None, :], axis=1)
        out["episode_final_goal_dist_m"] = _finite_or_none(dist[-1])
        finite = dist[np.isfinite(dist)]
        out["episode_min_goal_dist_m"] = float(np.min(finite)) if finite.size else None
        if self._goal_radius is not None:
            inside = np.where(np.isfinite(dist) & (dist <= self._goal_radius))[0]
            if inside.size > 0:
                out["episode_steps_to_goal"] = int(inside[0]) + 1
        return out

    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        goal = self._goal if self._goal is not None else self._read_goal(env)
        metrics = self._position_metrics(rows, goal)
        metrics["episode_goal_reached"] = _goal_reached_from_result(result)
        metrics["goal_x"] = float(goal[0]) if goal is not None and goal.size > 0 else None
        metrics["goal_y"] = float(goal[1]) if goal is not None and goal.size > 1 else None
        return metrics

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        final_d = metrics.get("episode_final_goal_dist_m")
        min_d = metrics.get("episode_min_goal_dist_m")
        steps = metrics.get("episode_steps_to_goal")
        return (
            f"goal_reached={int(bool(metrics['episode_goal_reached']))} "
            f"final_dist={'-' if final_d is None else f'{final_d:.3f}'}m "
            f"min_dist={'-' if min_d is None else f'{min_d:.3f}'}m "
            f"steps_to_goal={'-' if steps is None else int(steps)}"
        )


class PaddleReachVelocityEvalHooks(PaddleReachEvalHooks):
    """``paddle_reach_position_velocity``: +10 when the paddle is inside the
    goal radius *and* its velocity is within ``goal_velocity_radius`` of the
    goal velocity on the same step.

    Same reset as ``PaddleReachEvalHooks``. Adds velocity-side metrics from
    the rows' ``speed`` column (paddle velocity, m/s):

      * ``episode_final_goal_vel_dist_mps`` / ``episode_min_goal_vel_dist_mps``
      * ``episode_steps_to_goal`` — first row meeting BOTH tolerances.
      * ``goal_vx`` / ``goal_vy`` — record only.
    """

    numeric_series_fields: Tuple[str, ...] = (
        "episode_return",
        "episode_final_goal_dist_m",
        "episode_min_goal_dist_m",
        "episode_final_goal_vel_dist_mps",
        "episode_min_goal_vel_dist_mps",
        "episode_steps_to_goal",
        "episode_reward",
        "episode_length",
    )
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {
        **PaddleReachEvalHooks.field_format_overrides,
        "episode_final_goal_vel_dist_mps": (".3f", ".3f", ".3f", ".3f"),
        "episode_min_goal_vel_dist_mps":   (".3f", ".3f", ".3f", ".3f"),
    }

    def __init__(self) -> None:
        super().__init__()
        self._goal_velocity_radius: Optional[float] = None

    def on_episode_start(self, env: Any) -> None:
        super().on_episode_start(env)
        vel_radius = getattr(env, "goal_velocity_radius", None)
        self._goal_velocity_radius = float(vel_radius) if vel_radius is not None else None

    def compute_episode_metrics(
        self, *, result: Any, rows: list, env: Any = None
    ) -> Dict[str, Any]:
        goal = self._goal if self._goal is not None else self._read_goal(env)
        metrics = self._position_metrics(rows, goal)
        metrics["episode_final_goal_vel_dist_mps"] = None
        metrics["episode_min_goal_vel_dist_mps"] = None
        metrics["goal_vx"] = float(goal[2]) if goal is not None and goal.size > 2 else None
        metrics["goal_vy"] = float(goal[3]) if goal is not None and goal.size > 3 else None
        if goal is not None and goal.size >= 4 and rows:
            paddle = _rows_xy(rows, "pose")
            speed = _rows_xy(rows, "speed")
            pos_dist = np.linalg.norm(paddle - goal[:2][None, :], axis=1)
            vel_dist = np.linalg.norm(speed - goal[2:4][None, :], axis=1)
            metrics["episode_final_goal_vel_dist_mps"] = _finite_or_none(vel_dist[-1])
            finite = vel_dist[np.isfinite(vel_dist)]
            metrics["episode_min_goal_vel_dist_mps"] = (
                float(np.min(finite)) if finite.size else None
            )
            # Joint criterion overrides the position-only steps_to_goal.
            metrics["episode_steps_to_goal"] = None
            if self._goal_radius is not None and self._goal_velocity_radius is not None:
                ok = (
                    np.isfinite(pos_dist)
                    & np.isfinite(vel_dist)
                    & (pos_dist <= self._goal_radius)
                    & (vel_dist <= self._goal_velocity_radius)
                )
                hit = np.where(ok)[0]
                if hit.size > 0:
                    metrics["episode_steps_to_goal"] = int(hit[0]) + 1
        metrics["episode_goal_reached"] = _goal_reached_from_result(result)
        return metrics

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        base = super().format_kept_console_extras(metrics)
        vel_d = metrics.get("episode_final_goal_vel_dist_mps")
        return f"{base} final_vel_dist={'-' if vel_d is None else f'{vel_d:.3f}'}m/s"


# ---------------------------------------------------------------------------
# Registry + factory.
# ---------------------------------------------------------------------------


_JUGGLE_TASKS: Tuple[str, ...] = (
    "puck_juggle",
    "multipuck_juggle",
    "puck_juggle_linear_top",
    "multipuck_juggle_linear_top",
    "puck_juggle_no_base_reward",
    "multipuck_juggle_no_base_reward",
    "puck_juggle_upper_half_reward",
    "multipuck_juggle_upper_half_reward",
    "puck_juggle_pinball_triangle_sides",
    "multipuck_juggle_pinball_triangle_sides",
    "puck_juggle_upper_half_mid_band_reward",
    "multipuck_juggle_upper_half_mid_band_reward",
)

TASK_EVAL_HOOKS: Dict[str, type] = {task: JuggleEvalHooks for task in _JUGGLE_TASKS}
TASK_EVAL_HOOKS.update(
    {
        "puck_touch":                      PuckTouchEvalHooks,
        "puck_velocity":                   PuckVelocityEvalHooks,
        "paddle_reach_position":           PaddleReachEvalHooks,
        "paddle_reach_position_neg":       PaddleReachEvalHooks,
        "paddle_reach_position_velocity":  PaddleReachVelocityEvalHooks,
    }
)


def get_task_eval_hooks(task: str) -> TaskEvalHooks:
    """Return a hooks instance for the given task name.

    Falls back to ``GenericEvalHooks`` when the task is not registered, so
    plugging a new task into the eval pipeline works without registry edits
    (you only register hooks when you want task-specific metrics or a
    non-puck reset).
    """
    cls = TASK_EVAL_HOOKS.get(str(task), GenericEvalHooks)
    return cls()
