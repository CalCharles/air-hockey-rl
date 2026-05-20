"""Task-specific eval hooks for the real-world TD3 eval pipeline.

The eval orchestrator (``extras/async_td3_real_eval.py``) is task-agnostic
in its episode loop — termination is driven by the env's ``terminations`` /
``truncations`` / ``step_info``. The hooks plugged in here supply the
task-specific bits the orchestrator can't know:

  * which extra per-episode metrics to compute (juggles / contacts / …);
  * which numeric and rate fields to summarize in ``eval_summary.json``;
  * the per-task minimum episode length for ``clean_episode_hdf5``;
  * any per-field precision overrides for the printed console summary.

Two implementations ship:

  * ``JuggleEvalHooks``  — juggle-success eval (the historical default).
                            Bit-identical to the pre-refactor pipeline.
  * ``GenericEvalHooks`` — task-agnostic baseline (no task-specific
                            metrics). Plugs in for any new task without
                            edits to the eval orchestrator.

A registry keyed on the ``task`` field of the sim YAML picks one. Unknown
tasks fall through to ``GenericEvalHooks``, so adding a new task to the
eval pipeline only requires registering a hooks class when the task has
something beyond ``episode_return`` / ``episode_success`` worth tracking.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

from .juggle_counter import count_juggles_from_rows


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

    def compute_episode_metrics(
        self, *, result: Any, rows: list
    ) -> Dict[str, Any]:
        """Return task-specific fields to splat into the per-episode record
        and the ``episode_summaries.jsonl`` row. Keys must include every
        task-specific entry in ``numeric_series_fields`` + ``rate_fields``."""

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        """Per-episode console fragment appended after ``return=…`` in the
        ``[eval] kept …`` line. Return ``""`` to add nothing."""


# ---------------------------------------------------------------------------
# Juggle (historical default).
# ---------------------------------------------------------------------------


class JuggleEvalHooks:
    """Hooks for the puck-juggle family of tasks.

    Computes paddle-puck contacts + long-direction-flip juggles using
    ``helper.juggle_counter`` and exposes the same fields the eval pipeline
    has tracked since the juggle-only era. Juggle eval output is bit-identical
    before and after the refactor: same record keys, same summary fields,
    same console precision, same 50-step ``min_timesteps`` floor.
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
        self, *, result: Any, rows: list
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


class GenericEvalHooks:
    """Default for any task not in the registry.

    Emits no task-specific fields; the eval summary reduces to
    ``episode_return`` / ``episode_*_reward`` / ``episode_length`` plus the
    standard rate fields (``episode_success``, e-stop flags). Plenty of
    signal for a first-pass eval on a new task; register a richer hooks
    class once you know what to measure.
    """

    numeric_series_fields: Tuple[str, ...] = BASE_NUMERIC_SERIES_FIELDS
    rate_fields: Tuple[str, ...] = BASE_RATE_FIELDS
    field_format_overrides: Dict[str, Tuple[str, str, str, str]] = {}
    # 10 is a permissive floor — short success-terminating tasks
    # (puck_strike, paddle_reach_position) routinely end well before
    # juggle's 50.
    min_timesteps: int = 10

    def compute_episode_metrics(
        self, *, result: Any, rows: list
    ) -> Dict[str, Any]:
        return {}

    def format_kept_console_extras(self, metrics: Dict[str, Any]) -> str:
        return ""


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


def get_task_eval_hooks(task: str) -> TaskEvalHooks:
    """Return a hooks instance for the given task name.

    Falls back to ``GenericEvalHooks`` when the task is not registered, so
    plugging a new task into the eval pipeline works without registry edits
    (you only register hooks when you want task-specific metrics).
    """
    cls = TASK_EVAL_HOOKS.get(str(task), GenericEvalHooks)
    return cls()
