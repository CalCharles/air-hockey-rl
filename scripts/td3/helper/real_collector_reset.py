"""Reset-path helpers for the real TD3 collector (FSM artifact bookkeeping, soft re-init)."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class _ResetArtifactView(Protocol):
    """Minimal surface used for merge / log lines (matches PendingResetArtifact)."""

    episode_id: int
    partition: str
    done_reason: str
    step_count: int


def merge_reset_fsm_artifact_into_pending(
    artifact: _ResetArtifactView | None,
    pending_reset_artifact: _ResetArtifactView | None,
    next_reset_file_id: int,
    *,
    startup_buffered_message: bool,
) -> tuple[_ResetArtifactView | None, int]:
    """Apply reset FSM artifact to pending buffer and bump file id when rows were recorded.

    Preserves warning/print semantics: optional overwrite warning, then buffered line
    (startup vs episode wording).
    """
    if artifact is None:
        return pending_reset_artifact, next_reset_file_id
    if pending_reset_artifact is not None:
        print(
            "[collector_reset_artifact] "
            f"WARNING: overwriting unsaved pending reset artifact "
            f"episode_id={pending_reset_artifact.episode_id}"
        )
    pending_reset_artifact = artifact
    next_reset_file_id += 1
    if startup_buffered_message:
        print(
            "[collector_reset_artifact] "
            f"buffered startup episode_id={pending_reset_artifact.episode_id} "
            f"partition={pending_reset_artifact.partition} "
            f"reason={pending_reset_artifact.done_reason} "
            f"steps={pending_reset_artifact.step_count}"
        )
    else:
        print(
            "[collector_reset_artifact] "
            f"buffered episode_id={pending_reset_artifact.episode_id} "
            f"partition={pending_reset_artifact.partition} "
            f"reason={pending_reset_artifact.done_reason} "
            f"steps={pending_reset_artifact.step_count}"
        )
    return pending_reset_artifact, next_reset_file_id


def soft_reset_and_prime_paddle(
    env: Any,
    *,
    prime_paddle_history_stand_still_non_occluded: Callable[[Any], Any],
    post_soft_reset_hook: Callable[[Any], None] | None = None,
) -> Any:
    """env.soft_reset, then prime paddle history; returns the primed observation.

    ``post_soft_reset_hook`` (optional) runs between ``env.soft_reset()`` and
    the priming call — the slot task-specific eval hooks use to resample a
    goal, so the primed observation already carries the new goal. ``None``
    (the default, and what the training collector passes) keeps the
    historical two-step sequence bit-identical.
    """
    env.soft_reset()
    if post_soft_reset_hook is not None:
        post_soft_reset_hook(env)
    return prime_paddle_history_stand_still_non_occluded(env)
