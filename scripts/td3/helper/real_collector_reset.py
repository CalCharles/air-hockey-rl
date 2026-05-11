"""Reset-path helpers for the real TD3 collector (FSM artifact bookkeeping, soft re-init)."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Tuple


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


def soft_reset_prime_paddle_and_extract_previous_puck(
    env: Any,
    *,
    device: Any,
    prime_paddle_history_stand_still_non_occluded: Callable[[Any], Any],
    extract_primitive_state_tensors: Callable[[Any, Any], Tuple[Any, Any, Any]],
) -> tuple[Any, Any]:
    """env.soft_reset, prime paddle history, then previous puck tensor for primitives."""
    obs, _ = env.soft_reset()
    obs = prime_paddle_history_stand_still_non_occluded(env)
    _, previous_puck_position_for_primitive, _ = extract_primitive_state_tensors(env, device=device)
    return obs, previous_puck_position_for_primitive
