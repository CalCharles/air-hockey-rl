"""Helpers for real-collector episode buffers (truncation, row shaping).

See notes/docs/environments/real-world/episode-lifecycle.md for the truncation flow.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_episode_collection import EpisodeTrajectory


def vector_with_width(values: np.ndarray | list | tuple, width: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros((int(width),), dtype=np.float64)
    copy_width = min(int(width), int(vector.shape[0]))
    if copy_width > 0:
        out[:copy_width] = vector[:copy_width]
    return out


def _truncate_episode_trajectory_inplace(episode_trajectory: EpisodeTrajectory, keep_count: int) -> int:
    """Keep the first keep_count transitions and drop the rest."""
    keep_count = max(0, int(keep_count))
    original_count = len(episode_trajectory.observations)
    if original_count <= keep_count:
        return 0
    episode_trajectory.observations = episode_trajectory.observations[:keep_count]
    episode_trajectory.next_observations = episode_trajectory.next_observations[:keep_count]
    episode_trajectory.actions = episode_trajectory.actions[:keep_count]
    episode_trajectory.task_rewards = episode_trajectory.task_rewards[:keep_count]
    episode_trajectory.motion_rewards = episode_trajectory.motion_rewards[:keep_count]
    episode_trajectory.dones = episode_trajectory.dones[:keep_count]
    episode_trajectory.bootstrap_terminals = episode_trajectory.bootstrap_terminals[:keep_count]
    episode_trajectory.prev_actions = episode_trajectory.prev_actions[:keep_count]
    if keep_count > 0:
        episode_trajectory.episode_return = float(
            torch.stack(episode_trajectory.task_rewards, dim=0).sum().item()
        )
    else:
        episode_trajectory.episode_return = 0.0
    return int(original_count - keep_count)


def truncate_collector_episode_for_readiness_fail(
    episode_trajectory: EpisodeTrajectory,
    episode_readiness_first_fail_step_idx: int,
    episode_rows: List[Any],
    episode_images: List[np.ndarray],
    episode_puck_detection_latency_ms: List[float],
    episode_model_inference_latency_ms: List[float],
    episode_block_sleep_latency_ms: List[float],
    episode_other_latency_ms: List[float],
    episode_camera_null_frames: int,
    device: torch.device,
) -> tuple[
    int,
    List[Any],
    List[np.ndarray],
    List[float],
    List[float],
    List[float],
    List[float],
    int,
]:
    keep_count = min(
        len(episode_trajectory.observations),
        max(0, int(episode_readiness_first_fail_step_idx) + 1),
    )
    readiness_fail_dropped_steps = _truncate_episode_trajectory_inplace(
        episode_trajectory,
        keep_count=keep_count,
    )
    if keep_count > 0 and len(episode_trajectory.dones) >= keep_count:
        episode_trajectory.dones[keep_count - 1] = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        )
    if keep_count > 0 and len(episode_trajectory.bootstrap_terminals) >= keep_count:
        episode_trajectory.bootstrap_terminals[keep_count - 1] = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        )
    if len(episode_rows) > keep_count:
        episode_rows = episode_rows[:keep_count]
    if len(episode_images) > keep_count:
        episode_images = episode_images[:keep_count]
    if len(episode_puck_detection_latency_ms) > keep_count:
        episode_puck_detection_latency_ms = episode_puck_detection_latency_ms[:keep_count]
    if len(episode_model_inference_latency_ms) > keep_count:
        episode_model_inference_latency_ms = episode_model_inference_latency_ms[:keep_count]
    if len(episode_block_sleep_latency_ms) > keep_count:
        episode_block_sleep_latency_ms = episode_block_sleep_latency_ms[:keep_count]
    if len(episode_other_latency_ms) > keep_count:
        episode_other_latency_ms = episode_other_latency_ms[:keep_count]
    new_camera_null_frames = max(
        0,
        int(episode_camera_null_frames) - int(readiness_fail_dropped_steps),
    )
    if keep_count > 0 and len(episode_rows) >= keep_count:
        cutoff_row = dict(episode_rows[keep_count - 1])
        cutoff_row["estop"] = np.array([1.0], dtype=np.float64)
        stop_flags = np.asarray(
            cutoff_row.get("stop_flags", np.zeros((3,), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if stop_flags.shape[0] < 3:
            stop_flags = vector_with_width(stop_flags, 3)
        stop_flags[2] = 1.0
        cutoff_row["stop_flags"] = stop_flags
        episode_rows[keep_count - 1] = cutoff_row
    return (
        readiness_fail_dropped_steps,
        episode_rows,
        episode_images,
        episode_puck_detection_latency_ms,
        episode_model_inference_latency_ms,
        episode_block_sleep_latency_ms,
        episode_other_latency_ms,
        new_camera_null_frames,
    )
