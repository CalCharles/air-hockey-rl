"""Rolling window (rolling50) metrics shared by episode-end and periodic collector logging."""

from __future__ import annotations

from typing import MutableMapping, NamedTuple, Sequence

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def rolling_mean(values: Sequence[float]) -> float:
    if len(values) <= 0:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


class Rolling50Metrics(NamedTuple):
    window_count: int
    task_reward_avg: float
    motion_reward_avg: float
    episode_length_avg: float
    estop_episode_count: float


def compute_rolling50_metrics(
    rolling50_task_reward_values: Sequence[float],
    rolling50_motion_reward_values: Sequence[float],
    rolling50_episode_length_values: Sequence[float],
    rolling50_estop_episode_flags: Sequence[float],
) -> Rolling50Metrics:
    window_count = len(rolling50_episode_length_values)
    return Rolling50Metrics(
        window_count=window_count,
        task_reward_avg=rolling_mean(rolling50_task_reward_values),
        motion_reward_avg=rolling_mean(rolling50_motion_reward_values),
        episode_length_avg=rolling_mean(rolling50_episode_length_values),
        estop_episode_count=float(sum(rolling50_estop_episode_flags)),
    )


def update_stats_dict_rolling50(
    stats: MutableMapping[str, object],
    m: Rolling50Metrics,
    *,
    window_size: int,
    rolling50_task_reward_values: Sequence[float],
    rolling50_motion_reward_values: Sequence[float],
    rolling50_episode_length_values: Sequence[float],
    rolling50_estop_episode_flags: Sequence[float],
) -> None:
    stats["rolling50_window_size"] = float(window_size)
    stats["rolling50_window_count"] = float(m.window_count)
    stats["rolling50_task_reward_avg"] = float(m.task_reward_avg)
    stats["rolling50_motion_reward_avg"] = float(m.motion_reward_avg)
    stats["rolling50_episode_length_avg"] = float(m.episode_length_avg)
    stats["rolling50_estop_episode_count"] = float(m.estop_episode_count)
    stats["rolling50_task_reward_values"] = list(rolling50_task_reward_values)
    stats["rolling50_motion_reward_values"] = list(rolling50_motion_reward_values)
    stats["rolling50_episode_length_values"] = list(rolling50_episode_length_values)
    stats["rolling50_estop_episode_flags"] = list(rolling50_estop_episode_flags)


def write_rolling50_tensorboard_scalars(
    writer: SummaryWriter,
    m: Rolling50Metrics,
    total_steps: int,
) -> None:
    writer.add_scalar("rolling50/task_reward_avg", float(m.task_reward_avg), total_steps)
    writer.add_scalar("rolling50/motion_reward_avg", float(m.motion_reward_avg), total_steps)
    writer.add_scalar("rolling50/episode_length_avg", float(m.episode_length_avg), total_steps)
    writer.add_scalar("rolling50/estop_episode_count", float(m.estop_episode_count), total_steps)
    writer.add_scalar("rolling50/window_count", float(m.window_count), total_steps)
