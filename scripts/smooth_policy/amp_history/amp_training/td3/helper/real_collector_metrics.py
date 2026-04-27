"""Rolling window metrics shared by episode-end and periodic collector logging.

See notes/docs/environments/real-world/episode-lifecycle.md for the metrics context.

Each numeric series tracked over the rolling window (task reward, motion
reward, episode length, total episode return) is summarized as a
``SeriesStats`` — count, average, min, max, standard deviation, median.
This lets the user read both *level* (``avg`` / ``median``) and *spread*
(``min`` / ``max`` / ``std``) of the policy's recent performance from
TensorBoard or the console without having to inspect the raw window.

Stats are emitted at multiple window sizes (default ``ROLLING_WINDOW_SIZES
= (5, 10, 25, 50)``) so the user can compare short-horizon
responsiveness (``rolling5``) against long-horizon trend
(``rolling50``). Internally the orchestrator only keeps a single
maxlen-50 deque per series; smaller windows are derived by slicing the
last N entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import numpy as np
from torch.utils.tensorboard import SummaryWriter


# Default window sizes emitted to TensorBoard / stats / console. Listed
# smallest → largest so logs read in increasing-stability order.
ROLLING_WINDOW_SIZES: tuple[int, ...] = (5, 10, 25, 50)


def rolling_mean(values: Sequence[float]) -> float:
    if len(values) <= 0:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


@dataclass(frozen=True)
class SeriesStats:
    """Distribution summary for one rolling-window numeric series.

    All fields are 0.0 for an empty window; this keeps TensorBoard /
    console output unobtrusive before any episode has been recorded.
    ``std`` is the population standard deviation (numpy default).
    """

    count: int
    avg: float
    min: float
    max: float
    std: float
    median: float

    @classmethod
    def empty(cls) -> "SeriesStats":
        return cls(count=0, avg=0.0, min=0.0, max=0.0, std=0.0, median=0.0)


def compute_series_stats(values: Sequence[float]) -> SeriesStats:
    if len(values) <= 0:
        return SeriesStats.empty()
    arr = np.asarray(values, dtype=np.float32)
    return SeriesStats(
        count=int(arr.shape[0]),
        avg=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        std=float(np.std(arr)),
        median=float(np.median(arr)),
    )


@dataclass(frozen=True)
class RollingWindowMetrics:
    """Distribution stats for a single rolling window of size ``window_size``.

    ``window_count`` is the actual number of episodes summarized — equal
    to ``window_size`` once the window has filled, less otherwise.
    """

    window_size: int
    window_count: int
    task_reward: SeriesStats
    motion_reward: SeriesStats
    episode_length: SeriesStats
    episode_return: SeriesStats
    estop_episode_count: float

    # Backwards-compat scalar aliases used by collector log lines that
    # were written before per-series distribution stats existed.
    @property
    def task_reward_avg(self) -> float:
        return self.task_reward.avg

    @property
    def motion_reward_avg(self) -> float:
        return self.motion_reward.avg

    @property
    def episode_length_avg(self) -> float:
        return self.episode_length.avg


# Legacy alias — the metrics object originally tracked only the rolling-50
# window; today it can represent any window size. Kept so that callers
# that imported ``Rolling50Metrics`` continue to type-check.
Rolling50Metrics = RollingWindowMetrics


def _last_n(values: Sequence[float] | None, n: int) -> list[float]:
    if values is None:
        return []
    seq = list(values)
    if len(seq) <= n:
        return seq
    return seq[-n:]


def compute_rolling_window_metrics(
    *,
    window_size: int,
    task_reward_values: Sequence[float],
    motion_reward_values: Sequence[float],
    episode_length_values: Sequence[float],
    estop_episode_flags: Sequence[float],
    episode_return_values: Sequence[float] | None = None,
) -> RollingWindowMetrics:
    """Compute distribution stats over the most-recent ``window_size`` items.

    Inputs may hold up to ``max(ROLLING_WINDOW_SIZES)`` items (typically
    50); this helper slices the last ``window_size`` items from each
    series so all sizes can be derived from a single source-of-truth
    deque.
    """
    task_w = _last_n(task_reward_values, window_size)
    motion_w = _last_n(motion_reward_values, window_size)
    length_w = _last_n(episode_length_values, window_size)
    estop_w = _last_n(estop_episode_flags, window_size)
    return_w = _last_n(episode_return_values, window_size)
    return RollingWindowMetrics(
        window_size=int(window_size),
        window_count=len(length_w),
        task_reward=compute_series_stats(task_w),
        motion_reward=compute_series_stats(motion_w),
        episode_length=compute_series_stats(length_w),
        episode_return=compute_series_stats(return_w),
        estop_episode_count=float(sum(estop_w)),
    )


def compute_rolling_window_metrics_multi(
    *,
    task_reward_values: Sequence[float],
    motion_reward_values: Sequence[float],
    episode_length_values: Sequence[float],
    estop_episode_flags: Sequence[float],
    episode_return_values: Sequence[float] | None = None,
    window_sizes: Sequence[int] = ROLLING_WINDOW_SIZES,
) -> dict[int, RollingWindowMetrics]:
    """Compute per-window stats for several window sizes from one input."""
    return {
        int(n): compute_rolling_window_metrics(
            window_size=int(n),
            task_reward_values=task_reward_values,
            motion_reward_values=motion_reward_values,
            episode_length_values=episode_length_values,
            estop_episode_flags=estop_episode_flags,
            episode_return_values=episode_return_values,
        )
        for n in window_sizes
    }


# Backwards-compat wrapper for callers that only want the rolling-50 view.
def compute_rolling50_metrics(
    rolling50_task_reward_values: Sequence[float],
    rolling50_motion_reward_values: Sequence[float],
    rolling50_episode_length_values: Sequence[float],
    rolling50_estop_episode_flags: Sequence[float],
    rolling50_episode_return_values: Sequence[float] | None = None,
) -> RollingWindowMetrics:
    return compute_rolling_window_metrics(
        window_size=50,
        task_reward_values=rolling50_task_reward_values,
        motion_reward_values=rolling50_motion_reward_values,
        episode_length_values=rolling50_episode_length_values,
        estop_episode_flags=rolling50_estop_episode_flags,
        episode_return_values=rolling50_episode_return_values,
    )


def _series_to_stats_keys(prefix: str, s: SeriesStats) -> dict[str, float]:
    return {
        f"{prefix}_avg": float(s.avg),
        f"{prefix}_min": float(s.min),
        f"{prefix}_max": float(s.max),
        f"{prefix}_std": float(s.std),
        f"{prefix}_median": float(s.median),
    }


def _write_window_to_stats(
    stats: MutableMapping[str, object], m: RollingWindowMetrics, *, key_prefix: str
) -> None:
    stats[f"{key_prefix}_window_size"] = float(m.window_size)
    stats[f"{key_prefix}_window_count"] = float(m.window_count)
    stats[f"{key_prefix}_estop_episode_count"] = float(m.estop_episode_count)
    for series_prefix, series_stats in (
        (f"{key_prefix}_task_reward", m.task_reward),
        (f"{key_prefix}_motion_reward", m.motion_reward),
        (f"{key_prefix}_episode_length", m.episode_length),
        (f"{key_prefix}_episode_return", m.episode_return),
    ):
        for k, v in _series_to_stats_keys(series_prefix, series_stats).items():
            stats[k] = v


def update_stats_dict_rolling_windows(
    stats: MutableMapping[str, object],
    multi: Mapping[int, RollingWindowMetrics],
    *,
    raw_task_reward_values: Sequence[float],
    raw_motion_reward_values: Sequence[float],
    raw_episode_length_values: Sequence[float],
    raw_estop_episode_flags: Sequence[float],
    raw_episode_return_values: Sequence[float] | None = None,
) -> None:
    """Write per-window stats + raw-deque values into ``stats``.

    Only the largest window's raw values are persisted (under
    ``rolling50_*_values``) — smaller windows are derived from those at
    load time, so resume/checkpoint round-trip is unchanged.
    """
    for window_size, m in multi.items():
        _write_window_to_stats(stats, m, key_prefix=f"rolling{int(window_size)}")
    stats["rolling50_task_reward_values"] = list(raw_task_reward_values)
    stats["rolling50_motion_reward_values"] = list(raw_motion_reward_values)
    stats["rolling50_episode_length_values"] = list(raw_episode_length_values)
    stats["rolling50_estop_episode_flags"] = list(raw_estop_episode_flags)
    stats["rolling50_episode_return_values"] = list(raw_episode_return_values or ())


# Backwards-compat wrapper used by the legacy single-window collector.
def update_stats_dict_rolling50(
    stats: MutableMapping[str, object],
    m: RollingWindowMetrics,
    *,
    window_size: int,
    rolling50_task_reward_values: Sequence[float],
    rolling50_motion_reward_values: Sequence[float],
    rolling50_episode_length_values: Sequence[float],
    rolling50_estop_episode_flags: Sequence[float],
    rolling50_episode_return_values: Sequence[float] | None = None,
) -> None:
    _ = window_size  # always 50; retained for API compatibility
    _write_window_to_stats(stats, m, key_prefix="rolling50")
    stats["rolling50_task_reward_values"] = list(rolling50_task_reward_values)
    stats["rolling50_motion_reward_values"] = list(rolling50_motion_reward_values)
    stats["rolling50_episode_length_values"] = list(rolling50_episode_length_values)
    stats["rolling50_estop_episode_flags"] = list(rolling50_estop_episode_flags)
    stats["rolling50_episode_return_values"] = list(rolling50_episode_return_values or ())


def _write_window_to_tb(
    writer: SummaryWriter,
    m: RollingWindowMetrics,
    total_steps: int,
    *,
    tag_prefix: str,
) -> None:
    for series_prefix, series_stats in (
        (f"{tag_prefix}/task_reward", m.task_reward),
        (f"{tag_prefix}/motion_reward", m.motion_reward),
        (f"{tag_prefix}/episode_length", m.episode_length),
        (f"{tag_prefix}/episode_return", m.episode_return),
    ):
        writer.add_scalar(f"{series_prefix}_avg", float(series_stats.avg), total_steps)
        writer.add_scalar(f"{series_prefix}_min", float(series_stats.min), total_steps)
        writer.add_scalar(f"{series_prefix}_max", float(series_stats.max), total_steps)
        writer.add_scalar(f"{series_prefix}_std", float(series_stats.std), total_steps)
        writer.add_scalar(f"{series_prefix}_median", float(series_stats.median), total_steps)
    writer.add_scalar(f"{tag_prefix}/estop_episode_count", float(m.estop_episode_count), total_steps)
    writer.add_scalar(f"{tag_prefix}/window_count", float(m.window_count), total_steps)


def write_rolling_windows_tensorboard_scalars(
    writer: SummaryWriter,
    multi: Mapping[int, RollingWindowMetrics],
    total_steps: int,
) -> None:
    """Emit ``rolling{N}/<series>_<stat>`` scalars for every window in ``multi``.

    Each window groups under its own TensorBoard tag prefix
    (``rolling5``, ``rolling10``, ``rolling25``, ``rolling50``) so they
    can be overlaid on a single chart in the UI.
    """
    for window_size, m in multi.items():
        _write_window_to_tb(writer, m, total_steps, tag_prefix=f"rolling{int(window_size)}")


# Backwards-compat wrapper for the legacy single-window collector.
def write_rolling50_tensorboard_scalars(
    writer: SummaryWriter,
    m: RollingWindowMetrics,
    total_steps: int,
) -> None:
    _write_window_to_tb(writer, m, total_steps, tag_prefix="rolling50")


def format_rolling_window_console_line(m: RollingWindowMetrics) -> str:
    """One-line summary of a single window's distribution.

    Layout: ``count=<n> estops=<k> | <series>: avg=… std=… [min..max] median=… | …``
    The configured window size is conveyed by the caller's log prefix
    (``[collector_rolling<N>]``); ``count`` is the *actual* number of
    items so the reader can tell when the window is still warming up.
    """

    def _fmt_reward(s: SeriesStats) -> str:
        return (
            f"avg={s.avg:.3f} std={s.std:.3f} "
            f"[{s.min:.3f}..{s.max:.3f}] median={s.median:.3f}"
        )

    def _fmt_length(s: SeriesStats) -> str:
        return (
            f"avg={s.avg:.1f} std={s.std:.1f} "
            f"[{s.min:.0f}..{s.max:.0f}] median={s.median:.1f}"
        )

    return (
        f"count={int(m.window_count)} estops={m.estop_episode_count:.0f} "
        f"| return: {_fmt_reward(m.episode_return)} "
        f"| task: {_fmt_reward(m.task_reward)} "
        f"| motion: {_fmt_reward(m.motion_reward)} "
        f"| length: {_fmt_length(m.episode_length)}"
    )


# Backwards-compat alias (older orchestrator code referenced this name).
format_rolling50_console_line = format_rolling_window_console_line
