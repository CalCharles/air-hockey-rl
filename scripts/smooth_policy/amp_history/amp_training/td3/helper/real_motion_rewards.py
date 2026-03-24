"""Motion reward state and component computation for real TD3 training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


def velocity_reward_from_magnitude(
    velocity_mag: float,
    velocity_at_one: float,
    velocity_at_zero: float,
) -> float:
    denom = max(float(velocity_at_zero) - float(velocity_at_one), 1e-6)
    reward = 1.0 - (float(velocity_mag) - float(velocity_at_one)) / denom
    return min(reward, 1.0)


def jerk_reward_from_magnitude(
    jerk_mag: float,
    jerk_at_one: float,
    jerk_at_zero: float,
) -> float:
    denom = max(float(jerk_at_zero) - float(jerk_at_one), 1e-6)
    return 1.0 - (float(jerk_mag) - float(jerk_at_one)) / denom


@dataclass
class MotionRewardState:
    temporal_horizon: int
    paddle_history: deque[np.ndarray]
    puck_history: deque[np.ndarray]
    steps_since_reset: int = 0
    current_velocity_mag: float = 0.0
    current_acceleration_mag: float = 0.0
    current_jerk_mag: float = 0.0


def _finite_or_fallback(value: object, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if np.isfinite(out) else float(fallback)


def _extract_motion_positions_from_state_info(state_info: dict | None) -> tuple[np.ndarray, np.ndarray]:
    zero_xy = np.zeros((2,), dtype=np.float64)
    if not isinstance(state_info, dict):
        return zero_xy.copy(), zero_xy.copy()
    try:
        paddle_xy = np.asarray(
            state_info["paddles"]["paddle_ego"]["position"],
            dtype=np.float64,
        ).reshape(-1)[:2]
    except Exception:
        paddle_xy = zero_xy.copy()
    try:
        puck_xy = np.asarray(
            state_info["pucks"][0]["position"],
            dtype=np.float64,
        ).reshape(-1)[:2]
    except Exception:
        puck_xy = zero_xy.copy()
    if paddle_xy.shape[0] < 2:
        paddle_xy = zero_xy.copy()
    if puck_xy.shape[0] < 2:
        puck_xy = zero_xy.copy()
    return paddle_xy.copy(), puck_xy.copy()


def _extract_motion_magnitudes_from_step_info(
    step_info: dict | None,
    state: MotionRewardState,
) -> tuple[float, float, float]:
    velocity_mag = state.current_velocity_mag
    acceleration_mag = state.current_acceleration_mag
    jerk_mag = state.current_jerk_mag
    if isinstance(step_info, dict):
        velocity_mag = _finite_or_fallback(step_info.get("paddle_velocity_mag"), velocity_mag)
        acceleration_mag = _finite_or_fallback(
            step_info.get("paddle_acceleration_mag"),
            acceleration_mag,
        )
        jerk_mag = _finite_or_fallback(step_info.get("paddle_jerk_mag"), jerk_mag)
    state.current_velocity_mag = float(velocity_mag)
    state.current_acceleration_mag = float(acceleration_mag)
    state.current_jerk_mag = float(jerk_mag)
    return float(velocity_mag), float(acceleration_mag), float(jerk_mag)


def _extract_motion_magnitudes_from_state_info(
    state_info: dict | None,
    state: MotionRewardState,
) -> tuple[float, float, float]:
    velocity_mag = state.current_velocity_mag
    acceleration_mag = state.current_acceleration_mag
    jerk_mag = state.current_jerk_mag
    if isinstance(state_info, dict):
        paddle = state_info.get("paddles", {}).get("paddle_ego", {})
        if isinstance(paddle, dict):
            if "velocity" in paddle:
                velocity_mag = _finite_or_fallback(
                    np.linalg.norm(np.asarray(paddle["velocity"], dtype=np.float64).reshape(-1)[:2]),
                    velocity_mag,
                )
            if "acceleration" in paddle:
                acceleration_mag = _finite_or_fallback(
                    np.linalg.norm(np.asarray(paddle["acceleration"], dtype=np.float64).reshape(-1)[:2]),
                    acceleration_mag,
                )
            if "jerk" in paddle:
                jerk_mag = _finite_or_fallback(
                    np.linalg.norm(np.asarray(paddle["jerk"], dtype=np.float64).reshape(-1)[:2]),
                    jerk_mag,
                )
    state.current_velocity_mag = float(velocity_mag)
    state.current_acceleration_mag = float(acceleration_mag)
    state.current_jerk_mag = float(jerk_mag)
    return float(velocity_mag), float(acceleration_mag), float(jerk_mag)


def _init_motion_reward_state(
    temporal_horizon: int,
    anchor_paddle_xy: np.ndarray | None = None,
    anchor_puck_xy: np.ndarray | None = None,
) -> MotionRewardState:
    horizon = max(int(temporal_horizon), 1)
    history_len = horizon + 1
    zeros = np.zeros((2,), dtype=np.float64)
    paddle_anchor = np.asarray(anchor_paddle_xy, dtype=np.float64).reshape(-1)[:2] if anchor_paddle_xy is not None else zeros
    puck_anchor = np.asarray(anchor_puck_xy, dtype=np.float64).reshape(-1)[:2] if anchor_puck_xy is not None else zeros
    if paddle_anchor.shape[0] < 2:
        paddle_anchor = zeros
    if puck_anchor.shape[0] < 2:
        puck_anchor = zeros
    paddle_history = deque(
        [zeros.copy() for _ in range(history_len - 1)] + [paddle_anchor.copy()],
        maxlen=history_len,
    )
    puck_history = deque(
        [zeros.copy() for _ in range(history_len - 1)] + [puck_anchor.copy()],
        maxlen=history_len,
    )
    return MotionRewardState(
        temporal_horizon=horizon,
        paddle_history=paddle_history,
        puck_history=puck_history,
    )


def _reset_motion_reward_state(
    state: MotionRewardState,
    anchor_paddle_xy: np.ndarray | None,
    anchor_puck_xy: np.ndarray | None,
) -> None:
    state.steps_since_reset = 0
    state.current_velocity_mag = 0.0
    state.current_acceleration_mag = 0.0
    state.current_jerk_mag = 0.0
    refreshed = _init_motion_reward_state(
        state.temporal_horizon,
        anchor_paddle_xy=anchor_paddle_xy,
        anchor_puck_xy=anchor_puck_xy,
    )
    state.paddle_history = refreshed.paddle_history
    state.puck_history = refreshed.puck_history


def _compute_motion_reward_components(
    *,
    args: Any,
    motion_state: MotionRewardState,
    paddle_xy: np.ndarray,
    puck_xy: np.ndarray,
    velocity_mag: float,
    jerk_mag: float,
) -> dict[str, float]:
    motion_state.paddle_history.append(np.asarray(paddle_xy, dtype=np.float64).reshape(-1)[:2].copy())
    motion_state.puck_history.append(np.asarray(puck_xy, dtype=np.float64).reshape(-1)[:2].copy())
    motion_state.steps_since_reset = int(motion_state.steps_since_reset) + 1

    paddle_hist = np.asarray(list(motion_state.paddle_history), dtype=np.float64)
    puck_hist = np.asarray(list(motion_state.puck_history), dtype=np.float64)
    realized_movement = paddle_hist[-1, :] - paddle_hist[0, :]
    movement_norm = float(np.linalg.norm(realized_movement))
    eps = 1e-8

    temporal_valid = float(1.0 if motion_state.steps_since_reset >= int(motion_state.temporal_horizon) else 0.0)
    stand_still_reward_raw = float(
        1.0 if (movement_norm <= float(args.stand_still_threshold) and temporal_valid > 0.5) else 0.0
    )

    target_direction = puck_hist[0, :] - paddle_hist[0, :]
    movement_norm_safe = max(movement_norm, eps)
    target_norm_safe = max(float(np.linalg.norm(target_direction)), eps)
    temporal_cosine = float(np.dot(realized_movement, target_direction) / (movement_norm_safe * target_norm_safe))
    temporal_alignment_reward_raw = float(np.clip((temporal_cosine + 1.0) * 0.5, 0.0, 1.0)) * temporal_valid
    if stand_still_reward_raw > 0.5:
        temporal_alignment_reward_raw = 1.0

    movement_unit = realized_movement / movement_norm_safe
    max_axis_cosine = max(abs(float(movement_unit[0])), abs(float(movement_unit[1])))
    min_axis_cosine = float(1.0 / np.sqrt(2.0))
    axis_alignment_reward_raw = (
        (max_axis_cosine - min_axis_cosine) / (1.0 - min_axis_cosine + eps)
    )
    axis_alignment_reward_raw = float(np.clip(axis_alignment_reward_raw, 0.0, 1.0)) * temporal_valid
    if stand_still_reward_raw > 0.5:
        axis_alignment_reward_raw = 1.0

    velocity_reward_raw = float(
        velocity_reward_from_magnitude(
            velocity_mag,
            velocity_at_one=float(args.velocity_at_one),
            velocity_at_zero=float(args.velocity_at_zero),
        )
    )
    jerk_reward_raw = float(
        jerk_reward_from_magnitude(
            jerk_mag,
            jerk_at_one=float(args.jerk_at_one),
            jerk_at_zero=float(args.jerk_at_zero),
        )
    )

    stand_still_reward_weighted = float(args.stand_still_reward_weight) * stand_still_reward_raw
    temporal_alignment_reward_weighted = (
        float(args.temporal_alignment_reward_weight) * temporal_alignment_reward_raw
    )
    axis_alignment_reward_weighted = float(args.axis_alignment_reward_weight) * axis_alignment_reward_raw
    velocity_reward_weighted = float(args.velocity_reward_weight) * velocity_reward_raw
    jerk_reward_weighted = float(args.jerk_reward_weight) * jerk_reward_raw
    motion_reward_total = (
        stand_still_reward_weighted
        + temporal_alignment_reward_weighted
        + axis_alignment_reward_weighted
        + velocity_reward_weighted
        + jerk_reward_weighted
    )
    return {
        "temporal_valid_fraction": temporal_valid,
        "stand_still_reward_raw": stand_still_reward_raw,
        "temporal_alignment_reward_raw": temporal_alignment_reward_raw,
        "axis_alignment_reward_raw": axis_alignment_reward_raw,
        "velocity_reward_raw": velocity_reward_raw,
        "jerk_reward_raw": jerk_reward_raw,
        "stand_still_reward_weighted": stand_still_reward_weighted,
        "temporal_alignment_reward_weighted": temporal_alignment_reward_weighted,
        "axis_alignment_reward_weighted": axis_alignment_reward_weighted,
        "velocity_reward_weighted": velocity_reward_weighted,
        "jerk_reward_weighted": jerk_reward_weighted,
        "motion_reward_total": motion_reward_total,
    }
