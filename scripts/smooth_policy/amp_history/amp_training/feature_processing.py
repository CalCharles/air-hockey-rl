"""Shared feature processing utilities for AMP discriminator inputs."""

from __future__ import annotations

import numpy as np
import torch


PUCK_FEATURE_DIM = 4  # [noised_x, noised_y, direction_sign, downward_speed_bin]


def normalize_position_history_batch(position_history: torch.Tensor) -> torch.Tensor:
    """Normalize [B, 5, 2] paddle position history to relative [B, 8] features."""
    pos1 = position_history[:, 0, :]
    translated = position_history - pos1.unsqueeze(1)
    return translated[:, 1:, :].reshape(-1, 8)


def normalize_action_history_batch(action_history: torch.Tensor) -> torch.Tensor:
    """Normalize [B, 4, 2] transition actions to unit norm and flatten to [B, 8]."""
    action_norms = torch.norm(action_history, dim=-1, keepdim=True)
    normalized_actions = action_history / (action_norms + 1e-8)
    return normalized_actions.reshape(action_history.shape[0], 8)


def _to_valid_index(index: int, window_len: int) -> int:
    if index < 0:
        index = window_len + index
    if index < 0 or index >= window_len:
        raise ValueError(f"current_index {index} is out of range for window_len {window_len}")
    return index


def _downward_speed_bin_torch(
    downward_speed: torch.Tensor,
    downward_speed_max: float,
) -> torch.Tensor:
    if downward_speed_max <= 0:
        raise ValueError(f"downward_speed_max must be > 0, got {downward_speed_max}")
    bin_width = downward_speed_max / 3.0
    clipped = torch.clamp(downward_speed, min=0.0, max=downward_speed_max)
    bin_idx = torch.clamp((clipped / (bin_width + 1e-8)).floor().long(), max=2)
    mapping = torch.tensor([-1.0, 0.0, 1.0], device=downward_speed.device, dtype=downward_speed.dtype)
    return mapping[bin_idx]


def _downward_speed_bin_numpy(
    downward_speed: np.ndarray,
    downward_speed_max: float,
) -> np.ndarray:
    if downward_speed_max <= 0:
        raise ValueError(f"downward_speed_max must be > 0, got {downward_speed_max}")
    edges = np.array([downward_speed_max / 3.0, 2.0 * downward_speed_max / 3.0], dtype=np.float32)
    clipped = np.clip(downward_speed, 0.0, downward_speed_max)
    bin_idx = np.digitize(clipped, bins=edges, right=False)
    return np.take(np.array([-1.0, 0.0, 1.0], dtype=np.float32), bin_idx)


def build_puck_discriminator_features_torch(
    puck_position_window: torch.Tensor,
    *,
    current_index: int = 2,
    vertical_axis: int = 0,
    downward_positive_direction: float = 1.0,
    downward_speed_max: float = 0.75,
    speed_dt: float = 0.05,
    noise_std: float = 0.03,
) -> torch.Tensor:
    """
    Build puck features [B, 4] from position windows [B, T, 2].

    Features:
    1) noised current x, 2) noised current y, 3) direction_sign, 4) downward_speed_bin.
    """
    if puck_position_window.dim() != 3 or puck_position_window.shape[-1] != 2:
        raise ValueError(
            f"Expected puck_position_window with shape [B, T, 2], got {tuple(puck_position_window.shape)}"
        )
    if vertical_axis not in (0, 1):
        raise ValueError(f"vertical_axis must be 0 or 1, got {vertical_axis}")
    if speed_dt <= 0:
        raise ValueError(f"speed_dt must be > 0, got {speed_dt}")

    batch_size, window_len, _ = puck_position_window.shape
    if window_len < 2:
        raise ValueError(f"puck_position_window must have at least 2 timesteps, got {window_len}")
    idx = _to_valid_index(current_index, window_len)

    current_pos = puck_position_window[:, idx, :]
    if noise_std > 0:
        noised_current_pos = current_pos + torch.randn_like(current_pos) * noise_std
    else:
        noised_current_pos = current_pos.clone()

    axis_series = puck_position_window[:, :, vertical_axis]
    direction_delta = (axis_series[:, -1] - axis_series[:, 0]) * downward_positive_direction
    direction_sign = torch.sign(direction_delta)

    net_velocity = direction_delta / ((window_len - 1) * speed_dt)
    downward_speed = torch.clamp(net_velocity, min=0.0)
    downward_speed_bin = _downward_speed_bin_torch(downward_speed, downward_speed_max)

    return torch.cat(
        [noised_current_pos, direction_sign.unsqueeze(-1), downward_speed_bin.unsqueeze(-1)],
        dim=-1,
    ).reshape(batch_size, PUCK_FEATURE_DIM)


def build_puck_discriminator_features_numpy(
    puck_position_window: np.ndarray,
    *,
    current_index: int = 2,
    vertical_axis: int = 0,
    downward_positive_direction: float = 1.0,
    downward_speed_max: float = 0.75,
    speed_dt: float = 0.05,
    noise_std: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Numpy equivalent of build_puck_discriminator_features_torch."""
    if puck_position_window.ndim != 3 or puck_position_window.shape[-1] != 2:
        raise ValueError(
            f"Expected puck_position_window with shape [B, T, 2], got {puck_position_window.shape}"
        )
    if vertical_axis not in (0, 1):
        raise ValueError(f"vertical_axis must be 0 or 1, got {vertical_axis}")
    if speed_dt <= 0:
        raise ValueError(f"speed_dt must be > 0, got {speed_dt}")

    window_len = puck_position_window.shape[1]
    if window_len < 2:
        raise ValueError(f"puck_position_window must have at least 2 timesteps, got {window_len}")
    idx = _to_valid_index(current_index, window_len)

    current_pos = puck_position_window[:, idx, :].astype(np.float32)
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng()
        noised_current_pos = current_pos + rng.normal(
            loc=0.0, scale=noise_std, size=current_pos.shape
        ).astype(np.float32)
    else:
        noised_current_pos = current_pos.copy()

    axis_series = puck_position_window[:, :, vertical_axis].astype(np.float32)
    direction_delta = (axis_series[:, -1] - axis_series[:, 0]) * float(downward_positive_direction)
    direction_sign = np.sign(direction_delta).astype(np.float32)

    net_velocity = direction_delta / float((window_len - 1) * speed_dt)
    downward_speed = np.clip(net_velocity, 0.0, None)
    downward_speed_bin = _downward_speed_bin_numpy(downward_speed, downward_speed_max).astype(np.float32)

    return np.concatenate(
        [noised_current_pos, direction_sign[:, None], downward_speed_bin[:, None]],
        axis=-1,
    ).astype(np.float32)
