"""Shared feature processing utilities for AMP discriminator inputs."""

from __future__ import annotations

import numpy as np
import torch


PUCK_FEATURE_DIM = 3  # [direction_sign, downward_speed_bin, vertical_pos_bin_5]


def normalize_position_history_batch(position_history: torch.Tensor) -> torch.Tensor:
    """Normalize [B, 5, 2] paddle position history to relative [B, 8] features."""
    return normalize_position_sequence_batch(position_history)


def normalize_position_sequence_batch(position_sequence: torch.Tensor) -> torch.Tensor:
    """Normalize [B, T, 2] paddle position sequence to relative [B, (T-1)*2] features."""
    if position_sequence.dim() != 3 or position_sequence.shape[-1] != 2:
        raise ValueError(
            f"Expected position_sequence with shape [B, T, 2], got {tuple(position_sequence.shape)}"
        )
    pos1 = position_sequence[:, 0, :]
    translated = position_sequence - pos1.unsqueeze(1)
    return translated[:, 1:, :].reshape(position_sequence.shape[0], -1)


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


def _vertical_pos_bin_5_torch(
    vertical_position: torch.Tensor,
    vertical_pos_min: float,
    vertical_pos_max: float,
) -> torch.Tensor:
    if vertical_pos_max <= vertical_pos_min:
        raise ValueError(
            f"vertical_pos_max must be > vertical_pos_min, got [{vertical_pos_min}, {vertical_pos_max}]"
        )
    norm = (vertical_position - vertical_pos_min) / (vertical_pos_max - vertical_pos_min + 1e-8)
    clipped = torch.clamp(norm, 0.0, 1.0)
    bin_idx = torch.clamp((clipped * 5.0).floor().long(), max=4)
    mapping = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=vertical_position.device, dtype=vertical_position.dtype)
    return mapping[bin_idx]


def _vertical_pos_bin_5_numpy(
    vertical_position: np.ndarray,
    vertical_pos_min: float,
    vertical_pos_max: float,
) -> np.ndarray:
    if vertical_pos_max <= vertical_pos_min:
        raise ValueError(
            f"vertical_pos_max must be > vertical_pos_min, got [{vertical_pos_min}, {vertical_pos_max}]"
        )
    norm = (vertical_position - vertical_pos_min) / (vertical_pos_max - vertical_pos_min + 1e-8)
    clipped = np.clip(norm, 0.0, 1.0)
    bin_idx = np.minimum(np.floor(clipped * 5.0).astype(np.int64), 4)
    return np.take(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32), bin_idx)


def sample_bucketed_indices_torch(
    batch_size: int,
    *,
    window_len: int = 30,
    num_bins: int = 3,
    samples_per_bin: int = 2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample sorted index sets with endpoints + bucketed interior samples."""
    if window_len < 4:
        raise ValueError(f"window_len must be at least 4, got {window_len}")
    if num_bins <= 0 or samples_per_bin <= 0:
        raise ValueError(f"num_bins and samples_per_bin must be positive, got {num_bins}, {samples_per_bin}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    interior_start = 1
    interior_end = window_len - 1
    interior_count = interior_end - interior_start
    if interior_count <= 0:
        raise ValueError("Not enough interior timesteps for bucketed sampling.")
    if interior_count < num_bins:
        raise ValueError(f"Not enough interior timesteps ({interior_count}) for {num_bins} bins.")

    base_size = interior_count // num_bins
    remainder = interior_count % num_bins
    bin_ranges = []
    cursor = interior_start
    for bin_idx in range(num_bins):
        current_size = base_size + (1 if bin_idx < remainder else 0)
        bin_low = cursor
        bin_high = cursor + current_size
        cursor = bin_high
        if current_size < samples_per_bin:
            raise ValueError(
                f"Bin {bin_idx} has only {current_size} elements but needs {samples_per_bin} samples."
            )
        bin_ranges.append((bin_low, bin_high))

    sampled_chunks = []
    for bin_low, bin_high in bin_ranges:
        sampled = torch.randint(
            low=bin_low,
            high=bin_high,
            size=(batch_size, samples_per_bin),
            device=device,
        )
        sampled_chunks.append(sampled)

    interior_samples = torch.cat(sampled_chunks, dim=-1)
    endpoints = torch.tensor([0, window_len - 1], dtype=torch.long, device=device).view(1, 2).expand(batch_size, -1)
    all_indices = torch.cat([endpoints, interior_samples], dim=-1)
    all_indices, _ = torch.sort(all_indices, dim=-1)
    return all_indices


def build_puck_discriminator_features_torch(
    puck_position_window: torch.Tensor,
    *,
    current_index: int = 2,
    vertical_axis: int = 0,
    downward_positive_direction: float = 1.0,
    downward_speed_max: float = 0.75,
    speed_dt: float = 0.05,
    noise_std: float = 0.03,
    vertical_pos_min: float = -1.0,
    vertical_pos_max: float = 1.0,
) -> torch.Tensor:
    """
    Build puck features [B, 3] from position windows [B, T, 2].

    Features:
    1) direction_sign, 2) downward_speed_bin, 3) vertical_pos_bin_5.
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

    del noise_std
    current_pos = puck_position_window[:, idx, :]

    axis_series = puck_position_window[:, :, vertical_axis]
    direction_delta = (axis_series[:, -1] - axis_series[:, 0]) * downward_positive_direction
    direction_sign = torch.sign(direction_delta)

    net_velocity = direction_delta / ((window_len - 1) * speed_dt)
    downward_speed = torch.clamp(net_velocity, min=0.0)
    downward_speed_bin = _downward_speed_bin_torch(downward_speed, downward_speed_max)
    vertical_position = current_pos[:, vertical_axis]
    vertical_pos_bin = _vertical_pos_bin_5_torch(vertical_position, vertical_pos_min, vertical_pos_max)

    return torch.cat(
        [direction_sign.unsqueeze(-1), downward_speed_bin.unsqueeze(-1), vertical_pos_bin.unsqueeze(-1)],
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
    vertical_pos_min: float = -1.0,
    vertical_pos_max: float = 1.0,
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

    del noise_std, rng
    current_pos = puck_position_window[:, idx, :].astype(np.float32)

    axis_series = puck_position_window[:, :, vertical_axis].astype(np.float32)
    direction_delta = (axis_series[:, -1] - axis_series[:, 0]) * float(downward_positive_direction)
    direction_sign = np.sign(direction_delta).astype(np.float32)

    net_velocity = direction_delta / float((window_len - 1) * speed_dt)
    downward_speed = np.clip(net_velocity, 0.0, None)
    downward_speed_bin = _downward_speed_bin_numpy(downward_speed, downward_speed_max).astype(np.float32)
    vertical_position = current_pos[:, vertical_axis]
    vertical_pos_bin = _vertical_pos_bin_5_numpy(
        vertical_position,
        vertical_pos_min=vertical_pos_min,
        vertical_pos_max=vertical_pos_max,
    ).astype(np.float32)

    return np.concatenate(
        [direction_sign[:, None], downward_speed_bin[:, None], vertical_pos_bin[:, None]],
        axis=-1,
    ).astype(np.float32)
