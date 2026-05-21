"""Primitive action generators for TD3 exploration takeover."""

from __future__ import annotations

import math

import torch


def sample_unit_directions(
    count: int,
    device: torch.device | str,
    dtype: torch.dtype,
    y_component_weight: float = 1.0,
) -> torch.Tensor:
    if count <= 0:
        return torch.zeros((0, 2), device=device, dtype=dtype)
    angles = 2.0 * torch.pi * torch.rand(count, device=device, dtype=dtype)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    directions[:, 1] = directions[:, 1] * float(max(y_component_weight, 1e-6))
    norm = torch.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)
    return directions / norm


def sample_directions_from_angle_range(
    count: int,
    min_angle_deg: float,
    max_angle_deg: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        return torch.zeros((0, 2), device=device, dtype=dtype)

    min_angle = float(min_angle_deg)
    max_angle = float(max_angle_deg)
    span_deg = max_angle - min_angle
    if abs(span_deg) >= 360.0:
        span_deg = 360.0
    elif span_deg < 0.0:
        span_deg = math.fmod(span_deg, 360.0)
        if span_deg < 0.0:
            span_deg += 360.0

    sampled_angles_deg = min_angle + torch.rand(count, device=device, dtype=dtype) * span_deg
    sampled_angles_rad = sampled_angles_deg * (torch.pi / 180.0)
    return torch.stack((torch.cos(sampled_angles_rad), torch.sin(sampled_angles_rad)), dim=-1)


def sample_uniform_magnitude(
    count: int,
    max_magnitude: float,
    min_magnitude: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    max_mag = float(max_magnitude)
    min_mag = float(min_magnitude)
    if max_mag < min_mag:
        raise ValueError("max_magnitude must be >= min_magnitude")
    return min_mag + torch.rand(count, device=device, dtype=dtype) * (max_mag - min_mag)


def stand_still_actions(count: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((count, 2), device=device, dtype=dtype)


def actions_from_direction_and_magnitude(directions: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
    return directions * magnitudes.unsqueeze(-1)


def max_magnitude_for_directions_in_action_box(
    directions: torch.Tensor,
    max_delta_x: float,
    max_delta_y: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if directions.ndim != 2 or directions.shape[-1] != 2:
        raise ValueError("directions must have shape (N, 2)")

    max_dx = max(float(max_delta_x), eps)
    max_dy = max(float(max_delta_y), eps)
    abs_dir = torch.abs(directions)
    limit_x = torch.full(
        (directions.shape[0],),
        float("inf"),
        dtype=directions.dtype,
        device=directions.device,
    )
    limit_y = torch.full(
        (directions.shape[0],),
        float("inf"),
        dtype=directions.dtype,
        device=directions.device,
    )
    nonzero_x = abs_dir[:, 0] > eps
    nonzero_y = abs_dir[:, 1] > eps
    if torch.any(nonzero_x):
        limit_x[nonzero_x] = max_dx / abs_dir[nonzero_x, 0]
    if torch.any(nonzero_y):
        limit_y[nonzero_y] = max_dy / abs_dir[nonzero_y, 1]
    return torch.minimum(limit_x, limit_y)


def sample_simulator_displacements_from_ranges(
    count: int,
    min_angle_deg: float,
    max_angle_deg: float,
    min_magnitude: float,
    max_magnitude: float,
    max_delta_x: float,
    max_delta_y: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    directions = sample_directions_from_angle_range(
        count=count,
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        device=device,
        dtype=dtype,
    )
    if count <= 0:
        return directions, torch.zeros((0,), device=device, dtype=dtype), directions

    requested_min = float(min_magnitude)
    requested_max = float(max_magnitude)
    if requested_max < requested_min:
        raise ValueError("max_magnitude must be >= min_magnitude")

    feasible_max = max_magnitude_for_directions_in_action_box(
        directions=directions,
        max_delta_x=max_delta_x,
        max_delta_y=max_delta_y,
    )
    capped_max = torch.clamp(feasible_max, min=0.0, max=requested_max)
    lower = torch.minimum(
        capped_max,
        torch.full_like(capped_max, requested_min),
    )
    sampled_magnitudes = lower + torch.rand(count, device=device, dtype=dtype) * (capped_max - lower)
    sampled_displacements = actions_from_direction_and_magnitude(directions, sampled_magnitudes)
    return directions, sampled_magnitudes, sampled_displacements


def project_displacement_to_action_box(
    target_displacements: torch.Tensor,
    max_delta_x: float,
    max_delta_y: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert desired per-step displacements in meters to normalized actions in [-1, 1]^2.

    The simulator uses anisotropic linear scaling:
      dx = action_x * max_delta_x
      dy = action_y * max_delta_y

    If the desired displacement is outside this rectangle, this projects to the farthest
    same-direction command on the rectangle boundary.
    """
    if target_displacements.ndim != 2 or target_displacements.shape[-1] != 2:
        raise ValueError("target_displacements must have shape (N, 2)")
    max_dx = max(float(max_delta_x), eps)
    max_dy = max(float(max_delta_y), eps)
    scale = torch.as_tensor([max_dx, max_dy], dtype=target_displacements.dtype, device=target_displacements.device)
    normalized = target_displacements / scale.unsqueeze(0)
    max_abs = torch.amax(torch.abs(normalized), dim=-1, keepdim=True).clamp_min(1.0)
    actions = normalized / max_abs
    achieved_displacements = actions * scale.unsqueeze(0)
    projection_scale = 1.0 / max_abs.squeeze(-1)
    return actions, achieved_displacements, projection_scale

