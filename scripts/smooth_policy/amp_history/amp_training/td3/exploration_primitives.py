"""Primitive action generators for TD3 exploration takeover."""

from __future__ import annotations

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


def sample_target_distances(
    count: int,
    min_distance: float,
    max_distance: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    min_dist = float(min_distance)
    max_dist = float(max_distance)
    if max_dist < min_dist:
        raise ValueError("max_distance must be >= min_distance")
    return min_dist + torch.rand(count, device=device, dtype=dtype) * (max_dist - min_dist)


def stand_still_actions(count: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((count, 2), device=device, dtype=dtype)


def actions_from_direction_and_magnitude(directions: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
    return directions * magnitudes.unsqueeze(-1)


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

