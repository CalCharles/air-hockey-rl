"""Primitive action generators for TD3 exploration takeover."""

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


def stand_still_actions(count: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((count, 2), device=device, dtype=dtype)


def actions_from_direction_and_magnitude(directions: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
    return directions * magnitudes.unsqueeze(-1)

