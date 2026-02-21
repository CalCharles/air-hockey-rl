"""Soft gates for smooth/small single-direction exploration rewards."""

from __future__ import annotations

import math
import torch


def _small_motion_gate(
    displacement: torch.Tensor,
    inner_threshold_m: float,
    outer_threshold_m: float,
) -> torch.Tensor:
    if inner_threshold_m <= 0 or outer_threshold_m <= inner_threshold_m:
        raise ValueError(
            f"Expected 0 < inner < outer, got inner={inner_threshold_m}, outer={outer_threshold_m}"
        )
    rel = (displacement - inner_threshold_m) / (outer_threshold_m - inner_threshold_m)
    clamped = torch.clamp(rel, min=0.0, max=1.0)
    # Cosine shoulder keeps high reward near 2cm and smooth decay to 7cm.
    shoulder = 0.5 * (1.0 + torch.cos(math.pi * clamped))
    tail = torch.exp(-torch.clamp(displacement - outer_threshold_m, min=0.0) / (outer_threshold_m + 1e-8))
    return torch.where(displacement <= outer_threshold_m, shoulder, 0.2 * tail)


def _weighted_cosine_consistency(deltas: torch.Tensor, ref_dir: torch.Tensor) -> torch.Tensor:
    # deltas: [B, T-1, 2], ref_dir: [B, 2]
    step_norm = torch.norm(deltas, dim=-1).clamp_min(1e-8)
    ref_norm = torch.norm(ref_dir, dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (deltas * ref_dir.unsqueeze(1)).sum(dim=-1) / (step_norm * ref_norm.squeeze(-1))
    weight = step_norm / step_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (cosine * weight).sum(dim=-1)


def _consecutive_delta_consistency(deltas: torch.Tensor) -> torch.Tensor:
    if deltas.shape[1] <= 1:
        return torch.ones(deltas.shape[0], device=deltas.device, dtype=deltas.dtype)
    prev = deltas[:, :-1, :]
    nxt = deltas[:, 1:, :]
    prev_norm = torch.norm(prev, dim=-1).clamp_min(1e-8)
    next_norm = torch.norm(nxt, dim=-1).clamp_min(1e-8)
    cosine = (prev * nxt).sum(dim=-1) / (prev_norm * next_norm)
    return cosine.mean(dim=-1)


def compute_smooth_exploration_gate(
    paddle_window: torch.Tensor,
    *,
    inner_threshold_m: float = 0.02,
    outer_threshold_m: float = 0.07,
    direction_center: float = 0.35,
    direction_temperature: float = 10.0,
) -> torch.Tensor:
    """
    Compute a soft gate [0, 1] for small, mostly single-direction paddle motion.

    Args:
        paddle_window: [B, T, 2] paddle positions.
    """
    if paddle_window.dim() != 3 or paddle_window.shape[-1] != 2:
        raise ValueError(f"Expected paddle_window [B, T, 2], got {tuple(paddle_window.shape)}")

    deltas = paddle_window[:, 1:, :] - paddle_window[:, :-1, :]
    total_disp_vec = paddle_window[:, -1, :] - paddle_window[:, 0, :]
    total_disp = torch.norm(total_disp_vec, dim=-1)
    motion_gate = _small_motion_gate(total_disp, inner_threshold_m, outer_threshold_m)

    weighted_consistency = _weighted_cosine_consistency(deltas, total_disp_vec)
    consecutive_consistency = _consecutive_delta_consistency(deltas)
    direction_score = 0.5 * ((weighted_consistency + 1.0) * 0.5 + (consecutive_consistency + 1.0) * 0.5)
    direction_gate = torch.sigmoid((direction_score - direction_center) * direction_temperature)

    tiny_disp = total_disp < 1e-4
    direction_gate = torch.where(tiny_disp, torch.ones_like(direction_gate), direction_gate)
    return torch.clamp(motion_gate * direction_gate, min=0.0, max=1.0)
