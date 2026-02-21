"""Feature construction for smooth-constrained RND."""

from __future__ import annotations

import torch

from scripts.smooth_policy.amp_history.amp_training.feature_processing import (
    normalize_position_sequence_batch,
)


def build_rnd_features(
    paddle_window: torch.Tensor,
    puck_window: torch.Tensor,
) -> torch.Tensor:
    """
    Build RND inputs from paddle and puck history windows.

    Args:
        paddle_window: Tensor [B, T, 2]
        puck_window: Tensor [B, T, 2]

    Returns:
        Tensor [B, 2 + 2 + (T - 1) * 2] made of:
            - current paddle position [2]
            - paddle displacement from first to last [2]
            - normalized puck sequence relative to first puck point [(T-1)*2]
    """
    if paddle_window.dim() != 3 or paddle_window.shape[-1] != 2:
        raise ValueError(f"Expected paddle_window [B, T, 2], got {tuple(paddle_window.shape)}")
    if puck_window.dim() != 3 or puck_window.shape[-1] != 2:
        raise ValueError(f"Expected puck_window [B, T, 2], got {tuple(puck_window.shape)}")
    if paddle_window.shape != puck_window.shape:
        raise ValueError(
            f"paddle_window and puck_window must have the same shape, got "
            f"{tuple(paddle_window.shape)} and {tuple(puck_window.shape)}"
        )

    current_paddle = paddle_window[:, -1, :]
    paddle_delta = paddle_window[:, -1, :] - paddle_window[:, 0, :]
    normalized_puck = normalize_position_sequence_batch(puck_window)
    return torch.cat([current_paddle, paddle_delta, normalized_puck], dim=-1)
