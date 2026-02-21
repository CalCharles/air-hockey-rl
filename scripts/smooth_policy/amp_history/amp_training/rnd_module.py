"""RND modules for intrinsic exploration rewards."""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class RNDModel(nn.Module):
    """Random Network Distillation model with fixed target and trainable predictor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.target = _build_mlp(input_dim, hidden_dims, output_dim)
        self.predictor = _build_mlp(input_dim, hidden_dims, output_dim)
        for param in self.target.parameters():
            param.requires_grad_(False)

    def prediction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared prediction error."""
        pred = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        return torch.mean((pred - target) ** 2, dim=-1)

    def predictor_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Mean predictor training loss over a batch."""
        pred = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        return torch.mean((pred - target) ** 2)
