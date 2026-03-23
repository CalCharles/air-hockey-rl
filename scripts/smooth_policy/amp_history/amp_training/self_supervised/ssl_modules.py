"""Self-supervised learning modules for AMP training."""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_hidden_stack(
    input_dim: int, hidden_layer_size: int, num_hidden_layers: int
) -> tuple[nn.Sequential, int]:
    if hidden_layer_size <= 0:
        raise ValueError(f"hidden_layer_size must be positive, got {hidden_layer_size}")
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for _ in range(int(num_hidden_layers)):
        layers.append(nn.Linear(prev_dim, int(hidden_layer_size)))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        prev_dim = int(hidden_layer_size)
    return nn.Sequential(*layers), prev_dim


class SharedStateEncoder(nn.Module):
    """Encodes state observations into a compact latent representation."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.backbone, hidden_dim = _build_hidden_stack(
            obs_dim, hidden_layer_size, num_hidden_layers
        )
        self.projection = nn.Linear(hidden_dim, int(latent_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(obs))


class ActionConditionedRewardHead(nn.Module):
    """Predicts immediate combined reward from latent state and action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.backbone, hidden_dim = _build_hidden_stack(
            latent_dim + action_dim, hidden_layer_size, num_hidden_layers
        )
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, action], dim=-1)
        return self.projection(self.backbone(x)).squeeze(-1)


class ActionConditionedDynamicsHead(nn.Module):
    """Predicts position deltas from latent state, action, and current positions."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        position_dim: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.backbone, hidden_dim = _build_hidden_stack(
            latent_dim + action_dim + position_dim, hidden_layer_size, num_hidden_layers
        )
        self.projection = nn.Linear(hidden_dim, int(position_dim))

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([latent, action, current_positions], dim=-1)
        return self.projection(self.backbone(x))
