"""
TD3 dual-head Q networks with transformed Bellman outputs.

Each Q-network predicts two transformed-value heads:
- task head for environment/task rewards
- motion head for auxiliary motion rewards

The trunk mirrors the current policy style by using residual blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.smooth_policy.agent import ResidualMLPTrunk, layer_init


class TD3DualHeadQNetwork(nn.Module):
    """Single TD3 critic with residual trunk and two scalar Q heads."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_layer_size: int = 128,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

        input_dim = int(obs_dim + act_dim)
        units_per_residual_block = 4
        num_residual_blocks = int(num_hidden_layers)

        self.network_depth = int(num_residual_blocks * units_per_residual_block)
        self.hidden_layer_size = int(hidden_layer_size)
        self.num_hidden_layers = int(num_hidden_layers)

        self.trunk = ResidualMLPTrunk(
            input_dim=input_dim,
            hidden_layer_size=hidden_layer_size,
            num_residual_blocks=num_residual_blocks,
            units_per_block=units_per_residual_block,
        )

        # Heads output Q values in transformed (h) space.
        self.task_head = layer_init(nn.Linear(hidden_layer_size, 1), std=1.0)
        self.motion_head = layer_init(nn.Linear(hidden_layer_size, 1), std=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        x = self.trunk(x)
        q_task_h = self.task_head(x)
        q_motion_h = self.motion_head(x)
        return q_task_h, q_motion_h

