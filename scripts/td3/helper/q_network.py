"""TD3 critic with transformed Bellman output (single scalar head).

The trunk mirrors the policy style by using residual blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.td3.agent import ResidualMLPTrunk, layer_init


class TD3QNetwork(nn.Module):
    """TD3 critic with residual trunk and one scalar Q head."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_layer_size: int = 128,
        num_hidden_layers: int = 2,
        use_context=False,
        context_vector_dim=0,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

        input_dim = int(obs_dim + act_dim)
        
        if use_context:
            input_dim += context_vector_dim

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

        # Small output-head init keeps initial Q estimates close to zero.
        self.head = layer_init(nn.Linear(hidden_layer_size, 1), std=0.01)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = self.trunk(x)
        return self.head(x)
