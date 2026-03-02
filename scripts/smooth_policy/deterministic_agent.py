import numpy as np
import torch
import torch.nn as nn

from scripts.smooth_policy.agent import ResidualMLPTrunk, layer_init


class DeterministicAgent(nn.Module):
    """
    Deterministic actor that mirrors Agent's actor architecture.

    This class intentionally keeps actor module names (`actor`, `actor_mean_head`)
    compatible with `Agent` so actor weights can be copied directly.
    """

    def __init__(
        self,
        envs,
        action_scale=1.0,
        action_bias=0.0,
        hidden_layer_size=64,
        num_hidden_layers=2,
        hidden_size=None,
    ):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        # Backward compatibility for existing callers while preferring hidden_layer_size.
        if hidden_size is not None:
            hidden_layer_size = hidden_size

        units_per_residual_block = 4
        num_residual_blocks = int(num_hidden_layers)
        if num_residual_blocks < 1:
            raise ValueError(f"num_residual_blocks must be >= 1, got {num_residual_blocks}")

        self.network_depth = int(num_residual_blocks * units_per_residual_block)
        self.num_residual_blocks = int(num_residual_blocks)
        self.hidden_layer_size = int(hidden_layer_size)

        self.actor = ResidualMLPTrunk(
            input_dim=obs_dim,
            hidden_layer_size=hidden_layer_size,
            num_residual_blocks=num_residual_blocks,
            units_per_block=units_per_residual_block,
        )
        self.actor_mean_head = layer_init(nn.Linear(hidden_layer_size, act_dim), std=1)

        self.register_buffer("action_scale", torch.tensor(action_scale))
        self.register_buffer("action_bias", torch.tensor(action_bias))

    def get_action_mean(self, x):
        x = self.actor(x)
        return self.actor_mean_head(x)

    def get_action(self, x):
        mean = self.get_action_mean(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def forward(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            return self.get_action(x)

