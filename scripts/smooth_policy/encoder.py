import torch
import torch.nn as nn

from scripts.smooth_policy.agent import layer_init


class EnvEncoder(nn.Module):
    """
    Compact environment encoder for RMA.
    Maps environment variable vectors to a latent conditioning code.
    """

    def __init__(self, env_var_dim, latent_dim=8, hidden_size=64):
        super().__init__()
        if env_var_dim <= 0:
            raise ValueError(f"env_var_dim must be positive, got {env_var_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.net = nn.Sequential(
            layer_init(nn.Linear(env_var_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, latent_dim), std=1.0),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, env_vars):
        return self.net(env_vars)
