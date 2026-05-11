import torch
import torch.nn as nn

from scripts.td3.agent import layer_init


class EnvEncoder(nn.Module):
    """
    Compact environment encoder for RMA.
    Maps environment variable vectors to a latent conditioning code.
    """

    def __init__(self, env_var_dim, latent_dim=8, hidden_size=(128, 128)):
        super().__init__()
        if env_var_dim <= 0:
            raise ValueError(f"env_var_dim must be positive, got {env_var_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if isinstance(hidden_size, int):
            hidden_dims = [hidden_size]
        else:
            hidden_dims = [int(x) for x in hidden_size]
        if len(hidden_dims) == 0:
            raise ValueError("EnvEncoder hidden_size must have at least one layer.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"EnvEncoder hidden sizes must be positive, got {hidden_dims}")

        layers = []
        in_dim = env_var_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, latent_dim), std=1.0))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, env_vars):
        return self.net(env_vars)
