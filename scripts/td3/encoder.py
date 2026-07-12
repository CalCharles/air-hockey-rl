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


# Alias for RMA privileged-property encoder (same API as EnvEncoder).
ObjectPropEncoder = EnvEncoder


class AdaptationModule(nn.Module):
    """
    RMA adaptation module: maps flattened observation/action history to a
    latent code that approximates the privileged EnvEncoder output.
    """

    def __init__(self, history_input_dim, latent_dim=8, hidden_size=(256, 128)):
        super().__init__()
        if history_input_dim <= 0:
            raise ValueError(f"history_input_dim must be positive, got {history_input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if isinstance(hidden_size, int):
            hidden_dims = [hidden_size]
        else:
            hidden_dims = [int(x) for x in hidden_size]
        if len(hidden_dims) == 0:
            raise ValueError("AdaptationModule hidden_size must have at least one layer.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"AdaptationModule hidden sizes must be positive, got {hidden_dims}")

        layers = []
        in_dim = history_input_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, latent_dim), std=1.0))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, history):
        # Accept (B, T, D) or already-flattened (B, T*D).
        if history.dim() == 3:
            history = history.reshape(history.shape[0], -1)
        elif history.dim() != 2:
            raise ValueError(
                f"AdaptationModule expects (B, T, D) or (B, T*D), got shape {tuple(history.shape)}"
            )
        return self.net(history)
