import torch
import torch.nn as nn

from scripts.smooth_policy.agent import layer_init


class ActionStateEmbedder(nn.Module):
    """Embed per-timestep (action, state) pairs into a latent feature vector."""

    def __init__(self, action_dim, state_dim, embed_dim=16, hidden_size=64):
        super().__init__()
        if action_dim <= 0 or state_dim <= 0:
            raise ValueError("action_dim and state_dim must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        input_dim = action_dim + state_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, embed_dim), std=1.0),
            nn.Tanh(),
        )

    def forward(self, actions, states):
        if actions.ndim != 3 or states.ndim != 3:
            raise ValueError("actions and states must both be rank-3 tensors [batch, time, dim].")
        if actions.shape[:2] != states.shape[:2]:
            raise ValueError("actions and states must match on [batch, time].")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {actions.shape[-1]}.")
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {states.shape[-1]}.")

        x = torch.cat([actions, states], dim=-1)  # [B, T, action_dim + state_dim]
        return self.net(x)  # [B, T, embed_dim]


class TemporalConvEncoder(nn.Module):
    """Temporal Conv1d stack followed by mean pooling."""

    def __init__(self, conv_specs=None, activation=nn.ReLU):
        super().__init__()
        if conv_specs is None:
            conv_specs = [
                (8, 8, 8, 1),
                (8, 8, 5, 1),
                (8, 8, 5, 1),
            ]

        layers = []
        for in_channels, out_channels, kernel_size, stride in conv_specs:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                )
            )
            layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, temporal_features):
        # temporal_features: [B, C, T]
        if temporal_features.ndim != 3:
            raise ValueError("temporal_features must be rank-3 tensor [batch, channels, time].")
        x = self.network(temporal_features)
        return x.mean(dim=-1)  # [B, C_out]


class RMAAdaptationModule(nn.Module):
    """
    Stage-2 RMA adaptation module.
    (action, state) -> per-step embedding -> temporal conv -> mean pool -> 8D latent
    """

    def __init__(
        self,
        action_dim,
        state_dim,
        embed_dim=16,
        conv_in_channels=8,
        latent_dim=8,
        hidden_size=64,
    ):
        super().__init__()
        self.embedder = ActionStateEmbedder(
            action_dim=action_dim,
            state_dim=state_dim,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
        )
        # Bridge from 16D embedding to conv input channels (8D per timestep by default).
        self.pre_conv_projection = layer_init(nn.Linear(embed_dim, conv_in_channels), std=1.0)
        self.temporal_encoder = TemporalConvEncoder(
            conv_specs=[
                (conv_in_channels, 8, 8, 1),
                (8, 8, 5, 1),
                (8, 8, 5, 1),
            ]
        )
        self.latent_head = layer_init(nn.Linear(8, latent_dim), std=1.0)

    def forward(self, actions, states, return_intermediates=False):
        embedded = self.embedder(actions, states)  # [B, T, 16]
        projected = self.pre_conv_projection(embedded)  # [B, T, 8]
        temporal_in = projected.transpose(1, 2)  # [B, 8, T]
        pooled = self.temporal_encoder(temporal_in)  # [B, 8]
        latent = self.latent_head(pooled)  # [B, 8]

        if not return_intermediates:
            return latent
        return {
            "embedded": embedded,
            "projected": projected,
            "temporal_in": temporal_in,
            "pooled": pooled,
            "latent": latent,
        }
