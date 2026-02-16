import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.smooth_policy.agent import layer_init


class ActionStateEmbedder(nn.Module):
    """Embed per-timestep (action, state) pairs into a latent feature vector."""

    def __init__(self, action_dim, state_dim, embed_dim=16, hidden_size=128):
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
        )

    def forward(self, actions, states):
        x = torch.cat([actions, states], dim=-1)  # [B, T, action_dim + state_dim]
        return self.net(x)  # [B, T, embed_dim]


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over channels for each timestep of a [B, C, T] tensor."""

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # [B, C, T] -> [B, T, C] -> LayerNorm(C) -> [B, C, T]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """Conv1d block with LayerNorm, ReLU, and residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.norm = ChannelLayerNorm1d(out_channels)
        self.activation = activation()

        # Match residual branch shape to the main branch when temporal/channel sizes differ.
        if in_channels == out_channels and kernel_size == 1 and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=False,
            )

    @staticmethod
    def _align_time_dim(main, residual):
        if main.shape[-1] == residual.shape[-1]:
            return main, residual
        min_len = min(main.shape[-1], residual.shape[-1])
        return main[..., -min_len:], residual[..., -min_len:]

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        residual = self.residual(x)
        out, residual = self._align_time_dim(out, residual)
        return out + residual


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

        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation,
                )
                for in_channels, out_channels, kernel_size, stride in conv_specs
            ]
        )
        self._mask_kernel_cache = {}

    def _get_mask_kernel(self, kernel_size, dtype, device):
        cache_key = (int(kernel_size), dtype, device)
        kernel = self._mask_kernel_cache.get(cache_key)
        if kernel is None:
            kernel = torch.ones((1, 1, int(kernel_size)), dtype=dtype, device=device)
            self._mask_kernel_cache[cache_key] = kernel
        return kernel

    def forward(self, temporal_features):
        # temporal_features: [B, C, T]
        x = temporal_features
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=-1)  # [B, C_out]

    def forward_masked(self, temporal_features, valid_mask):
        """
        Mask-aware temporal encoding and pooling.
        valid_mask: [B, T] (1 for valid timestep, 0 for padded timestep).
        """

        x = temporal_features
        mask = valid_mask.to(dtype=x.dtype).unsqueeze(1)  # [B, 1, T]

        # Zero padded inputs before temporal convolutions.
        x = x * mask

        for block in self.blocks:
            main = block.conv(x)
            k = int(block.conv.kernel_size[0])
            s = int(block.conv.stride[0])
            p = int(block.conv.padding[0])
            d = int(block.conv.dilation[0])

            # A timestep stays valid only if the whole conv receptive field was valid.
            main_kernel = self._get_mask_kernel(k, mask.dtype, mask.device)
            main_mask = F.conv1d(mask, main_kernel, stride=s, padding=p, dilation=d)
            main_mask = (main_mask >= float(k)).to(dtype=x.dtype)

            main = block.norm(main)
            main = block.activation(main)
            main = main * main_mask

            residual = block.residual(x)
            if isinstance(block.residual, nn.Identity):
                residual_mask = mask
            else:
                rk = int(block.residual.kernel_size[0])
                rs = int(block.residual.stride[0])
                rp = int(block.residual.padding[0])
                rd = int(block.residual.dilation[0])
                residual_kernel = self._get_mask_kernel(rk, mask.dtype, mask.device)
                residual_mask = F.conv1d(mask, residual_kernel, stride=rs, padding=rp, dilation=rd)
                residual_mask = (residual_mask >= float(rk)).to(dtype=x.dtype)
            residual = residual * residual_mask

            main, residual = block._align_time_dim(main, residual)
            main_mask, residual_mask = block._align_time_dim(main_mask, residual_mask)
            x = (main + residual) * main_mask * residual_mask
            mask = main_mask * residual_mask

        valid_count = mask.sum(dim=-1).clamp_min(1.0)  # [B, 1]
        pooled = x.sum(dim=-1) / valid_count  # [B, C_out]
        return pooled


class RMAAdaptationModule(nn.Module):
    """
    Stage-2 RMA adaptation module.
    (action, state) -> per-step embedding -> temporal conv -> mean pool -> 8D latent
    """

    def __init__(
        self,
        action_dim,
        state_dim,
        conv_in_channels=32,
        latent_dim=12,
        hidden_size=128,
    ):
        super().__init__()
        self.embedder = ActionStateEmbedder(
            action_dim=action_dim,
            state_dim=state_dim,
            embed_dim=conv_in_channels,
            hidden_size=hidden_size,
        )
        self.temporal_encoder = TemporalConvEncoder(
            conv_specs=[
                (conv_in_channels, conv_in_channels, 8, 2),
                (conv_in_channels, conv_in_channels, 5, 1),
                (conv_in_channels, conv_in_channels, 5, 1),
            ]
        )
        self.latent_head = layer_init(nn.Linear(conv_in_channels, latent_dim), std=1.0)

    def forward(self, actions, states, valid_mask=None, return_intermediates=False):
        embedded = self.embedder(actions, states)  # [B, T, conv_in_channels]
        temporal_in = embedded.transpose(1, 2)  # [B, conv_in_channels, T]
        if valid_mask is None:
            pooled = self.temporal_encoder(temporal_in)  # [B, conv_in_channels]
        else:
            pooled = self.temporal_encoder.forward_masked(temporal_in, valid_mask=valid_mask)  # [B, conv_in_channels]
        latent = self.latent_head(pooled)  # [B, latent_dim]

        if not return_intermediates:
            return latent
        return {
            "embedded": embedded,
            "temporal_in": temporal_in,
            "pooled": pooled,
            "latent": latent,
        }
