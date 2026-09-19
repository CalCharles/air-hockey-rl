"""Networks for the RMA baseline (Kumar, Fu, Pathak, Malik — RSS 2021).

RMA has three learned modules:

* ``EnvFactorEncoder`` — mu(e_t) -> z_t.  Privileged encoder of the
  per-episode environment factors e_t (here the three randomized physics
  parameters, normalised to [-1, 1] over their DR ranges) into a low-dim
  latent z_t.  Trained jointly with the base policy through the RL objective
  (phase 1).  Sim-only.
* ``RMAActor`` — pi(x_t, a_{t-1}, z_t) -> a_t.  The base policy.  For
  convenience the actor *owns* the encoder: ``get_action`` takes the
  privileged policy observation ``[x_t, e_t, a_{t-1}]`` and computes z_t
  internally, so the unmodified TD3 machinery of the project (target actor,
  Polyak averaging, CPU rollout replica, CUDA-graph update, checkpointing)
  trains pi and mu end-to-end with no special cases.
* ``AdaptationModule`` — phi(x_{t-H:t-1}, a_{t-H:t-1}) -> z_hat_t.  Deployed
  replacement for mu that infers the latent from the recent state/action
  history alone (phase 2, supervised regression to z_t on on-policy
  rollouts).  Architecture follows the paper: per-step MLP embedding, three
  1-D convolutions over time (channels 32, kernels 8/5/5, strides 4/1/1),
  linear read-out.

``AdaptedActor`` composes phi + pi for deployment / evaluation.

``HistoryActor`` is the long-history TD3 control baseline: the deployable RMA
architecture (window -> conv encoder -> z -> pi) trained end-to-end by RL with
no privileged signal, so RMA's only remaining difference is the supervised
latent.

Widths / depths of the base policy follow the project's canonical TD3 actor
(``DeterministicAgent``: residual MLP trunk, 64 wide, 2 blocks) so the only
difference to the plain-DR baseline is the latent input.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from scripts.td3.agent import ResidualMLPTrunk, layer_init


# --------------------------------------------------------------------------- mu
class EnvFactorEncoder(nn.Module):
    """mu: e_t (env_param_dim) -> z_t (latent_dim).  MLP, ELU, linear output."""

    def __init__(self, env_param_dim: int, latent_dim: int = 8, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        if env_param_dim <= 0 or latent_dim <= 0:
            raise ValueError(f"env_param_dim / latent_dim must be positive, got {env_param_dim} / {latent_dim}")
        layers: list[nn.Module] = []
        in_dim = int(env_param_dim)
        for h in hidden:
            layers.append(layer_init(nn.Linear(in_dim, int(h))))
            layers.append(nn.ELU())
            in_dim = int(h)
        layers.append(layer_init(nn.Linear(in_dim, int(latent_dim)), std=1.0))
        self.net = nn.Sequential(*layers)
        self.env_param_dim = int(env_param_dim)
        self.latent_dim = int(latent_dim)

    def forward(self, env_params: torch.Tensor) -> torch.Tensor:
        return self.net(env_params)


# --------------------------------------------------------------------------- pi
class _LatentConditionedActor(nn.Module):
    """Shared trunk for actors of the form pi(x_t, a_{t-1}, z): residual MLP + tanh head.

    Subclasses own an ``encoder`` that produces z from their conditioning
    input and implement ``get_action(policy_obs)`` / ``split_policy_obs``.
    Module names ``actor`` / ``actor_mean_head`` match ``DeterministicAgent``.
    """

    def _build_policy(self, *, policy_in: int, act_dim: int, hidden_layer_size: int, num_hidden_layers: int, action_scale: float, action_bias: float) -> None:
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
        self.actor = ResidualMLPTrunk(
            input_dim=int(policy_in),
            hidden_layer_size=int(hidden_layer_size),
            num_residual_blocks=int(num_hidden_layers),
            units_per_block=4,
        )
        self.actor_mean_head = layer_init(nn.Linear(int(hidden_layer_size), int(act_dim)), std=1)
        self.register_buffer("action_scale", torch.tensor(float(action_scale)))
        self.register_buffer("action_bias", torch.tensor(float(action_bias)))

    def get_action_from_latent(
        self, obs: torch.Tensor, prev_action: torch.Tensor | None, latent: torch.Tensor
    ) -> torch.Tensor:
        parts = [obs]
        if self.use_last_action:
            if prev_action is None:
                raise ValueError("use_last_action=True but prev_action is None")
            parts.append(prev_action)
        parts.append(latent)
        x = self.actor(torch.cat(parts, dim=-1))
        mean = self.actor_mean_head(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.get_action(x)


class RMAActor(_LatentConditionedActor):
    """Base policy pi(x_t, a_{t-1}, z_t) with the privileged encoder mu attached.

    ``get_action(policy_obs)`` expects the *privileged* policy observation
    layout produced by ``RMAEnvVector`` + the trainer::

        [ x_t (obs_dim) | e_t (env_param_dim) | a_{t-1} (act_dim, optional) ]

    and returns tanh-squashed actions like ``DeterministicAgent``.
    ``get_action_from_latent(x_t, a_{t-1}, z)`` is the deployment path used
    with z_hat from the adaptation module.
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        env_param_dim: int,
        latent_dim: int = 8,
        use_last_action: bool = True,
        hidden_layer_size: int = 64,
        num_hidden_layers: int = 2,
        encoder_hidden: Sequence[int] = (256, 128),
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.env_param_dim = int(env_param_dim)
        self.latent_dim = int(latent_dim)
        self.use_last_action = bool(use_last_action)
        self.encoder = EnvFactorEncoder(self.env_param_dim, self.latent_dim, encoder_hidden)
        policy_in = self.obs_dim + (self.act_dim if self.use_last_action else 0) + self.latent_dim
        self._build_policy(
            policy_in=policy_in, act_dim=self.act_dim, hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers, action_scale=action_scale, action_bias=action_bias,
        )

    @property
    def policy_obs_dim(self) -> int:
        """Dim of the privileged policy observation ``[x, e, a_prev]``."""
        return self.obs_dim + self.env_param_dim + (self.act_dim if self.use_last_action else 0)

    def split_policy_obs(self, policy_obs: torch.Tensor):
        obs = policy_obs[..., : self.obs_dim]
        env_params = policy_obs[..., self.obs_dim : self.obs_dim + self.env_param_dim]
        prev_action = policy_obs[..., self.obs_dim + self.env_param_dim :] if self.use_last_action else None
        return obs, env_params, prev_action

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        obs, env_params, prev_action = self.split_policy_obs(policy_obs)
        latent = self.encoder(env_params)
        return self.get_action_from_latent(obs, prev_action, latent)


# -------------------------------------------------------------------------- phi
class AdaptationModule(nn.Module):
    """phi: history (B, H, F) of per-step features [x_t-features, a_t] -> z_hat (B, latent_dim).

    Paper architecture: each (x, a) pair -> MLP -> 32-dim embedding; three
    1-D conv layers over the time axis (32 channels, kernels [8, 5, 5],
    strides [4, 1, 1]); flatten; linear to the latent.  With H = 50 the conv
    stack yields 3 time steps x 32 channels = 96 features before the read-out.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        history_len: int = 50,
        latent_dim: int = 8,
        embed_dim: int = 32,
        conv_channels: int = 32,
        conv_kernels: Sequence[int] = (8, 5, 5),
        conv_strides: Sequence[int] = (4, 1, 1),
    ):
        super().__init__()
        if len(conv_kernels) != len(conv_strides):
            raise ValueError("conv_kernels and conv_strides must have the same length")
        self.feature_dim = int(feature_dim)
        self.history_len = int(history_len)
        self.latent_dim = int(latent_dim)
        self.embed = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, int(embed_dim))),
            nn.ELU(),
            layer_init(nn.Linear(int(embed_dim), int(embed_dim))),
            nn.ELU(),
        )
        convs: list[nn.Module] = []
        in_ch = int(embed_dim)
        for k, s in zip(conv_kernels, conv_strides):
            convs.append(nn.Conv1d(in_ch, int(conv_channels), kernel_size=int(k), stride=int(s)))
            convs.append(nn.ELU())
            in_ch = int(conv_channels)
        self.convs = nn.Sequential(*convs)
        try:
            with torch.no_grad():
                probe = torch.zeros(1, int(embed_dim), self.history_len)
                conv_out = self.convs(probe)
        except RuntimeError as exc:  # kernel larger than the (shrinking) time axis
            raise ValueError(
                f"history_len={self.history_len} is too short for the conv stack "
                f"(kernels {tuple(conv_kernels)}, strides {tuple(conv_strides)}): {exc}"
            ) from exc
        if conv_out.shape[-1] < 1:
            raise ValueError(
                f"history_len={self.history_len} is too short for the conv stack "
                f"(kernels {tuple(conv_kernels)}, strides {tuple(conv_strides)})"
            )
        self.conv_out_dim = int(conv_out.numel())
        self.head = layer_init(nn.Linear(self.conv_out_dim, self.latent_dim), std=1.0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 2:
            history = history.unsqueeze(0)
        if history.shape[-1] != self.feature_dim or history.shape[-2] != self.history_len:
            raise ValueError(
                f"expected history of shape (B, {self.history_len}, {self.feature_dim}), got {tuple(history.shape)}"
            )
        emb = self.embed(history)  # (B, H, E)
        emb = emb.transpose(1, 2)  # (B, E, H) for Conv1d
        feat = self.convs(emb).flatten(1)
        return self.head(feat)


class AdaptedActor(nn.Module):
    """Deployment policy pi(x_t, a_{t-1}, phi(history)).  mu is not used."""

    def __init__(self, base: RMAActor, adaptation: AdaptationModule):
        super().__init__()
        if adaptation.latent_dim != base.latent_dim:
            raise ValueError("adaptation module latent_dim must match the base policy's latent_dim")
        self.base = base
        self.adaptation = adaptation

    def predict_latent(self, history: torch.Tensor) -> torch.Tensor:
        return self.adaptation(history)

    def get_action(self, obs: torch.Tensor, prev_action: torch.Tensor | None, history: torch.Tensor) -> torch.Tensor:
        latent = self.adaptation(history)
        return self.base.get_action_from_latent(obs, prev_action, latent)


class HistoryActor(_LatentConditionedActor):
    """Long-history TD3 baseline: pi(x_t, a_{t-1}, phi(x_{t-H:t-1}, a_{t-H:t-1})), no privileged input.

    The fair control for RMA: exactly the deployable RMA policy's inputs and
    architecture (same H-step window, same conv encoder ``AdaptationModule``,
    same trunk), but the encoder is trained end-to-end by the RL objective
    instead of being supervised on a privileged latent.  Policy observation
    layout produced by ``HistoryEnvVector`` + the trainer::

        [ x_t (obs_dim) | window (history_len * feature_dim, flattened (H, F)) | a_{t-1} (act_dim, optional) ]
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        history_len: int = 50,
        feature_dim: int = 8,
        latent_dim: int = 8,
        use_last_action: bool = True,
        hidden_layer_size: int = 64,
        num_hidden_layers: int = 2,
        embed_dim: int = 32,
        conv_channels: int = 32,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.window_dim = self.history_len * self.feature_dim
        self.latent_dim = int(latent_dim)
        self.use_last_action = bool(use_last_action)
        self.encoder = AdaptationModule(
            feature_dim=self.feature_dim, history_len=self.history_len, latent_dim=self.latent_dim,
            embed_dim=int(embed_dim), conv_channels=int(conv_channels),
        )
        policy_in = self.obs_dim + (self.act_dim if self.use_last_action else 0) + self.latent_dim
        self._build_policy(
            policy_in=policy_in, act_dim=self.act_dim, hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers, action_scale=action_scale, action_bias=action_bias,
        )

    @property
    def policy_obs_dim(self) -> int:
        return self.obs_dim + self.window_dim + (self.act_dim if self.use_last_action else 0)

    def split_policy_obs(self, policy_obs: torch.Tensor):
        obs = policy_obs[..., : self.obs_dim]
        window = policy_obs[..., self.obs_dim : self.obs_dim + self.window_dim]
        window = window.reshape(*window.shape[:-1], self.history_len, self.feature_dim)
        prev_action = policy_obs[..., self.obs_dim + self.window_dim :] if self.use_last_action else None
        return obs, window, prev_action

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        obs, window, prev_action = self.split_policy_obs(policy_obs)
        if window.dim() == 2:
            window = window.unsqueeze(0)
        latent = self.encoder(window)
        return self.get_action_from_latent(obs, prev_action, latent)


# ------------------------------------------------------------ step features
STEP_FEATURE_MODES = ("latest_frame", "full_obs")

# Canonical 30-dim ``history`` observation layout (see CLAUDE.md):
# [0:15] paddle 5 x [x, y, valid] oldest->newest, [15:30] puck 5 x [x, y, valid].
_LATEST_PADDLE = slice(12, 15)
_LATEST_PUCK = slice(27, 30)


def step_feature_dim(obs_dim: int, act_dim: int, mode: str) -> int:
    if mode == "latest_frame":
        # The canonical 30-dim history observation, optionally followed by a desired goal
        # (goal-conditioned tasks append it after the 30 dims, so the slices below still apply).
        if obs_dim < 30:
            raise ValueError(
                "step_features='latest_frame' assumes the canonical 30-dim history observation "
                f"(plus an optional goal suffix); got obs_dim={obs_dim}. Use step_features='full_obs'."
            )
        return 6 + int(act_dim)
    if mode == "full_obs":
        return int(obs_dim) + int(act_dim)
    raise ValueError(f"unknown step feature mode {mode!r}; choose from {STEP_FEATURE_MODES}")


def step_state_features(obs: torch.Tensor, mode: str) -> torch.Tensor:
    """The x_t part of a history entry (torch, any leading batch dims)."""
    if mode == "latest_frame":
        return torch.cat([obs[..., _LATEST_PADDLE], obs[..., _LATEST_PUCK]], dim=-1)
    if mode == "full_obs":
        return obs
    raise ValueError(f"unknown step feature mode {mode!r}; choose from {STEP_FEATURE_MODES}")
