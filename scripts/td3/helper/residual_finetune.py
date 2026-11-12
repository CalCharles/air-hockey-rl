"""Residual-policy fine-tuning helpers for online (real-robot / sim dry run) fine-tuning.

The sim-trained actor is frozen (the *base*) and a second network of the **same class and
architecture** (the *residual*) is trained on top of it:

    a = clamp(base(s) + residual(s), action_low, action_high)

* residual = a fresh instance built by the same factory as the base (random init, as in
  ``scripts/td3/helper/td3_residual.py``), its output layer ``actor_mean_head`` zeroed
  (``zero_init_residual_head``) so ``residual(s) == 0`` and the wrapped policy equals the
  base at the start, and its tanh output scaled by ``residual_scale`` (``action_scale``
  buffer) so each action dimension moves by at most ±residual_scale;
* only the residual's parameters get an optimizer; the base runs under ``no_grad`` with
  ``requires_grad=False`` (``scripts.td3.residual_agent.ResidualActor``).

For the RMA ``history_privileged`` policies both networks are ``PrivilegedHistoryActor`` s,
so the residual has its own 50-step conv encoder and reads the same policy observation
``[x_t | window | s_t | a_{t-1}]`` (it skips ``s_t`` exactly as the base does).

Checkpoints: the wrapped actor's state dict holds ``base.*`` / ``residual.*`` /
``action_low`` / ``action_high``. ``rma_meta.json`` next to an online checkpoint gets a
``"residual": {"scale": ...}`` entry (``RESIDUAL_META_KEY``) so ``scripts.rma.bundle.load_phase1``
rebuilds the pair as a ``ResidualHistoryActor`` for evaluation (``async_td3_real_eval.py
--agent history`` and the sim evals that go through ``HistoryAgent``).
"""

from __future__ import annotations

import copy
from typing import Callable, Mapping, Union

import torch
import torch.nn as nn

from scripts.td3.residual_agent import ResidualActor, zero_init_residual_head

RESIDUAL_META_KEY = "residual"
_BASE_PREFIX = "base."
_RESIDUAL_PREFIX = "residual."

TensorLike = Union[float, torch.Tensor]


def is_residual_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """True for a ``ResidualActor`` state dict (an online residual checkpoint)."""
    return any(key.startswith(_RESIDUAL_PREFIX) for key in state_dict)


def base_state_from_residual(state_dict: Mapping[str, torch.Tensor]) -> dict:
    return {key[len(_BASE_PREFIX):]: value for key, value in state_dict.items() if key.startswith(_BASE_PREFIX)}


def make_residual_head(make_actor: Callable[[], nn.Module], residual_scale: float) -> nn.Module:
    """Same class / architecture as the base, random init, zero output layer, output × residual_scale."""
    head = make_actor()
    if not hasattr(head, "action_scale") or not hasattr(head, "actor_mean_head"):
        raise TypeError(f"{type(head).__name__} has no action_scale / actor_mean_head; cannot be a residual head")
    with torch.no_grad():
        head.action_scale.fill_(float(residual_scale))
        if hasattr(head, "action_bias"):
            head.action_bias.fill_(0.0)
    zero_init_residual_head(head)
    return head


def build_residual_actor_pair(
    make_actor: Callable[[], nn.Module],
    *,
    base_state: Mapping[str, torch.Tensor] | None,
    residual_scale: float,
    action_low: TensorLike,
    action_high: TensorLike,
    device: torch.device | str,
) -> tuple[ResidualActor, ResidualActor]:
    """(actor, actor_target): frozen base (``base_state`` loaded strictly) + fresh zero-output
    residual; the target shares the frozen base and holds a copy of the residual."""
    base = make_actor()
    if base_state is not None:
        base.load_state_dict(dict(base_state), strict=True)
    residual = make_residual_head(make_actor, residual_scale)
    actor = ResidualActor(base, residual, action_low, action_high).to(device)
    actor_target = ResidualActor(actor.base, copy.deepcopy(actor.residual), action_low, action_high).to(device)
    return actor, actor_target


def build_residual_actor_structure(
    make_actor: Callable[[], nn.Module],
    *,
    residual_scale: float,
    action_low: TensorLike,
    action_high: TensorLike,
    device: torch.device | str,
) -> ResidualActor:
    """The wrapper with placeholder weights (the collector's copy; filled by a strict state-dict sync)."""
    return ResidualActor(make_actor(), make_residual_head(make_actor, residual_scale), action_low, action_high).to(device)


class ResidualHistoryActor(nn.Module):
    """Evaluation form of a residual ``HistoryActor`` pair for ``scripts.rma.evaluate.HistoryAgent``.

    ``HistoryAgent.act`` calls ``actor.encoder(window)`` once and then
    ``actor.get_action_from_latent(obs, prev, z)``. Here ``encoder`` returns the two
    networks' latents concatenated ``[z_base | z_residual]`` and ``get_action_from_latent``
    splits them, so the output is exactly ``clamp(base + residual)`` of the training wrapper.
    The state dict (``base.*`` / ``residual.*`` / ``action_low`` / ``action_high``) is the
    training ``ResidualActor``'s, so ``model.pth`` loads strictly.
    """

    def __init__(self, base: nn.Module, residual: nn.Module):
        super().__init__()
        if type(base) is not type(residual):
            raise TypeError(f"base {type(base).__name__} and residual {type(residual).__name__} must be the same class")
        self.base = base
        self.residual = residual
        # Placeholders of the training wrapper's shape (act_dim,); the real bounds load from model.pth.
        self.register_buffer("action_low", torch.full((int(base.act_dim),), -1.0))
        self.register_buffer("action_high", torch.full((int(base.act_dim),), 1.0))
        for name in ("obs_dim", "act_dim", "history_len", "feature_dim", "use_last_action"):
            setattr(self, name, getattr(base, name))
        self.step_features = getattr(base, "step_features", "latest_frame")
        self._base_latent_dim = int(base.latent_dim)

    @property
    def latent_dim(self) -> int:
        return self._base_latent_dim + int(self.residual.latent_dim)

    def encoder(self, window: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.base.encoder(window), self.residual.encoder(window)], dim=-1)

    def get_action_from_latent(self, obs: torch.Tensor, prev_action: torch.Tensor | None, latent: torch.Tensor) -> torch.Tensor:
        z_base, z_res = latent[..., : self._base_latent_dim], latent[..., self._base_latent_dim :]
        a = self.base.get_action_from_latent(obs, prev_action, z_base) + self.residual.get_action_from_latent(obs, prev_action, z_res)
        return torch.clamp(a, self.action_low, self.action_high)

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        a = self.base.get_action(policy_obs) + self.residual.get_action(policy_obs)
        return torch.clamp(a, self.action_low, self.action_high)
