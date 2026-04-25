"""Residual actor wrapper for residual RL fine-tuning.

Method: π(s) = clip(π_base(s) + π_residual(s), action_low, action_high), with
π_base frozen and π_residual trained from scratch. Initialised so that
π_residual(s) == 0 at t=0, i.e. the wrapped policy starts equal to the base.

References:
    Silver et al., "Residual Policy Learning" (arXiv:1812.06298, 2018).
    Johannink et al., "Residual RL for Robot Control" (ICRA 2019).

See notes/scratch/residual_rl_plan.md for the design rationale.
"""

from typing import Union

import torch
import torch.nn as nn

from scripts.smooth_policy.deterministic_agent import DeterministicAgent


class ResidualActor(nn.Module):
    """Wraps a frozen base actor and a trainable residual actor.

    Exposes the same get_action / forward API as DeterministicAgent so it is a
    drop-in replacement at every TD3 call site that goes through
    `deterministic_actor_action(actor, ...)`.
    """

    def __init__(
        self,
        base_actor: DeterministicAgent,
        residual_actor: DeterministicAgent,
        action_low: Union[float, torch.Tensor],
        action_high: Union[float, torch.Tensor],
    ):
        super().__init__()
        self.base = base_actor.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.residual = residual_actor
        self.register_buffer(
            "action_low", torch.as_tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(action_high, dtype=torch.float32)
        )

    @property
    def action_scale(self) -> torch.Tensor:
        # Surfaced for callers that read actor.action_scale (e.g. the fallback
        # branch in deterministic_actor_action). The base's value is the one
        # trained against, not the residual's small bound.
        return self.base.action_scale

    @property
    def action_bias(self) -> torch.Tensor:
        return self.base.action_bias

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a_base = self.base.get_action(x)
        a_res = self.residual.get_action(x)
        return torch.clamp(a_base + a_res, self.action_low, self.action_high)

    def get_action_mean(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "ResidualActor exposes get_action only — pre-tanh means are not"
            " well-defined for the additive residual."
        )

    def forward(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self.get_action(x)


def zero_init_residual_head(residual: DeterministicAgent) -> None:
    """Zero the residual's final linear head so residual(s) == 0 at init.

    With tanh(0) == 0, this guarantees the wrapped policy starts equal to the
    base policy and there is no behavior regression when fine-tuning begins.
    """
    nn.init.zeros_(residual.actor_mean_head.weight)
    nn.init.zeros_(residual.actor_mean_head.bias)
