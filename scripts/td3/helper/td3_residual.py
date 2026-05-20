"""Residual-RL accommodation for TD3 training.

Bundles the residual actor construction and residual-only optimizer so the
training loop does not have to branch on `checkpoint_load_mode == "residual"`
in multiple places.

Use:

    if residual_mode:
        actor, actor_target, actor_optimizer = build_residual_training(...)
    else:
        # normal actor / actor_target / actor_optimizer construction
"""

from typing import Tuple

import torch
import torch.optim as optim

from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.residual_agent import ResidualActor, zero_init_residual_head


def build_residual_training(
    *,
    base_actor: DeterministicAgent,
    policy_env_view,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    device: str,
    residual_scale: float,
    residual_weight_decay: float,
    agent_hidden_layer_size: int,
    agent_num_hidden_layers: int,
    policy_lr: float,
) -> Tuple[ResidualActor, ResidualActor, optim.Optimizer]:
    """Wrap a (already-base-weights-loaded) actor in ResidualActor and build the
    residual-only optimizer.

    Returns (actor, actor_target, actor_optimizer). The caller is responsible
    for having loaded the base policy weights into `base_actor` before calling
    this.
    """
    residual_online = DeterministicAgent(
        policy_env_view,
        action_scale=residual_scale,
        action_bias=0.0,
        hidden_layer_size=agent_hidden_layer_size,
        num_hidden_layers=agent_num_hidden_layers,
    ).to(device)
    residual_target = DeterministicAgent(
        policy_env_view,
        action_scale=residual_scale,
        action_bias=0.0,
        hidden_layer_size=agent_hidden_layer_size,
        num_hidden_layers=agent_num_hidden_layers,
    ).to(device)
    zero_init_residual_head(residual_online)
    zero_init_residual_head(residual_target)
    residual_target.load_state_dict(residual_online.state_dict())

    actor = ResidualActor(
        base_actor=base_actor,
        residual_actor=residual_online,
        action_low=action_low,
        action_high=action_high,
    ).to(device)
    actor_target = ResidualActor(
        base_actor=actor.base,
        residual_actor=residual_target,
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    actor_optimizer = optim.Adam(
        actor.residual.parameters(),
        lr=policy_lr,
        weight_decay=residual_weight_decay,
    )
    if residual_weight_decay > 0:
        print(
            f"Residual actor optimizer: Adam(lr={policy_lr}, "
            f"weight_decay={residual_weight_decay}) — residual head L2 active"
        )
    print(
        f"Residual mode: base frozen, residual_scale={residual_scale},"
        " critic from scratch."
    )

    return actor, actor_target, actor_optimizer
