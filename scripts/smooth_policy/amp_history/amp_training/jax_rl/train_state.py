"""Generic training state container for JAX RL algorithms.

RLTrainState holds all mutable training objects as a plain NamedTuple so it
can be passed through jax.jit boundaries as a pytree.  Algorithm-specific
fields (e.g. log_alpha for SAC) go in the subclass or alongside it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import optax
from flax.training.train_state import TrainState


class ActorCriticTrainState(NamedTuple):
    """Mutable state for a twin-critic actor-critic algorithm (TD3, SAC, …).

    Fields:
        actor:              Flax TrainState — actor params + optimizer state.
        qf1, qf2:          Flax TrainState — critic params + optimizer states.
        actor_target:      Frozen copy of actor params for target network.
        qf1_target,
        qf2_target:        Frozen copies of critic params for target networks.
        step:              Global gradient-update step counter.
    """

    actor: TrainState
    qf1: TrainState
    qf2: TrainState
    actor_target: Any   # pytree of params
    qf1_target: Any
    qf2_target: Any
    step: int


def make_train_state(
    actor_module,
    qf_module,
    obs_dim: int,
    act_dim: int,
    policy_obs_dim: int,
    actor_lr: float,
    q_lr: float,
    q_weight_decay: float,
    key,
) -> ActorCriticTrainState:
    """Initialise networks and wrap in ActorCriticTrainState.

    Args:
        actor_module:    Flax module instance (DeterministicActor or similar).
        qf_module:       Flax module instance (DualHeadQNetwork or similar).
                         Two independent copies are created for qf1/qf2.
        obs_dim:         Raw observation dimension (used for critic input).
        act_dim:         Action dimension.
        policy_obs_dim:  Observation dimension seen by the actor
                         (= obs_dim + act_dim when use_last_action is True).
        actor_lr:        Learning rate for the actor Adam optimizer.
        q_lr:            Learning rate for the critic Adam optimizer.
        q_weight_decay:  L2 weight decay for the critic optimizer.
        key:             JAX PRNG key — split internally for each network init.
    """
    import jax
    import jax.numpy as jnp

    key, k1, k2, k3 = jax.random.split(key, 4)

    dummy_policy_obs = jnp.zeros((1, policy_obs_dim))
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, act_dim))

    actor_params = actor_module.init(k1, dummy_policy_obs)
    qf1_params   = qf_module.init(k2, dummy_obs, dummy_act)
    qf2_params   = qf_module.init(k3, dummy_obs, dummy_act)

    actor_tx = optax.adam(actor_lr)
    q_tx = optax.adamw(q_lr, weight_decay=q_weight_decay)

    actor_ts = TrainState.create(apply_fn=actor_module.apply, params=actor_params, tx=actor_tx)
    qf1_ts   = TrainState.create(apply_fn=qf_module.apply,   params=qf1_params,   tx=q_tx)
    qf2_ts   = TrainState.create(apply_fn=qf_module.apply,   params=qf2_params,   tx=q_tx)

    return ActorCriticTrainState(
        actor=actor_ts,
        qf1=qf1_ts,
        qf2=qf2_ts,
        actor_target=actor_params,
        qf1_target=qf1_params,
        qf2_target=qf2_params,
        step=0,
    )
