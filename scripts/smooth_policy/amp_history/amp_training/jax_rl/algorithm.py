"""Protocol defining the interface every JAX RL algorithm must implement.

A concrete algorithm (e.g. TD3, SAC) satisfies this protocol by providing
the four methods below.  The training loop in *_training.py only calls these
methods plus the shared replay-buffer / logging infrastructure, so swapping
algorithms is a one-line change.

Usage::

    from jax_rl.algorithm import RLAlgorithm
    from jax_rl.td3.td3_algorithm import TD3

    algo: RLAlgorithm = TD3(config)
    state = algo.init_train_state(key, obs_dim, act_dim)
    state, metrics = algo.update_critic(state, batch, key)
    state, metrics = algo.update_actor(state, batch, key)
    action = algo.select_action(state.actor.params, obs)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax


@runtime_checkable
class RLAlgorithm(Protocol):
    def init_train_state(self, key: jax.Array, obs_dim: int, act_dim: int) -> Any:
        """Initialise all networks and optimizers, return an ActorCriticTrainState."""
        ...

    def update_critic(
        self, state: Any, batch: dict[str, jax.Array], key: jax.Array
    ) -> tuple[Any, dict[str, float]]:
        """One critic gradient step. Returns (new_state, scalar_metrics)."""
        ...

    def update_actor(
        self, state: Any, batch: dict[str, jax.Array], key: jax.Array
    ) -> tuple[Any, dict[str, float]]:
        """One actor gradient step. Returns (new_state, scalar_metrics)."""
        ...

    def select_action(self, actor_params: Any, obs: jax.Array) -> jax.Array:
        """Deterministic action for a batch of observations (no exploration noise)."""
        ...
