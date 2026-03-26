"""Flax neural network modules.

Ports of the PyTorch originals:
  - ResidualDenseNormSwishBlock  →  ResidualBlock
  - ResidualMLPTrunk             →  ResidualTrunk
  - DeterministicAgent           →  DeterministicActor
  - TD3DualHeadQNetwork          →  DualHeadQNetwork
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


def layer_init(module: nn.Module, std: float = 1.0):
    """Return an initializer that scales weights by `std` and zeros biases.

    Used as `kernel_init` / `bias_init` kwargs in nn.Dense.
    """
    # Flax Dense accepts callables conforming to (key, shape, dtype) -> array.
    kernel_init = nn.initializers.variance_scaling(
        scale=std ** 2,
        mode="fan_in",
        distribution="truncated_normal",
    )
    bias_init = nn.initializers.zeros
    return kernel_init, bias_init

class ResidualBlock(nn.Module):
    """Dense residual block with LayerNorm + Swish activation.

    Mirrors PyTorch ResidualDenseNormSwishBlock:
      Linear(in→h) → LayerNorm → Swish
      Linear(h→h)  → LayerNorm → Swish
      Linear(h→h)  → LayerNorm → Swish
      Linear(h→h)  → LayerNorm → Swish
      + residual projection (Linear(in→h)) if in_dim != hidden_dim
    """

    hidden_dim: int
    units_per_block: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_dim = x.shape[-1]
        h = x
        for _ in range(self.units_per_block):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.LayerNorm()(h)
            h = nn.swish(h)

        # Residual projection when dimensions differ
        if in_dim != self.hidden_dim:
            x = nn.Dense(self.hidden_dim)(x)
        return h + x


class ResidualTrunk(nn.Module):
    """Stack of ResidualBlocks — shared backbone for actor and critic.

    Args:
        hidden_dim: Width of each hidden layer.
        num_blocks: Number of residual blocks.
        units_per_block: Dense layers per block (default 4).
    """

    hidden_dim: int
    num_blocks: int
    units_per_block: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.num_blocks):
            x = ResidualBlock(
                hidden_dim=self.hidden_dim,
                units_per_block=self.units_per_block,
            )(x)
        return x


class DeterministicActor(nn.Module):
    """Deterministic policy: tanh(linear(trunk(obs))) * scale + bias.

    Mirrors DeterministicAgent from deterministic_agent.py.

    Args:
        act_dim: Dimensionality of the action space.
        hidden_dim: Width of trunk hidden layers.
        num_blocks: Number of residual blocks in the trunk.
        action_scale: Multiplier applied after tanh.
        action_bias: Additive bias applied after scale.
    """

    act_dim: int
    hidden_dim: int = 64
    num_blocks: int = 2
    action_scale: float = 0.02
    action_bias: float = 0.0

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = ResidualTrunk(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks)(obs)
        # Mean head: std=1 init (same as layer_init in PyTorch version)
        mean = nn.Dense(self.act_dim)(x)
        return jnp.tanh(mean) * self.action_scale + self.action_bias


class DualHeadQNetwork(nn.Module):
    """TD3 critic with one trunk and two scalar heads (task + motion).

    Mirrors TD3DualHeadQNetwork from helper/dual_head_q.py.

    Args:
        hidden_dim: Width of trunk hidden layers.
        num_blocks: Number of residual blocks in the trunk.
    """

    hidden_dim: int = 128
    num_blocks: int = 2

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([obs, action], axis=-1)
        x = ResidualTrunk(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks)(x)
        # Small init (std=0.01) keeps initial Q-estimates near zero
        q_task = nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.01))(x)
        q_motion = nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.01))(x)
        return q_task, q_motion
