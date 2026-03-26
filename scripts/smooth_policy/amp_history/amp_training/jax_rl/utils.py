"""Shared JAX utilities: value transforms, reward shaping, training helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def h_transform(x: jax.Array, eps: float = 1e-3) -> jax.Array:
    """Invertible value scaling that compresses large magnitudes."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def h_inverse(x: jax.Array, eps: float = 1e-3) -> jax.Array:
    """Inverse of h_transform."""
    abs_x = jnp.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    quotient = (jnp.sqrt(inner) - 1) / (2 * eps)
    return jnp.sign(x) * (quotient ** 2 - 1)


def soft_update(target_params, online_params, tau: float):
    """Polyak-average online params into target params: target = tau*online + (1-tau)*target."""
    return jax.tree_util.tree_map(
        lambda t, o: tau * o + (1.0 - tau) * t,
        target_params,
        online_params,
    )


def linear_anneal(start: float, end: float, step: int, anneal_steps: int) -> float:
    """Linearly interpolate from start to end over anneal_steps."""
    if anneal_steps <= 0:
        return end
    progress = min(max(step, 0) / float(anneal_steps), 1.0)
    return start + progress * (end - start)


def velocity_reward(velocity_mag: jax.Array, at_one: float, at_zero: float) -> jax.Array:
    """Reward in [−∞, 1] that equals 1 when velocity ≤ at_one and 0 when velocity ≥ at_zero."""
    denom = max(at_zero - at_one, 1e-6)
    return jnp.clip(1.0 - (velocity_mag - at_one) / denom, a_max=1.0)


def jerk_reward(jerk_mag: jax.Array, at_one: float, at_zero: float) -> jax.Array:
    """Reward in [−∞, 1] that equals 1 when jerk ≤ at_one and 0 when jerk ≥ at_zero."""
    denom = max(at_zero - at_one, 1e-6)
    return 1.0 - (jerk_mag - at_one) / denom
