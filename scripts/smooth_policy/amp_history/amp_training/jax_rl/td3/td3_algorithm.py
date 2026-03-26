"""TD3 algorithm: JIT-compiled critic/actor update steps.

Implements the RLAlgorithm protocol from jax_rl/algorithm.py.

Update logic mirrors td3_training.py:
  - Transformed Bellman targets via h_transform / h_inverse
  - Twin critics with separate task and motion heads
  - Deterministic actor maximising (1-gamma)*Q
  - Polyak-averaged target networks
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from jax_rl.networks import DeterministicActor, DualHeadQNetwork
from jax_rl.train_state import ActorCriticTrainState, make_train_state
from jax_rl.utils import h_inverse, h_transform, soft_update
from jax_rl.td3.td3_config import TD3Config


class TD3:
    """TD3 algorithm.

    Args:
        config:      TD3Config holding all hyperparameters.
        action_low:  Lower bound on actions, shape (act_dim,).
        action_high: Upper bound on actions, shape (act_dim,).
    """

    def __init__(self, config: TD3Config, action_low: jax.Array, action_high: jax.Array):
        self.config = config
        self.action_low = action_low
        self.action_high = action_high
        self._actor_module: DeterministicActor | None = None

    def init_train_state(
        self, key: jax.Array, obs_dim: int, act_dim: int
    ) -> ActorCriticTrainState:
        cfg = self.config
        policy_obs_dim = obs_dim + act_dim if cfg.use_last_action_in_policy_state else obs_dim
        actor_module = DeterministicActor(
            act_dim=act_dim,
            hidden_dim=cfg.actor_hidden_dim,
            num_blocks=cfg.actor_num_blocks,
            action_scale=cfg.action_scale,
            action_bias=0.0,
        )
        qf_module = DualHeadQNetwork(hidden_dim=cfg.q_hidden_dim, num_blocks=cfg.q_num_blocks)
        self._actor_module = actor_module
        self._qf_module = qf_module
        return make_train_state(
            actor_module=actor_module,
            qf_module=qf_module,
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_obs_dim=policy_obs_dim,
            actor_lr=cfg.policy_lr,
            q_lr=cfg.q_lr,
            q_weight_decay=cfg.q_weight_decay,
            key=key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_critic(
        self,
        state: ActorCriticTrainState,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[ActorCriticTrainState, dict[str, jax.Array], jax.Array]:
        """One critic gradient step.

        Returns (new_state, scalar_metrics, td_errors) where td_errors is
        shape (batch_size,) and used to update PER priorities.
        """
        cfg = self.config
        obs          = batch["observations"]
        next_obs     = batch["next_observations"]
        actions      = batch["actions"]
        task_r       = batch["task_rewards"]
        motion_r     = batch["motion_rewards"]
        dones        = batch["dones"]
        weights      = batch.get("weights", jnp.ones_like(task_r))

        next_prev_actions = actions * (1.0 - dones[:, None])
        next_policy_obs = (
            jnp.concatenate([next_obs, next_prev_actions], axis=-1)
            if cfg.use_last_action_in_policy_state
            else next_obs
        )

        target_next_action = state.actor.apply_fn(state.actor_target, next_policy_obs)
        noise = jnp.clip(
            jax.random.normal(key, target_next_action.shape) * cfg.policy_noise,
            -cfg.noise_clip,
            cfg.noise_clip,
        )
        target_next_action = jnp.clip(target_next_action + noise, self.action_low, self.action_high)

        q1_next_task_h, q1_next_motion_h = state.qf1.apply_fn(state.qf1_target, next_obs, target_next_action)
        q2_next_task_h, q2_next_motion_h = state.qf2.apply_fn(state.qf2_target, next_obs, target_next_action)

        min_next_task   = h_inverse(jnp.minimum(q1_next_task_h,   q2_next_task_h  ).squeeze(-1), cfg.h_transform_eps)
        min_next_motion = h_inverse(jnp.minimum(q1_next_motion_h, q2_next_motion_h).squeeze(-1), cfg.h_transform_eps)

        target_task_h   = h_transform(task_r   + (1.0 - dones) * cfg.task_gamma   * min_next_task,   cfg.h_transform_eps)
        target_motion_h = h_transform(motion_r + (1.0 - dones) * cfg.motion_gamma * min_next_motion, cfg.h_transform_eps)

        def critic_loss_fn(params):
            qf1_params, qf2_params = params
            q1_task_h, q1_motion_h = state.qf1.apply_fn(qf1_params, obs, actions)
            q2_task_h, q2_motion_h = state.qf2.apply_fn(qf2_params, obs, actions)
            q1_task_h   = q1_task_h.squeeze(-1)
            q1_motion_h = q1_motion_h.squeeze(-1)
            q2_task_h   = q2_task_h.squeeze(-1)
            q2_motion_h = q2_motion_h.squeeze(-1)

            q1_task_loss   = (weights * jnp.square(q1_task_h   - target_task_h  )).mean()
            q2_task_loss   = (weights * jnp.square(q2_task_h   - target_task_h  )).mean()
            q1_motion_loss = (weights * jnp.square(q1_motion_h - target_motion_h)).mean()
            q2_motion_loss = (weights * jnp.square(q2_motion_h - target_motion_h)).mean()
            total = q1_task_loss + q2_task_loss + q1_motion_loss + q2_motion_loss

            td_error = 0.25 * (
                jnp.abs(q1_task_h - target_task_h)
                + jnp.abs(q2_task_h - target_task_h)
                + jnp.abs(q1_motion_h - target_motion_h)
                + jnp.abs(q2_motion_h - target_motion_h)
            )
            metrics = {
                "losses/q_task_loss":    (q1_task_loss   + q2_task_loss)   / 2.0,
                "losses/q_motion_loss":  (q1_motion_loss + q2_motion_loss) / 2.0,
                "losses/q_total_loss":   total,
                "losses/q1_task_mean":   q1_task_h.mean(),
                "losses/q1_motion_mean": q1_motion_h.mean(),
            }
            return total, (td_error, metrics)

        (_, (td_error, metrics)), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )((state.qf1.params, state.qf2.params))

        new_qf1 = state.qf1.apply_gradients(grads=grads[0])
        new_qf2 = state.qf2.apply_gradients(grads=grads[1])
        new_qf1_target   = soft_update(state.qf1_target,   new_qf1.params,      cfg.tau)
        new_qf2_target   = soft_update(state.qf2_target,   new_qf2.params,      cfg.tau)
        new_actor_target = soft_update(state.actor_target, state.actor.params,   cfg.tau)

        new_state = state._replace(
            qf1=new_qf1,
            qf2=new_qf2,
            qf1_target=new_qf1_target,
            qf2_target=new_qf2_target,
            actor_target=new_actor_target,
            step=state.step + 1,
        )
        return new_state, metrics, td_error

    @partial(jax.jit, static_argnums=(0,))
    def update_actor(
        self,
        state: ActorCriticTrainState,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[ActorCriticTrainState, dict[str, jax.Array]]:
        """One actor gradient step. Returns (new_state, scalar_metrics)."""
        cfg = self.config
        obs          = batch["observations"]
        prev_actions = batch["prev_actions"]

        policy_obs = (
            jnp.concatenate([obs, prev_actions], axis=-1)
            if cfg.use_last_action_in_policy_state
            else obs
        )

        def actor_loss_fn(actor_params):
            current_actions = state.actor.apply_fn(actor_params, policy_obs)
            q1_task_h, q1_motion_h = state.qf1.apply_fn(state.qf1.params, obs, current_actions)
            q1_task   = h_inverse(q1_task_h.squeeze(-1),   cfg.h_transform_eps)
            q1_motion = h_inverse(q1_motion_h.squeeze(-1), cfg.h_transform_eps)
            norm_task   = (1.0 - cfg.task_gamma)   * q1_task
            norm_motion = (1.0 - cfg.motion_gamma) * q1_motion
            objective = cfg.task_reward_weight * norm_task + cfg.motion_reward_weight * norm_motion
            loss = -objective.mean()
            metrics = {
                "losses/actor_loss":             loss,
                "losses/actor_norm_task_mean":   norm_task.mean(),
                "losses/actor_norm_motion_mean": norm_motion.mean(),
            }
            return loss, metrics

        (_, metrics), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(state.actor.params)
        new_actor = state.actor.apply_gradients(grads=grads)
        return state._replace(actor=new_actor), metrics

    @partial(jax.jit, static_argnums=(0,))
    def select_action(self, actor_params: Any, obs: jax.Array) -> jax.Array:
        return self._actor_module.apply(actor_params, obs)
