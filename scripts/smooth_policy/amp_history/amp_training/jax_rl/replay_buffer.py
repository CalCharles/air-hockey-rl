"""jax_rl/buffers.py — replay buffers and episode routing.

Three classes + one function, all in one file:

    ReplayBuffer             — uniform ring-buffer
    PrioritizedReplayBuffer  — proportional PER with IS weights
    EpisodeBuffer            — per-episode accumulator
    finalize_episode_if_done — routes completed episode to success/failure buffer

Design notes:
  - All arrays are NumPy on CPU. Conversion to jnp happens once at sample time
    in the training loop, not here. This avoids host↔device transfers on every
    env step write.
  - ReplayBuffer.sample() always returns the same keys as
    PrioritizedReplayBuffer.sample() (indices, weights, sampled_priorities)
    so concat_replay_samples() in jax_rl/sampling.py never needs key-presence
    checks when mixing uniform and PER batches.
  - task_rewards / motion_rewards support dual-reward algorithms (TD3 with
    motion shaping). Single-reward algorithms pass motion_reward=0.0 and
    ignore the motion_rewards key in sampled batches — zero overhead.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np


class ReplayBuffer:
    """Uniform ring-buffer.

    Fields stored per transition:
        observations, next_observations  — env obs before/after step
        actions, prev_actions            — action taken; previous action
                                           (needed for use_last_action_in_policy_state)
        task_rewards                     — primary env reward
        motion_rewards                   — auxiliary shaping reward (0 if unused)
        dones                            — terminal flag (float32, 1.0 = terminal)

    Args:
        buffer_size:  Maximum number of transitions.
        obs_shape:    Shape of a single observation, e.g. (30,).
        action_shape: Shape of a single action, e.g. (2,).
    """

    def __init__(self, buffer_size: int, obs_shape: tuple, action_shape: tuple):
        self.buffer_size  = int(buffer_size)
        self.obs_shape    = tuple(obs_shape)
        self.action_shape = tuple(action_shape)

        self.observations      = np.zeros((buffer_size, *obs_shape),    dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape),    dtype=np.float32)
        self.actions           = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.prev_actions      = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.task_rewards      = np.zeros(buffer_size,                  dtype=np.float32)
        self.motion_rewards    = np.zeros(buffer_size,                  dtype=np.float32)
        self.dones             = np.zeros(buffer_size,                  dtype=np.float32)

        self.position = 0
        self.size     = 0

    def add(
        self,
        obs,
        next_obs,
        action,
        task_reward,
        motion_reward,
        done,
        prev_action,
    ) -> None:
        obs         = np.asarray(obs,         dtype=np.float32)
        next_obs    = np.asarray(next_obs,    dtype=np.float32)
        action      = np.asarray(action,      dtype=np.float32)
        prev_action = np.asarray(prev_action, dtype=np.float32)

        def _check(arr, expected_shape, name):
            if arr.shape == expected_shape:
                return arr.reshape(1, *expected_shape)
            if arr.ndim > 1 and arr.shape[1:] == expected_shape:
                return arr
            raise AssertionError(
                f"{name} shape mismatch: expected {expected_shape} or "
                f"(B, {expected_shape}), got {arr.shape}"
            )

        obs         = _check(obs,         self.obs_shape,    "obs")
        next_obs    = _check(next_obs,    self.obs_shape,    "next_obs")
        action      = _check(action,      self.action_shape, "action")
        prev_action = _check(prev_action, self.action_shape, "prev_action")

        task_reward   = np.asarray(task_reward,   dtype=np.float32).reshape(-1)
        motion_reward = np.asarray(motion_reward, dtype=np.float32).reshape(-1)
        done          = np.asarray(done,          dtype=np.float32).reshape(-1)

        batch = int(obs.shape[0])
        first = min(batch, self.buffer_size - self.position)
        s     = slice(self.position, self.position + first)

        self.observations[s]      = obs[:first]
        self.next_observations[s] = next_obs[:first]
        self.actions[s]           = action[:first]
        self.prev_actions[s]      = prev_action[:first]
        self.task_rewards[s]      = task_reward[:first]
        self.motion_rewards[s]    = motion_reward[:first]
        self.dones[s]             = done[:first]

        second = batch - first
        if second > 0:
            self.observations[:second]      = obs[first:]
            self.next_observations[:second] = next_obs[first:]
            self.actions[:second]           = action[first:]
            self.prev_actions[:second]      = prev_action[first:]
            self.task_rewards[:second]      = task_reward[first:]
            self.motion_rewards[:second]    = motion_reward[first:]
            self.dones[:second]             = done[first:]

        self.position = (self.position + batch) % self.buffer_size
        self.size     = min(self.size + batch, self.buffer_size)


    def _gather(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Collect fields at given indices. Shared by all samplers."""
        return {
            "observations":      self.observations[idx],
            "next_observations": self.next_observations[idx],
            "actions":           self.actions[idx],
            "prev_actions":      self.prev_actions[idx],
            "task_rewards":      self.task_rewards[idx],
            "motion_rewards":    self.motion_rewards[idx],
            "dones":             self.dones[idx],
        }

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Uniform random sample.

        Returns numpy arrays — caller converts to jnp.
        Includes indices/weights/sampled_priorities stubs so the return dict
        is always structurally identical to PrioritizedReplayBuffer.sample(),
        making concat_replay_samples() safe with mixed buffer types.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        idx    = np.random.randint(0, self.size, size=batch_size)
        result = self._gather(idx)
        result["indices"]            = idx
        result["weights"]            = np.ones(batch_size,  dtype=np.float32)
        result["sampled_priorities"] = np.zeros(batch_size, dtype=np.float32)
        return result


    def state_dict(self) -> dict:
        return {
            "buffer_size":       self.buffer_size,
            "obs_shape":         self.obs_shape,
            "action_shape":      self.action_shape,
            "position":          self.position,
            "size":              self.size,
            "observations":      self.observations.copy(),
            "next_observations": self.next_observations.copy(),
            "actions":           self.actions.copy(),
            "prev_actions":      self.prev_actions.copy(),
            "task_rewards":      self.task_rewards.copy(),
            "motion_rewards":    self.motion_rewards.copy(),
            "dones":             self.dones.copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.position               = int(sd["position"])
        self.size                   = int(sd["size"])
        self.observations[:]        = sd["observations"]
        self.next_observations[:]   = sd["next_observations"]
        self.actions[:]             = sd["actions"]
        self.prev_actions[:]        = sd["prev_actions"]
        self.task_rewards[:]        = sd["task_rewards"]
        self.motion_rewards[:]      = sd["motion_rewards"]
        self.dones[:]               = sd["dones"]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER ring-buffer with importance-sampling weights.

    Extends ReplayBuffer with per-transition priorities stored in a parallel
    numpy array. Priority updates must be called after each critic step:

        td_errors_np = jax.device_get(td_errors)          # move off GPU first
        rb.update_priorities(indices, td_errors_np + eps)

    sample() and sample_uniform() return identical key sets so batches from
    both can be safely concatenated by concat_replay_samples().

    Args:
        alpha:        Priority exponent (0 = uniform, 1 = fully prioritised).
        priority_eps: Floor added to every priority (prevents zero-prob sampling).
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        action_shape: tuple,
        alpha: float = 0.6,
        priority_eps: float = 1e-6,
    ):
        super().__init__(buffer_size, obs_shape, action_shape)
        self.alpha        = float(alpha)
        self.priority_eps = float(priority_eps)
        self.priorities   = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0



    def add(self, obs, next_obs, action, task_reward, motion_reward, done, prev_action) -> None:
        # Compute batch size and position before super().add() advances self.position.
        batch   = int(np.asarray(obs, dtype=np.float32).reshape(-1, *self.obs_shape).shape[0])
        old_pos = self.position
        super().add(obs, next_obs, action, task_reward, motion_reward, done, prev_action)
        # New transitions get max priority so they are sampled at least once.
        priority_value = max(self.max_priority, self.priority_eps)
        for i in range(batch):
            self.priorities[(old_pos + i) % self.buffer_size] = priority_value



    def sample(self, batch_size: int, beta: float = 0.4) -> dict[str, np.ndarray]:
        """Priority-weighted sample with IS correction weights.

        Extra keys vs ReplayBuffer.sample():
            indices          — buffer positions (pass to update_priorities)
            weights          — IS correction weights normalised to [0, 1]
            sampled_priorities — raw priorities at sampled positions (for logging)
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        valid        = np.clip(self.priorities[: self.size], self.priority_eps, None)
        scaled       = valid ** self.alpha
        probs        = scaled / scaled.sum()
        idx          = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
        sample_probs = np.clip(probs[idx], 1e-12, None)
        weights      = (self.size * sample_probs) ** (-beta)
        weights      = (weights / weights.max()).astype(np.float32)
        result                       = self._gather(idx)
        result["indices"]            = idx
        result["weights"]            = weights
        result["sampled_priorities"] = valid[idx]
        return result

    def sample_uniform(self, batch_size: int) -> dict[str, np.ndarray]:
        """Uniform sample with the same keys as sample() for concat safety.

        Used for the uniform fraction of the PER+uniform critic batch mix.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        idx                          = np.random.randint(0, self.size, size=batch_size)
        result                       = self._gather(idx)
        result["indices"]            = idx
        result["weights"]            = np.ones(batch_size,  dtype=np.float32)
        result["sampled_priorities"] = np.clip(
            self.priorities[: self.size], self.priority_eps, None
        )[idx]
        return result



    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update stored priorities after a critic gradient step.

        Args:
            indices:    Buffer positions from the last sample() call.
                        Must already be on CPU (call jax.device_get first).
            priorities: Per-sample magnitudes, typically td_error + per_eps.
        """
        indices    = np.asarray(indices,    dtype=np.int64).reshape(-1)
        priorities = np.clip(
            np.asarray(priorities, dtype=np.float32).reshape(-1),
            self.priority_eps, None,
        )
        if indices.size == 0:
            return
        self.priorities[indices] = priorities
        self.max_priority        = max(self.max_priority, float(priorities.max()))


    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd.update(
            alpha=self.alpha,
            priority_eps=self.priority_eps,
            max_priority=self.max_priority,
            priorities=self.priorities.copy(),
        )
        return sd

    def load_state_dict(self, sd: dict) -> None:
        super().load_state_dict(sd)
        self.alpha        = float(sd.get("alpha",        self.alpha))
        self.priority_eps = float(sd.get("priority_eps", self.priority_eps))
        self.max_priority = float(sd.get("max_priority", 1.0))
        priorities        = sd.get("priorities")
        if priorities is not None:
            self.priorities[:] = priorities
        else:
            self.priorities[:] = 0.0
            if self.size > 0:
                self.priorities[: self.size] = 1.0
        if self.size > 0:
            self.max_priority = max(
                self.max_priority, float(self.priorities[: self.size].max())
            )



@dataclass
class EpisodeBuffer:
    """Accumulates one episode of transitions, flushes to a replay buffer on done.

    Transitions stay in Python lists until flush_to_buffer() is called so the
    episode return is known for success/failure routing before anything is written.

    episode_return tracks task_rewards only — success/failure routing is based
    on task performance, not motion quality.
    """

    observations:      List[np.ndarray] = field(default_factory=list)
    next_observations: List[np.ndarray] = field(default_factory=list)
    actions:           List[np.ndarray] = field(default_factory=list)
    prev_actions:      List[np.ndarray] = field(default_factory=list)
    task_rewards:      List[float]      = field(default_factory=list)
    motion_rewards:    List[float]      = field(default_factory=list)
    dones:             List[float]      = field(default_factory=list)
    episode_return:    float            = 0.0

    @staticmethod
    def empty() -> "EpisodeBuffer":
        return EpisodeBuffer()

    def append_step(
        self,
        obs:           np.ndarray,
        next_obs:      np.ndarray,
        action:        np.ndarray,
        prev_action:   np.ndarray,
        task_reward:   float,
        motion_reward: float,
        done:          float,
    ) -> None:
        self.observations.append(np.asarray(obs,         dtype=np.float32))
        self.next_observations.append(np.asarray(next_obs,    dtype=np.float32))
        self.actions.append(np.asarray(action,      dtype=np.float32))
        self.prev_actions.append(np.asarray(prev_action, dtype=np.float32))
        self.task_rewards.append(float(task_reward))
        self.motion_rewards.append(float(motion_reward))
        self.dones.append(float(done))
        self.episode_return += float(task_reward)

    def flush_to_buffer(self, replay_buffer: ReplayBuffer) -> int:
        """Write all accumulated transitions to replay_buffer, then reset.

        Returns the number of transitions written (0 if episode was empty).
        """
        n = len(self.observations)
        if n == 0:
            return 0
        replay_buffer.add(
            obs=np.stack(self.observations),
            next_obs=np.stack(self.next_observations),
            action=np.stack(self.actions),
            prev_action=np.stack(self.prev_actions),
            task_reward=np.array(self.task_rewards,   dtype=np.float32),
            motion_reward=np.array(self.motion_rewards, dtype=np.float32),
            done=np.array(self.dones,           dtype=np.float32),
        )
        self.reset()
        return n

    def reset(self) -> None:
        self.observations.clear()
        self.next_observations.clear()
        self.actions.clear()
        self.prev_actions.clear()
        self.task_rewards.clear()
        self.motion_rewards.clear()
        self.dones.clear()
        self.episode_return = 0.0

    def state_dict(self) -> dict:
        return {
            "observations":      [o.copy() for o in self.observations],
            "next_observations": [o.copy() for o in self.next_observations],
            "actions":           [a.copy() for a in self.actions],
            "prev_actions":      [a.copy() for a in self.prev_actions],
            "task_rewards":      list(self.task_rewards),
            "motion_rewards":    list(self.motion_rewards),
            "dones":             list(self.dones),
            "episode_return":    self.episode_return,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.observations      = [np.asarray(o, dtype=np.float32) for o in sd.get("observations",      [])]
        self.next_observations = [np.asarray(o, dtype=np.float32) for o in sd.get("next_observations", [])]
        self.actions           = [np.asarray(a, dtype=np.float32) for a in sd.get("actions",           [])]
        self.prev_actions      = [np.asarray(a, dtype=np.float32) for a in sd.get("prev_actions",      [])]
        self.task_rewards      = list(sd.get("task_rewards",   []))
        self.motion_rewards    = list(sd.get("motion_rewards", []))
        self.dones             = list(sd.get("dones",          []))
        self.episode_return    = float(sd.get("episode_return", 0.0))



def finalize_episode_if_done(
    episode_done: bool,
    episode_buffer: EpisodeBuffer,
    recent_episode_returns: deque,
    success_top_fraction: float,
    episode_return_success_threshold: float,
    success_rb: ReplayBuffer,
    failure_rb: ReplayBuffer,
) -> float:
    """Flush episode to success or failure buffer based on rolling return quantile.

    No-op when episode_done=False. Routing rule:
        episode_return >= threshold  →  success_rb   (top success_top_fraction)
        episode_return <  threshold  →  failure_rb

    The threshold is updated each episode as the (1 - success_top_fraction)
    quantile of recent_episode_returns, so routing adapts as the policy improves.

    Returns:
        Updated episode_return_success_threshold.
    """
    if not episode_done:
        return episode_return_success_threshold

    recent_episode_returns.append(episode_buffer.episode_return)
    episode_return_success_threshold = float(
        np.quantile(
            np.asarray(recent_episode_returns, dtype=np.float32),
            1.0 - float(success_top_fraction),
        )
    )
    target_buffer = (
        success_rb
        if episode_buffer.episode_return >= episode_return_success_threshold
        else failure_rb
    )
    episode_buffer.flush_to_buffer(target_buffer)
    return episode_return_success_threshold