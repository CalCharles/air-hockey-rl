"""NumPy ring-buffer replay buffers with task/motion reward decomposition.

ReplayBuffer             — uniform sampling
PrioritizedReplayBuffer  — proportional PER with IS weights
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Uniform ring-buffer storing (obs, next_obs, action, prev_action, task_reward, motion_reward, done).

    Args:
        buffer_size: Maximum number of transitions.
        obs_shape: Shape of a single observation.
        action_shape: Shape of a single action.
    """

    def __init__(self, buffer_size: int, obs_shape: tuple, action_shape: tuple):
        self.buffer_size = int(buffer_size)
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.prev_actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.task_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.motion_rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(self, obs, next_obs, action, task_reward, motion_reward, done, prev_action):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, *self.obs_shape)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1, *self.obs_shape)
        action = np.asarray(action, dtype=np.float32).reshape(-1, *self.action_shape)
        prev_action = np.asarray(prev_action, dtype=np.float32).reshape(-1, *self.action_shape)
        task_reward = np.asarray(task_reward, dtype=np.float32).reshape(-1)
        motion_reward = np.asarray(motion_reward, dtype=np.float32).reshape(-1)
        done = np.asarray(done, dtype=np.float32).reshape(-1)

        batch = int(obs.shape[0])
        first = min(batch, self.buffer_size - self.position)
        s = slice(self.position, self.position + first)
        self.observations[s] = obs[:first]
        self.next_observations[s] = next_obs[:first]
        self.actions[s] = action[:first]
        self.prev_actions[s] = prev_action[:first]
        self.task_rewards[s] = task_reward[:first]
        self.motion_rewards[s] = motion_reward[:first]
        self.dones[s] = done[:first]

        second = batch - first
        if second > 0:
            self.observations[:second] = obs[first:]
            self.next_observations[:second] = next_obs[first:]
            self.actions[:second] = action[first:]
            self.prev_actions[:second] = prev_action[first:]
            self.task_rewards[:second] = task_reward[first:]
            self.motion_rewards[:second] = motion_reward[first:]
            self.dones[:second] = done[first:]

        self.position = (self.position + batch) % self.buffer_size
        self.size = min(self.size + batch, self.buffer_size)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._gather(idx)

    def _gather(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations[idx],
            "next_observations": self.next_observations[idx],
            "actions": self.actions[idx],
            "prev_actions": self.prev_actions[idx],
            "task_rewards": self.task_rewards[idx],
            "motion_rewards": self.motion_rewards[idx],
            "dones": self.dones[idx],
        }

    def state_dict(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "position": self.position,
            "size": self.size,
            "observations": self.observations.copy(),
            "next_observations": self.next_observations.copy(),
            "actions": self.actions.copy(),
            "prev_actions": self.prev_actions.copy(),
            "task_rewards": self.task_rewards.copy(),
            "motion_rewards": self.motion_rewards.copy(),
            "dones": self.dones.copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.position = int(sd["position"])
        self.size = int(sd["size"])
        self.observations[:] = sd["observations"]
        self.next_observations[:] = sd["next_observations"]
        self.actions[:] = sd["actions"]
        self.prev_actions[:] = sd["prev_actions"]
        self.task_rewards[:] = sd["task_rewards"]
        self.motion_rewards[:] = sd["motion_rewards"]
        self.dones[:] = sd["dones"]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER ring-buffer with importance-sampling weights.

    Args:
        buffer_size: Maximum number of transitions.
        obs_shape: Shape of a single observation.
        action_shape: Shape of a single action.
        alpha: Priority exponent (0 = uniform, 1 = fully prioritized).
        priority_eps: Floor added to all priorities to prevent zero sampling.
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
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs, next_obs, action, task_reward, motion_reward, done, prev_action):
        priority_value = max(self.max_priority, self.priority_eps)
        old_pos = self.position
        batch = int(np.asarray(obs, dtype=np.float32).reshape(-1, *self.obs_shape).shape[0])
        super().add(obs, next_obs, action, task_reward, motion_reward, done, prev_action)
        for i in range(batch):
            self.priorities[(old_pos + i) % self.buffer_size] = priority_value

    def sample(self, batch_size: int, beta: float = 0.4) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        valid = np.clip(self.priorities[: self.size], self.priority_eps, None)
        scaled = valid ** self.alpha
        probs = scaled / scaled.sum()
        idx = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
        sample_probs = np.clip(probs[idx], 1e-12, None)
        weights = (self.size * sample_probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        result = self._gather(idx)
        result["indices"] = idx
        result["weights"] = weights
        result["sampled_priorities"] = valid[idx]
        return result

    def sample_uniform(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        result = self._gather(idx)
        result["indices"] = idx
        result["weights"] = np.ones(batch_size, dtype=np.float32)
        result["sampled_priorities"] = np.clip(self.priorities[: self.size], self.priority_eps, None)[idx]
        return result

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        priorities = np.clip(np.asarray(priorities, dtype=np.float32).reshape(-1), self.priority_eps, None)
        if indices.size == 0:
            return
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))

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
        self.alpha = float(sd.get("alpha", self.alpha))
        self.priority_eps = float(sd.get("priority_eps", self.priority_eps))
        self.max_priority = float(sd.get("max_priority", 1.0))
        priorities = sd.get("priorities")
        if priorities is not None:
            self.priorities[:] = priorities
        else:
            self.priorities[:] = 0.0
            if self.size > 0:
                self.priorities[: self.size] = 1.0
        if self.size > 0:
            self.max_priority = max(self.max_priority, float(self.priorities[: self.size].max()))
