"""Prioritized replay buffer for TD3 with task/motion reward decomposition."""

import torch


class TD3PrioritizedReplayBuffer:
    """Proportional PER buffer with vectorized sampling and priority updates."""

    def __init__(
        self,
        buffer_size,
        obs_shape,
        action_shape,
        device="cuda",
        n_envs=1,
        alpha=0.6,
        priority_eps=1e-6,
        age_decay=0.0,
    ):
        self.buffer_size = int(buffer_size)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)
        # Age-weighted sampling: priority is multiplied by exp(-age_decay * age_in_slots)
        # before alpha-scaling at sample time. age_decay=0.0 disables. Reasonable
        # values: 1e-5 (very gentle, half-life ~70k slots) to 1e-3 (aggressive,
        # half-life ~700 slots). Implements "stochastic recency-weighted sampling"
        # from residual_rl_drift_fix_log.md open follow-ups.
        self.age_decay = float(age_decay)

        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.prev_actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.task_rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.motion_rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.priorities = torch.zeros((buffer_size,), dtype=torch.float32, device=device)

        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, obs, next_obs, actions, task_rewards, motion_rewards, dones, prev_action):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        prev_action = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        task_rewards = torch.as_tensor(task_rewards, dtype=torch.float32, device=self.device).reshape(-1)
        motion_rewards = torch.as_tensor(motion_rewards, dtype=torch.float32, device=self.device).reshape(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).reshape(-1)

        batch_size = int(obs.shape[0])
        priority_value = max(self.max_priority, self.priority_eps)

        first_chunk = min(batch_size, self.buffer_size - self.position)
        first_slice = slice(self.position, self.position + first_chunk)
        self.observations[first_slice] = obs[:first_chunk]
        self.next_observations[first_slice] = next_obs[:first_chunk]
        self.actions[first_slice] = actions[:first_chunk]
        self.prev_actions[first_slice] = prev_action[:first_chunk]
        self.task_rewards[first_slice] = task_rewards[:first_chunk]
        self.motion_rewards[first_slice] = motion_rewards[:first_chunk]
        self.dones[first_slice] = dones[:first_chunk]
        self.priorities[first_slice] = priority_value

        second_chunk = batch_size - first_chunk
        if second_chunk > 0:
            second_slice = slice(0, second_chunk)
            self.observations[second_slice] = obs[first_chunk:]
            self.next_observations[second_slice] = next_obs[first_chunk:]
            self.actions[second_slice] = actions[first_chunk:]
            self.prev_actions[second_slice] = prev_action[first_chunk:]
            self.task_rewards[second_slice] = task_rewards[first_chunk:]
            self.motion_rewards[second_slice] = motion_rewards[first_chunk:]
            self.dones[second_slice] = dones[first_chunk:]
            self.priorities[second_slice] = priority_value

        self.position = (self.position + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        valid_priorities = self.priorities[: self.size].clamp_min(self.priority_eps)
        # Age-weighted sampling. Each slot's age in "slots since added" is
        # computed from its index relative to self.position (write head).
        # Newer slots (higher index, freshly written) get age ~0 and full priority.
        # Older slots get exponentially down-weighted: w = exp(-age_decay * age).
        # This implements proper stochastic recency-weighted sampling — orthogonal
        # to FIFO eviction (which is binary "in / out") and to TD-error PER (which
        # ignores age).
        if self.age_decay > 0.0:
            if self.size < self.buffer_size:
                # Buffer not yet full: indices 0..size-1 contain entries in
                # temporal order. age(i) = (size-1) - i.
                ages = (self.size - 1) - torch.arange(
                    self.size, device=self.device, dtype=torch.float32
                )
            else:
                # Buffer full: write head at self.position.
                # Most recent write was at (position - 1) mod N → age 0.
                # Oldest live entry is at self.position → age N-1.
                idx = torch.arange(self.buffer_size, device=self.device, dtype=torch.long)
                ages = ((self.position - 1 - idx) % self.buffer_size).to(torch.float32)
            age_weights = torch.exp(-self.age_decay * ages)
            valid_priorities = valid_priorities * age_weights

        scaled = valid_priorities.pow(self.alpha)
        scaled_sum = scaled.sum()
        if not torch.isfinite(scaled_sum) or scaled_sum.item() <= 0.0:
            probs = torch.full_like(scaled, 1.0 / float(self.size))
        else:
            probs = scaled / scaled_sum

        indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        sample_probs = probs[indices].clamp_min(1e-12)

        beta = float(beta)
        weights = (self.size * sample_probs).pow(-beta)
        weights = weights / weights.max().clamp_min(1e-12)

        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "prev_actions": self.prev_actions[indices],
            "task_rewards": self.task_rewards[indices],
            "motion_rewards": self.motion_rewards[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights,
            "sampled_priorities": valid_priorities[indices],
        }

    def sample_uniform(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "prev_actions": self.prev_actions[indices],
            "task_rewards": self.task_rewards[indices],
            "motion_rewards": self.motion_rewards[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": torch.ones((batch_size,), dtype=torch.float32, device=self.device),
            "sampled_priorities": self.priorities[: self.size].clamp_min(self.priority_eps)[indices],
        }

    def update_priorities(self, indices, priorities):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        priorities = torch.as_tensor(priorities, dtype=torch.float32, device=self.device).reshape(-1)
        if indices.numel() == 0:
            return
        priorities = priorities.clamp_min(self.priority_eps)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())

    def state_dict(self):
        return {
            "buffer_size": self.buffer_size,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "n_envs": self.n_envs,
            "position": self.position,
            "size": self.size,
            "alpha": self.alpha,
            "priority_eps": self.priority_eps,
            "max_priority": self.max_priority,
            "observations": self.observations.detach().clone().cpu(),
            "next_observations": self.next_observations.detach().clone().cpu(),
            "actions": self.actions.detach().clone().cpu(),
            "prev_actions": self.prev_actions.detach().clone().cpu(),
            "task_rewards": self.task_rewards.detach().clone().cpu(),
            "motion_rewards": self.motion_rewards.detach().clone().cpu(),
            "dones": self.dones.detach().clone().cpu(),
            "priorities": self.priorities.detach().clone().cpu(),
        }

    def load_state_dict(self, state_dict):
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.alpha = float(state_dict.get("alpha", self.alpha))
        self.priority_eps = float(state_dict.get("priority_eps", self.priority_eps))
        self.max_priority = float(state_dict.get("max_priority", 1.0))
        self.observations.copy_(state_dict["observations"].to(self.device))
        self.next_observations.copy_(state_dict["next_observations"].to(self.device))
        self.actions.copy_(state_dict["actions"].to(self.device))
        self.prev_actions.copy_(state_dict["prev_actions"].to(self.device))
        self.task_rewards.copy_(state_dict["task_rewards"].to(self.device))
        self.motion_rewards.copy_(state_dict["motion_rewards"].to(self.device))
        self.dones.copy_(state_dict["dones"].to(self.device))

        priorities = state_dict.get("priorities")
        if priorities is not None:
            self.priorities.copy_(priorities.to(self.device))
        else:
            self.priorities.zero_()
            if self.size > 0:
                self.priorities[: self.size] = 1.0
        if self.size > 0:
            self.max_priority = max(self.max_priority, self.priorities[: self.size].max().item())
        else:
            self.max_priority = max(self.max_priority, 1.0)

    def __len__(self):
        return self.size
