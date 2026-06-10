"""Prioritized replay buffer for TD3."""

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
        use_history=False,
        history_entry_dim=4,
        context_len=0,
    ):
        self.buffer_size = int(buffer_size)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)
        # Age-weighted sampling: priority is multiplied by exp(-age_decay * age_in_slots)
        # before alpha-scaling at sample time. age_decay=0.0 disables.
        self.age_decay = float(age_decay)

        self.use_history = bool(use_history)
        self.context_len = int(context_len)
        self.history_entry_dim = int(history_entry_dim)


        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.prev_actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.priorities = torch.zeros((buffer_size,), dtype=torch.float32, device=device)

        # TODO: we need to think more on how to support using history and normal mode
        # Maybe we don't worry about it since for now we want to collect history no matter what
        # if self.use_history:
        #     # obs_dim = obs_shape[0]

        #     self.history = torch.zeros(
        #         (buffer_size, self.context_len, self.history_entry_dim),
        #         dtype=torch.float32,
        #         device=device,
        #     )
        # else:
        #     self.history = None

        # TODO: Note that we init this unconditionally bc in td3_training.py I choose
        #       to always gather history.
        self.history = torch.zeros(
            (buffer_size, self.context_len, self.history_entry_dim),
            dtype=torch.float32,
            device=device,
        )


        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, obs, next_obs, actions, rewards, dones, prev_action, history=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        prev_action = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).reshape(-1)

        batch_size = int(obs.shape[0])
        priority_value = max(self.max_priority, self.priority_eps)

        first_chunk = min(batch_size, self.buffer_size - self.position)
        first_slice = slice(self.position, self.position + first_chunk)
        self.observations[first_slice] = obs[:first_chunk]
        self.next_observations[first_slice] = next_obs[:first_chunk]
        self.actions[first_slice] = actions[:first_chunk]
        self.prev_actions[first_slice] = prev_action[:first_chunk]
        self.rewards[first_slice] = rewards[:first_chunk]
        self.dones[first_slice] = dones[:first_chunk]
        self.priorities[first_slice] = priority_value

        if self.history is not None and history is not None:
            self.history[first_slice] = history[:first_chunk]

        second_chunk = batch_size - first_chunk
        if second_chunk > 0:
            second_slice = slice(0, second_chunk)
            self.observations[second_slice] = obs[first_chunk:]
            self.next_observations[second_slice] = next_obs[first_chunk:]
            self.actions[second_slice] = actions[first_chunk:]
            self.prev_actions[second_slice] = prev_action[first_chunk:]
            self.rewards[second_slice] = rewards[first_chunk:]
            self.dones[second_slice] = dones[first_chunk:]
            self.priorities[second_slice] = priority_value

            # NEW: wrap-around write for sequences
            if self.history is not None and history is not None:
                self.history[second_slice] = history[first_chunk:]

        self.position = (self.position + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        valid_priorities = self.priorities[: self.size].clamp_min(self.priority_eps)
        if self.age_decay > 0.0:
            if self.size < self.buffer_size:
                ages = (self.size - 1) - torch.arange(
                    self.size, device=self.device, dtype=torch.float32
                )
            else:
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

        result = {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "prev_actions": self.prev_actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights,
            "sampled_priorities": valid_priorities[indices],
        }

        if self.history is not None:
            result["history"] = self.history[indices]

        return result

    def sample_uniform(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        result = {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "prev_actions": self.prev_actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": torch.ones((batch_size,), dtype=torch.float32, device=self.device),
            "sampled_priorities": self.priorities[: self.size].clamp_min(self.priority_eps)[indices],
        }

        if self.history is not None:
            result["history"] = self.history[indices]

        return result

    def update_priorities(self, indices, priorities):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        priorities = torch.as_tensor(priorities, dtype=torch.float32, device=self.device).reshape(-1)
        if indices.numel() == 0:
            return
        priorities = priorities.clamp_min(self.priority_eps)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())

    def state_dict(self):
        state_dict =  {
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
            "rewards": self.rewards.detach().clone().cpu(),
            "dones": self.dones.detach().clone().cpu(),
            "priorities": self.priorities.detach().clone().cpu(),
        }

        if self.history is not None:
            state_dict["history"] = self.history.detach().clone().cpu()

        return state_dict

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
        if "rewards" in state_dict:
            self.rewards.copy_(state_dict["rewards"].to(self.device))
        elif "task_rewards" in state_dict:
            self.rewards.copy_(state_dict["task_rewards"].to(self.device))
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
        
        if "history" in state_dict and self.history is not None:
            self.history.copy_(state_dict["history"].to(self.device))

    def __len__(self):
        return self.size
