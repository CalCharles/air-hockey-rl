"""Uniform replay buffer for TD3."""

import torch


class TD3ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        obs_shape,
        action_shape,
        device="cuda",
        n_envs=1,
        use_history=False,
        history_entry_dim=0,
        context_len=0,
        use_env_props=False,
        env_prop_dim=0,
    ):
        self.buffer_size = int(buffer_size)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.use_history = bool(use_history)
        self.history_entry_dim = int(history_entry_dim)
        self.context_len = int(context_len)
        self.use_env_props = bool(use_env_props)
        self.env_prop_dim = int(env_prop_dim)

        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.prev_actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.history = (
            torch.zeros(
                (buffer_size, self.context_len, self.history_entry_dim),
                dtype=torch.float32,
                device=device,
            )
            if self.use_history
            else None
        )

        if self.use_env_props:
            if self.env_prop_dim <= 0:
                raise ValueError(
                    f"env_prop_dim must be positive when use_env_props=True, got {self.env_prop_dim}"
                )
            self.env_props = torch.zeros(
                (buffer_size, self.env_prop_dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            self.env_props = None

        self.position = 0
        self.size = 0

    def add(
        self, obs, next_obs, actions, rewards, dones, prev_action,
        history=None, env_props=None,
    ):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        prev_action = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).reshape(-1)
        if self.use_env_props and env_props is None:
            raise ValueError("env_props are required when use_env_props=True.")
        if self.use_history and history is None:
            raise ValueError("history is required when use_history=True.")

        batch_size = int(obs.shape[0])

        first_chunk = min(batch_size, self.buffer_size - self.position)
        first_slice = slice(self.position, self.position + first_chunk)
        self.observations[first_slice] = obs[:first_chunk]
        self.next_observations[first_slice] = next_obs[:first_chunk]
        self.actions[first_slice] = actions[:first_chunk]
        self.prev_actions[first_slice] = prev_action[:first_chunk]
        self.rewards[first_slice] = rewards[:first_chunk]
        self.dones[first_slice] = dones[:first_chunk]
        if self.use_history and history is not None:
            self.history[first_slice] = torch.as_tensor(
                history[:first_chunk], dtype=torch.float32, device=self.device
            )

        if self.use_env_props and env_props is not None and self.env_props is not None:
            env_props_t = torch.as_tensor(env_props, dtype=torch.float32, device=self.device)
            self.env_props[first_slice] = env_props_t[:first_chunk]

        second_chunk = batch_size - first_chunk
        if second_chunk > 0:
            second_slice = slice(0, second_chunk)
            self.observations[second_slice] = obs[first_chunk:]
            self.next_observations[second_slice] = next_obs[first_chunk:]
            self.actions[second_slice] = actions[first_chunk:]
            self.prev_actions[second_slice] = prev_action[first_chunk:]
            self.rewards[second_slice] = rewards[first_chunk:]
            self.dones[second_slice] = dones[first_chunk:]
            if self.use_history and history is not None:
                self.history[second_slice] = torch.as_tensor(
                    history[first_chunk:], dtype=torch.float32, device=self.device
                )

            if self.use_env_props and env_props is not None and self.env_props is not None:
                env_props_t = torch.as_tensor(env_props, dtype=torch.float32, device=self.device)
                self.env_props[second_slice] = env_props_t[first_chunk:]

        self.position = (self.position + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size):
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
        }
        if self.use_env_props and self.env_props is not None:
            result["env_props"] = self.env_props[indices]
        if self.use_history and self.history is not None:
            result["history"] = self.history[indices]
        return result

    def state_dict(self):
        state = {
            "buffer_size": self.buffer_size,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "n_envs": self.n_envs,
            "position": self.position,
            "size": self.size,
            "observations": self.observations.detach().clone().cpu(),
            "next_observations": self.next_observations.detach().clone().cpu(),
            "actions": self.actions.detach().clone().cpu(),
            "prev_actions": self.prev_actions.detach().clone().cpu(),
            "rewards": self.rewards.detach().clone().cpu(),
            "dones": self.dones.detach().clone().cpu(),
        }
        if self.use_env_props and self.env_props is not None:
            state["env_props"] = self.env_props.detach().clone().cpu()
            state["use_env_props"] = True
            state["env_prop_dim"] = self.env_prop_dim
        if self.use_history and self.history is not None:
            state["history"] = self.history.detach().clone().cpu()
            state["use_history"] = True
            state["history_entry_dim"] = self.history_entry_dim
            state["context_len"] = self.context_len
        return state

    def load_state_dict(self, state_dict):
        self.position = int(state_dict["position"])
        self.size = int(state_dict["size"])
        self.observations.copy_(state_dict["observations"].to(self.device))
        self.next_observations.copy_(state_dict["next_observations"].to(self.device))
        self.actions.copy_(state_dict["actions"].to(self.device))
        self.prev_actions.copy_(state_dict["prev_actions"].to(self.device))
        # Old checkpoints stored separate task / motion reward channels.
        if "rewards" in state_dict:
            self.rewards.copy_(state_dict["rewards"].to(self.device))
        elif "task_rewards" in state_dict:
            self.rewards.copy_(state_dict["task_rewards"].to(self.device))
        self.dones.copy_(state_dict["dones"].to(self.device))
        # Backward compatible: older checkpoints omit env_props.
        if self.use_env_props and self.env_props is not None and "env_props" in state_dict:
            self.env_props.copy_(state_dict["env_props"].to(self.device))
        if self.use_history and self.history is not None and "history" in state_dict:
            self.history.copy_(state_dict["history"].to(self.device))

    def __len__(self):
        return self.size

    def clear(self):
        """Discard all stored transitions without reallocating storage."""
        self.position = 0
        self.size = 0
