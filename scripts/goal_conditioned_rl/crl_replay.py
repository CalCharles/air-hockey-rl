"""Episode replay for contrastive RL, on the GPU.

Each row is one transition (s_t, a_t, g_{t+1}) where g_{t+1} is the goal view of the next state, plus
``remaining`` = transitions from this row to the end of its episode (inclusive). Whole episodes are inserted
when they finish, as the official code's reverb EpisodeAdder does, so ``remaining`` is always known.

Goal relabelling is the official one (contrastive_rl builder.py ``flatten_fn``): for row t the goal is the
state at index j > t of the same episode with probability proportional to gamma^(j - t), i.e. offset
k = j - t - 1 in {0, ..., remaining - 1} with P(k) proportional to gamma^k, and goal = g_next[t + k].
Goals never cross an episode boundary. Rows sit in a FIFO ring; a row's future rows are always newer than
the row itself, so they are only overwritten after it.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class EpisodeReplay:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, goal_dim: int, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.next_goals = torch.zeros((capacity, goal_dim), dtype=torch.float32, device=self.device)
        self.remaining = torch.ones(capacity, dtype=torch.int64, device=self.device)
        self.written = 0
        self.episodes = 0

    @property
    def size(self) -> int:
        return min(self.written, self.capacity)

    def add_episode(self, states: np.ndarray, actions: np.ndarray, next_goals: np.ndarray):
        n = len(states)
        if n == 0:
            return
        if n > self.capacity:
            raise ValueError("episode longer than the replay capacity")
        slots = (self.written + torch.arange(n, device=self.device)) % self.capacity
        self.states[slots] = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self.actions[slots] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.next_goals[slots] = torch.as_tensor(next_goals, dtype=torch.float32, device=self.device)
        self.remaining[slots] = torch.arange(n, 0, -1, device=self.device)
        self.written += n
        self.episodes += 1

    def sample_offsets(self, remaining: torch.Tensor, gamma: float) -> torch.Tensor:
        """k ~ P(k) proportional to gamma^k on {0, ..., remaining - 1} (inverse CDF of the truncated geometric)."""
        u = torch.rand(remaining.shape, device=self.device).clamp_min(1e-12)
        log_g = math.log(gamma)
        k = torch.floor(torch.log1p(u * torch.expm1(remaining.double() * log_g)) / log_g)
        return torch.minimum(k.long().clamp_min(0), remaining - 1)

    def sample(self, batch_size: int, gamma: float):
        rows = torch.randint(0, self.size, (batch_size,), device=self.device)
        offsets = self.sample_offsets(self.remaining[rows], gamma)
        goal_rows = (rows + offsets) % self.capacity
        return self.states[rows], self.actions[rows], self.next_goals[goal_rows], rows, offsets

    def random_goals(self, n: int) -> np.ndarray:
        """Goal views of n uniformly random stored states (the per-episode collection goal)."""
        rows = torch.randint(0, self.size, (n,), device=self.device)
        return self.next_goals[rows].cpu().numpy()
