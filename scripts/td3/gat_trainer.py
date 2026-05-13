"""
Grounded Action Transform (GAT) components — Hanna & Stone, 2017.

Pipeline
--------
1. Collect rollouts in source sim → train SourceDynamicsModel f_S.
2. Collect rollouts in target sim → train ActionTransformer τ to minimise
   ||f_S(s, τ(s,a)) − s'_target||².
3. Wrap the source sim with GATEnvWrapper(τ) for policy fine-tuning.
   The policy proposes action a; the wrapper applies a' = τ(s,a) before
   env.step(), making source transitions look like target transitions.
4. Evaluate the fine-tuned policy directly in the target sim (no τ at
   deployment time).

References
----------
  Hanna & Stone. "Grounded Action Transformation for Robot Learning in
  Simulation." AAAI 2017.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# Default dims for the hist-2 juggle task (30-dim obs, 2-dim action).
_OBS_DIM = 30
_ACT_DIM = 2


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class SourceDynamicsModel(nn.Module):
    """Forward model of the source sim: (obs, action) → next_obs.

    Parameterised as a residual: next_obs = obs + net(obs ‖ action).
    """

    def __init__(self, obs_dim: int = _OBS_DIM, act_dim: int = _ACT_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return obs + self.net(x)


class ActionTransformer(nn.Module):
    """Action transformer τ: (obs, action) → transformed_action in [−1, 1]².

    Parameterised as a residual: a' = clip(a + net(obs ‖ a), −1, 1).
    Initialising net to zero output gives τ = identity at the start of
    training, making early fine-tuning identical to zero-shot transfer.
    """

    def __init__(self, obs_dim: int = _OBS_DIM, act_dim: int = _ACT_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        # Zero-init output layer → τ starts as identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        delta = self.net(x)
        return torch.clamp(action + delta, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_transitions(
    env: gym.Env,
    policy: nn.Module,
    n_steps: int,
    use_last_action: bool = True,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll *policy* in *env* for *n_steps* environment steps.

    Returns
    -------
    obs_arr      (N, obs_dim)   – observation at step t
    action_arr   (N, act_dim)   – action taken at step t
    next_obs_arr (N, obs_dim)   – observation at step t+1
    """
    act_dim = int(np.prod(env.action_space.shape))
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    nobs_list: list[np.ndarray] = []

    obs, _ = env.reset()
    last_action = np.zeros(act_dim, dtype=np.float32)

    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_a_t = torch.tensor(last_action, dtype=torch.float32, device=device).unsqueeze(0)
        policy_obs = torch.cat([obs_t, last_a_t], dim=-1) if use_last_action else obs_t

        with torch.no_grad():
            action = policy(policy_obs).cpu().numpy().squeeze(0)

        next_obs, _, term, trunc, _ = env.step(action)

        obs_list.append(obs.copy())
        act_list.append(action.copy())
        nobs_list.append(next_obs.copy())

        if term or trunc:
            obs, _ = env.reset()
            last_action = np.zeros(act_dim, dtype=np.float32)
        else:
            obs = next_obs
            last_action = action.copy()

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(nobs_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _random_batches(N: int, batch_size: int, device: str):
    idx = torch.randperm(N, device=device)
    for start in range(0, N, batch_size):
        yield idx[start : start + batch_size]


def train_dynamics_model(
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    model: SourceDynamicsModel,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """Fit *model* on (obs, action) → next_obs supervised pairs.

    Returns the final epoch's average MSE loss.
    """
    obs_t = torch.tensor(obs, device=device)
    act_t = torch.tensor(actions, device=device)
    nobs_t = torch.tensor(next_obs, device=device)
    N = len(obs_t)

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        total, cnt = 0.0, 0
        for b in _random_batches(N, batch_size, device):
            pred = model(obs_t[b], act_t[b])
            loss = nn.functional.mse_loss(pred, nobs_t[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            cnt += 1
        final_loss = total / max(cnt, 1)
    model.eval()
    return final_loss


def train_action_transformer(
    target_obs: np.ndarray,
    target_actions: np.ndarray,
    target_next_obs: np.ndarray,
    dynamics_model: SourceDynamicsModel,
    transformer: ActionTransformer,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """Minimise ||f_S(s, τ(s,a)) − s'_target||² over target-env transitions.

    *dynamics_model* is frozen (eval mode, no grad) throughout.

    Returns the final epoch's average MSE loss.
    """
    obs_t = torch.tensor(target_obs, device=device)
    act_t = torch.tensor(target_actions, device=device)
    nobs_t = torch.tensor(target_next_obs, device=device)
    N = len(obs_t)

    # Freeze dynamics model parameters: gradients flow *through* it (needed
    # to reach the transformer), but we don't update its weights here.
    for p in dynamics_model.parameters():
        p.requires_grad_(False)
    dynamics_model.eval()
    transformer.train()
    final_loss = 0.0
    try:
        for _ in range(epochs):
            total, cnt = 0.0, 0
            for b in _random_batches(N, batch_size, device):
                a_prime = transformer(obs_t[b], act_t[b])
                pred_next = dynamics_model(obs_t[b], a_prime)
                loss = nn.functional.mse_loss(pred_next, nobs_t[b])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                cnt += 1
            final_loss = total / max(cnt, 1)
    finally:
        # Always restore grad tracking on the dynamics model.
        for p in dynamics_model.parameters():
            p.requires_grad_(True)
    transformer.eval()
    return final_loss


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class GATEnvWrapper(gym.Wrapper):
    """Applies action transformer τ(obs, action) before env.step().

    The transformer always runs on CPU so that this wrapper is safe to use
    inside AsyncVectorEnv worker processes (CUDA is not available there).

    The observation and action spaces exposed to the training loop are
    identical to the underlying env — the transformation is internal.
    """

    def __init__(self, env: gym.Env, transformer: ActionTransformer):
        super().__init__(env)
        # CPU copy: safe for pickling / subprocess execution.
        self._transformer = transformer.cpu().eval()
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray):
        if self._last_obs is None:
            raise RuntimeError("GATEnvWrapper.step() called before reset().")
        obs_t = torch.tensor(self._last_obs, dtype=torch.float32).unsqueeze(0)
        act_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_prime = self._transformer(obs_t, act_t).numpy().squeeze(0)
        next_obs, reward, term, trunc, info = self.env.step(action_prime)
        self._last_obs = next_obs
        return next_obs, reward, term, trunc, info
