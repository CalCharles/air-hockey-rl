"""State/action history for the adaptation module (phase 2 + deployment).

The adaptation module phi consumes the last ``H`` (x, a) pairs *before* the
current step: at time t it sees (x_{t-H}, a_{t-H}), ..., (x_{t-1}, a_{t-1}).
The first steps of an episode have fewer than H pairs; they are padded by
repeating the first observed state with a zero action (a stationary
"nothing happened yet" prefix).  The same padding is used online
(``StepHistoryBuffer``) and when windows are cut from stored episodes
(``AdaptationDataset``), so train and deployment inputs match exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class StepHistoryBuffer:
    """Online rolling window of per-step features for one environment."""

    def __init__(self, history_len: int, feature_dim: int, act_dim: int) -> None:
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.act_dim = int(act_dim)
        self._buf = np.zeros((self.history_len, self.feature_dim), dtype=np.float32)

    def reset(self, first_state_features: np.ndarray) -> None:
        """Start an episode: fill the window with (x_0, a=0)."""
        first_state_features = np.asarray(first_state_features, dtype=np.float32).reshape(-1)
        if first_state_features.shape[0] != self.feature_dim - self.act_dim:
            raise ValueError(
                f"expected {self.feature_dim - self.act_dim} state features, got {first_state_features.shape[0]}"
            )
        self._buf[:, : self.feature_dim - self.act_dim] = first_state_features[None, :]
        self._buf[:, self.feature_dim - self.act_dim :] = 0.0

    def push(self, state_features: np.ndarray, action: np.ndarray) -> None:
        """Append the (x_t, a_t) pair once a_t has been chosen."""
        entry = np.concatenate(
            [np.asarray(state_features, dtype=np.float32).reshape(-1), np.asarray(action, dtype=np.float32).reshape(-1)]
        )
        if entry.shape[0] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {entry.shape[0]}")
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = entry

    def window(self) -> np.ndarray:
        """(H, F) window with the newest pair last."""
        return self._buf.copy()


def pad_episode(state_features: np.ndarray, actions: np.ndarray, history_len: int) -> np.ndarray:
    """(T, F) episode of (x_t, a_t) pairs -> (H + T, F) with the repeat-first prefix.

    The window for step t is ``padded[t : t + H]`` = the H pairs before t.
    """
    state_features = np.asarray(state_features, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if state_features.shape[0] != actions.shape[0]:
        raise ValueError("state_features and actions must have the same number of steps")
    pairs = np.concatenate([state_features, actions], axis=1)
    prefix = np.concatenate(
        [np.repeat(state_features[:1], history_len, axis=0), np.zeros((history_len, actions.shape[1]), dtype=np.float32)],
        axis=1,
    )
    return np.concatenate([prefix, pairs], axis=0)


class AdaptationDataset:
    """Episodes of (x_t, a_t) pairs with per-episode latent targets.

    Windows are gathered lazily from one contiguous, padded feature array so
    a 400k-step dataset costs ~ (400k + H * n_episodes) x F floats, not
    400k x H x F.
    """

    def __init__(self, history_len: int, feature_dim: int, latent_dim: int, env_param_dim: int, max_steps: int | None = None):
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.env_param_dim = int(env_param_dim)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self._episodes: List[Dict[str, np.ndarray]] = []
        self._cache: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------- building
    def add_episode(
        self,
        state_features: np.ndarray,
        actions: np.ndarray,
        latent_target: np.ndarray,
        env_params: np.ndarray,
        episode_return: float = 0.0,
    ) -> None:
        state_features = np.asarray(state_features, dtype=np.float32)
        if state_features.shape[0] == 0:
            return
        padded = pad_episode(state_features, actions, self.history_len)
        self._episodes.append(
            {
                "padded": padded,
                "T": int(state_features.shape[0]),
                "latent": np.asarray(latent_target, dtype=np.float32).reshape(self.latent_dim),
                "env_params": np.asarray(env_params, dtype=np.float32).reshape(self.env_param_dim),
                "episode_return": float(episode_return),
            }
        )
        self._cache = None
        if self.max_steps is not None:
            while self.num_steps > self.max_steps and len(self._episodes) > 1:
                self._episodes.pop(0)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_steps(self) -> int:
        return int(sum(ep["T"] for ep in self._episodes))

    def _build(self) -> Dict[str, np.ndarray]:
        if self._cache is not None:
            return self._cache
        if not self._episodes:
            raise ValueError("AdaptationDataset is empty")
        feats = np.concatenate([ep["padded"] for ep in self._episodes], axis=0)
        starts, latents, env_params, t_in_ep = [], [], [], []
        offset = 0
        for ep in self._episodes:
            T = ep["T"]
            starts.append(offset + np.arange(T, dtype=np.int64))
            latents.append(np.repeat(ep["latent"][None], T, axis=0))
            env_params.append(np.repeat(ep["env_params"][None], T, axis=0))
            t_in_ep.append(np.arange(T, dtype=np.int64))
            offset += self.history_len + T
        self._cache = {
            "feats": feats,
            "starts": np.concatenate(starts),
            "latents": np.concatenate(latents, axis=0),
            "env_params": np.concatenate(env_params, axis=0),
            "t_in_ep": np.concatenate(t_in_ep),
        }
        return self._cache

    # -------------------------------------------------------------- reading
    def windows(self, sample_idx: np.ndarray) -> np.ndarray:
        """(N, H, F) windows for dataset rows ``sample_idx``."""
        data = self._build()
        starts = data["starts"][sample_idx]
        gather = starts[:, None] + np.arange(self.history_len, dtype=np.int64)[None, :]
        return data["feats"][gather]

    def targets(self, sample_idx: np.ndarray) -> np.ndarray:
        return self._build()["latents"][sample_idx]

    def env_params(self, sample_idx: np.ndarray) -> np.ndarray:
        return self._build()["env_params"][sample_idx]

    def steps_in_episode(self, sample_idx: np.ndarray) -> np.ndarray:
        return self._build()["t_in_ep"][sample_idx]

    def all_indices(self) -> np.ndarray:
        return np.arange(self.num_steps, dtype=np.int64)

    def batch(self, sample_idx: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(self.windows(sample_idx), dtype=torch.float32, device=device)
        y = torch.as_tensor(self.targets(sample_idx), dtype=torch.float32, device=device)
        return x, y

    def state_dict(self) -> Dict[str, object]:
        return {
            "history_len": self.history_len,
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "env_param_dim": self.env_param_dim,
            "episodes": [dict(ep) for ep in self._episodes],
        }
