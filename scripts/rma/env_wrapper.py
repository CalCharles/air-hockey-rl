"""Gymnasium AirHockey → hora-style RMA observation dict.

Policy / encoder inputs (hora-equivalent roles):
  obs          — standard air-hockey observation (e.g. 30-d history obs)
  priv_info    — privileged physics props → μ → z  (phase 1)
  proprio_hist — context buffer for φ: T × 6 paddle/puck pos+valid
                 (no actions; TD3's last-action-in-policy-state is not used)

reset() -> {'obs', 'priv_info', 'proprio_hist'}
step(actions) -> (obs_dict, rewards, dones, infos)
"""

from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch

from airhockey import AirHockeyEnv
from scripts.rma.history_buffer import HISTORY_ENTRY_DIM, HistoryBuffer
from scripts.rma.rma_utils import (
    build_prop_normalizer,
    extract_env_props_from_vec_info,
    normalize_env_props,
    privileged_keys_from_config,
    read_env_props_from_vector_env,
)


class RMAVecEnv:
    """Thin vector-env adapter with hora-compatible dict observations."""

    def __init__(
        self,
        air_hockey_params: Dict[str, Any],
        num_envs: int,
        device: str | torch.device,
        prop_hist_len: int = 30,
        seed: int = 0,
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.prop_hist_len = int(prop_hist_len)
        self.air_hockey_params = air_hockey_params

        self.priv_keys = privileged_keys_from_config(air_hockey_params)
        ranges = air_hockey_params.get("random_variable_ranges", {})
        self.prop_lows, self.prop_highs = build_prop_normalizer(ranges, self.priv_keys)
        self.priv_info_dim = len(self.priv_keys)

        def _make(rank: int):
            def thunk():
                cfg = dict(air_hockey_params)
                cfg["seed"] = int(seed) + rank
                return AirHockeyEnv(cfg)

            return thunk

        self._venv = gym.vector.SyncVectorEnv([_make(i) for i in range(self.num_envs)])
        self.observation_space = self._venv.single_observation_space
        self.action_space = self._venv.single_action_space
        self.action_dim = int(np.prod(self.action_space.shape))
        # Context entries are paddle/puck pos+valid only (no action dims).
        self.proprio_hist_entry_dim = HISTORY_ENTRY_DIM

        self._history: List[HistoryBuffer] = [
            HistoryBuffer(
                context_len=self.prop_hist_len,
                device="cpu",
                include_action=False,
            )
            for _ in range(self.num_envs)
        ]

    def close(self):
        self._venv.close()

    def _raw_priv(self, infos: Any) -> np.ndarray:
        try:
            return extract_env_props_from_vec_info(infos, self.priv_keys, self.num_envs)
        except Exception:
            return read_env_props_from_vector_env(self._venv, self.priv_keys)

    def _normalize_priv(self, raw: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        return normalize_env_props(t, self.prop_lows, self.prop_highs)

    # TODO: Why are we stacking proprio history?
    def _stack_proprio_hist(self) -> torch.Tensor:
        # (N, T, 6)
        samples = [buf.sample().squeeze(0) for buf in self._history]
        return torch.stack(samples, dim=0).to(self.device)

    def _pack_obs(
        self,
        obs_np: np.ndarray,
        infos: Any,
    ) -> Dict[str, torch.Tensor]:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        priv = self._normalize_priv(self._raw_priv(infos))
        proprio = self._stack_proprio_hist()
        return {
            "obs": obs,
            "priv_info": priv,
            "proprio_hist": proprio,
        }

    def reset(self) -> Dict[str, torch.Tensor]:
        obs_np, infos = self._venv.reset()
        for i, buf in enumerate(self._history):
            buf.reset_env()
            buf.add(obs_np[i])
        return self._pack_obs(obs_np, infos)

    def step(self, actions: torch.Tensor):
        if isinstance(actions, torch.Tensor):
            act_np = actions.detach().cpu().numpy().astype(np.float32)
        else:
            act_np = np.asarray(actions, dtype=np.float32)
        if act_np.ndim == 1:
            act_np = act_np.reshape(1, -1)

        obs_np, rewards, terminations, truncations, infos = self._venv.step(act_np)
        dones_np = np.logical_or(terminations, truncations)

        for i in range(self.num_envs):
            self._history[i].add(obs_np[i])
            if bool(dones_np[i]):
                self._history[i].reset_env()
                self._history[i].add(obs_np[i])

        obs_dict = self._pack_obs(obs_np, infos)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones_np, dtype=torch.uint8, device=self.device)

        flat_infos: Dict[str, Any] = {}
        if isinstance(infos, dict):
            for k, v in infos.items():
                if k == "final_info":
                    continue
                if isinstance(v, (float, int)) and not isinstance(v, bool):
                    flat_infos[k] = v
                elif isinstance(v, np.ndarray) and v.size == 1:
                    # SyncVectorEnv packs string infos (e.g. transition_hold_reason
                    # == "none") as size-1 object arrays when num_envs=1; only
                    # promote numeric scalars so eval/GIF rollouts don't crash.
                    x = v.reshape(-1)[0]
                    if isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(
                        x, (bool, np.bool_)
                    ):
                        flat_infos[k] = float(x)

        return obs_dict, rewards_t, dones_t, flat_infos
