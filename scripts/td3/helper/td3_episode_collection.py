"""Episode trajectory staging and success/failure replay routing.

See notes/docs/training/replay-and-episodes.md for high-level design.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().cpu()


@dataclass
class EpisodeTrajectory:
    observations: List[torch.Tensor]
    next_observations: List[torch.Tensor]
    actions: List[torch.Tensor]
    rewards: List[torch.Tensor]
    dones: List[torch.Tensor]
    bootstrap_terminals: List[torch.Tensor]
    prev_actions: List[torch.Tensor]
    history: List[torch.Tensor] | None = None  # (T, entry_dim) snapshot per step
    env_props: List[torch.Tensor] | None = None  # RMA privileged props per step
    episode_return: float = 0.0

    @staticmethod
    def empty() -> "EpisodeTrajectory":
        return EpisodeTrajectory(
            observations=[],
            next_observations=[],
            actions=[],
            rewards=[],
            dones=[],
            bootstrap_terminals=[],
            prev_actions=[],
            history=[],
            env_props=[],
            episode_return=0.0,
        )

    def append_step(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        prev_action: torch.Tensor,
        history: torch.Tensor | None = None,    # (T, obs_dim)
        env_props: torch.Tensor | None = None,
        bootstrap_terminal: torch.Tensor | None = None,
    ) -> None:
        self.observations.append(obs.detach().clone())
        self.next_observations.append(next_obs.detach().clone())
        self.actions.append(action.detach().clone())
        self.rewards.append(reward.detach().clone())
        done_tensor = done.detach().clone()
        self.dones.append(done_tensor)
        if bootstrap_terminal is None:
            self.bootstrap_terminals.append(done_tensor.detach().clone())
        else:
            self.bootstrap_terminals.append(bootstrap_terminal.detach().clone())
        self.prev_actions.append(prev_action.detach().clone())

        # Store history if given
        if history is not None:
            if self.history is None:
                self.history = []
            self.history.append(history.detach().clone())

        if env_props is not None:
            if self.env_props is None:
                self.env_props = []
            self.env_props.append(env_props.detach().clone())

        self.episode_return += float(reward.item())

    def flush_to_buffer(self, replay_buffer) -> int:
        transition_count = len(self.observations)
        if transition_count == 0:
            return 0
        replay_buffer.add(
            obs=torch.stack(self.observations, dim=0),
            next_obs=torch.stack(self.next_observations, dim=0),
            actions=torch.stack(self.actions, dim=0),
            rewards=torch.stack(self.rewards, dim=0).view(-1),
            dones=torch.stack(self.dones, dim=0).view(-1),
            prev_action=torch.stack(self.prev_actions, dim=0),
            history=torch.stack(self.history, dim=0) if self.history else None,
            env_props=(
                torch.stack(self.env_props, dim=0)
                if self.env_props
                else None
            ),
        )

        self.reset()
        return transition_count

    def reset(self) -> None:
        self.observations.clear()
        self.next_observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.bootstrap_terminals.clear()
        self.prev_actions.clear()
        if self.history is None:
            self.history = []
        else:
            self.history.clear()
        if self.env_props is None:
            self.env_props = []
        else:
            self.env_props.clear()
        self.episode_return = 0.0

    def state_dict(self) -> Dict[str, Any]:
        result = {
            "observations": [_cpu_tensor(item) for item in self.observations],
            "next_observations": [_cpu_tensor(item) for item in self.next_observations],
            "actions": [_cpu_tensor(item) for item in self.actions],
            "rewards": [_cpu_tensor(item) for item in self.rewards],
            "dones": [_cpu_tensor(item) for item in self.dones],
            "bootstrap_terminals": [_cpu_tensor(item) for item in self.bootstrap_terminals],
            "prev_actions": [_cpu_tensor(item) for item in self.prev_actions],
            "episode_return": float(self.episode_return),
        }

        if self.history:
            result["history"] = [_cpu_tensor(item) for item in self.history]

        if self.env_props:
            result["env_props"] = [_cpu_tensor(item) for item in self.env_props]

        return result

    @classmethod
    def from_state_dict(cls, state_dict: Any, device: str) -> "EpisodeTrajectory":
        trajectory = cls.empty()
        if not isinstance(state_dict, dict):
            return trajectory
        # Old checkpoints (pre motion-reward removal) stored the per-step
        # reward array as `task_rewards`; current code uses `rewards`.
        reward_key = "rewards" if "rewards" in state_dict else "task_rewards"
        for attr, key in (
            ("observations", "observations"),
            ("next_observations", "next_observations"),
            ("actions", "actions"),
            ("rewards", reward_key),
            ("dones", "dones"),
            ("prev_actions", "prev_actions"),
        ):
            values = state_dict.get(key, [])
            if isinstance(values, list):
                setattr(
                    trajectory,
                    attr,
                    [torch.as_tensor(item, dtype=torch.float32, device=device) for item in values],
                )
        bootstrap_terminal_values = state_dict.get("bootstrap_terminals", None)
        if isinstance(bootstrap_terminal_values, list):
            trajectory.bootstrap_terminals = [
                torch.as_tensor(item, dtype=torch.float32, device=device)
                for item in bootstrap_terminal_values
            ]
        else:
            trajectory.bootstrap_terminals = [
                item.detach().clone() for item in trajectory.dones
            ]

        history_values = state_dict.get("history", [])
        if isinstance(history_values, list) and len(history_values) > 0:
            trajectory.history = [
                torch.as_tensor(item, dtype=torch.float32, device=device)
                for item in history_values
            ]

        env_props_values = state_dict.get("env_props", [])
        if isinstance(env_props_values, list) and len(env_props_values) > 0:
            trajectory.env_props = [
                torch.as_tensor(item, dtype=torch.float32, device=device)
                for item in env_props_values
            ]

        trajectory.episode_return = float(state_dict.get("episode_return", 0.0))
        return trajectory


def load_episode_trajectory_from_checkpoint(
    resume_checkpoint: Dict[str, Any], device: str
) -> EpisodeTrajectory:
    if "episode_trajectory" in resume_checkpoint:
        return EpisodeTrajectory.from_state_dict(resume_checkpoint["episode_trajectory"], device=device)

    # Backward compatibility for older format that kept per-env list staging.
    legacy_staging = resume_checkpoint.get("episode_transition_staging")
    legacy_returns = resume_checkpoint.get("episode_return_staging")
    if isinstance(legacy_staging, list) and len(legacy_staging) > 0:
        trajectory = EpisodeTrajectory.from_state_dict(legacy_staging[0], device=device)
        if isinstance(legacy_returns, list) and len(legacy_returns) > 0:
            trajectory.episode_return = float(legacy_returns[0])
        return trajectory

    return EpisodeTrajectory.empty()


def finalize_episode_if_done(
    episode_done: bool,
    episode_trajectory: EpisodeTrajectory,
    recent_episode_returns: deque,
    success_top_fraction: float,
    episode_return_success_threshold: float,
    success_rb,
    failure_rb,
) -> float:
    if not episode_done:
        return float(episode_return_success_threshold)
    episode_return = float(episode_trajectory.episode_return)
    recent_episode_returns.append(episode_return)
    if len(recent_episode_returns) > 0:
        success_threshold_quantile = 1.0 - float(success_top_fraction)
        episode_return_success_threshold = float(
            np.quantile(
                np.asarray(recent_episode_returns, dtype=np.float32),
                success_threshold_quantile,
            )
        )
    target_buffer = success_rb if episode_return >= episode_return_success_threshold else failure_rb
    episode_trajectory.flush_to_buffer(target_buffer)
    return float(episode_return_success_threshold)
