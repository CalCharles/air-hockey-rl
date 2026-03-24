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
    task_rewards: List[torch.Tensor]
    motion_rewards: List[torch.Tensor]
    dones: List[torch.Tensor]
    bootstrap_terminals: List[torch.Tensor]
    prev_actions: List[torch.Tensor]
    episode_return: float = 0.0

    @staticmethod
    def empty() -> "EpisodeTrajectory":
        return EpisodeTrajectory(
            observations=[],
            next_observations=[],
            actions=[],
            task_rewards=[],
            motion_rewards=[],
            dones=[],
            bootstrap_terminals=[],
            prev_actions=[],
            episode_return=0.0,
        )

    def append_step(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        task_reward: torch.Tensor,
        motion_reward: torch.Tensor,
        done: torch.Tensor,
        prev_action: torch.Tensor,
        bootstrap_terminal: torch.Tensor | None = None,
    ) -> None:
        self.observations.append(obs.detach().clone())
        self.next_observations.append(next_obs.detach().clone())
        self.actions.append(action.detach().clone())
        self.task_rewards.append(task_reward.detach().clone())
        self.motion_rewards.append(motion_reward.detach().clone())
        done_tensor = done.detach().clone()
        self.dones.append(done_tensor)
        if bootstrap_terminal is None:
            self.bootstrap_terminals.append(done_tensor.detach().clone())
        else:
            self.bootstrap_terminals.append(bootstrap_terminal.detach().clone())
        self.prev_actions.append(prev_action.detach().clone())
        self.episode_return += float(task_reward.item())

    def flush_to_buffer(self, replay_buffer) -> int:
        transition_count = len(self.observations)
        if transition_count == 0:
            return 0
        replay_buffer.add(
            obs=torch.stack(self.observations, dim=0),
            next_obs=torch.stack(self.next_observations, dim=0),
            actions=torch.stack(self.actions, dim=0),
            task_rewards=torch.stack(self.task_rewards, dim=0).view(-1),
            motion_rewards=torch.stack(self.motion_rewards, dim=0).view(-1),
            dones=torch.stack(self.dones, dim=0).view(-1),
            prev_action=torch.stack(self.prev_actions, dim=0),
        )
        self.reset()
        return transition_count

    def reset(self) -> None:
        self.observations.clear()
        self.next_observations.clear()
        self.actions.clear()
        self.task_rewards.clear()
        self.motion_rewards.clear()
        self.dones.clear()
        self.bootstrap_terminals.clear()
        self.prev_actions.clear()
        self.episode_return = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "observations": [_cpu_tensor(item) for item in self.observations],
            "next_observations": [_cpu_tensor(item) for item in self.next_observations],
            "actions": [_cpu_tensor(item) for item in self.actions],
            "task_rewards": [_cpu_tensor(item) for item in self.task_rewards],
            "motion_rewards": [_cpu_tensor(item) for item in self.motion_rewards],
            "dones": [_cpu_tensor(item) for item in self.dones],
            "bootstrap_terminals": [_cpu_tensor(item) for item in self.bootstrap_terminals],
            "prev_actions": [_cpu_tensor(item) for item in self.prev_actions],
            "episode_return": float(self.episode_return),
        }

    @classmethod
    def from_state_dict(cls, state_dict: Any, device: str) -> "EpisodeTrajectory":
        trajectory = cls.empty()
        if not isinstance(state_dict, dict):
            return trajectory
        for attr, key in (
            ("observations", "observations"),
            ("next_observations", "next_observations"),
            ("actions", "actions"),
            ("task_rewards", "task_rewards"),
            ("motion_rewards", "motion_rewards"),
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
