from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict

import numpy as np
import torch

from scripts.smooth_policy.amp_history.amp_training.td3.td3_episode_collection import (
    EpisodeTrajectory,
    load_episode_trajectory_from_checkpoint,
)


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().cpu()


def get_rng_states() -> Dict[str, object]:
    states: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return states


def set_rng_states(states: Dict[str, object]) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def load_resume_training_state(
    resume_checkpoint: Dict[str, Any],
    *,
    device: str,
    recent_episode_window_size: int,
    success_rb,
    failure_rb,
    primitive_selector,
    q_optimizer,
    actor_optimizer,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    q_optimizer.load_state_dict(resume_checkpoint["q_optimizer"])
    actor_optimizer.load_state_dict(resume_checkpoint["actor_optimizer"])
    if "success_replay_buffer" in resume_checkpoint and "failure_replay_buffer" in resume_checkpoint:
        success_rb.load_state_dict(resume_checkpoint["success_replay_buffer"])
        failure_rb.load_state_dict(resume_checkpoint["failure_replay_buffer"])
    elif "replay_buffer" in resume_checkpoint:
        # Backward compatibility with older checkpoints that stored a single replay buffer.
        failure_rb.load_state_dict(resume_checkpoint["replay_buffer"])
    if "primitive_selector" in resume_checkpoint:
        primitive_selector.load_state_dict(resume_checkpoint["primitive_selector"])

    loaded_recent_returns = resume_checkpoint.get("recent_episode_returns")
    if isinstance(loaded_recent_returns, list):
        recent_episode_returns = deque(
            [float(val) for val in loaded_recent_returns],
            maxlen=recent_episode_window_size,
        )
    else:
        recent_episode_returns = defaults["recent_episode_returns"]

    if "rng_states" in resume_checkpoint:
        set_rng_states(resume_checkpoint["rng_states"])

    return {
        "global_step": int(resume_checkpoint["global_step"]),
        "iteration": int(resume_checkpoint["iteration"]),
        "obs": np.asarray(resume_checkpoint["obs"], dtype=np.float32),
        "last_action_for_policy": resume_checkpoint["last_action_for_policy"].to(device),
        "warm_policy_last_action": resume_checkpoint.get(
            "warm_policy_last_action", defaults["warm_policy_last_action"]
        ).to(device),
        "temporal_paddle_history": resume_checkpoint["temporal_paddle_history"].to(device),
        "temporal_puck_history": resume_checkpoint["temporal_puck_history"].to(device),
        "temporal_done_history": resume_checkpoint["temporal_done_history"].to(device),
        "temporal_position_count": resume_checkpoint["temporal_position_count"].to(device),
        "current_velocity_mag": resume_checkpoint["current_velocity_mag"].to(device),
        "current_acceleration_mag": resume_checkpoint["current_acceleration_mag"].to(device),
        "current_jerk_mag": resume_checkpoint["current_jerk_mag"].to(device),
        "train_metrics": dict(resume_checkpoint.get("train_metrics", defaults["train_metrics"])),
        "episodic_returns": list(resume_checkpoint.get("episodic_returns", defaults["episodic_returns"])),
        "success_rates": list(resume_checkpoint.get("success_rates", defaults["success_rates"])),
        "velocity_magnitudes": list(
            resume_checkpoint.get("velocity_magnitudes", defaults["velocity_magnitudes"])
        ),
        "acceleration_magnitudes": list(
            resume_checkpoint.get("acceleration_magnitudes", defaults["acceleration_magnitudes"])
        ),
        "jerk_magnitudes": list(resume_checkpoint.get("jerk_magnitudes", defaults["jerk_magnitudes"])),
        "interval_paddle_puck_collisions": float(
            resume_checkpoint.get(
                "interval_paddle_puck_collisions",
                defaults["interval_paddle_puck_collisions"],
            )
        ),
        "interval_env_steps": int(resume_checkpoint.get("interval_env_steps", defaults["interval_env_steps"])),
        "interval_primitive_env_steps": int(
            resume_checkpoint.get("interval_primitive_env_steps", defaults["interval_primitive_env_steps"])
        ),
        "interval_primitive_horizontal_env_steps": int(
            resume_checkpoint.get(
                "interval_primitive_horizontal_env_steps",
                defaults["interval_primitive_horizontal_env_steps"],
            )
        ),
        "interval_policy_takeover_env_steps": int(
            resume_checkpoint.get(
                "interval_policy_takeover_env_steps",
                defaults["interval_policy_takeover_env_steps"],
            )
        ),
        "interval_target_position_directional_env_steps": int(
            resume_checkpoint.get(
                "interval_target_position_directional_env_steps",
                defaults["interval_target_position_directional_env_steps"],
            )
        ),
        "episode_trajectory": load_episode_trajectory_from_checkpoint(
            resume_checkpoint,
            device=device,
        ),
        "recent_episode_returns": recent_episode_returns,
        "episode_return_success_threshold": float(
            resume_checkpoint.get(
                "episode_return_success_threshold",
                defaults["episode_return_success_threshold"],
            )
        ),
    }


def build_training_state(
    *,
    global_step: int,
    iteration: int,
    actor,
    actor_target,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    q_optimizer,
    actor_optimizer,
    success_rb,
    failure_rb,
    primitive_selector,
    obs: np.ndarray,
    last_action_for_policy: torch.Tensor,
    warm_policy_last_action: torch.Tensor,
    temporal_paddle_history: torch.Tensor,
    temporal_puck_history: torch.Tensor,
    temporal_done_history: torch.Tensor,
    temporal_position_count: torch.Tensor,
    current_velocity_mag: torch.Tensor,
    current_acceleration_mag: torch.Tensor,
    current_jerk_mag: torch.Tensor,
    train_metrics: Dict[str, float],
    episodic_returns,
    success_rates,
    velocity_magnitudes,
    acceleration_magnitudes,
    jerk_magnitudes,
    interval_paddle_puck_collisions: float,
    interval_env_steps: int,
    interval_primitive_env_steps: int,
    interval_primitive_horizontal_env_steps: int,
    interval_policy_takeover_env_steps: int,
    interval_target_position_directional_env_steps: int,
    episode_trajectory: EpisodeTrajectory,
    recent_episode_returns,
    episode_return_success_threshold: float,
    args_dict: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "global_step": global_step,
        "iteration": iteration,
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "q_optimizer": q_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "success_replay_buffer": success_rb.state_dict(),
        "failure_replay_buffer": failure_rb.state_dict(),
        "primitive_selector": primitive_selector.state_dict(),
        "obs": np.array(obs, copy=True),
        "last_action_for_policy": _cpu_tensor(last_action_for_policy),
        "warm_policy_last_action": _cpu_tensor(warm_policy_last_action),
        "temporal_paddle_history": _cpu_tensor(temporal_paddle_history),
        "temporal_puck_history": _cpu_tensor(temporal_puck_history),
        "temporal_done_history": _cpu_tensor(temporal_done_history),
        "temporal_position_count": _cpu_tensor(temporal_position_count),
        "current_velocity_mag": _cpu_tensor(current_velocity_mag),
        "current_acceleration_mag": _cpu_tensor(current_acceleration_mag),
        "current_jerk_mag": _cpu_tensor(current_jerk_mag),
        "train_metrics": dict(train_metrics),
        "episodic_returns": list(episodic_returns),
        "success_rates": list(success_rates),
        "velocity_magnitudes": list(velocity_magnitudes),
        "acceleration_magnitudes": list(acceleration_magnitudes),
        "jerk_magnitudes": list(jerk_magnitudes),
        "interval_paddle_puck_collisions": interval_paddle_puck_collisions,
        "interval_env_steps": interval_env_steps,
        "interval_primitive_env_steps": interval_primitive_env_steps,
        "interval_primitive_horizontal_env_steps": interval_primitive_horizontal_env_steps,
        "interval_policy_takeover_env_steps": interval_policy_takeover_env_steps,
        "interval_target_position_directional_env_steps": interval_target_position_directional_env_steps,
        "episode_trajectory": episode_trajectory.state_dict(),
        "recent_episode_returns": list(recent_episode_returns),
        "episode_return_success_threshold": episode_return_success_threshold,
        "rng_states": get_rng_states(),
        "args": args_dict,
    }
