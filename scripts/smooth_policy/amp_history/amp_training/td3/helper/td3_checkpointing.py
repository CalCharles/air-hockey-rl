"""Checkpoint save/restore for TD3 training (RNG, optimizers, replay, episode state).

See notes/docs/training/checkpointing.md for the schema and migration details.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_episode_collection import (
    EpisodeTrajectory,
    load_episode_trajectory_from_checkpoint,
)


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().cpu()


def _to_rng_byte_tensor(state: object) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.detach().clone().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


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
        torch.set_rng_state(_to_rng_byte_tensor(states["torch_cpu"]))
    # Intentionally skip CUDA RNG restore on resume to avoid GPU topology/device-count
    # mismatch failures across runs (e.g., different CUDA_VISIBLE_DEVICES settings).


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

    if "steps_since_done" in resume_checkpoint:
        steps_since_done = resume_checkpoint["steps_since_done"].to(device)
    else:
        # Backward compatibility for older checkpoints that used
        # temporal_done_history + temporal_position_count.
        legacy_position_count = resume_checkpoint.get("temporal_position_count")
        if legacy_position_count is None:
            steps_since_done = defaults["steps_since_done"]
        else:
            steps_since_done = torch.clamp(legacy_position_count.to(device) - 1, min=0)
        legacy_done_history = resume_checkpoint.get("temporal_done_history")
        if legacy_done_history is not None:
            recent_done = legacy_done_history.to(device).any(dim=1)
            steps_since_done = torch.where(
                recent_done,
                torch.zeros_like(steps_since_done),
                steps_since_done,
            )

    return {
        "global_step": int(resume_checkpoint["global_step"]),
        "iteration": int(resume_checkpoint["iteration"]),
        "total_critic_updates": int(resume_checkpoint.get("total_critic_updates", 0)),
        "obs": np.asarray(resume_checkpoint["obs"], dtype=np.float32),
        "last_action_for_policy": resume_checkpoint["last_action_for_policy"].to(device),
        "warm_policy_last_action": resume_checkpoint.get(
            "warm_policy_last_action", defaults["warm_policy_last_action"]
        ).to(device),
        "temporal_paddle_history": resume_checkpoint["temporal_paddle_history"].to(device),
        "temporal_puck_history": resume_checkpoint["temporal_puck_history"].to(device),
        "steps_since_done": steps_since_done,
        "current_velocity_mag": resume_checkpoint["current_velocity_mag"].to(device),
        "current_acceleration_mag": resume_checkpoint["current_acceleration_mag"].to(device),
        "current_jerk_mag": resume_checkpoint["current_jerk_mag"].to(device),
        "train_metrics": dict(resume_checkpoint.get("train_metrics", defaults["train_metrics"])),
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
        "rolling_step_stats_window": deque(
            [
                (
                    int(item[0]),
                    int(item[1]),
                    float(item[2]),
                    float(item[3]),
                )
                for item in resume_checkpoint.get(
                    "rolling_step_stats_window",
                    defaults["rolling_step_stats_window"],
                )
                if isinstance(item, (list, tuple)) and len(item) >= 4
            ]
        ),
        "rolling_episode_stats_window": deque(
            [
                (
                    int(item[0]),
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                )
                for item in resume_checkpoint.get(
                    "rolling_episode_stats_window",
                    defaults["rolling_episode_stats_window"],
                )
                if isinstance(item, (list, tuple)) and len(item) >= 4
            ]
        ),
    }


def load_fine_tune_optimizer_state(
    checkpoint: Dict[str, Any], *, q_optimizer, actor_optimizer
) -> None:
    if "q_optimizer" not in checkpoint:
        raise KeyError("Missing 'q_optimizer' in checkpoint; cannot run fine-tune optimizer restore.")
    if "actor_optimizer" not in checkpoint:
        raise KeyError("Missing 'actor_optimizer' in checkpoint; cannot run fine-tune optimizer restore.")
    q_optimizer.load_state_dict(checkpoint["q_optimizer"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])


def seed_fine_tune_replay_from_source(
    *,
    success_rb,
    failure_rb,
    source_success: Dict[str, Any] | None,
    source_failure: Dict[str, Any] | None,
    keep_total: int,
    seed: int = 0,
) -> Dict[str, int]:
    """Seed fresh target replay buffers with up to ``keep_total`` source samples.

    Subsamples uniformly within each source buffer, splitting the keep budget
    proportionally to how full each source buffer is. Source samples are added
    via the buffers' ``add()`` API so positions/sizes stay consistent.
    """
    keep_total = int(keep_total)
    if keep_total <= 0:
        return {"success_kept": 0, "failure_kept": 0}

    success_size = int(source_success["size"]) if source_success is not None else 0
    failure_size = int(source_failure["size"]) if source_failure is not None else 0
    total_source = success_size + failure_size
    if total_source == 0:
        return {"success_kept": 0, "failure_kept": 0}

    keep_total = min(keep_total, total_source)
    success_keep = int(round(keep_total * success_size / total_source))
    failure_keep = keep_total - success_keep
    success_keep = min(success_keep, success_size)
    failure_keep = min(failure_keep, failure_size)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    def _seed_one(target_rb, source: Dict[str, Any], n_keep: int) -> int:
        if n_keep <= 0 or source is None:
            return 0
        size = int(source["size"])
        if size == 0:
            return 0
        n_keep = min(n_keep, size)
        idx = torch.randperm(size, generator=rng)[:n_keep]
        target_rb.add(
            obs=source["observations"][idx],
            next_obs=source["next_observations"][idx],
            actions=source["actions"][idx],
            task_rewards=source["task_rewards"][idx],
            motion_rewards=source["motion_rewards"][idx],
            dones=source["dones"][idx],
            prev_action=source["prev_actions"][idx],
        )
        return n_keep

    kept_success = _seed_one(success_rb, source_success, success_keep)
    kept_failure = _seed_one(failure_rb, source_failure, failure_keep)
    return {"success_kept": kept_success, "failure_kept": kept_failure}


def build_training_state(
    *,
    global_step: int,
    iteration: int,
    total_critic_updates: int = 0,
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
    include_replay_buffer: bool = True,
    obs: np.ndarray,
    last_action_for_policy: torch.Tensor,
    warm_policy_last_action: torch.Tensor,
    temporal_paddle_history: torch.Tensor,
    temporal_puck_history: torch.Tensor,
    steps_since_done: torch.Tensor,
    current_velocity_mag: torch.Tensor,
    current_acceleration_mag: torch.Tensor,
    current_jerk_mag: torch.Tensor,
    train_metrics: Dict[str, float],
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
    rolling_step_stats_window,
    rolling_episode_stats_window,
    args_dict: Dict[str, Any],
    extra_qfs: Optional[List[Any]] = None,
    extra_qfs_target: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "checkpoint_version": 2,
        "global_step": global_step,
        "iteration": iteration,
        "total_critic_updates": int(total_critic_updates),
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "q_optimizer": q_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "primitive_selector": primitive_selector.state_dict(),
        "obs": np.array(obs, copy=True),
        "last_action_for_policy": _cpu_tensor(last_action_for_policy),
        "warm_policy_last_action": _cpu_tensor(warm_policy_last_action),
        "temporal_paddle_history": _cpu_tensor(temporal_paddle_history),
        "temporal_puck_history": _cpu_tensor(temporal_puck_history),
        "steps_since_done": _cpu_tensor(steps_since_done),
        "current_velocity_mag": _cpu_tensor(current_velocity_mag),
        "current_acceleration_mag": _cpu_tensor(current_acceleration_mag),
        "current_jerk_mag": _cpu_tensor(current_jerk_mag),
        "train_metrics": dict(train_metrics),
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
        "rolling_step_stats_window": [
            (
                int(item[0]),
                int(item[1]),
                float(item[2]),
                float(item[3]),
            )
            for item in rolling_step_stats_window
        ],
        "rolling_episode_stats_window": [
            (
                int(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
            )
            for item in rolling_episode_stats_window
        ],
        "rng_states": get_rng_states(),
        "args": args_dict,
    }
    if include_replay_buffer:
        state["success_replay_buffer"] = success_rb.state_dict()
        state["failure_replay_buffer"] = failure_rb.state_dict()
    # Critics 3+ (only present for ensemble runs; backwards compat for N=2 ckpts).
    if extra_qfs is not None and extra_qfs_target is not None:
        if len(extra_qfs) != len(extra_qfs_target):
            raise ValueError("extra_qfs and extra_qfs_target must have same length")
        for offset, (q, qt) in enumerate(zip(extra_qfs, extra_qfs_target)):
            ci = offset + 3  # qf3, qf4, ...
            state[f"qf{ci}"] = q.state_dict()
            state[f"qf{ci}_target"] = qt.state_dict()
    return state
