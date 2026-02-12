"""
Collect trajectory and per-timestep policy rollout data for smooth_policy models.

This script mirrors the loading/evaluation pattern in `evaluate.py`:
- load model state dict and environment config
- instantiate `AirHockeyEnv` + `Agent`
- run policy rollouts

It exports:
- per_timestep.csv: flattened per-step records
- trajectory.npz: compact array-based trajectory tensors
- metadata.yaml: run metadata and collection settings
- model_used.pth / config_used.yaml: copied input artifacts
- example_episode.gif: one rollout visualization capped at 250 steps
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.agent import Agent


@dataclass
class CollectionLimits:
    """Stopping limits for data collection."""

    num_episodes: int | None
    total_timesteps: int | None


def _require_positive_or_none(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{name} must be > 0 when provided, got {value}.")
    return value


def _to_float_list(values: Any, length: int = 2, fill: float = np.nan) -> list[float]:
    """Convert nested values to a fixed-length float list with fallback fill values."""
    if values is None:
        return [fill] * length
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= length:
        return arr[:length].tolist()
    out = arr.tolist()
    out.extend([fill] * (length - arr.size))
    return out


def _extract_state_components(state_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract paddle/puck position/velocity/acceleration from env state_info."""
    paddle = state_info.get("paddles", {}).get("paddle_ego", {})
    pucks = state_info.get("pucks", [])
    puck = pucks[0] if len(pucks) > 0 else {}

    paddle_pos = np.asarray(_to_float_list(paddle.get("position")), dtype=np.float32)
    paddle_vel = np.asarray(_to_float_list(paddle.get("velocity")), dtype=np.float32)
    paddle_acc = np.asarray(_to_float_list(paddle.get("acceleration")), dtype=np.float32)
    puck_pos = np.asarray(_to_float_list(puck.get("position")), dtype=np.float32)
    puck_vel = np.asarray(_to_float_list(puck.get("velocity")), dtype=np.float32)
    return paddle_pos, paddle_vel, paddle_acc, puck_pos, puck_vel


def _predict_action(agent: Agent, obs: np.ndarray, device: torch.device) -> np.ndarray:
    """Run one policy forward pass and return a single action vector."""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_tensor, _, _, _ = agent.get_action_and_value(obs_tensor)
        return action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


def save_example_gif(
    env,
    agent: Agent,
    device: torch.device,
    renderer: AirHockeyRenderer,
    gif_path: str,
    max_steps: int = 250,
) -> int:
    """Save one rollout GIF capped by max_steps. Returns rendered frame count."""
    frames = []
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
        frames.append(frame)

        action = _predict_action(agent, obs, device)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

    if len(frames) > 0:
        imageio.mimsave(gif_path, frames, format="GIF", loop=0, duration=int(1000 * (1 / 20)))
    return len(frames)


def _should_stop(limits: CollectionLimits, episodes_completed: int, total_steps: int) -> bool:
    """Stop when either configured limit is reached (whichever-first semantics)."""
    hit_episode_limit = limits.num_episodes is not None and episodes_completed >= limits.num_episodes
    hit_step_limit = limits.total_timesteps is not None and total_steps >= limits.total_timesteps
    return hit_episode_limit or hit_step_limit


def collect_policy_data(
    model_path: str,
    config_path: str,
    save_dir: str,
    limits: CollectionLimits,
    agent_hidden_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run rollouts and export trajectory/per-step metrics and artifacts."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    air_hockey_params = config["air_hockey"]

    def make_env():
        return AirHockeyEnv(air_hockey_params)

    envs = gym.vector.SyncVectorEnv([make_env])
    env = envs.envs[0]
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)

    state_dict = torch.load(model_path, map_location=device)
    if "action_scale" not in state_dict:
        raise KeyError("Model checkpoint does not contain 'action_scale'; cannot infer policy action scaling.")
    action_scale = float(torch.as_tensor(state_dict["action_scale"]).item())
    agent = Agent(envs, action_scale=action_scale, action_bias=0.0, hidden_size=agent_hidden_size).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Store exact artifacts used for reproducibility.
    shutil.copy2(model_path, os.path.join(save_dir, "model_used.pth"))
    shutil.copy2(config_path, os.path.join(save_dir, "config_used.yaml"))

    step_records: list[dict[str, Any]] = []
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "episode_index": [],
        "episode_step": [],
        "global_timestep": [],
        "paddle_position": [],
        "target_position": [],
        "paddle_velocity": [],
        "paddle_acceleration": [],
        "puck_position": [],
        "puck_velocity": [],
        "puck_acceleration": [],
    }

    total_steps = 0
    episodes_started = 0
    episodes_completed = 0

    while not _should_stop(limits, episodes_completed, total_steps):
        obs, _ = env.reset()
        prev_puck_vel: np.ndarray | None = None
        episode_step = 0
        done = False
        episodes_started += 1
        episode_index = episodes_started - 1
        dt = float(getattr(env.simulator, "time_per_step", 1.0))

        while not done:
            action = _predict_action(agent, obs, device)
            pre_state_info = env.current_state
            pre_paddle_pos, _, _, _, _ = _extract_state_components(pre_state_info)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            state_info = env.current_state
            paddle_pos, paddle_vel, paddle_acc, puck_pos, puck_vel = _extract_state_components(state_info)

            if prev_puck_vel is None:
                puck_acc = np.zeros_like(puck_vel, dtype=np.float32)
            else:
                puck_acc = ((puck_vel - prev_puck_vel) / max(dt, 1e-8)).astype(np.float32)
            prev_puck_vel = puck_vel.copy()

            target_pos = (pre_paddle_pos + action).astype(np.float32)
            obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)

            trajectory["observations"].append(obs_vec)
            trajectory["actions"].append(action_vec)
            trajectory["rewards"].append(float(reward))
            trajectory["terminated"].append(bool(terminated))
            trajectory["truncated"].append(bool(truncated))
            trajectory["episode_index"].append(int(episode_index))
            trajectory["episode_step"].append(int(episode_step))
            trajectory["global_timestep"].append(int(total_steps))
            trajectory["paddle_position"].append(paddle_pos)
            trajectory["target_position"].append(target_pos)
            trajectory["paddle_velocity"].append(paddle_vel)
            trajectory["paddle_acceleration"].append(paddle_acc)
            trajectory["puck_position"].append(puck_pos)
            trajectory["puck_velocity"].append(puck_vel)
            trajectory["puck_acceleration"].append(puck_acc)

            row: dict[str, Any] = {
                "global_timestep": int(total_steps),
                "episode_index": int(episode_index),
                "episode_step": int(episode_step),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "paddle_pos_x": float(paddle_pos[0]),
                "paddle_pos_y": float(paddle_pos[1]),
                "target_pos_x": float(target_pos[0]),
                "target_pos_y": float(target_pos[1]),
                "paddle_vel_x": float(paddle_vel[0]),
                "paddle_vel_y": float(paddle_vel[1]),
                "paddle_acc_x": float(paddle_acc[0]),
                "paddle_acc_y": float(paddle_acc[1]),
                "puck_pos_x": float(puck_pos[0]),
                "puck_pos_y": float(puck_pos[1]),
                "puck_vel_x": float(puck_vel[0]),
                "puck_vel_y": float(puck_vel[1]),
                "puck_acc_x": float(puck_acc[0]),
                "puck_acc_y": float(puck_acc[1]),
            }
            for i, val in enumerate(obs_vec):
                row[f"obs_{i}"] = float(val)
            for i, val in enumerate(action_vec):
                row[f"action_{i}"] = float(val)
            step_records.append(row)

            total_steps += 1
            episode_step += 1
            obs = next_obs

            # Episode limits are checked between episodes; timestep limit is immediate.
            if limits.total_timesteps is not None and total_steps >= limits.total_timesteps:
                break

        if done:
            episodes_completed += 1

    if len(step_records) == 0:
        raise RuntimeError("No rollout steps were collected. Check episode/timestep limits.")

    csv_path = os.path.join(save_dir, "per_timestep.csv")
    csv_fieldnames = list(step_records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(step_records)

    npz_path = os.path.join(save_dir, "trajectory.npz")
    np.savez_compressed(
        npz_path,
        observations=np.asarray(trajectory["observations"], dtype=np.float32),
        actions=np.asarray(trajectory["actions"], dtype=np.float32),
        rewards=np.asarray(trajectory["rewards"], dtype=np.float32),
        terminated=np.asarray(trajectory["terminated"], dtype=np.bool_),
        truncated=np.asarray(trajectory["truncated"], dtype=np.bool_),
        episode_index=np.asarray(trajectory["episode_index"], dtype=np.int32),
        episode_step=np.asarray(trajectory["episode_step"], dtype=np.int32),
        global_timestep=np.asarray(trajectory["global_timestep"], dtype=np.int64),
        paddle_position=np.asarray(trajectory["paddle_position"], dtype=np.float32),
        target_position=np.asarray(trajectory["target_position"], dtype=np.float32),
        paddle_velocity=np.asarray(trajectory["paddle_velocity"], dtype=np.float32),
        paddle_acceleration=np.asarray(trajectory["paddle_acceleration"], dtype=np.float32),
        puck_position=np.asarray(trajectory["puck_position"], dtype=np.float32),
        puck_velocity=np.asarray(trajectory["puck_velocity"], dtype=np.float32),
        puck_acceleration=np.asarray(trajectory["puck_acceleration"], dtype=np.float32),
    )

    gif_path = os.path.join(save_dir, "example_episode.gif")
    gif_frames = save_example_gif(env, agent, device, renderer, gif_path, max_steps=250)

    metadata = {
        "inputs": {
            "model_path": os.path.abspath(model_path),
            "config_path": os.path.abspath(config_path),
        },
        "collection": {
            "requested_num_episodes": limits.num_episodes,
            "requested_total_timesteps": limits.total_timesteps,
            "stop_rule": "whichever_first",
            "action_scale_loaded_from_model": action_scale,
            "episodes_started": episodes_started,
            "episodes_completed": episodes_completed,
            "total_timesteps_collected": total_steps,
        },
        "shapes": {
            "observation_dim": int(np.asarray(trajectory["observations"][0]).shape[0]),
            "action_dim": int(np.asarray(trajectory["actions"][0]).shape[0]),
            "num_rows_csv": len(step_records),
        },
        "outputs": {
            "per_timestep_csv": "per_timestep.csv",
            "trajectory_npz": "trajectory.npz",
            "model_copy": "model_used.pth",
            "config_copy": "config_used.yaml",
            "example_gif": "example_episode.gif",
            "example_gif_frames": gif_frames,
        },
    }
    with open(os.path.join(save_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect smooth_policy rollout data into CSV/NPZ artifacts.")
    parser.add_argument("--model", type=str, required=True, help="Path to policy model state dict (.pth).")
    parser.add_argument("--config-path", type=str, required=True, help="Path to YAML config containing air_hockey settings.")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save exported artifacts.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Episode limit for collection.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Global timestep limit for collection.")
    parser.add_argument("--agent-hidden-size", type=int, default=64, help="Hidden layer width for Agent MLPs.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (for example: cpu, cuda:0).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    num_episodes = _require_positive_or_none(args.num_episodes, "--num-episodes")
    total_timesteps = _require_positive_or_none(args.total_timesteps, "--total-timesteps")

    if num_episodes is None and total_timesteps is None:
        raise ValueError("Provide at least one stopping limit: --num-episodes and/or --total-timesteps.")

    metadata = collect_policy_data(
        model_path=args.model,
        config_path=args.config_path,
        save_dir=args.save_dir,
        limits=CollectionLimits(num_episodes=num_episodes, total_timesteps=total_timesteps),
        agent_hidden_size=args.agent_hidden_size,
        device=torch.device(args.device),
    )
    print(yaml.safe_dump(metadata, sort_keys=False))
