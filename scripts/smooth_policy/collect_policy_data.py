"""
Collect trajectory data and generate rollout GIFs for smooth_policy models.

Handles both stochastic (Agent) and deterministic (DeterministicAgent) policies.
Can be pointed at a TD3 training run directory via --run-dir, which auto-reads
args.yaml and config.yaml so no manual architecture flags are needed.

Exports:
- eval_0.gif, eval_1.gif, ...: rollout visualizations
- per_timestep.csv: flattened per-step records
- trajectory.npz: compact array-based trajectory tensors
- metadata.yaml: run metadata and collection settings
- model_used.pth / config_used.yaml: copied input artifacts
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.evaluate import (
    _augment_policy_observation,
    _load_policy_for_evaluation,
)


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


def _predict_action(
    model,
    obs: np.ndarray,
    last_action: torch.Tensor,
    use_last_action: bool,
) -> np.ndarray:
    """Run one policy forward pass and return a single action vector."""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        policy_obs = _augment_policy_observation(obs_tensor.unsqueeze(0), last_action, use_last_action)
        result = model(policy_obs)
        action_tensor = result[0] if isinstance(result, tuple) else result
        return action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _should_stop(limits: CollectionLimits, episodes_completed: int, total_steps: int) -> bool:
    """Stop when either configured limit is reached (whichever-first semantics)."""
    hit_episode_limit = limits.num_episodes is not None and episodes_completed >= limits.num_episodes
    hit_step_limit = limits.total_timesteps is not None and total_steps >= limits.total_timesteps
    return hit_episode_limit or hit_step_limit


def save_rollout_gif(
    env,
    model,
    renderer: AirHockeyRenderer,
    gif_path: str,
    action_dim: int,
    use_last_action_in_policy_state: bool = False,
    max_steps: int = 250,
) -> int:
    """Save one rollout GIF. Returns rendered frame count."""
    frames = []
    obs, _ = env.reset()
    last_action = torch.zeros((1, action_dim), dtype=torch.float32)
    done = False
    steps = 0

    while not done and steps < max_steps:
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
        frames.append(frame)

        action = _predict_action(model, obs, last_action, use_last_action_in_policy_state)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
        if done:
            last_action.zero_()
        steps += 1

    if frames:
        imageio.mimsave(gif_path, frames, format="GIF", loop=0, duration=int(1000 * (1 / 20)))
    return len(frames)


def collect_policy_data(
    model_path: str,
    config_path: str,
    save_dir: str,
    limits: CollectionLimits,
    agent_hidden_layer_size: int = 64,
    agent_num_hidden_layers: int = 2,
    action_scale: float = 0.02,
    use_last_action_in_policy_state: bool = False,
    policy_type: str | None = None,
    n_gifs: int = 1,
) -> dict[str, Any]:
    """Run rollouts, export trajectory/per-step metrics, and save GIFs."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    air_hockey_params = config["air_hockey"]

    def make_env():
        return AirHockeyEnv(air_hockey_params)

    envs = gym.vector.SyncVectorEnv([make_env])
    env = envs.envs[0]
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    policy_obs_dim = obs_dim + action_dim if use_last_action_in_policy_state else obs_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32
        ),
        single_action_space=envs.single_action_space,
    )

    model = _load_policy_for_evaluation(
        model_path=model_path,
        policy_env_view=policy_env_view,
        action_scale=action_scale,
        agent_hidden_layer_size=agent_hidden_layer_size,
        agent_num_hidden_layers=agent_num_hidden_layers,
        policy_type=policy_type,
    )

    os.makedirs(save_dir, exist_ok=True)
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
        last_action = torch.zeros((1, action_dim), dtype=torch.float32)
        episode_step = 0
        done = False
        episodes_started += 1
        episode_index = episodes_started - 1
        dt = float(getattr(env.simulator, "time_per_step", 1.0))

        while not done:
            action = _predict_action(model, obs, last_action, use_last_action_in_policy_state)

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

            last_action = torch.tensor(action_vec, dtype=torch.float32).reshape(1, -1)
            if done:
                last_action.zero_()

            total_steps += 1
            episode_step += 1
            obs = next_obs

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

    gif_frame_counts = []
    for i in range(n_gifs):
        gif_path = os.path.join(save_dir, f"eval_{i}.gif")
        frames = save_rollout_gif(
            env, model, renderer, gif_path, action_dim,
            use_last_action_in_policy_state=use_last_action_in_policy_state,
        )
        gif_frame_counts.append(frames)

    metadata = {
        "inputs": {
            "model_path": os.path.abspath(model_path),
            "config_path": os.path.abspath(config_path),
        },
        "collection": {
            "requested_num_episodes": limits.num_episodes,
            "requested_total_timesteps": limits.total_timesteps,
            "stop_rule": "whichever_first",
            "action_scale": action_scale,
            "use_last_action_in_policy_state": use_last_action_in_policy_state,
            "episodes_started": episodes_started,
            "episodes_completed": episodes_completed,
            "total_timesteps_collected": total_steps,
        },
        "shapes": {
            "observation_dim": int(np.asarray(trajectory["observations"][0]).shape[0]),
            "action_dim": action_dim,
            "num_rows_csv": len(step_records),
        },
        "outputs": {
            "gifs": [f"eval_{i}.gif" for i in range(n_gifs)],
            "gif_frame_counts": gif_frame_counts,
            "per_timestep_csv": "per_timestep.csv",
            "trajectory_npz": "trajectory.npz",
            "model_copy": "model_used.pth",
            "config_copy": "config_used.yaml",
        },
    }
    with open(os.path.join(save_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return metadata


def _args_from_run_dir(run_dir: str) -> tuple[str, str, dict]:
    """Read model path, env config path, and policy kwargs from a TD3 run directory."""
    args_yaml = os.path.join(run_dir, "args.yaml")
    if not os.path.isfile(args_yaml):
        print(f"ERROR: args.yaml not found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    with open(args_yaml, "r") as f:
        run_args = yaml.safe_load(f)

    model_path = os.path.join(run_dir, "model.pth")
    if not os.path.isfile(model_path):
        print(f"ERROR: model.pth not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Env config: prefer path recorded in args.yaml, fall back to config.yaml in run dir
    config_path = run_args.get("config")
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    if not config_path or not os.path.isfile(config_path):
        config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"ERROR: env config not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    policy_kwargs = {
        "action_scale": float(run_args.get("action_scale", 0.02)),
        "agent_hidden_layer_size": int(run_args.get("agent_hidden_layer_size", 64)),
        "agent_num_hidden_layers": int(run_args.get("agent_num_hidden_layers", 2)),
        "use_last_action_in_policy_state": bool(run_args.get("use_last_action_in_policy_state", False)),
        "policy_type": "deterministic_agent",
    }
    return model_path, config_path, policy_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect smooth_policy rollout data into CSV/NPZ/GIF artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "TD3 run directory (auto-reads args.yaml/config.yaml):\n"
            "  uv run scripts/smooth_policy/collect_policy_data.py \\\n"
            "      --run-dir /data2/.../my_run --num-episodes 20 --n-gifs 5\n\n"
            "Explicit paths:\n"
            "  uv run scripts/smooth_policy/collect_policy_data.py \\\n"
            "      --model /path/model.pth --config-path /path/config.yaml \\\n"
            "      --num-episodes 20 --n-gifs 5 --action-scale 1.0 --use-last-action"
        ),
    )
    # --- run-dir shortcut (TD3) ---
    parser.add_argument("--run-dir", type=str, default=None,
                        help="TD3 training run directory. Auto-reads model.pth, args.yaml, config.yaml.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory. Defaults to <run-dir>/rollout when --run-dir is used.")

    # --- explicit model/config (used when --run-dir is not given) ---
    parser.add_argument("--model", type=str, default=None, help="Path to policy model state dict (.pth).")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to YAML config containing air_hockey settings.")
    parser.add_argument("--action-scale", type=float, default=0.02, help="Policy action scale.")
    parser.add_argument("--agent-hidden-layer-size", type=int, default=64)
    parser.add_argument("--agent-num-hidden-layers", type=int, default=2)
    parser.add_argument("--use-last-action", action="store_true",
                        help="Augment observations with last action.")
    parser.add_argument("--policy-type", type=str, default=None,
                        choices=["agent", "deterministic_agent"],
                        help="Policy class override. Inferred from checkpoint if not set.")

    # --- collection limits ---
    parser.add_argument("--num-episodes", type=int, default=None, help="Episode limit for collection.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Global timestep limit.")
    parser.add_argument("--n-gifs", type=int, default=3, help="Number of rollout GIFs to generate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_dir is not None:
        model_path, config_path, policy_kwargs = _args_from_run_dir(os.path.abspath(args.run_dir))
        save_dir = args.save_dir or os.path.join(os.path.abspath(args.run_dir), "rollout")
    else:
        if not args.model or not args.config_path:
            print("ERROR: provide --run-dir or both --model and --config-path.", file=sys.stderr)
            sys.exit(1)
        model_path = args.model
        config_path = args.config_path
        save_dir = args.save_dir
        if not save_dir:
            print("ERROR: --save-dir is required when not using --run-dir.", file=sys.stderr)
            sys.exit(1)
        policy_kwargs = {
            "action_scale": args.action_scale,
            "agent_hidden_layer_size": args.agent_hidden_layer_size,
            "agent_num_hidden_layers": args.agent_num_hidden_layers,
            "use_last_action_in_policy_state": args.use_last_action,
            "policy_type": args.policy_type,
        }

    num_episodes = _require_positive_or_none(args.num_episodes, "--num-episodes")
    total_timesteps = _require_positive_or_none(args.total_timesteps, "--total-timesteps")
    if num_episodes is None and total_timesteps is None:
        print("ERROR: provide at least one stopping limit: --num-episodes and/or --total-timesteps.", file=sys.stderr)
        sys.exit(1)

    print(f"Model:      {model_path}")
    print(f"Config:     {config_path}")
    print(f"Save dir:   {save_dir}")
    print(f"Policy:     {policy_kwargs}")

    metadata = collect_policy_data(
        model_path=model_path,
        config_path=config_path,
        save_dir=save_dir,
        limits=CollectionLimits(num_episodes=num_episodes, total_timesteps=total_timesteps),
        n_gifs=args.n_gifs,
        **policy_kwargs,
    )
    print(yaml.safe_dump(metadata, sort_keys=False))
