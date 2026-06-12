"""Rollout collection and offscreen video recording."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import imageio
import numpy as np


PolicyFn = Callable[[np.ndarray, bool], np.ndarray]


def collect_rollout(
    env,
    policy: PolicyFn,
    *,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """
    Run one episode and return trajectory statistics.

    policy(obs, deterministic) -> action
    """
    obs, info = env.reset(seed=seed)

    # Reset stateful policy (history buffer, last action) if supported
    if hasattr(policy, "reset"):
        policy.reset()

    max_steps = max_steps or env.unwrapped.horizon

    rewards: list[float] = []
    for _ in range(max_steps):
        action = policy(np.asarray(obs, dtype=np.float32), deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return {
        "return": float(sum(rewards)),
        "length": len(rewards),
        "success": bool(terminated),
        "task": info.get("task"),
    }

def _get_sim(env):
    """
    Robustly get the MuJoCo sim object regardless of wrapper depth.
    Handles: adapter -> robosuite env, or direct robosuite env.
    """
    # Our adapter: env._env is the RobosuiteAirHockeyEnv which has .sim
    if hasattr(env, "_env") and hasattr(env._env, "sim"):
        return env._env.sim
    # Direct robosuite env
    if hasattr(env, "sim"):
        return env.sim
    # Gymnasium unwrapped chain
    if hasattr(env, "unwrapped"):
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "_env") and hasattr(unwrapped._env, "sim"):
            return unwrapped._env.sim
        if hasattr(unwrapped, "sim"):
            return unwrapped.sim
    raise AttributeError(
        f"Cannot find MuJoCo sim on env of type {type(env)}. "
        "Expected env._env.sim or env.sim."
    )


def record_rollout_video(
    env,
    policy,
    output_path,
    *,
    camera_name: str = "overview",
    height: int = 512,
    width: int = 512,
    fps: int = 20,
    max_steps=None,
    deterministic: bool = True,
    seed=None,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not getattr(env, "has_offscreen_renderer", False):
        raise ValueError("record_rollout_video requires has_offscreen_renderer=True")

    obs, info = env.reset(seed=seed)

    if hasattr(policy, "reset"):
        policy.reset()

    sim = _get_sim(env)
    max_steps = max_steps or getattr(env, "max_episode_steps", 500)
    frames, rewards = [], []
    terminated = False

    for _ in range(max_steps):
        frame = sim.render(
            camera_name=camera_name,
            height=height,
            width=width,
        )[::-1]   # robosuite renders upside-down
        frames.append(frame)

        action = policy(np.asarray(obs, dtype=np.float32), deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    if frames:
        writer = imageio.get_writer(str(output_path), fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    return {
        "return": float(sum(rewards)),
        "length": len(rewards),
        "success": bool(info.get("success", False)),
        "task": info.get("task"),
        "video_path": str(output_path),
        "num_frames": len(frames),
    }


def evaluate_policy(
    env,
    policy: PolicyFn,
    *,
    n_episodes: int,
    seeds: Optional[Sequence[int]] = None,
    deterministic: bool = True,
) -> dict:
    """Run multiple evaluation episodes and aggregate metrics."""
    if seeds is None:
        seeds = list(range(n_episodes))
    if len(seeds) < n_episodes:
        raise ValueError("Need at least n_episodes seeds for evaluation.")

    stats = [collect_rollout(env, policy, deterministic=deterministic, seed=seeds[i]) for i in range(n_episodes)]
    returns = np.array([s["return"] for s in stats], dtype=np.float32)
    lengths = np.array([s["length"] for s in stats], dtype=np.float32)
    successes = np.array([s["success"] for s in stats], dtype=np.float32)

    return {
        "episodes": stats,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(lengths.mean()),
        "success_rate": float(successes.mean()),
    }


