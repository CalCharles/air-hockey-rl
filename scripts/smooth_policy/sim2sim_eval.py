"""Zero-shot sim2sim evaluation harness.

Load a base policy checkpoint, build an AirHockeyEnv from an arbitrary sim
config (the *target*), run N deterministic episodes, and write a
``metrics.json`` to ``--out-dir``.

This is pure metric collection — no training. GIFs are off by default; pass
``--save-gif`` to record qualitative rollouts.

Usage
-----
::

    python scripts/smooth_policy/sim2sim_eval.py \
        --checkpoint runs/td3/<run>/checkpoint_<step>/model.pth \
        --target-config scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_<tag>.yaml \
        --n-episodes 50 \
        --seed 0 \
        --out-dir runs/td3/sim2sim/<src_to_tgt>/zero_shot/

See also
--------
notes/scratch/sim2sim_infra_plan.md §2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from scripts.smooth_policy.eval_utils import (
    augment_policy_observation,
    build_policy_env_view,
    load_policy_for_evaluation,
)


def _resolve_source_args_path(checkpoint_path: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        return explicit
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    # checkpoint_<step>/model.pth → walk up to the run dir that holds args.yaml.
    for candidate_dir in (ckpt_dir, os.path.dirname(ckpt_dir)):
        candidate = os.path.join(candidate_dir, "args.yaml")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not auto-locate args.yaml near checkpoint {checkpoint_path}. "
        f"Pass --source-args explicitly."
    )


def _load_sim_params(target_config_path: str) -> dict:
    with open(target_config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if "air_hockey" not in cfg:
        raise KeyError(
            f"Target config {target_config_path} is missing the top-level "
            f"'air_hockey' key."
        )
    return cfg["air_hockey"]


def _make_env(sim_params: dict) -> AirHockeyEnv:
    return AirHockeyEnv(sim_params)


def _run_episode(env, policy, seed: int, use_last_action: bool, action_dim: int) -> float:
    obs, _ = env.reset(seed=seed)
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    last_action = torch.zeros((1, action_dim), dtype=torch.float32)
    total_return = 0.0
    done = False
    while not done:
        policy_obs = augment_policy_observation(
            obs_tensor.unsqueeze(0), last_action, use_last_action
        )
        with torch.no_grad():
            action = policy(policy_obs).cpu().numpy().squeeze()
        obs, rew, term, trunc, _ = env.step(action)
        total_return += float(rew)
        done = bool(term or trunc)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
    return total_return


def _save_rollout_gif(env, policy, seed: int, use_last_action: bool, action_dim: int,
                      out_path: str) -> None:
    import cv2
    import imageio
    from airhockey.renderers import AirHockeyRenderer

    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)
    frames = []
    obs, _ = env.reset(seed=seed)
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    last_action = torch.zeros((1, action_dim), dtype=torch.float32)
    done = False
    cum_rew = 0.0
    while not done:
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect)))
        cv2.putText(frame, f"Ret: {cum_rew:.2f}", (frame.shape[1] - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        frames.append(frame)

        policy_obs = augment_policy_observation(
            obs_tensor.unsqueeze(0), last_action, use_last_action
        )
        with torch.no_grad():
            action = policy(policy_obs).cpu().numpy().squeeze()
        obs, rew, term, trunc, _ = env.step(action)
        cum_rew += float(rew)
        done = bool(term or trunc)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)

    imageio.mimsave(out_path, frames, format="GIF", loop=0, duration=int(1000 / 20))


def evaluate_zero_shot(
    checkpoint_path: str,
    target_config_path: str,
    out_dir: str,
    n_episodes: int = 50,
    seed: int = 0,
    source_args_path: Optional[str] = None,
    save_gif: bool = False,
    n_gifs: int = 1,
) -> dict:
    source_args_path = _resolve_source_args_path(checkpoint_path, source_args_path)
    with open(source_args_path, "r") as f:
        src_args = yaml.load(f, Loader=yaml.FullLoader)

    use_last_action = bool(src_args.get("use_last_action_in_policy_state", False))
    hidden_layer_size = int(src_args.get("agent_hidden_layer_size", 64))
    num_hidden_layers = int(src_args.get("agent_num_hidden_layers", 2))
    action_scale = float(src_args.get("action_scale", 1.0))

    sim_params = _load_sim_params(target_config_path)
    envs = gym.vector.SyncVectorEnv([lambda: _make_env(sim_params)])
    action_dim = int(np.prod(envs.single_action_space.shape))
    policy_env_view = build_policy_env_view(envs, use_last_action)

    policy = load_policy_for_evaluation(
        model_path=checkpoint_path,
        policy_env_view=policy_env_view,
        action_scale=action_scale,
        agent_hidden_layer_size=hidden_layer_size,
        agent_num_hidden_layers=num_hidden_layers,
    )

    env = envs.envs[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    returns = []
    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        ret = _run_episode(env, policy, ep_seed, use_last_action, action_dim)
        returns.append(ret)

    returns_arr = np.asarray(returns, dtype=np.float64)
    metrics = {
        "n_episodes": int(n_episodes),
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std(ddof=0)),
        "median_return": float(np.median(returns_arr)),
        "tail10": float(returns_arr[-10:].mean()) if len(returns_arr) >= 1 else float("nan"),
        "max_return": float(returns_arr.max()),
        "source_checkpoint_path": os.path.abspath(checkpoint_path),
        "source_args_path": os.path.abspath(source_args_path),
        "target_config_path": os.path.abspath(target_config_path),
        "seed": int(seed),
        "per_episode_returns": [float(r) for r in returns_arr],
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if save_gif and n_gifs > 0:
        gif_dir = os.path.join(out_dir, "eval_rollouts")
        os.makedirs(gif_dir, exist_ok=True)
        for g_idx in range(n_gifs):
            _save_rollout_gif(
                env, policy,
                seed=seed + n_episodes + g_idx,
                use_last_action=use_last_action,
                action_dim=action_dim,
                out_path=os.path.join(gif_dir, f"rollout_{g_idx}.gif"),
            )

    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Zero-shot sim2sim policy evaluation.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to actor checkpoint (e.g. runs/.../model.pth).")
    p.add_argument("--target-config", required=True,
                   help="Path to target sim YAML (has top-level 'air_hockey' key).")
    p.add_argument("--source-args", default=None,
                   help="Path to the training args.yaml. Auto-located if omitted.")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True,
                   help="Directory to write metrics.json (and optional GIFs).")
    p.add_argument("--save-gif", action="store_true",
                   help="Also record qualitative rollout GIFs.")
    p.add_argument("--n-gifs", type=int, default=1)
    args = p.parse_args(argv)

    metrics = evaluate_zero_shot(
        checkpoint_path=args.checkpoint,
        target_config_path=args.target_config,
        out_dir=args.out_dir,
        n_episodes=args.n_episodes,
        seed=args.seed,
        source_args_path=args.source_args,
        save_gif=args.save_gif,
        n_gifs=args.n_gifs,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_episode_returns"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
