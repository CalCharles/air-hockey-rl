"""Evaluation of a goal-conditioned (HER-trained) TD3 actor.

Mirrors ``scripts/td3/evaluate.py`` for the ``GoalEnv``-dict tasks: builds
the env from the air_hockey config (``return_goal_obs: true``), flattens each
observation to ``[observation, desired_goal]`` exactly as
``scripts/td3/helper/td3_her.GoalEnvVector`` does during training, rolls
``n_eps`` episodes with the deterministic actor and writes

* ``eval_<i>.gif`` — the first ``n_gifs`` episodes rendered with the goal
  circle (``AirHockeyRenderer`` draws ``goal_pos`` / ``goal_radius``), and
* ``goal_eval.json`` — success rate, mean return, mean episode length,
  per-episode records, and the end-of-episode reasons.

Runnable as a subprocess for the trainer's async per-checkpoint eval::

    python -m scripts.td3.helper.her_eval --checkpoint-dir <run>/checkpoint_25000 \
        [--n-eps 20] [--n-gifs 1] [--eval-call-index 3]

``--eval-call-index`` shifts the env seed so successive checkpoints see
different start states / goals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import cv2
import imageio
import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.td3.eval_utils import (
    augment_policy_observation,
    build_policy_env_view,
    load_policy_for_evaluation,
)
from scripts.td3.helper.td3_her import GoalEnvVector


def _frame(renderer: AirHockeyRenderer, reward: float, cum_reward: float, goal_text: str) -> np.ndarray:
    frame = renderer.get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    aspect_ratio = frame.shape[1] / frame.shape[0]
    frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
    font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1
    cv2.putText(frame, f"r {reward:.1f} R {cum_reward:.1f}", (4, 14), font, scale, color, thick)
    if goal_text:
        cv2.putText(frame, goal_text, (4, 28), font, scale, color, thick)
    return frame


def evaluate_goal_policy(
    model_path: str,
    save_dir: str,
    air_hockey_params: Dict[str, Any],
    *,
    n_eps: int = 20,
    n_gifs: int = 1,
    action_scale: float = 1.0,
    agent_hidden_layer_size: int = 64,
    agent_num_hidden_layers: int = 2,
    use_last_action_in_policy_state: bool = True,
    seed: Optional[int] = None,
    fps: int = 20,
) -> Dict[str, Any]:
    params = dict(air_hockey_params)
    params["return_goal_obs"] = True
    if seed is not None:
        params["seed"] = int(seed)
    envs = GoalEnvVector(lambda: AirHockeyEnv(params))
    env = envs.env
    action_dim = int(np.prod(envs.single_action_space.shape))
    policy_env_view = build_policy_env_view(envs, use_last_action_in_policy_state)
    model = load_policy_for_evaluation(
        model_path=model_path,
        policy_env_view=policy_env_view,
        action_scale=action_scale,
        agent_hidden_layer_size=agent_hidden_layer_size,
        agent_num_hidden_layers=agent_num_hidden_layers,
    )
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False) if n_gifs > 0 else None
    os.makedirs(save_dir, exist_ok=True)

    episodes = []
    end_reasons: Dict[str, int] = {}
    gif_frames = []
    gifs_written = 0
    for ep in range(int(n_eps)):
        obs, _ = envs.reset()
        goal = env.get_desired_goal()
        goal_text = "g " + " ".join(f"{g:+.2f}" for g in goal)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        last_action = torch.zeros((1, action_dim), dtype=torch.float32)
        record = gifs_written < n_gifs
        cum_reward, steps, reward, done, contacts = 0.0, 0, 0.0, False, 0
        info: Dict[str, Any] = {}
        while not done:
            if record:
                gif_frames.append(_frame(renderer, reward, cum_reward, goal_text))
            policy_obs = augment_policy_observation(obs_tensor, last_action, use_last_action_in_policy_state)
            with torch.no_grad():
                action = model(policy_obs).numpy().reshape(1, -1)
            obs, r, term, trunc, infos = envs.step(action)
            reward = float(r[0])
            cum_reward += reward
            steps += 1
            contacts += int(infos["paddle_puck_collision_count"][0])
            done = bool(term[0] or trunc[0])
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            last_action = torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)
            if done:
                info = infos["final_info"][0]
                last_action.zero_()
        reason = info.get("episode_end_reason") or "unknown"
        end_reasons[reason] = end_reasons.get(reason, 0) + 1
        episodes.append(
            {
                "return": cum_reward,
                "success": int(bool(info.get("success", False))),
                "length": steps,
                "contacts": contacts,
                "end_reason": reason,
                "goal": [float(g) for g in goal],
            }
        )
        if record:
            gifs_written += 1
            if gifs_written == n_gifs or ep == n_eps - 1:
                path = os.path.join(save_dir, "eval_0.gif")
                imageio.mimsave(path, gif_frames, format="GIF", loop=0, duration=int(1000 / fps))
                gif_frames = []
    envs.close()

    summary = {
        "n_eps": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])) if episodes else float("nan"),
        "mean_return": float(np.mean([e["return"] for e in episodes])) if episodes else float("nan"),
        "mean_length": float(np.mean([e["length"] for e in episodes])) if episodes else float("nan"),
        "mean_contacts": float(np.mean([e["contacts"] for e in episodes])) if episodes else float("nan"),
        "end_reasons": end_reasons,
        "episodes": episodes,
    }
    with open(os.path.join(save_dir, "goal_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[her_eval] {os.path.basename(save_dir)}: success {summary['success_rate']:.3f} "
        f"return {summary['mean_return']:.2f} len {summary['mean_length']:.1f} "
        f"contacts {summary['mean_contacts']:.2f} ends {end_reasons} (n={len(episodes)})",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--n-eps", type=int, default=None)
    parser.add_argument("--n-gifs", type=int, default=1)
    parser.add_argument("--eval-call-index", type=int, default=1)
    cli = parser.parse_args()
    ckpt_dir = os.path.abspath(cli.checkpoint_dir)
    with open(os.path.join(ckpt_dir, "args.yaml"), "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_eps = int(cli.n_eps if cli.n_eps is not None else args.get("eval_n_eps", 20))
    evaluate_goal_policy(
        os.path.join(ckpt_dir, "model.pth"),
        ckpt_dir,
        config["air_hockey"],
        n_eps=n_eps,
        n_gifs=int(cli.n_gifs),
        action_scale=1.0,
        agent_hidden_layer_size=int(args["agent_hidden_layer_size"]),
        agent_num_hidden_layers=int(args["agent_num_hidden_layers"]),
        use_last_action_in_policy_state=bool(args["use_last_action_in_policy_state"]),
        seed=int(args.get("seed", 0)) * 100000 + 7919 * int(cli.eval_call_index),
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
