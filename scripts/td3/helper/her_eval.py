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

DR configs (``eval_param_seed`` set in ``args.yaml``) get the same fixed
multi-env evaluation as ``td3_training_dr``: ``eval_n_envs`` dynamics
overlays sampled once from the sim config's ``random_variable_ranges`` with
``eval_param_seed``, ``eval_eps_per_env`` episodes each, aggregated into
``multi_env_eval.json`` (same schema, so ``run_experiments`` summarises it)
next to a ``goal_eval.json`` for env 0.
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
    verbose: bool = True,
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
    if verbose:
        print(
            f"[her_eval] {os.path.basename(save_dir)}: success {summary['success_rate']:.3f} "
            f"return {summary['mean_return']:.2f} len {summary['mean_length']:.1f} "
            f"contacts {summary['mean_contacts']:.2f} ends {end_reasons} (n={len(episodes)})",
            flush=True,
        )
    return summary


def evaluate_goal_policy_multi_env(
    model_path: str,
    save_dir: str,
    air_hockey_params: Dict[str, Any],
    *,
    eval_param_seed: int,
    eval_n_envs: int,
    eval_eps_per_env: int,
    n_gifs: int = 1,
    seed: Optional[int] = None,
    log_parent_dir: Optional[str] = None,
    **policy_kwargs: Any,
) -> Dict[str, Any]:
    """Fixed multi-env evaluation for DR runs (mirrors ``td3_training_dr``).

    Samples ``eval_n_envs`` dynamics overlays once with ``eval_param_seed``
    (written to ``<log_parent_dir>/eval_envs.json`` on first use), rolls
    ``eval_eps_per_env`` goal-conditioned episodes on each with per-reset
    randomization off, writes ``multi_env_eval.json`` (``aggregate`` +
    ``per_env``) and returns the aggregate.  Env 0 also gets the GIF and
    ``goal_eval.json``.
    """
    from scripts.td3.td3_training_dr import (
        _apply_overrides_to_air_hockey_params,
        _sample_eval_env_overrides,
    )

    random_variables = list(air_hockey_params.get("random_variables", []))
    random_variable_ranges = dict(air_hockey_params.get("random_variable_ranges", {}))
    if not random_variables or not random_variable_ranges:
        raise ValueError("multi-env eval needs `random_variables` / `random_variable_ranges` in the air_hockey config")
    overrides = _sample_eval_env_overrides(
        seed=int(eval_param_seed), n_envs=int(eval_n_envs),
        random_variable_ranges=random_variable_ranges, random_variables=random_variables,
    )
    if log_parent_dir is not None:
        path = os.path.join(log_parent_dir, "eval_envs.json")
        if not os.path.exists(path):
            os.makedirs(log_parent_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"eval_param_seed": int(eval_param_seed), "n_envs": int(eval_n_envs),
                           "eps_per_env": int(eval_eps_per_env), "random_variables": random_variables,
                           "random_variable_ranges": {k: list(v) for k, v in random_variable_ranges.items()},
                           "overrides": overrides}, f, indent=2)
    per_env = []
    for env_idx, override in enumerate(overrides):
        cfg = _apply_overrides_to_air_hockey_params(air_hockey_params, override)
        env_seed = None if seed is None else int(seed) + 1000 * env_idx
        summary = evaluate_goal_policy(
            model_path, save_dir if env_idx == 0 else os.path.join(save_dir, f"env{env_idx}_tmp"), cfg,
            n_eps=int(eval_eps_per_env), n_gifs=int(n_gifs) if env_idx == 0 else 0, seed=env_seed,
            verbose=(env_idx == 0), **policy_kwargs,
        )
        if env_idx > 0:
            # Only env 0 keeps a goal_eval.json / GIF; drop the scratch dir.
            import shutil
            shutil.rmtree(os.path.join(save_dir, f"env{env_idx}_tmp"), ignore_errors=True)
        per_env.append({
            "env_idx": env_idx, "override": override,
            "returns": [e["return"] for e in summary["episodes"]],
            "successes": [e["success"] for e in summary["episodes"]],
            "episode_lengths": [e["length"] for e in summary["episodes"]],
            "mean_return": summary["mean_return"], "mean_success_rate": summary["success_rate"],
            "mean_episode_length": summary["mean_length"], "mean_contacts": summary["mean_contacts"],
        })
    aggregate = {
        "n_envs_used": len(per_env), "eps_per_env": int(eval_eps_per_env),
        "mean_return_across_envs": float(np.mean([r["mean_return"] for r in per_env])),
        "mean_success_across_envs": float(np.mean([r["mean_success_rate"] for r in per_env])),
        "mean_ep_length_across_envs": float(np.mean([r["mean_episode_length"] for r in per_env])),
        "per_env_mean_return": [r["mean_return"] for r in per_env],
        "per_env_mean_success": [r["mean_success_rate"] for r in per_env],
    }
    with open(os.path.join(save_dir, "multi_env_eval.json"), "w") as f:
        json.dump({"aggregate": aggregate, "per_env": per_env}, f, indent=2)
    print(
        f"[her_eval] {os.path.basename(save_dir)} multi-env (n_envs={len(per_env)}, eps/env={eval_eps_per_env}): "
        f"mean_success={aggregate['mean_success_across_envs']:.3f} mean_return={aggregate['mean_return_across_envs']:.2f} "
        f"per_env_success={[round(x, 2) for x in aggregate['per_env_mean_success']]}",
        flush=True,
    )
    return aggregate


def evaluate_checkpoint(ckpt_dir: str, *, n_eps: Optional[int], n_gifs: int, eval_call_index: int,
                        log_parent_dir: Optional[str] = None, final: bool = False) -> Dict[str, Any]:
    """Evaluate ``<ckpt_dir>/model.pth`` with the settings in its ``args.yaml`` /
    ``config.yaml``: plain goal eval, or the fixed multi-env eval when
    ``eval_param_seed`` is set."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    with open(os.path.join(ckpt_dir, "args.yaml"), "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    policy_kwargs = dict(
        action_scale=1.0,
        agent_hidden_layer_size=int(args["agent_hidden_layer_size"]),
        agent_num_hidden_layers=int(args["agent_num_hidden_layers"]),
        use_last_action_in_policy_state=bool(args["use_last_action_in_policy_state"]),
    )
    seed = int(args.get("seed", 0)) * 100000 + (424242 if final else 7919 * int(eval_call_index))
    model_path = os.path.join(ckpt_dir, "model.pth")
    if args.get("eval_param_seed") is not None:
        eps_per_env = int(args.get("eval_eps_per_env", 4))
        if final:
            n_final = int(n_eps if n_eps is not None else args.get("eval_n_eps_final", 100))
            eps_per_env = max(eps_per_env, n_final // max(int(args.get("eval_n_envs", 1)), 1))
        return evaluate_goal_policy_multi_env(
            model_path, ckpt_dir, config["air_hockey"],
            eval_param_seed=int(args["eval_param_seed"]), eval_n_envs=int(args.get("eval_n_envs", 1)),
            eval_eps_per_env=eps_per_env, n_gifs=n_gifs, seed=seed,
            log_parent_dir=log_parent_dir or os.path.dirname(ckpt_dir), **policy_kwargs,
        )
    default_n = args.get("eval_n_eps_final", 100) if final else args.get("eval_n_eps", 20)
    return evaluate_goal_policy(
        model_path, ckpt_dir, config["air_hockey"],
        n_eps=int(n_eps if n_eps is not None else default_n), n_gifs=n_gifs, seed=seed, **policy_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--n-eps", type=int, default=None)
    parser.add_argument("--n-gifs", type=int, default=1)
    parser.add_argument("--eval-call-index", type=int, default=1)
    parser.add_argument("--final", action="store_true", help="use eval_n_eps_final and the final-eval seed")
    cli = parser.parse_args()
    evaluate_checkpoint(cli.checkpoint_dir, n_eps=cli.n_eps, n_gifs=int(cli.n_gifs),
                        eval_call_index=int(cli.eval_call_index), final=bool(cli.final))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
