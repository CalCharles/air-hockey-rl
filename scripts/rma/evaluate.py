"""Evaluate an RMA ActorCritic checkpoint (phase 1 privileged or phase 2 adaptation).

Phase 1 (.pth): uses μ(priv_info) — requires privileged props from the env.
Phase 2 (.ckpt): uses φ(proprio_hist) — deploy-compatible, no privileged props needed
                 for the policy path (priv still used only if you force phase1 mode).

Launch:
  python -m scripts.rma.evaluate \\
    --checkpoint runs/rma/.../stage2_nn/model_best.ckpt \\
    --config configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml \\
    --save-dir runs/rma/eval
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio
import numpy as np
import torch
import tqdm
import yaml

from airhockey.renderers import AirHockeyRenderer
from scripts.rma.env_wrapper import RMAVecEnv
from scripts.rma.models import ActorCritic
from scripts.rma.running_mean_std import RunningMeanStd


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_actor_critic(
    obs_dim: int,
    action_dim: int,
    priv_info_dim: int,
    actor_units: List[int],
    priv_mlp_units: List[int],
    proprio_adapt: bool,
    proprio_hist_input_dim: int,
) -> ActorCritic:
    return ActorCritic(
        {
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "actor_units": list(actor_units),
            "priv_mlp_units": list(priv_mlp_units),
            "priv_info": True,
            "proprio_adapt": bool(proprio_adapt),
            "priv_info_dim": int(priv_info_dim),
            "proprio_hist_input_dim": int(proprio_hist_input_dim),
        }
    )


def load_rma_checkpoint(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    priv_info_dim: int,
    actor_units: List[int],
    priv_mlp_units: List[int],
    prop_hist_len: int,
    proprio_hist_input_dim: int,
    device: str,
    force_phase: Optional[str] = None,
):
    """Load ActorCritic + normalizers. Returns (model, obs_rms, hist_rms, stage)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    is_phase2 = checkpoint_path.endswith(".ckpt") or "sa_mean_std" in ckpt
    if force_phase == "phase1":
        is_phase2 = False
    elif force_phase == "phase2":
        is_phase2 = True

    model = build_actor_critic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        priv_info_dim=priv_info_dim,
        actor_units=actor_units,
        priv_mlp_units=priv_mlp_units,
        proprio_adapt=is_phase2,
        proprio_hist_input_dim=proprio_hist_input_dim,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    obs_rms = RunningMeanStd((obs_dim,)).to(device)
    if "running_mean_std" in ckpt:
        obs_rms.load_state_dict(ckpt["running_mean_std"])
    obs_rms.eval()

    hist_rms = None
    if is_phase2:
        hist_rms = RunningMeanStd((prop_hist_len, proprio_hist_input_dim)).to(device)
        if "sa_mean_std" in ckpt:
            hist_rms.load_state_dict(ckpt["sa_mean_std"])
        hist_rms.eval()

    stage = "phase2" if is_phase2 else "phase1"
    return model, obs_rms, hist_rms, stage


@torch.no_grad()
def rollout_returns(
    air_hockey_params: Dict[str, Any],
    checkpoint_path: str,
    n_eps: int,
    actor_units: List[int],
    priv_mlp_units: List[int],
    prop_hist_len: int = 30,
    device: str = "cpu",
    max_timesteps: int = 200,
    force_phase: Optional[str] = None,
) -> Dict[str, Any]:
    env = RMAVecEnv(
        air_hockey_params=air_hockey_params,
        num_envs=1,
        device=device,
        prop_hist_len=prop_hist_len,
        seed=int(air_hockey_params.get("seed", 0)),
    )
    # Cap episode length on the underlying env.
    env._venv.envs[0].max_timesteps = max_timesteps

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_dim
    model, obs_rms, hist_rms, stage = load_rma_checkpoint(
        checkpoint_path=checkpoint_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        priv_info_dim=env.priv_info_dim,
        actor_units=actor_units,
        priv_mlp_units=priv_mlp_units,
        prop_hist_len=prop_hist_len,
        proprio_hist_input_dim=env.proprio_hist_entry_dim,
        device=device,
        force_phase=force_phase,
    )

    returns: List[float] = []
    lengths: List[int] = []
    successes: List[int] = []

    for _ in range(n_eps):
        obs_dict = env.reset()
        done = False
        cum = 0.0
        steps = 0
        while not done:
            obs_n = obs_rms(obs_dict["obs"])
            if stage == "phase2":
                input_dict = {
                    "obs": obs_n,
                    "proprio_hist": hist_rms(obs_dict["proprio_hist"]),
                }
            else:
                input_dict = {
                    "obs": obs_n,
                    "priv_info": obs_dict["priv_info"],
                }
            mu = model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, rew, dones, _info = env.step(mu)
            cum += float(rew[0].item())
            steps += 1
            done = bool(dones[0].item())
        returns.append(cum)
        lengths.append(steps)
        successes.append(int(steps >= max_timesteps))

    env.close()
    return {
        "stage": stage,
        "returns": returns,
        "episode_lengths": lengths,
        "successes": successes,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_success_rate": float(np.mean(successes)) if successes else float("nan"),
        "mean_episode_length": float(np.mean(lengths)) if lengths else float("nan"),
    }


def save_eval_gifs(
    air_hockey_params: Dict[str, Any],
    checkpoint_path: str,
    save_dir: str,
    n_eps: int,
    n_gifs: int,
    actor_units: List[int],
    priv_mlp_units: List[int],
    prop_hist_len: int = 30,
    device: str = "cpu",
    max_timesteps: int = 200,
    force_phase: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    env = RMAVecEnv(
        air_hockey_params=air_hockey_params,
        num_envs=1,
        device=device,
        prop_hist_len=prop_hist_len,
        seed=int(air_hockey_params.get("seed", 0)),
    )
    raw_env = env._venv.envs[0]
    raw_env.max_timesteps = max_timesteps
    renderer = AirHockeyRenderer(raw_env, show_target_position=True, show_acceleration_arrow=False)

    obs_dim = int(np.prod(env.observation_space.shape))
    model, obs_rms, hist_rms, stage = load_rma_checkpoint(
        checkpoint_path=checkpoint_path,
        obs_dim=obs_dim,
        action_dim=env.action_dim,
        priv_info_dim=env.priv_info_dim,
        actor_units=actor_units,
        priv_mlp_units=priv_mlp_units,
        prop_hist_len=prop_hist_len,
        proprio_hist_input_dim=env.proprio_hist_entry_dim,
        device=device,
        force_phase=force_phase,
    )

    for gif_idx in range(n_gifs):
        frames = []
        for _ in tqdm.tqdm(range(n_eps), desc=f"gif {gif_idx}"):
            obs_dict = env.reset()
            done = False
            cum = 0.0
            rew_val = 0.0
            while not done:
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aspect = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect)))
                cv2.putText(
                    frame, f"Reward: {rew_val:.2f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                )
                cv2.putText(
                    frame, f"Return: {cum:.2f}", (frame.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                )
                frames.append(frame)

                obs_n = obs_rms(obs_dict["obs"])
                if stage == "phase2":
                    input_dict = {
                        "obs": obs_n,
                        "proprio_hist": hist_rms(obs_dict["proprio_hist"]),
                    }
                else:
                    input_dict = {
                        "obs": obs_n,
                        "priv_info": obs_dict["priv_info"],
                    }
                mu = model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
                obs_dict, rew, dones, _ = env.step(mu)
                rew_val = float(rew[0].item())
                cum += rew_val
                done = bool(dones[0].item())

        path = os.path.join(save_dir, f"eval_{gif_idx}.gif")
        imageio.mimsave(path, frames, format="GIF", loop=0, duration=int(1000 / 20))
        print(f"[rma.evaluate] wrote {path}")

    env.close()
    return stage


def evaluate_agent(
    checkpoint_path: str,
    save_dir: str,
    air_hockey_params: Dict[str, Any],
    actor_units=(512, 256, 128),
    priv_mlp_units=(128, 64, 8),
    prop_hist_len: int = 30,
    n_eps: int = 4,
    n_gifs: int = 1,
    device: str = "cpu",
    force_phase: Optional[str] = None,
    write_metrics: bool = True,
):
    """Drop-in style evaluator used by rma_training_dr checkpoint hooks."""
    os.makedirs(save_dir, exist_ok=True)
    stage = force_phase
    if n_gifs > 0:
        stage = save_eval_gifs(
            air_hockey_params=air_hockey_params,
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            n_eps=n_eps,
            n_gifs=n_gifs,
            actor_units=list(actor_units),
            priv_mlp_units=list(priv_mlp_units),
            prop_hist_len=prop_hist_len,
            device=device,
            force_phase=force_phase,
        )
    stats = rollout_returns(
        air_hockey_params=air_hockey_params,
        checkpoint_path=checkpoint_path,
        n_eps=n_eps,
        actor_units=list(actor_units),
        priv_mlp_units=list(priv_mlp_units),
        prop_hist_len=prop_hist_len,
        device=device,
        force_phase=force_phase or stage,
    )
    if write_metrics:
        with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate an RMA checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Sim YAML with air_hockey block.")
    parser.add_argument("--save-dir", default="runs/rma/eval")
    parser.add_argument("--n-eps", type=int, default=4)
    parser.add_argument("--n-gifs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prop-hist-len", type=int, default=30)
    parser.add_argument("--actor-units", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--priv-mlp-units", type=int, nargs="+", default=[128, 64, 8])
    parser.add_argument("--force-phase", choices=["phase1", "phase2"], default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    air_hockey_params = copy.deepcopy(cfg["air_hockey"])
    air_hockey_params["domain_random"] = False

    stats = evaluate_agent(
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        air_hockey_params=air_hockey_params,
        actor_units=args.actor_units,
        priv_mlp_units=args.priv_mlp_units,
        prop_hist_len=args.prop_hist_len,
        n_eps=args.n_eps,
        n_gifs=args.n_gifs,
        device=args.device,
        force_phase=args.force_phase,
    )
    print(
        f"[rma.evaluate] stage={stats['stage']} mean_return={stats['mean_return']:.2f} "
        f"mean_success={stats['mean_success_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
