"""Offline state-estimation diagnostic on saved noisy juggling trajectories.

No environment rollouts or actor/critic training. Fits fresh supervised MLPs to
ask whether deployable control-step history predicts physical puck state better
than the canonical observation alone. Group split by episode seed across actors.
Targets are physical state BEFORE the action whose observation is the input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_data(roots, history_len, holdout_seed, dt=.05):
    episodes = []
    for root in roots:
        for path in sorted(root.glob("training_seed*.npz")):
            seed = int(path.stem.split("seed")[-1])
            with np.load(path) as data:
                obs, action, truth = data["observation"], data["action"], data["true_puck_xy"]
                if len(obs) < 3:
                    continue
                prev_action = np.concatenate([np.zeros((1, 2)), action[:-1]])
                short = np.concatenate([obs, prev_action], axis=1)
                # Past observation/action pairs only; never include action t.
                frame_action = np.concatenate([obs[:, 12:15], obs[:, 27:30], action], axis=1)
                indices = np.maximum(np.arange(len(obs))[:, None] - np.arange(history_len, 0, -1), 0)
                window = frame_action[indices].copy()
                # Before the first real action, duplicate initial state, zero action.
                padded = np.arange(len(obs))[:, None] - np.arange(history_len, 0, -1) < 0
                window[padded, -2:] = 0.
                long = np.concatenate([short, window.reshape(len(obs), -1)], axis=1)
                # truth[t] is post-action t. Inputs obs[t] precede that action.
                target = np.concatenate([truth[1:-1], (truth[1:-1] - truth[:-2]) / dt], axis=1)
                episodes.append(dict(seed=seed, source=str(path), heldout=seed >= holdout_seed,
                                     short=short[2:].astype(np.float32), long=long[2:].astype(np.float32),
                                     target=target.astype(np.float32)))
    if not episodes or not any(e["heldout"] for e in episodes) or not any(not e["heldout"] for e in episodes):
        raise ValueError("Need both training and held-out seeds")
    return episodes


def run(args):
    if args.history_len < 1 or args.dt <= 0 or args.updates < 1:
        raise ValueError("History length, timestep and update count must be positive")
    torch.set_num_threads(1)
    episodes = load_data(args.roots, args.history_len, args.holdout_seed, args.dt)
    target = torch.from_numpy(np.concatenate([e["target"] for e in episodes if not e["heldout"]]))
    target_mean, target_std = target.mean(0), target.std(0).clamp_min(.01)
    target = (target - target_mean) / target_std
    result = dict(created=datetime.now(timezone.utc).isoformat(), args=vars(args).copy(), models={})
    result["args"]["roots"] = [str(p) for p in args.roots]
    result["args"]["out"] = str(args.out)
    for name, feature, width in (("short64", "short", 64), ("short128", "short", 128), ("long64", "long", 64)):
        torch.manual_seed(args.seed)
        inputs = torch.from_numpy(np.concatenate([e[feature] for e in episodes if not e["heldout"]]))
        mean, std = inputs.mean(0), inputs.std(0).clamp_min(.1)
        inputs = (inputs - mean) / std
        model = nn.Sequential(nn.Linear(inputs.shape[1], width), nn.ELU(), nn.Linear(width, width), nn.ELU(), nn.Linear(width, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(args.updates):
            batch = torch.randint(len(inputs), (256,))
            loss = (model(inputs[batch]) - target[batch]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        rows = []
        with torch.no_grad():
            for e in episodes:
                if not e["heldout"]:
                    continue
                pred = model((torch.from_numpy(e[feature]) - mean) / std) * target_std + target_mean
                error = (pred.numpy() - e["target"]) ** 2
                rows.append(dict(seed=e["seed"], source=e["source"], steps=len(error),
                                 position_rmse_cm=float(100 * np.sqrt(error[:, :2].sum(1).mean())),
                                 prior_step_velocity_rmse_m_s=float(np.sqrt(error[:, 2:].sum(1).mean()))))
        result["models"][name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                      train_steps=len(inputs), heldout_episodes=rows,
                                      mean_episode_position_rmse_cm=float(np.mean([r["position_rmse_cm"] for r in rows])),
                                      mean_episode_velocity_rmse_m_s=float(np.mean([r["prior_step_velocity_rmse_m_s"] for r in rows])))
        print(name, result["models"][name]["mean_episode_position_rmse_cm"],
              result["models"][name]["mean_episode_velocity_rmse_m_s"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--holdout-seed", type=int, default=300024)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=.05, help="Control period of saved trajectories (seconds)")
    run(parser.parse_args())
