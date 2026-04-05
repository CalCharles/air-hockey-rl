"""
Phase 2 (position-based): collision adaptation loop using velocity estimation.

Like run_adaptation.py but estimates pre- and post-collision speeds from
noisy position trajectories (env.simulator.puck_history) rather than reading
privileged Box2D body velocities.  This simulates the real-world case where
only camera-tracked puck positions are available.

The same multiplicative scale update rule from adapt.py is used, so convergence
behaviour should be directionally identical to the privileged baseline.

Usage:
    python scripts/collision_adaptation/run_adaptation_position_based.py \
        --config scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params_heavy.yaml \
        --model-path runs/td3_training/some_run/model.pth \
        --oracle-paddle-scales 0.7 1.0 1.2 \
        --n-iterations 20 \
        --n-episodes 50 \
        --lr 0.2 \
        --output-dir runs/collision_adaptation_position_based

Convergence check: with oracle scales = [1.0, 1.0, 1.0], learner scales should
stay near 1.0.  With oracle scales = [0.7, 1.0, 1.2], learner scales should
drift toward matching the oracle speed ratios over ~10 iterations.

Differences from run_adaptation.py:
  - Uses rollout_position_based.rollout_episodes_position_based instead of rollout.rollout_episodes
  - Extra args: --window-frames, --min-snr, --timestep
  - Default output dir: runs/collision_adaptation_position_based
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey import AirHockeyEnv
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.collision_adaptation.rollout_position_based import rollout_episodes_position_based
from scripts.collision_adaptation.adapt import compute_scale_updates, max_abs_ratio_minus_one


# ---------------------------------------------------------------------------
# Helpers (identical to run_adaptation.py)
# ---------------------------------------------------------------------------

def _build_env(config_path: str, paddle_scales: list[float]) -> object:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["simulator_params"]["puck_density"] = 3000
    cfg["simulator_params"]["paddle_density"] = 3000
    env = AirHockeyEnv(cfg)
    env.simulator.set_collision_scales(
        wall_scales=[1.0, 1.0, 1.0],
        paddle_scales=paddle_scales,
    )
    return env


def _load_actor(model_path: str, device: str, obs_dim: int = 32, act_dim: int = 2) -> DeterministicAgent:
    """Load a DeterministicAgent from a TD3 checkpoint."""

    class _EnvView:
        class single_observation_space:
            shape = (obs_dim,)

        class single_action_space:
            shape = (act_dim,)

    actor = DeterministicAgent(
        _EnvView(),
        action_scale=1.0,
        hidden_layer_size=64,
        num_hidden_layers=5,
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # model.pth IS the actor state dict directly (not nested under "actor" key)
    actor.load_state_dict(ckpt)
    actor.eval()
    return actor.to(device)


def _json_default(obj):
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Position-based collision adaptation loop (Phase 2, no privileged velocities)."
    )
    parser.add_argument("--config", required=True, help="Path to YAML sim config.")
    parser.add_argument("--model-path", required=True, help="Path to TD3 checkpoint (.pth).")
    parser.add_argument(
        "--oracle-paddle-scales",
        nargs=3,
        type=float,
        default=[0.7, 1.0, 1.2],
        metavar=("LOW", "MID", "HIGH"),
        help="Fixed oracle paddle restitution scales [low, mid, high].",
    )
    parser.add_argument("--n-iterations", type=int, default=20, help="Adaptation iterations.")
    parser.add_argument("--n-episodes", type=int, default=50, help="Episodes per rollout per sim.")
    parser.add_argument("--lr", type=float, default=0.2, help="Scale update learning rate.")
    parser.add_argument("--min-count", type=int, default=3, help="Min collisions to update a tier.")
    parser.add_argument("--min-scale", type=float, default=0.3, help="Lower clamp for scales.")
    parser.add_argument("--max-scale", type=float, default=3.0, help="Upper clamp for scales.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--output-dir",
        default="runs/collision_adaptation_position_based",
        help="Output directory.",
    )
    parser.add_argument(
        "--use-last-action",
        action="store_true",
        default=True,
        help="Append last action to obs before actor (default: True).",
    )
    # Position-based specific args
    parser.add_argument(
        "--window-frames",
        type=int,
        default=10,
        help="Frames used for each velocity regression window (pre and post collision).",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=10.0,
        help="Minimum signal-to-noise ratio to accept a velocity estimate.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.05,
        help="Seconds per env step (default 0.05 = 20 Hz).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    oracle_scales = list(args.oracle_paddle_scales)
    learner_scales = [1.0, 1.0, 1.0]

    print(f"Oracle paddle scales : {oracle_scales}")
    print(f"Learner initial scales: {learner_scales}")
    print(f"Config  : {args.config}")
    print(f"Model   : {args.model_path}")
    print(f"window_frames={args.window_frames}  min_snr={args.min_snr}  timestep={args.timestep}")

    oracle_env = _build_env(args.config, oracle_scales)
    learner_env = _build_env(args.config, learner_scales)
    actor = _load_actor(args.model_path, device)

    rollout_kwargs = dict(
        use_last_action=args.use_last_action,
        window_frames=args.window_frames,
        min_snr=args.min_snr,
        timestep=args.timestep,
    )

    history = []

    for iteration in range(args.n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")
        print(f"  Learner scales: {[f'{s:.4f}' for s in learner_scales]}")

        oracle_stats = rollout_episodes_position_based(
            oracle_env, actor, args.n_episodes, device, **rollout_kwargs
        )
        learner_stats = rollout_episodes_position_based(
            learner_env, actor, args.n_episodes, device, **rollout_kwargs
        )

        new_scales, update_info = compute_scale_updates(
            oracle_stats,
            learner_stats,
            learner_scales,
            lr=args.lr,
            min_count=args.min_count,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )

        convergence = max_abs_ratio_minus_one(update_info)

        print(f"  max|ratio-1|: {convergence:.4f}")
        for tier, info in update_info.items():
            if info.get("skipped"):
                print(f"    {tier}: SKIPPED ({info['reason']})")
            else:
                print(
                    f"    {tier}: oracle_out={info['oracle_mean_out']:.3f}"
                    f"  learner_out={info['learner_mean_out']:.3f}"
                    f"  ratio={info['ratio']:.4f}"
                    f"  scale {info['scale_before']:.4f} → {info['scale_after']:.4f}"
                )

        scales_before_update = list(learner_scales)
        learner_scales = new_scales
        learner_env.simulator.set_collision_scales(
            wall_scales=[1.0, 1.0, 1.0],
            paddle_scales=learner_scales,
        )

        entry = {
            "iteration": iteration + 1,
            "learner_scales_before": [float(s) for s in scales_before_update],
            "learner_scales_after": [float(s) for s in learner_scales],
            "oracle_stats": oracle_stats,
            "learner_stats": learner_stats,
            "update_info": update_info,
            "convergence_max_ratio_minus_one": convergence,
        }
        history.append(entry)

    history_path = os.path.join(args.output_dir, "adaptation_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=_json_default)

    print(f"\nFinal learner scales: {[f'{s:.4f}' for s in learner_scales]}")
    print(f"Saved adaptation history → {history_path}")


if __name__ == "__main__":
    main()
