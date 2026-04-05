"""
Phase 2 (oracle-scenario replay): collision adaptation via scenario matching.

Collects collision scenarios from oracle rollouts (position-based velocity
estimation, simulating camera data), then replays each exact scenario in the
learner sim (privileged velocity read) for a direct apples-to-apples comparison.

This avoids the distribution-shift failure of run_adaptation_position_based.py:
because both oracle and learner evaluate the same physical inputs, non-uniform
oracle scales no longer contaminate tier statistics.

Usage:
    python scripts/collision_adaptation/run_adaptation_replay.py \
        --config scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params_heavy.yaml \
        --model-path runs/td3/final/task_only/checkpoint_350000/model.pth \
        --oracle-paddle-scales 0.7 1.0 1.2 \
        --n-iterations 30 \
        --n-episodes 100 \
        --lr 0.15 \
        --output-dir runs/collision_adaptation_replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey import AirHockeyEnv
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.collision_adaptation.rollout_replay import collect_oracle_scenarios, replay_scenarios
from scripts.collision_adaptation.adapt import compute_scale_updates, max_abs_ratio_minus_one


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
    actor.load_state_dict(ckpt)
    actor.eval()
    return actor.to(device)


def _json_default(obj):
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle-scenario replay collision adaptation."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--oracle-paddle-scales", nargs=3, type=float, default=[0.7, 1.0, 1.2],
        metavar=("LOW", "MID", "HIGH"),
    )
    p.add_argument("--n-iterations", type=int, default=30)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--min-scale", type=float, default=0.3)
    p.add_argument("--max-scale", type=float, default=3.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="runs/collision_adaptation_replay")
    p.add_argument("--use-last-action", action="store_true", default=True)
    p.add_argument("--window-frames", type=int, default=10)
    p.add_argument("--min-snr", type=float, default=8.0)
    p.add_argument("--max-replay-steps", type=int, default=10)
    p.add_argument("--min-approach-speed", type=float, default=0.1,
                   help="Minimum estimated relative approach speed (m/s) to keep a scenario.")
    p.add_argument("--timestep", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    oracle_scales = list(args.oracle_paddle_scales)
    learner_scales = [1.0, 1.0, 1.0]

    print(f"Oracle paddle scales : {oracle_scales}")
    print(f"Learner initial scales: {learner_scales}")
    print(f"Config  : {args.config}")
    print(f"Model   : {args.model_path}")
    print(f"window_frames={args.window_frames}  min_snr={args.min_snr}  "
          f"min_approach_speed={args.min_approach_speed}  max_replay_steps={args.max_replay_steps}")

    oracle_env = _build_env(args.config, oracle_scales)
    learner_env = _build_env(args.config, learner_scales)
    actor = _load_actor(args.model_path, args.device)

    collect_kwargs = dict(
        use_last_action=args.use_last_action,
        window_frames=args.window_frames,
        min_snr=args.min_snr,
        timestep=args.timestep,
        min_approach_speed=args.min_approach_speed,
    )

    history = []

    for iteration in range(args.n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.n_iterations} ===")
        print(f"  Learner scales: {[f'{s:.4f}' for s in learner_scales]}")

        scenarios = collect_oracle_scenarios(
            oracle_env, actor, args.n_episodes, args.device, **collect_kwargs
        )

        tier_counts = {t: sum(1 for s in scenarios if s["tier"] == t) for t in ("low", "mid", "high")}
        total_sc = len(scenarios)
        print(f"  Scenarios collected: {total_sc}  "
              f"(low={tier_counts['low']} mid={tier_counts['mid']} high={tier_counts['high']})")

        oracle_stats, learner_stats = replay_scenarios(
            learner_env, scenarios, max_replay_steps=args.max_replay_steps
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

        scales_before = list(learner_scales)
        learner_scales = new_scales
        learner_env.simulator.set_collision_scales(
            wall_scales=[1.0, 1.0, 1.0],
            paddle_scales=learner_scales,
        )

        history.append({
            "iteration": iteration + 1,
            "scenarios_collected": total_sc,
            "tier_counts": tier_counts,
            "learner_scales_before": [float(s) for s in scales_before],
            "learner_scales_after": [float(s) for s in learner_scales],
            "oracle_stats": oracle_stats,
            "learner_stats": learner_stats,
            "update_info": update_info,
            "convergence_max_ratio_minus_one": convergence,
        })

    history_path = os.path.join(args.output_dir, "adaptation_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=_json_default)

    print(f"\nFinal learner scales: {[f'{s:.4f}' for s in learner_scales]}")
    print(f"Saved adaptation history → {history_path}")


if __name__ == "__main__":
    main()
