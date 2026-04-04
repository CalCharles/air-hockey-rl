"""
Phase 1: render crafted collision scenarios for visual inspection.

Builds two envs (oracle and learner) and renders each of the 5 crafted scenarios
as GIFs so the user can confirm the setup makes physical sense before running
the adaptation algorithm.

Usage:
    python scripts/collision_adaptation/render_scenarios.py \
        --config scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params_heavy.yaml \
        --oracle-paddle-scales 0.7 1.0 1.2 \
        --output-dir runs/collision_adaptation \
        --fps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.collision_adaptation.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_env(config_path: str, paddle_scales: list[float]) -> object:
    """Build an AirHockeyEnv with density overrides and given paddle scales."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["air_hockey"]

    # Override densities as specified in the adaptation plan.
    cfg["simulator_params"]["puck_density"] = 3000
    cfg["simulator_params"]["paddle_density"] = 3000

    # Disable stochastic elements for clean, deterministic collision scenarios.
    cfg["simulator_params"]["puck_noise"] = False
    cfg["simulator_params"]["enable_random_occlusions"] = False
    cfg["simulator_params"]["enable_action_delay"] = False
    cfg["simulator_params"]["enable_observation_delay"] = False

    env = AirHockeyEnv(cfg)
    env.simulator.set_collision_scales(
        wall_scales=[1.0, 1.0, 1.0],
        paddle_scales=paddle_scales,
    )
    return env


def _set_scenario_state(env, scenario: dict) -> None:
    """After env.reset(), place puck and paddle at scenario positions/velocities."""
    sim = env.simulator
    puck_name = list(sim.pucks.keys())[0]

    # Puck
    sim.pucks[puck_name].position = sim.base_coord_to_box2d(scenario["puck_pos"])
    sim.pucks[puck_name].linearVelocity = sim.base_coord_to_box2d(scenario["puck_vel"])

    # Paddle
    sim.paddles["paddle_ego"].position = sim.base_coord_to_box2d(scenario["paddle_pos"])
    sim.paddles["paddle_ego"].linearVelocity = (0.0, 0.0)


def _is_new_paddle_collision(forces_before: int, sim) -> bool:
    """Return True if any new collision entry since forces_before involves the paddle."""
    forces = sim.get_collision_forces()
    for cf in forces[forces_before:]:
        if "paddle" in str(cf.get("bodyA", "")) or "paddle" in str(cf.get("bodyB", "")):
            return True
    return False


def _puck_speed(sim) -> float:
    """Return current puck speed in base-frame m/s."""
    puck_name = list(sim.pucks.keys())[0]
    # Box2D linearVelocity is in box2d frame; convert back to base speed (magnitude is invariant).
    vel = sim.pucks[puck_name].linearVelocity
    return float(np.sqrt(vel[0] ** 2 + vel[1] ** 2))


def _puck_base_vel(sim) -> tuple[float, float]:
    """Return (vx, vy) puck velocity in base frame."""
    puck_name = list(sim.pucks.keys())[0]
    vb = sim.pucks[puck_name].linearVelocity  # (box2d_x, box2d_y)
    # Inverse of base_coord_to_box2d: box2d = (base_y, -base_x) → base = (-box2d_y, box2d_x)
    vx = -float(vb[1])
    vy = float(vb[0])
    return vx, vy


def _run_scenario(
    env,
    scenario: dict,
    gif_path: str,
    fps: int,
) -> dict:
    """
    Run one scenario on env.  Returns a dict with pre/post puck velocity info.
    """
    env.reset()
    _set_scenario_state(env, scenario)

    renderer = AirHockeyRenderer(
        env,
        orientation="vertical",
        show_target_position=False,
        show_acceleration_arrow=False,
    )

    action = np.array(scenario["paddle_action"], dtype=np.float32)
    n_steps = scenario["n_steps"]
    settle_steps = 15  # steps to capture after first paddle collision

    sim = env.simulator
    frames: list[np.ndarray] = []
    collision_step: int | None = None
    pre_puck_vel: tuple | None = None
    post_puck_vel: tuple | None = None
    forces_baseline = len(sim.get_collision_forces())

    for step_i in range(n_steps + settle_steps):
        # Capture frame before step
        frame = cv2.cvtColor(renderer.get_frame(), cv2.COLOR_BGR2RGB)
        aspect = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect)))

        # Annotate
        label = f"step {step_i}"
        if collision_step is not None:
            label += f" [+{step_i - collision_step}]"
        cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(
            frame,
            f"|v_puck|={_puck_speed(sim):.2f}",
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        frames.append(frame)

        # Capture puck velocity just before this step for pre-collision recording
        vel_before = _puck_base_vel(sim)
        forces_before = len(sim.get_collision_forces())

        env.step(action)

        # Check for paddle collision this step
        if collision_step is None and _is_new_paddle_collision(forces_before, sim):
            collision_step = step_i
            pre_puck_vel = vel_before
            post_puck_vel = _puck_base_vel(sim)

        # Stop after settle_steps post-collision
        if collision_step is not None and (step_i - collision_step) >= settle_steps:
            # Capture one last frame
            frame = cv2.cvtColor(renderer.get_frame(), cv2.COLOR_BGR2RGB)
            aspect = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (160, int(160 / aspect)))
            frames.append(frame)
            break

    # Retrieve full episode stats (resets counters in sim)
    episode_stats = sim.get_episode_collision_stats()

    os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
    duration_ms = int(1000 / max(fps, 1))
    imageio.mimsave(gif_path, frames, format="GIF", loop=0, duration=duration_ms)

    return {
        "collision_detected": collision_step is not None,
        "collision_step": collision_step,
        "pre_puck_vel": list(pre_puck_vel) if pre_puck_vel is not None else None,
        "post_puck_vel": list(post_puck_vel) if post_puck_vel is not None else None,
        "pre_puck_speed": float(np.linalg.norm(pre_puck_vel)) if pre_puck_vel is not None else None,
        "post_puck_speed": float(np.linalg.norm(post_puck_vel)) if post_puck_vel is not None else None,
        "episode_collision_stats": episode_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render crafted collision scenarios (Phase 1).")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML sim config (air_hockey top-level key).",
    )
    parser.add_argument(
        "--oracle-paddle-scales",
        nargs=3,
        type=float,
        default=[0.7, 1.0, 1.2],
        metavar=("LOW", "MID", "HIGH"),
        help="Oracle paddle restitution scales [low, mid, high].",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/collision_adaptation",
        help="Root output directory.",
    )
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inspect_dir = os.path.join(args.output_dir, "inspect")
    os.makedirs(inspect_dir, exist_ok=True)

    oracle_scales = args.oracle_paddle_scales
    learner_scales = [1.0, 1.0, 1.0]

    envs = {
        "oracle": _build_env(args.config, oracle_scales),
        "learner": _build_env(args.config, learner_scales),
    }

    all_results: dict = {}

    for env_name, env in envs.items():
        all_results[env_name] = {}
        for sc in SCENARIOS:
            sc_name = sc["name"]
            gif_path = os.path.join(inspect_dir, f"{env_name}_scenario_{sc_name}.gif")
            print(f"  [{env_name}] {sc_name} → {gif_path}")
            result = _run_scenario(env, sc, gif_path, args.fps)
            all_results[env_name][sc_name] = result
            detected = result["collision_detected"]
            pre = result["pre_puck_speed"]
            post = result["post_puck_speed"]
            pre_str = f"{pre:.3f}" if pre is not None else "N/A"
            post_str = f"{post:.3f}" if post is not None else "N/A"
            print(
                f"    collision={'YES' if detected else 'NO'} "
                f"pre={pre_str} m/s  post={post_str} m/s"
            )

    # Save scenarios.json
    json_path = os.path.join(inspect_dir, "scenarios.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved scenarios.json → {json_path}")
    print("Inspect GIFs before running Phase 2 (run_adaptation.py).")


if __name__ == "__main__":
    main()
