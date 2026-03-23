#!/usr/bin/env python3
"""
Generate a hard-coded Box2D paddle-puck contact scenario and validate contact metrics.

Scenario:
- One puck starts above the ego paddle in the ego half.
- Puck moves downward and is affected by gravity.
- Paddle moves upward toward the puck.
- The script writes a GIF and prints contact validation stats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airhockey.renderers import AirHockeyRenderer
from airhockey.sims.airhockey_box2d import AirHockeyBox2D


def _parse_density_pairs(raw_pairs: list[str]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for item in raw_pairs:
        if ":" not in item:
            raise ValueError(f"Invalid density pair '{item}'. Expected format paddle:puck")
        paddle_raw, puck_raw = item.split(":", maxsplit=1)
        parsed.append((float(paddle_raw), float(puck_raw)))
    return parsed


def _load_simulator_params_from_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config at {config_path}: expected dictionary.")

    air_hockey_cfg = config.get("air_hockey", {})
    if not isinstance(air_hockey_cfg, dict):
        raise ValueError(f"Invalid config at {config_path}: expected 'air_hockey' dictionary.")

    sim_params = air_hockey_cfg.get("simulator_params", {})
    if not isinstance(sim_params, dict):
        raise ValueError(f"Invalid config at {config_path}: expected 'air_hockey.simulator_params' dictionary.")
    return dict(sim_params)


def _build_simulator(
    *,
    config_path: Path,
    seed: int,
    paddle_density: float,
    puck_density: float,
    gravity: float | None,
    time_frequency: int | None,
    paddle_restitution: float | None,
    puck_restitution: float | None,
) -> AirHockeyBox2D:
    params = _load_simulator_params_from_config(config_path)
    params["seed"] = int(seed)
    params["paddle_density"] = float(paddle_density)
    params["puck_density"] = float(puck_density)
    # Keep visuals truthful and stable for contact validation.
    params["puck_noise"] = False
    params["enable_random_occlusions"] = False
    if gravity is not None:
        params["gravity"] = float(gravity)
    if time_frequency is not None:
        params["time_frequency"] = int(time_frequency)
        params["step_frequency"] = int(time_frequency)
    if paddle_restitution is not None:
        params["paddle_restitution"] = float(paddle_restitution)
    if puck_restitution is not None:
        params["puck_restitution"] = float(puck_restitution)
    return AirHockeyBox2D.from_dict(params)


def _make_render_adapter(sim: AirHockeyBox2D) -> SimpleNamespace:
    return SimpleNamespace(
        render_width=sim.render_width,
        render_length=sim.render_length,
        render_masks=sim.render_masks,
        width=sim.width,
        length=sim.length,
        ppm=sim.ppm,
        goal_conditioned=False,
        reward_regions=[],
        multiagent=False,
        puck_radius=sim.puck_radius,
        paddle_radius=sim.paddle_radius,
        block_width=sim.block_width,
        current_state=sim.get_current_state(),
    )


def _extract_paddle_puck_velocities(state_info: dict) -> tuple[np.ndarray, np.ndarray]:
    paddle_vel = np.zeros(2, dtype=float)
    puck_vel = np.zeros(2, dtype=float)

    paddles = state_info.get("paddles", {})
    if "paddle_ego" in paddles:
        paddle_vel = np.asarray(paddles["paddle_ego"]["velocity"], dtype=float)

    pucks = state_info.get("pucks", [])
    if len(pucks) > 0:
        puck_vel = np.asarray(pucks[0]["velocity"], dtype=float)
    return paddle_vel, puck_vel


def _overlay_metadata(
    frame: np.ndarray,
    *,
    step_idx: int,
    seed: int,
    paddle_density: float,
    puck_density: float,
    collision_count: int,
) -> np.ndarray:
    text_lines = [
        f"step: {step_idx}",
        f"seed: {seed}",
        f"rho_paddle: {paddle_density:.2f}",
        f"rho_puck: {puck_density:.2f}",
        f"contact_count: {collision_count}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    line_height = 18
    x = 6
    y = 18
    for line in text_lines:
        cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_height
    return frame


def _state_to_metrics_dict(state_info: dict) -> dict[str, float]:
    paddle_vel, puck_vel = _extract_paddle_puck_velocities(state_info)
    return {
        "paddle_vx": float(paddle_vel[0]),
        "paddle_vy": float(paddle_vel[1]),
        "paddle_speed": float(np.linalg.norm(paddle_vel)),
        "puck_vx": float(puck_vel[0]),
        "puck_vy": float(puck_vel[1]),
        "puck_speed": float(np.linalg.norm(puck_vel)),
        "relative_speed": float(np.linalg.norm(paddle_vel - puck_vel)),
    }


def run_contact_scenario(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    fps: int,
    paddle_density: float,
    puck_density: float,
    gif_path: Path,
    gravity: float | None,
    time_frequency: int | None,
    paddle_restitution: float | None,
    puck_restitution: float | None,
    paddle_jitter_action: float,
    paddle_jitter_half_period_steps: int,
) -> dict:
    sim = _build_simulator(
        config_path=config_path,
        seed=seed,
        paddle_density=paddle_density,
        puck_density=puck_density,
        gravity=gravity,
        time_frequency=time_frequency,
        paddle_restitution=paddle_restitution,
        puck_restitution=puck_restitution,
    )
    sim.reset(seed=seed)

    # Hard-coded base-frame setup:
    # - puck starts "above" paddle on same y-lane
    # - puck heads downward (plus gravity), paddle heads upward
    # Box2D mapping is internal through spawn_* conversions.
    puck_initial_pos = (-0.5, 0.0)
    puck_initial_vel = (0.52, 0.0)
    paddle_initial_pos = (0.36, 0.0)
    paddle_initial_vel = (-0.65, 0.0)

    sim.spawn_puck(puck_initial_pos, puck_initial_vel, "puck_0", affected_by_gravity=True)
    sim.spawn_paddle(paddle_initial_pos, paddle_initial_vel, "paddle_ego", affected_by_gravity=False)
    sim.set_object_links()

    adapter = _make_render_adapter(sim)
    renderer = AirHockeyRenderer(adapter, show_target_position=False, show_acceleration_arrow=False)

    frames: list[np.ndarray] = []
    contact_step: int | None = None
    pre_contact_state: dict | None = None
    post_contact_state: dict | None = None

    action = np.zeros(2, dtype=float)
    last_collision_count = 0
    for step_idx in range(int(steps)):
        adapter.current_state = sim.get_current_state()
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))

        frame = _overlay_metadata(
            frame,
            step_idx=step_idx,
            seed=seed,
            paddle_density=paddle_density,
            puck_density=puck_density,
            collision_count=last_collision_count,
        )
        frames.append(frame)

        step_start_state = adapter.current_state
        jitter_sign = 1.0 if ((step_idx // max(1, int(paddle_jitter_half_period_steps))) % 2 == 0) else -1.0
        # Base-frame action: alternate "up/down" paddle nudges in x.
        action[0] = float(paddle_jitter_action) * jitter_sign
        action[1] = 0.0
        step_state = sim.get_transition(action)
        collision_count = int(step_state.get("paddle_puck_collision_count", 0))
        last_collision_count = collision_count

        if contact_step is None and collision_count > 0:
            contact_step = step_idx
            pre_contact_state = step_start_state
            post_contact_state = step_state

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        gif_path,
        frames,
        format="GIF",
        loop=0,
        duration=int(1000 / max(1, int(fps))),
    )

    final_state = sim.get_current_state()
    sim.world.contactListener = None

    result = {
        "seed": int(seed),
        "steps": int(steps),
        "fps": int(fps),
        "gif_path": str(gif_path.resolve()),
        "paddle_density": float(paddle_density),
        "puck_density": float(puck_density),
        "contact_step": None if contact_step is None else int(contact_step),
        "config_path": str(config_path.resolve()),
        "gravity_override": None if gravity is None else float(gravity),
        "time_frequency_override": None if time_frequency is None else int(time_frequency),
        "paddle_jitter_action": float(paddle_jitter_action),
        "paddle_jitter_half_period_steps": int(paddle_jitter_half_period_steps),
        "initial_conditions": {
            "puck_initial_pos_base": [float(x) for x in puck_initial_pos],
            "puck_initial_vel_base": [float(x) for x in puck_initial_vel],
            "paddle_initial_pos_base": [float(x) for x in paddle_initial_pos],
            "paddle_initial_vel_base": [float(x) for x in paddle_initial_vel],
        },
        "final_state": _state_to_metrics_dict(final_state),
    }
    if pre_contact_state is not None:
        result["pre_contact"] = _state_to_metrics_dict(pre_contact_state)
    if post_contact_state is not None:
        result["post_contact"] = _state_to_metrics_dict(post_contact_state)
    return result


def _result_basename(paddle_density: float, puck_density: float, seed: int) -> str:
    pd = str(int(paddle_density)) if float(paddle_density).is_integer() else f"{paddle_density:.2f}"
    pk = str(int(puck_density)) if float(puck_density).is_integer() else f"{puck_density:.2f}"
    return f"contact_pd{pd}_pk{pk}_seed{seed}.gif"


def _print_run_summary(result: dict) -> None:
    print("-" * 80)
    print(f"GIF: {result['gif_path']}")
    print(f"Densities: paddle={result['paddle_density']}, puck={result['puck_density']}")
    print(f"Contact step: {result['contact_step']}")
    if "pre_contact" in result:
        pre = result["pre_contact"]
        print(
            "Pre-contact | "
            f"paddle_speed={pre['paddle_speed']:.4f}, "
            f"puck_speed={pre['puck_speed']:.4f}, "
            f"relative_speed={pre['relative_speed']:.4f}"
        )
    if "post_contact" in result:
        post = result["post_contact"]
        print(
            "Post-contact | "
            f"paddle_speed={post['paddle_speed']:.4f}, "
            f"puck_speed={post['puck_speed']:.4f}, "
            f"relative_speed={post['relative_speed']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a hard-coded Box2D paddle-puck contact scenario with density validation."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml",
        help="YAML config used to source simulator_params (AMP-scale alignment).",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=str,
        default="runs/paddle_puck_contact/contact.gif",
        help="Output GIF path (single run) or output directory (sweep mode).",
    )
    parser.add_argument("--paddle-density", type=float, default=2500.0)
    parser.add_argument("--puck-density", type=float, default=250.0)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--time-frequency", type=int, default=None)
    parser.add_argument("--paddle-restitution", type=float, default=None)
    parser.add_argument("--puck-restitution", type=float, default=None)
    parser.add_argument(
        "--paddle-jitter-action",
        type=float,
        default=0.1,
        help="Per-step paddle action magnitude in base x; sign alternates by jitter period.",
    )
    parser.add_argument(
        "--paddle-jitter-half-period-steps",
        type=int,
        default=7,
        help="Timesteps to hold one jitter direction before flipping sign.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional JSON file for single-run metrics.",
    )
    parser.add_argument(
        "--sweep-jsonl",
        type=str,
        default="",
        help="If set, run sweep mode and append each run result as one JSON line.",
    )
    parser.add_argument(
        "--density-pairs",
        nargs="*",
        default=[],
        help="Optional sweep density pairs in 'paddle:puck' format.",
    )
    args = parser.parse_args()
    config_path = Path(args.config_path).expanduser().resolve()

    if args.sweep_jsonl:
        density_pairs = _parse_density_pairs(args.density_pairs)
        if not density_pairs:
            density_pairs = [(2500.0, 250.0), (250.0, 2500.0), (2500.0, 2500.0)]

        output_root = Path(args.output)
        output_root.mkdir(parents=True, exist_ok=True)
        sweep_results: list[dict] = []
        for paddle_density, puck_density in density_pairs:
            gif_name = _result_basename(paddle_density, puck_density, args.seed)
            gif_path = output_root / gif_name
            result = run_contact_scenario(
                config_path=config_path,
                seed=args.seed,
                steps=args.steps,
                fps=args.fps,
                paddle_density=paddle_density,
                puck_density=puck_density,
                gif_path=gif_path,
                gravity=args.gravity,
                time_frequency=args.time_frequency,
                paddle_restitution=args.paddle_restitution,
                puck_restitution=args.puck_restitution,
                paddle_jitter_action=args.paddle_jitter_action,
                paddle_jitter_half_period_steps=args.paddle_jitter_half_period_steps,
            )
            sweep_results.append(result)
            _print_run_summary(result)

        sweep_jsonl_path = Path(args.sweep_jsonl)
        sweep_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with sweep_jsonl_path.open("w", encoding="utf-8") as handle:
            for row in sweep_results:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote sweep metrics to: {sweep_jsonl_path.resolve()}")
        return

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path / _result_basename(args.paddle_density, args.puck_density, args.seed)

    result = run_contact_scenario(
        config_path=config_path,
        seed=args.seed,
        steps=args.steps,
        fps=args.fps,
        paddle_density=args.paddle_density,
        puck_density=args.puck_density,
        gif_path=output_path,
        gravity=args.gravity,
        time_frequency=args.time_frequency,
        paddle_restitution=args.paddle_restitution,
        puck_restitution=args.puck_restitution,
        paddle_jitter_action=args.paddle_jitter_action,
        paddle_jitter_half_period_steps=args.paddle_jitter_half_period_steps,
    )
    _print_run_summary(result)

    if args.json_out:
        json_out_path = Path(args.json_out)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        with json_out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote JSON metrics to: {json_out_path.resolve()}")


if __name__ == "__main__":
    main()
