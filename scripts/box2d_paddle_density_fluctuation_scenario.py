#!/usr/bin/env python3
"""
Render and validate Box2D paddle-density fluctuation behavior.

The script:
- loads simulator params from a YAML config,
- enables paddle-density fluctuation,
- runs a short paddle-only rollout,
- writes a GIF with per-step density metadata overlays,
- asserts the hold-window behavior (default: 5 env steps),
- and optionally writes JSON diagnostics.
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
        raise ValueError(
            f"Invalid config at {config_path}: expected 'air_hockey.simulator_params' dictionary."
        )
    return dict(sim_params)


def _build_simulator(
    *,
    config_path: Path,
    seed: int,
    relative_range: float,
    hold_steps: int,
    enabled: bool,
    paddle_density: float | None,
) -> AirHockeyBox2D:
    params = _load_simulator_params_from_config(config_path)
    params["seed"] = int(seed)
    params["enable_paddle_density_fluctuation"] = bool(enabled)
    params["paddle_density_fluctuation_relative_range"] = float(relative_range)
    params["paddle_density_fluctuation_hold_steps"] = int(hold_steps)
    if paddle_density is not None:
        params["paddle_density"] = float(paddle_density)

    # Keep observation/visualization deterministic for diagnostics.
    params["puck_noise"] = False
    params["enable_random_occlusions"] = False
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


def _overlay_step(
    frame: np.ndarray,
    *,
    step_idx: int,
    seed: int,
    density_base: float,
    multiplier: float,
    density_active: float,
    paddle_mass: float,
    hold_remaining: int,
) -> np.ndarray:
    lines = [
        f"step: {step_idx}",
        f"seed: {seed}",
        f"rho_base: {density_base:.2f}",
        f"rho_mult: {multiplier:.4f}",
        f"rho_active: {density_active:.2f}",
        f"mass: {paddle_mass:.5f}",
        f"hold_remaining: {hold_remaining}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 18
    for line in lines:
        cv2.putText(frame, line, (6, y), font, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (6, y), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        y += 18
    return frame


def _validate_fluctuation(
    *,
    multipliers: list[float],
    hold_remaining: list[int],
    relative_range: float,
    hold_steps: int,
) -> dict:
    arr_mult = np.asarray(multipliers, dtype=float)
    arr_hold = np.asarray(hold_remaining, dtype=int)
    lower = 1.0 - float(relative_range)
    upper = 1.0 + float(relative_range)
    in_range = bool(np.all(arr_mult >= lower - 1e-10) and np.all(arr_mult <= upper + 1e-10))

    per_segment_constant = True
    for start in range(0, len(arr_mult), int(hold_steps)):
        segment = arr_mult[start : start + int(hold_steps)]
        if len(segment) <= 1:
            continue
        if not bool(np.allclose(segment, segment[0], rtol=0.0, atol=1e-12)):
            per_segment_constant = False
            break

    expected_hold = np.asarray(
        [(int(hold_steps) - 1 - (i % int(hold_steps))) for i in range(len(arr_hold))],
        dtype=int,
    )
    hold_pattern_ok = bool(np.array_equal(arr_hold, expected_hold))

    return {
        "in_range": in_range,
        "per_segment_constant": per_segment_constant,
        "hold_pattern_ok": hold_pattern_ok,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "expected_hold_remaining": expected_hold.tolist(),
    }


def run_density_fluctuation_scenario(
    *,
    config_path: Path,
    seed: int,
    steps: int,
    fps: int,
    relative_range: float,
    hold_steps: int,
    paddle_density: float | None,
    output_gif: Path,
    output_json: Path | None,
) -> dict:
    sim = _build_simulator(
        config_path=config_path,
        seed=seed,
        relative_range=relative_range,
        hold_steps=hold_steps,
        enabled=True,
        paddle_density=paddle_density,
    )
    sim.reset(seed=seed)
    sim.spawn_paddle(pos=(-0.52, 0.0), vel=(0.0, 0.0), name="paddle_ego", affected_by_gravity=False)
    sim.set_object_links()

    adapter = _make_render_adapter(sim)
    renderer = AirHockeyRenderer(
        adapter,
        orientation="vertical",
        show_target_position=False,
        show_acceleration_arrow=False,
    )

    frames: list[np.ndarray] = []
    multipliers: list[float] = []
    active_densities: list[float] = []
    masses: list[float] = []
    hold_remaining: list[int] = []

    action = np.zeros(2, dtype=float)
    for step_idx in range(int(steps)):
        sign = 1.0 if ((step_idx // 6) % 2 == 0) else -1.0
        action[0] = 0.05 * sign
        action[1] = 0.0
        sim.get_transition(action)

        adapter.current_state = sim.get_current_state()
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))

        paddle_mass = float(sim.paddles["paddle_ego"].mass)
        multiplier = float(sim._current_paddle_density_multiplier)
        density_base = float(sim._paddle_density_base)
        density_active = density_base * multiplier
        hold = int(sim._paddle_density_hold_remaining)

        multipliers.append(multiplier)
        active_densities.append(density_active)
        masses.append(paddle_mass)
        hold_remaining.append(hold)

        frame = _overlay_step(
            frame,
            step_idx=step_idx,
            seed=seed,
            density_base=density_base,
            multiplier=multiplier,
            density_active=density_active,
            paddle_mass=paddle_mass,
            hold_remaining=hold,
        )
        frames.append(frame)

    validation = _validate_fluctuation(
        multipliers=multipliers,
        hold_remaining=hold_remaining,
        relative_range=relative_range,
        hold_steps=hold_steps,
    )
    if not (validation["in_range"] and validation["per_segment_constant"] and validation["hold_pattern_ok"]):
        raise AssertionError(f"Density fluctuation validation failed: {validation}")

    # Quick regression check: disabled mode should remain at nominal density.
    baseline = _build_simulator(
        config_path=config_path,
        seed=seed,
        relative_range=relative_range,
        hold_steps=hold_steps,
        enabled=False,
        paddle_density=paddle_density,
    )
    baseline.reset(seed=seed)
    baseline.spawn_paddle(pos=(-0.52, 0.0), vel=(0.0, 0.0), name="paddle_ego", affected_by_gravity=False)
    baseline.set_object_links()
    baseline_masses: list[float] = []
    for _ in range(int(steps)):
        baseline.get_transition(np.zeros(2, dtype=float))
        baseline_masses.append(float(baseline.paddles["paddle_ego"].mass))
    baseline_constant = bool(np.allclose(baseline_masses, baseline_masses[0], rtol=0.0, atol=1e-12))
    if not baseline_constant:
        raise AssertionError("Baseline (fluctuation disabled) paddle mass changed unexpectedly.")

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output_gif,
        frames,
        format="GIF",
        loop=0,
        duration=int(1000 / max(1, int(fps))),
    )

    result = {
        "seed": int(seed),
        "steps": int(steps),
        "fps": int(fps),
        "config_path": str(config_path.resolve()),
        "gif_path": str(output_gif.resolve()),
        "relative_range": float(relative_range),
        "hold_steps": int(hold_steps),
        "configured_paddle_density": (
            None if paddle_density is None else float(paddle_density)
        ),
        "base_density": float(sim._paddle_density_base),
        "validation": validation,
        "baseline_constant_mass": bool(baseline_constant),
        "multiplier_series": [float(v) for v in multipliers],
        "hold_remaining_series": [int(v) for v in hold_remaining],
        "density_series": [float(v) for v in active_densities],
        "mass_series": [float(v) for v in masses],
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and visualize Box2D paddle density fluctuation."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml",
        help="Config path for air_hockey.simulator_params source of truth.",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--relative-range", type=float, default=0.25)
    parser.add_argument("--hold-steps", type=int, default=5)
    parser.add_argument(
        "--paddle-density",
        type=float,
        default=100.0,
        help="Nominal paddle density used as fluctuation base (much lower than default).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/paddle_density_fluctuation/density_fluctuation.gif",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="runs/paddle_density_fluctuation/density_fluctuation_metrics.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path).expanduser().resolve()
    output_gif = Path(args.output).expanduser()
    output_json = Path(args.json_out).expanduser() if args.json_out else None

    result = run_density_fluctuation_scenario(
        config_path=config_path,
        seed=int(args.seed),
        steps=int(args.steps),
        fps=int(args.fps),
        relative_range=float(args.relative_range),
        hold_steps=int(args.hold_steps),
        paddle_density=float(args.paddle_density),
        output_gif=output_gif,
        output_json=output_json,
    )

    print("-" * 80)
    print(f"GIF: {result['gif_path']}")
    print(f"JSON: {str(output_json.resolve()) if output_json is not None else '(disabled)'}")
    print(
        "Validation | "
        f"in_range={result['validation']['in_range']}, "
        f"per_segment_constant={result['validation']['per_segment_constant']}, "
        f"hold_pattern_ok={result['validation']['hold_pattern_ok']}, "
        f"baseline_constant_mass={result['baseline_constant_mass']}"
    )


if __name__ == "__main__":
    main()
