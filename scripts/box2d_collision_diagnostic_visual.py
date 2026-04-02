#!/usr/bin/env python3
"""
Visual Box2D collision diagnostic with GIF output.

Re-runs the same density x speed sweep as box2d_collision_diagnostic.py, but
renders each collision as a GIF using AirHockeyRenderer (same format as
TD3 evaluation checkpoints).

The script uses AirHockeyBox2D for table geometry and rendering, but steps
the Box2D world directly (no PID, no force application) so collisions
reflect pure Box2D resolution — matching the analytical diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airhockey.renderers import AirHockeyRenderer
from airhockey.sims.airhockey_box2d import AirHockeyBox2D

PADDLE_RADIUS = 0.0508
PUCK_RADIUS = 0.03175
RESTITUTION = 1.0

DENSITY_RATIOS = [
    ("1:1", 250.0, 250.0),
    ("5:1", 1250.0, 250.0),
    ("25:1", 6250.0, 250.0),
]

REL_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.05, 1.1, 1.5, 2.0, 4.0]

DEFAULT_CONFIG = (
    "scripts/smooth_policy/amp_history/configs/new_juggle/"
    "pid_noise_constant_upper_half_custom_sim_params.yaml"
)


@dataclass
class VisualCollisionResult:
    paddle_density: float
    puck_density: float
    mass_ratio: float
    rel_speed: float
    speed_label: str
    paddle_v_in: float
    puck_v_in: float
    paddle_v_out: float
    puck_v_out: float
    paddle_v_expected: float
    puck_v_expected: float
    puck_error_pct: float
    paddle_error_pct: float
    ke_ratio: float
    p_ratio: float
    collision_detected: bool
    gif_path: str


def analytical_1d_collision(
    m1: float, m2: float, v1: float, v2: float, e: float = 1.0
) -> tuple[float, float]:
    denom = m1 + m2
    v1f = ((m1 - e * m2) / denom) * v1 + ((1 + e) * m2 / denom) * v2
    v2f = ((1 + e) * m1 / denom) * v1 + ((m2 - e * m1) / denom) * v2
    return v1f, v2f


def build_speed_cases(rel_speed: float) -> list[tuple[str, float, float]]:
    """
    Return (label, paddle_vx_base, puck_vx_base) tuples for head-on collisions
    along the base-frame x-axis. Paddle moves in -x, puck in +x.
    """
    return [
        ("paddle_only", -rel_speed, 0.0),
        ("mostly_paddle", -0.75 * rel_speed, 0.25 * rel_speed),
        ("equal_split", -0.5 * rel_speed, 0.5 * rel_speed),
        ("mostly_puck", -0.25 * rel_speed, 0.75 * rel_speed),
        ("puck_only", 0.0, rel_speed),
    ]


def _load_sim_params(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return dict(config.get("air_hockey", {}).get("simulator_params", {}))


def _build_sim(
    config_path: Path,
    paddle_density: float,
    puck_density: float,
    seed: int = 0,
) -> AirHockeyBox2D:
    params = _load_sim_params(config_path)
    params["seed"] = seed
    params["paddle_density"] = paddle_density
    params["puck_density"] = puck_density
    params["gravity"] = 0.0
    params["paddle_damping"] = 0.0
    params["puck_damping"] = 0.0
    params["paddle_restitution"] = RESTITUTION
    params["puck_restitution"] = RESTITUTION
    params["puck_noise"] = False
    params["enable_random_occlusions"] = False
    params["enable_action_delay"] = False
    params["enable_observation_delay"] = False
    params["use_pid"] = False
    params["max_paddle_vel"] = 10.0
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


def _overlay_text(
    frame: np.ndarray,
    *,
    step_idx: int,
    paddle_density: float,
    puck_density: float,
    rel_speed: float,
    speed_label: str,
    collision_step: int | None,
    paddle_speed: float,
    puck_speed: float,
) -> np.ndarray:
    lines = [
        f"rho_pd:{paddle_density:.0f} rho_pk:{puck_density:.0f}",
        f"rel_spd:{rel_speed:.2f} [{speed_label}]",
        f"step:{step_idx}",
        f"pd_spd:{paddle_speed:.3f} pk_spd:{puck_speed:.3f}",
    ]
    if collision_step is not None:
        lines.append(f"contact@step:{collision_step}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.32
    y = 12
    for line in lines:
        cv2.putText(frame, line, (4, y), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (4, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += 13
    return frame


def run_visual_collision(
    *,
    config_path: Path,
    paddle_density: float,
    puck_density: float,
    rel_speed: float,
    speed_label: str,
    paddle_vx_base: float,
    puck_vx_base: float,
    gif_path: Path,
    steps: int = 150,
    fps: int = 20,
    seed: int = 0,
    sim_dt: float = 1.0 / 50.0,
) -> VisualCollisionResult:
    sim = _build_sim(config_path, paddle_density, puck_density, seed=seed)
    sim.reset(seed=seed)

    paddle_base_pos = (0.30, 0.0)
    puck_base_pos = (-0.20, 0.0)
    paddle_base_vel = (paddle_vx_base, 0.0)
    puck_base_vel = (puck_vx_base, 0.0)

    sim.spawn_puck(puck_base_pos, puck_base_vel, "puck_0", affected_by_gravity=False)
    sim.spawn_paddle(paddle_base_pos, paddle_base_vel, "paddle_ego", affected_by_gravity=False)
    sim.set_object_links()

    paddle_body = sim.paddles["paddle_ego"]
    puck_body = sim.pucks["puck_0"]
    paddle_body.linearDamping = 0.0
    puck_body.linearDamping = 0.0

    paddle_mass = paddle_body.mass
    puck_mass = puck_body.mass

    adapter = _make_render_adapter(sim)
    renderer = AirHockeyRenderer(
        adapter,
        orientation="vertical",
        show_target_position=False,
        show_acceleration_arrow=False,
    )

    # Collision axis is box2d-y (base_coord_to_box2d maps base (vx,vy)->(vy,-vx)).
    # Pre-collision box2d-y velocities:
    pre_paddle_vy = float(paddle_body.linearVelocity[1])
    pre_puck_vy = float(puck_body.linearVelocity[1])

    frames: list[np.ndarray] = []
    collision_step: int | None = None
    pre_collision_forces_len = len(sim.collision_listener.collision_forces)
    post_paddle_vy: float | None = None
    post_puck_vy: float | None = None
    settle_counter = 0

    for step_i in range(steps):
        adapter.current_state = sim.get_current_state()
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))

        frame = _overlay_text(
            frame,
            step_idx=step_i,
            paddle_density=paddle_density,
            puck_density=puck_density,
            rel_speed=rel_speed,
            speed_label=speed_label,
            collision_step=collision_step,
            paddle_speed=float(
                np.linalg.norm(
                    [paddle_body.linearVelocity.x, paddle_body.linearVelocity.y]
                )
            ),
            puck_speed=float(
                np.linalg.norm(
                    [puck_body.linearVelocity.x, puck_body.linearVelocity.y]
                )
            ),
        )
        frames.append(frame)

        sim.world.Step(sim_dt, 100, 100)
        sim.world.ClearForces()

        new_forces_len = len(sim.collision_listener.collision_forces)
        if collision_step is None and new_forces_len > pre_collision_forces_len:
            collision_step = step_i
            settle_counter = 0

        if collision_step is not None:
            settle_counter += 1
            if settle_counter == 10 and post_paddle_vy is None:
                post_paddle_vy = float(paddle_body.linearVelocity[1])
                post_puck_vy = float(puck_body.linearVelocity[1])

    if post_paddle_vy is None:
        post_paddle_vy = float(paddle_body.linearVelocity[1])
        post_puck_vy = float(puck_body.linearVelocity[1])

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(gif_path),
        frames,
        format="GIF",
        loop=0,
        duration=int(1000 / max(1, fps)),
    )

    paddle_v_exp, puck_v_exp = analytical_1d_collision(
        paddle_mass, puck_mass, pre_paddle_vy, pre_puck_vy, RESTITUTION
    )

    def pct_err(actual: float, expected: float) -> float:
        if abs(expected) < 1e-8:
            return abs(actual - expected) * 100.0
        return (actual - expected) / abs(expected) * 100.0

    ke_pre = 0.5 * paddle_mass * pre_paddle_vy**2 + 0.5 * puck_mass * pre_puck_vy**2
    ke_post = (
        0.5 * paddle_mass * post_paddle_vy**2 + 0.5 * puck_mass * post_puck_vy**2
    )
    ke_ratio = ke_post / ke_pre if ke_pre > 1e-12 else float("nan")

    p_pre = paddle_mass * pre_paddle_vy + puck_mass * pre_puck_vy
    p_post = paddle_mass * post_paddle_vy + puck_mass * post_puck_vy
    p_ratio = p_post / p_pre if abs(p_pre) > 1e-12 else float("nan")

    sim.world.contactListener = None

    return VisualCollisionResult(
        paddle_density=paddle_density,
        puck_density=puck_density,
        mass_ratio=paddle_mass / puck_mass if puck_mass > 0 else float("inf"),
        rel_speed=rel_speed,
        speed_label=speed_label,
        paddle_v_in=pre_paddle_vy,
        puck_v_in=pre_puck_vy,
        paddle_v_out=post_paddle_vy,
        puck_v_out=post_puck_vy,
        paddle_v_expected=paddle_v_exp,
        puck_v_expected=puck_v_exp,
        puck_error_pct=pct_err(post_puck_vy, puck_v_exp),
        paddle_error_pct=pct_err(post_paddle_vy, paddle_v_exp),
        ke_ratio=ke_ratio,
        p_ratio=p_ratio,
        collision_detected=collision_step is not None,
        gif_path=str(gif_path),
    )


def print_results_table(results: list[VisualCollisionResult]) -> None:
    header = (
        f"{'rel_spd':>8} {'distribution':>14} "
        f"{'pdl_v_in':>9} {'pck_v_in':>9} "
        f"{'pdl_v_out':>10} {'pck_v_out':>10} "
        f"{'pck_v_exp':>10} {'pck_err%':>9} "
        f"{'KE_ratio':>9} {'p_ratio':>8} {'hit':>4}"
    )
    sep = "-" * len(header)

    grouped: dict[str, list[VisualCollisionResult]] = {}
    for r in results:
        key = f"{r.paddle_density:.0f}:{r.puck_density:.0f}"
        grouped.setdefault(key, []).append(r)

    for density_key, group in grouped.items():
        ratio = group[0].mass_ratio
        print(f"\n{'=' * len(header)}")
        print(f"  Density paddle:puck = {density_key}   |   Mass ratio = {ratio:.2f}:1")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        prev_rs = None
        for r in group:
            if prev_rs is not None and r.rel_speed != prev_rs:
                print(sep)
            prev_rs = r.rel_speed
            hit_mark = "Y" if r.collision_detected else "N"
            print(
                f"{r.rel_speed:>8.2f} {r.speed_label:>14} "
                f"{r.paddle_v_in:>9.4f} {r.puck_v_in:>9.4f} "
                f"{r.paddle_v_out:>10.4f} {r.puck_v_out:>10.4f} "
                f"{r.puck_v_expected:>10.4f} {r.puck_error_pct:>8.2f}% "
                f"{r.ke_ratio:>9.4f} {r.p_ratio:>8.4f} {hit_mark:>4}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual Box2D collision diagnostic with GIF output."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/collision_diagnostic_visual",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--density-filter",
        type=str,
        default=None,
        help="Only run a single density ratio label, e.g. '1:1' or '5:1'.",
    )
    parser.add_argument(
        "--speed-filter",
        type=float,
        default=None,
        help="Only run a single relative speed value.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Save all results as JSON to this path (default: <output-dir>/results.json).",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path).expanduser().resolve()
    output_root = Path(args.output_dir)

    density_ratios = DENSITY_RATIOS
    if args.density_filter:
        density_ratios = [d for d in DENSITY_RATIOS if d[0] == args.density_filter]
        if not density_ratios:
            print(f"No density ratio matching '{args.density_filter}'")
            sys.exit(1)

    rel_speeds = REL_SPEEDS
    if args.speed_filter is not None:
        rel_speeds = [s for s in REL_SPEEDS if abs(s - args.speed_filter) < 1e-6]
        if not rel_speeds:
            rel_speeds = [args.speed_filter]

    total_cases = len(density_ratios) * len(rel_speeds) * 5
    print(f"Running visual collision diagnostic: {total_cases} cases")
    print(f"  Density ratios: {[d[0] for d in density_ratios]}")
    print(f"  Relative speeds: {rel_speeds}")
    print(f"  Output: {output_root}")

    all_results: list[VisualCollisionResult] = []
    case_idx = 0

    for ratio_label, pd, pkd in density_ratios:
        density_dir = output_root / f"d{pd:.0f}_{pkd:.0f}"
        for rs in rel_speeds:
            for label, paddle_vx, puck_vx in build_speed_cases(rs):
                case_idx += 1
                gif_name = f"s{rs:.2f}_{label}.gif"
                gif_path = density_dir / gif_name
                print(
                    f"  [{case_idx}/{total_cases}] "
                    f"density={pd:.0f}:{pkd:.0f} speed={rs:.2f} dist={label} ... ",
                    end="",
                    flush=True,
                )
                result = run_visual_collision(
                    config_path=config_path,
                    paddle_density=pd,
                    puck_density=pkd,
                    rel_speed=rs,
                    speed_label=label,
                    paddle_vx_base=paddle_vx,
                    puck_vx_base=puck_vx,
                    gif_path=gif_path,
                    steps=args.steps,
                    fps=args.fps,
                    seed=args.seed,
                )
                hit = "HIT" if result.collision_detected else "MISS"
                print(
                    f"{hit}  pck_err={result.puck_error_pct:+.2f}%  "
                    f"KE={result.ke_ratio:.4f}"
                )
                all_results.append(result)

    print_results_table(all_results)

    json_out = args.json_out or str(output_root / "results.json")
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nJSON results saved to: {json_path.resolve()}")
    print(f"GIFs saved under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
