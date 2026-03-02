import argparse
import copy
import os

import cv2
import imageio
import numpy as np
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer


def _load_base_env_cfg(config_path):
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)["air_hockey"]
    cfg = copy.deepcopy(cfg)
    cfg["max_timesteps"] = 120
    cfg["terminate_on_enemy_goal"] = False
    cfg["terminate_on_out_of_bounds"] = False
    cfg["terminate_on_puck_hit_bottom"] = False
    cfg["terminate_on_puck_hit_paddle"] = False
    cfg["terminate_on_puck_pass_paddle"] = False
    cfg["terminate_on_puck_stop"] = False
    cfg["obs_type"] = "vel"
    cfg["num_pucks"] = 1
    cfg["num_paddles"] = 1
    cfg["num_blocks"] = 0
    cfg["num_targets"] = 0
    cfg["num_obstacles"] = 0
    sim = cfg["simulator_params"]
    sim["puck_noise"] = False
    sim["enable_random_occlusions"] = False
    sim["action_lag"] = 0.0
    # Keep baseline behavior from config (including legacy wall_bounce_scale).
    return cfg


def _make_initial_state(speed_profile, launch_angle_deg):
    # [paddle_x, paddle_y, paddle_vx, paddle_vy, puck_x, puck_y, puck_vx, puck_vy]
    paddle_state = [0.60, 0.0, 0.0, 0.0]
    # Trajectory into side wall with configurable launch angle.
    speed_map = {
        "low": 0.12,
        "medium": 0.35,
        "high": 0.80,
        "super_high": 1.20,
    }
    if speed_profile not in speed_map:
        raise ValueError(f"Unknown speed_profile '{speed_profile}'. Expected one of {list(speed_map.keys())}.")
    speed = float(speed_map[speed_profile])
    theta = np.deg2rad(float(launch_angle_deg))
    vx = speed * np.cos(theta)
    vy = speed * np.sin(theta)
    puck_state = [-0.20, 0.28, vx, vy]
    return [float(v) for v in (paddle_state + puck_state)]


def _capture_rollout_gif(env_cfg, save_path, label, speed_profile, launch_angle_deg, frames=90):
    env = AirHockeyEnv(env_cfg)
    renderer = AirHockeyRenderer(
        env,
        show_target_position=True,
        show_acceleration_arrow=False,
    )
    env.reset_from_state(_make_initial_state(speed_profile, launch_angle_deg), seed=0)

    zero_action = np.zeros(2, dtype=np.float32)
    out_frames = []
    incoming_speed = None
    outgoing_speed = None
    collision_seen = False
    last_vel_box2d = np.array(
        [env.simulator.pucks["puck_0"].linearVelocity[0], env.simulator.pucks["puck_0"].linearVelocity[1]],
        dtype=float,
    )

    for step in range(frames):
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        state_info = env.simulator.get_current_state()
        puck_vel = np.array(state_info["pucks"][0]["velocity"], dtype=float)
        speed = float(np.linalg.norm(puck_vel))
        vy = float(puck_vel[1])

        _ = vy  # retained for overlay only

        cv2.putText(
            frame,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            f"step={step:03d} | puck_v=({puck_vel[0]:+.3f},{puck_vel[1]:+.3f}) | |v|={speed:.3f}",
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
        )
        out_frames.append(frame)
        env.step(zero_action)

        # Measure first bounce using actual wall contact normal in Box2D frame.
        current_vel_box2d = np.array(
            [env.simulator.pucks["puck_0"].linearVelocity[0], env.simulator.pucks["puck_0"].linearVelocity[1]],
            dtype=float,
        )
        if not collision_seen:
            for collision in env.simulator.get_collision_forces():
                body_a = str(collision.get("bodyA", ""))
                body_b = str(collision.get("bodyB", ""))
                puck_wall = (
                    (body_a.startswith("puck") and body_b == "table_wall")
                    or (body_b.startswith("puck") and body_a == "table_wall")
                )
                if not puck_wall:
                    continue
                normal = np.array(collision.get("contact_normal", (0.0, 0.0)), dtype=float)
                n_norm = float(np.linalg.norm(normal))
                if n_norm <= 1e-8:
                    continue
                contact_normal = normal / n_norm
                pre_n = float(np.dot(last_vel_box2d, contact_normal))
                incoming_speed = abs(pre_n)
                incoming_sign = 1.0 if pre_n >= 0.0 else -1.0
                v_n = float(np.dot(current_vel_box2d, contact_normal))
                outgoing_speed = max(0.0, -incoming_sign * v_n)
                collision_seen = True
                break

        last_vel_box2d = current_vel_box2d

    imageio.mimsave(save_path, out_frames, format="GIF", loop=0, duration=50)
    env.close()
    return incoming_speed, outgoing_speed


def _parse_float_list(text):
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def _format_threshold_tag(value):
    return f"{float(value):.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Low-speed puck-wall bounce GIF sweep")
    parser.add_argument(
        "--config-path",
        type=str,
        default="scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/wall_bounce_sweeps",
    )
    parser.add_argument(
        "--min-normal-impulses",
        type=str,
        default="0.0,0.01,0.03,0.05",
        help="Comma-separated minimum normal impulse values for option 3 sweep",
    )
    parser.add_argument(
        "--speed-profiles",
        type=str,
        default="low,medium,high,super_high",
        help="Comma-separated speed profiles to render. Choices: low,medium,high,super_high",
    )
    parser.add_argument(
        "--launch-angle-deg",
        type=float,
        default=90.0,
        help="Initial puck launch angle in base frame degrees. 90=horizontal toward side wall, 45=diagonal.",
    )
    parser.add_argument(
        "--velocity-thresholds",
        type=str,
        default="",
        help="Comma-separated Box2D velocity threshold values for threshold sweep (empty disables sweep).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_cfg = _load_base_env_cfg(args.config_path)

    impulse_values = _parse_float_list(args.min_normal_impulses)
    velocity_threshold_values = _parse_float_list(args.velocity_thresholds)
    speed_profiles = [s.strip().lower() for s in args.speed_profiles.split(",") if s.strip()]
    valid_profiles = {"low", "medium", "high", "super_high"}
    for profile in speed_profiles:
        if profile not in valid_profiles:
            raise ValueError(
                f"Invalid speed profile '{profile}'. Choices are: {sorted(valid_profiles)}."
            )

    option3_dir = os.path.join(args.out_dir, "option3_min_normal_impulse")
    os.makedirs(option3_dir, exist_ok=True)

    angle_tag = f"angle_{args.launch_angle_deg:.0f}deg"

    summary_rows = []

    # Baseline (no option 3).
    baseline_cfg = copy.deepcopy(base_cfg)
    baseline_sim = baseline_cfg["simulator_params"]
    baseline_sim["enable_min_wall_rebound_impulse"] = False
    for speed_profile in speed_profiles:
        baseline_path = os.path.join(args.out_dir, f"baseline_{speed_profile}_{angle_tag}.gif")
        in_s, out_s = _capture_rollout_gif(
            baseline_cfg,
            baseline_path,
            f"baseline ({speed_profile}, {args.launch_angle_deg:.0f}deg, no option3)",
            speed_profile=speed_profile,
            launch_angle_deg=args.launch_angle_deg,
        )
        summary_rows.append(
            {
                "variant": f"baseline_{speed_profile}_{angle_tag}",
                "mode": "baseline",
                "speed_profile": speed_profile,
                "launch_angle_deg": float(args.launch_angle_deg),
                "box2d_velocity_threshold": baseline_sim.get("box2d_velocity_threshold", ""),
                "gif_path": baseline_path,
                "incoming_normal_speed": in_s,
                "outgoing_normal_speed": out_s,
            }
        )

    # Option 3: vary minimum normal impulse.
    for speed_profile in speed_profiles:
        speed_dir = os.path.join(option3_dir, f"{speed_profile}_{angle_tag}")
        os.makedirs(speed_dir, exist_ok=True)
        for impulse in impulse_values:
            cfg = copy.deepcopy(base_cfg)
            sim = cfg["simulator_params"]
            # Isolate option-3 sweeps from legacy wall-bounce impulse logic.
            sim["wall_bounce_scale"] = 0.0
            sim["enable_min_wall_rebound_impulse"] = impulse > 0.0
            sim["min_wall_rebound_impulse"] = float(impulse)
            sim["min_wall_rebound_max_pre_speed"] = 0.35

            out_path = os.path.join(speed_dir, f"option3_min_impulse_{impulse:.3f}.gif")
            label = (
                f"option3: {speed_profile}, {args.launch_angle_deg:.0f}deg, "
                f"min_normal_impulse={impulse:.3f}"
            )
            in_s, out_s = _capture_rollout_gif(
                cfg,
                out_path,
                label,
                speed_profile=speed_profile,
                launch_angle_deg=args.launch_angle_deg,
            )
            summary_rows.append(
                {
                    "variant": f"option3_{speed_profile}_{angle_tag}",
                    "mode": "option3_min_normal_impulse",
                    "speed_profile": speed_profile,
                    "launch_angle_deg": float(args.launch_angle_deg),
                    "box2d_velocity_threshold": sim.get("box2d_velocity_threshold", ""),
                    "gif_path": out_path,
                    "incoming_normal_speed": in_s,
                    "outgoing_normal_speed": out_s,
                }
            )

    if velocity_threshold_values:
        threshold_root = os.path.join(args.out_dir, "threshold_sweep")
        os.makedirs(threshold_root, exist_ok=True)
        for threshold in velocity_threshold_values:
            cfg = copy.deepcopy(base_cfg)
            sim = cfg["simulator_params"]
            sim["box2d_velocity_threshold"] = float(threshold)
            threshold_tag = _format_threshold_tag(threshold)
            threshold_dir = os.path.join(threshold_root, f"threshold_{threshold_tag}")
            os.makedirs(threshold_dir, exist_ok=True)
            for speed_profile in speed_profiles:
                out_path = os.path.join(threshold_dir, f"{speed_profile}_{angle_tag}.gif")
                label = (
                    f"threshold sweep: {speed_profile}, {args.launch_angle_deg:.0f}deg, "
                    f"vth={float(threshold):.3f}"
                )
                in_s, out_s = _capture_rollout_gif(
                    cfg,
                    out_path,
                    label,
                    speed_profile=speed_profile,
                    launch_angle_deg=args.launch_angle_deg,
                )
                summary_rows.append(
                    {
                        "variant": f"threshold_{threshold_tag}_{speed_profile}_{angle_tag}",
                        "mode": "velocity_threshold_sweep",
                        "speed_profile": speed_profile,
                        "launch_angle_deg": float(args.launch_angle_deg),
                        "box2d_velocity_threshold": float(threshold),
                        "gif_path": out_path,
                        "incoming_normal_speed": in_s,
                        "outgoing_normal_speed": out_s,
                    }
                )

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", encoding="ascii") as f:
        f.write(
            "variant,mode,speed_profile,launch_angle_deg,box2d_velocity_threshold,"
            "gif_path,incoming_normal_speed,outgoing_normal_speed\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['variant']},{row['mode']},{row['speed_profile']},{row['launch_angle_deg']},"
                f"{row['box2d_velocity_threshold']},{row['gif_path']},{row['incoming_normal_speed']},"
                f"{row['outgoing_normal_speed']}\n"
            )
    print(f"Wrote sweep GIFs to: {args.out_dir}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
