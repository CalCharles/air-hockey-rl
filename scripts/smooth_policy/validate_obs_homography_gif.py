import argparse
import copy
import os

import cv2
import imageio
import numpy as np
import yaml

from airhockey import AirHockeyEnv
from airhockey.observation_homography import pixel_homography_from_world_homography
from airhockey.renderers import AirHockeyRenderer


def _annotate_reward(frame_rgb, reward, cumulative_reward):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    line_type = 2
    text_position = (frame_rgb.shape[1] - 150, 30)
    cv2.putText(
        frame_rgb,
        f"Reward: {reward:.2f}",
        text_position,
        font,
        font_scale,
        font_color,
        line_type,
    )
    text_position = (frame_rgb.shape[1] - 150, 60)
    cv2.putText(
        frame_rgb,
        f"Return: {cumulative_reward:.2f}",
        text_position,
        font,
        font_scale,
        font_color,
        line_type,
    )
    return frame_rgb


def _resize_for_gif(frame_rgb, width=160):
    aspect_ratio = frame_rgb.shape[1] / frame_rgb.shape[0]
    return cv2.resize(frame_rgb, (width, int(width / aspect_ratio)))


def main():
    parser = argparse.ArgumentParser(
        description="Render Box2D frames and warp them with observation homography."
    )
    parser.add_argument(
        "--env-config-path",
        type=str,
        default=(
            "scripts/smooth_policy/amp_history/configs/new_juggle/"
            "pid_noise_constant_upper_half_custom_sim_params.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=str, default="runs/obs_homography_viz")
    parser.add_argument("--output-name", type=str, default="eval_0.gif")
    parser.add_argument("--n-eps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--matrix",
        type=float,
        nargs=9,
        default=None,
        help="Optional 9-value row-major world homography matrix.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.env_config_path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    env_params = copy.deepcopy(config["air_hockey"])
    if env_params.get("simulator") != "box2d":
        raise ValueError("validate_obs_homography_gif.py currently supports simulator=box2d only.")
    env_params["seed"] = int(args.seed)
    simulator_params = env_params.setdefault("simulator_params", {})
    simulator_params["enable_obs_position_homography"] = True
    simulator_params["obs_position_homography_seed"] = int(args.seed)
    if args.matrix is not None:
        simulator_params["obs_position_homography_matrix"] = list(args.matrix)

    env = AirHockeyEnv(env_params)
    renderer = AirHockeyRenderer(env, orientation="vertical")
    homography_world = getattr(env.simulator, "obs_position_homography", None)
    if homography_world is None:
        raise RuntimeError("Observation homography is not available on simulator.")
    pixel_homography = pixel_homography_from_world_homography(homography_world, env, renderer)

    frames = []
    for _ in range(int(args.n_eps)):
        obs, _ = env.reset(seed=int(args.seed))
        del obs  # observation is not used in this visualization.
        cumulative_reward = 0.0
        reward = 0.0
        for _step in range(int(args.max_steps)):
            bgr_frame = renderer.get_frame()
            warped_bgr = cv2.warpPerspective(
                bgr_frame,
                pixel_homography,
                (bgr_frame.shape[1], bgr_frame.shape[0]),
            )
            rgb_frame = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
            rgb_frame = _resize_for_gif(rgb_frame, width=160)
            rgb_frame = _annotate_reward(rgb_frame, reward, cumulative_reward)
            frames.append(rgb_frame)
            action = np.zeros(2, dtype=np.float32)
            _, reward, terminated, truncated, _info = env.step(action)
            cumulative_reward += float(reward)
            if terminated or truncated:
                break

    output_path = os.path.join(output_dir, args.output_name)
    fps = 20
    imageio.mimsave(
        output_path,
        frames,
        format="GIF",
        loop=0,
        duration=int(1000 / fps),
    )
    print(f"Saved homography-warped GIF to: {output_path}")


if __name__ == "__main__":
    main()
