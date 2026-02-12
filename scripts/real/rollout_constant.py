import argparse
import numpy as np
import yaml
from pathlib import Path

from airhockey import AirHockeyEnv
from airhockey.sims.real.trajectory_merging import (
    clear_images,
    merge_trajectory,
    write_trajectory,
)


def build_eval_env(config_path: str, timesteps: int, save_path_override: str = None) -> AirHockeyEnv:
    with open(config_path, "r") as f:
        air_hockey_cfg = yaml.safe_load(f)

    air_hockey_params = air_hockey_cfg["air_hockey"]
    air_hockey_params["n_training_steps"] = air_hockey_cfg["n_training_steps"]

    if air_hockey_cfg["algorithm"] == "sac" and "goal" in air_hockey_cfg["air_hockey"]["task"]:
        air_hockey_cfg["air_hockey"]["return_goal_obs"] = True
    else:
        air_hockey_cfg["air_hockey"]["return_goal_obs"] = False

    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp["seed"] = 42
    # Avoid episode truncation before the requested rollout length.
    air_hockey_params_cp["max_timesteps"] = max(timesteps + 5, air_hockey_params_cp.get("max_timesteps", 0))
    if save_path_override is not None:
        air_hockey_params_cp["simulator_params"]["save_path"] = save_path_override

    return AirHockeyEnv(air_hockey_params_cp)


def write_current_trajectory(env: AirHockeyEnv) -> bool:
    simulator = env.simulator
    imgs, vals = merge_trajectory(simulator.image_path, simulator.images, simulator.vals)
    clear_images(folder=simulator.image_path)

    if imgs is None:
        return False

    write_trajectory(simulator.save_path, simulator.tidx, imgs, vals)
    simulator.tidx += 1
    simulator.images = []
    simulator.vals = []
    return True


def maybe_generate_gifs_for_saved_trajectory(
    env: AirHockeyEnv,
    auto_gif: bool,
    gif_fps: int,
    gif_max_frames_per_file: int,
) -> None:
    if not auto_gif:
        return

    simulator = env.simulator
    saved_idx = simulator.tidx - 1
    if saved_idx < 0:
        print("No saved trajectory index found for GIF generation.")
        return

    hdf5_path = (Path(simulator.save_path) / f"trajectory_data{saved_idx}.hdf5").resolve()
    if not hdf5_path.exists():
        print(f"Skipping GIF generation; missing file: {hdf5_path}")
        return

    output_dir = hdf5_path.parent / f"{hdf5_path.stem}_gifs"
    try:
        from visualize_saved_trajectory import generate_gifs_from_hdf5
    except ModuleNotFoundError as exc:
        print(f"Skipping GIF generation; dependency missing: {exc}")
        return

    try:
        outputs = generate_gifs_from_hdf5(
            input_hdf5=hdf5_path,
            output_dir=output_dir,
            fps=gif_fps,
            max_frames_per_gif=gif_max_frames_per_file,
        )
    except Exception as exc:
        print(f"GIF generation failed for {hdf5_path}: {exc}")
        return

    if outputs:
        print(f"Generated {len(outputs)} GIF(s) in {output_dir}")
    else:
        print(f"No GIF frames produced for {hdf5_path}")


def run_constant_rollout(env: AirHockeyEnv, action: np.ndarray, timesteps: int) -> int:
    executed_steps = 0
    for t in range(timesteps):
        _, _, is_finished, truncated, _ = env.step(action)
        executed_steps += 1
        # if is_finished or truncated:
        #     print(f"Environment ended early at step {t + 1}. Stopping rollout.")
        #     break
    return executed_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a constant-action rollout and auto-save trajectory data.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/real_configs/rollout_config.yaml",
        help="Path to rollout config YAML.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Override trajectory save path (defaults to config value).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200,
        help="Number of environment timesteps to run.",
    )
    parser.add_argument(
        "--action",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("AX", "AY"),
        help="Constant action to apply at each timestep.",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Clip constant action into [-1, 1] before rollout.",
    )
    parser.add_argument(
        "--auto-gif",
        action="store_true",
        help="Generate GIF visualization(s) automatically after saving trajectory.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=20,
        help="GIF playback FPS used when --auto-gif is enabled.",
    )
    parser.add_argument(
        "--gif-max-frames-per-file",
        type=int,
        default=250,
        help="Maximum rendered frames per GIF file when --auto-gif is enabled.",
    )
    args = parser.parse_args()

    action = np.array(args.action, dtype=np.float32)
    if args.clip:
        action = np.clip(action, -1.0, 1.0)

    print(f"Starting rollout immediately for {args.timesteps} timesteps with action {action}.")
    env = build_eval_env(args.config_path, args.timesteps, save_path_override=args.save_path)
    print(f"Trajectory save path: {env.simulator.save_path}")

    # Clear stale temporary images so only this rollout is written.
    clear_images(folder=env.simulator.image_path)
    env.simulator.images = []
    env.simulator.vals = []

    steps = run_constant_rollout(env, action, args.timesteps)
    saved = write_current_trajectory(env)

    if saved:
        print(f"Saved trajectory index {env.simulator.tidx - 1} after {steps} steps.")
        maybe_generate_gifs_for_saved_trajectory(
            env=env,
            auto_gif=args.auto_gif,
            gif_fps=args.gif_fps,
            gif_max_frames_per_file=args.gif_max_frames_per_file,
        )
    else:
        print(f"No trajectory data saved after {steps} steps.")
