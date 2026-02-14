import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from airhockey import AirHockeyEnv
from airhockey.sims.real.multiprocessing import NonBlockingConsole


# train_vals layout from scripts/real/README.md
IDX_POSE_X = 5
IDX_POSE_Y = 6
IDX_DESIRED_X = 26
IDX_DESIRED_Y = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imitate a saved real-robot demonstration trajectory."
    )
    parser.add_argument(
        "--demo-hdf5",
        type=str,
        required=True,
        help="Path to trajectory_data*.hdf5 containing train_vals.",
    )
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
        help="Override trajectory save path in simulator config.",
    )
    parser.add_argument(
        "--start-tol",
        type=float,
        default=0.01,
        help="Distance tolerance (m) to consider start alignment complete.",
    )
    parser.add_argument(
        "--final-tol",
        type=float,
        default=0.001,
        help="Distance tolerance (m) to consider final state reached.",
    )
    parser.add_argument(
        "--max-align-steps",
        type=int,
        default=300,
        help="Max control steps spent aligning to demo start.",
    )
    parser.add_argument(
        "--max-imitation-steps",
        type=int,
        default=1200,
        help="Max control steps spent in nearest-neighbor imitation.",
    )
    parser.add_argument(
        "--backtrack-window",
        type=int,
        default=4,
        help="Allow nearest-index lookup this many frames behind latest progress.",
    )
    parser.add_argument(
        "--lookahead-steps",
        type=int,
        default=2,
        help="Number of demo positions to target ahead of nearest index.",
    )
    parser.add_argument(
        "--nearest-recovery-threshold",
        type=float,
        default=0.02,
        help="If farther than this (m) from nearest demo pose, target nearest pose first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load demo and print reconstructed actions; do not step environment.",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=10,
        help="Number of reconstructed actions to print when --dry-run is used.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step imitation diagnostics.",
    )
    return parser.parse_args()


def load_rollout_config(config_path: str, save_path_override: str = None) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    air_hockey_params = cfg["air_hockey"]
    air_hockey_params["n_training_steps"] = cfg["n_training_steps"]

    if cfg["algorithm"] == "sac" and "goal" in cfg["air_hockey"]["task"]:
        cfg["air_hockey"]["return_goal_obs"] = True
    else:
        cfg["air_hockey"]["return_goal_obs"] = False

    params = air_hockey_params.copy()
    params["seed"] = 42
    if save_path_override is not None:
        params["simulator_params"]["save_path"] = save_path_override
    return params


def build_eval_env(
    config_path: str,
    min_timesteps: int,
    save_path_override: str = None,
) -> AirHockeyEnv:
    params = load_rollout_config(config_path, save_path_override=save_path_override)
    params["max_timesteps"] = max(min_timesteps + 5, params.get("max_timesteps", 0))
    return AirHockeyEnv(params)


def load_demo_vals(demo_hdf5: str) -> np.ndarray:
    demo_path = Path(demo_hdf5).expanduser().resolve()
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    with h5py.File(demo_path, "r") as f:
        if "train_vals" not in f:
            raise ValueError(f"{demo_path} does not contain 'train_vals'")
        vals = f["train_vals"][:]
    if vals.ndim != 2 or vals.shape[1] < 35:
        raise ValueError(f"Unexpected train_vals shape: {vals.shape}")
    if vals.shape[0] < 2:
        raise ValueError("Demonstration must contain at least 2 timesteps.")
    return vals


def extract_demo_pose_and_actions(
    vals: np.ndarray,
    rmax_x: float,
    rmax_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    if rmax_x <= 0 or rmax_y <= 0:
        raise ValueError(f"Invalid rmax values: rmax_x={rmax_x}, rmax_y={rmax_y}")

    demo_pose_xy = vals[:, [IDX_POSE_X, IDX_POSE_Y]].astype(np.float32)
    desired_xy = vals[:, [IDX_DESIRED_X, IDX_DESIRED_Y]].astype(np.float32)
    scale = np.array([rmax_x, rmax_y], dtype=np.float32)
    demo_actions = (desired_xy - demo_pose_xy) / scale
    demo_actions = np.clip(demo_actions, -1.0, 1.0)
    return demo_pose_xy, demo_actions.astype(np.float32)


def target_position_to_action(
    current_xy: np.ndarray,
    target_xy: np.ndarray,
    rmax_x: float,
    rmax_y: float,
) -> np.ndarray:
    scale = np.array([rmax_x, rmax_y], dtype=np.float32)
    action = (target_xy - current_xy) / scale
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def get_live_pose_xy(env: AirHockeyEnv) -> np.ndarray:
    state = env.simulator.get_current_state()
    # get_current_state() returns table-frame paddle x (offset added). Convert
    # back to robot frame so it matches demo pose/desired_pose from train_vals.
    live_xy = np.array(state["paddles"]["paddle_ego"]["position"], dtype=np.float32)
    x_offset = float(getattr(env.simulator, "x_offset", 0.0))
    live_xy[0] -= x_offset
    return live_xy


def get_rmax_from_config(config_path: str) -> tuple[float, float]:
    params = load_rollout_config(config_path)
    sim_params = params.get("simulator_params", {})
    rmax_x = float(sim_params.get("rmax_x", 0.26))
    rmax_y = float(sim_params.get("rmax_y", 0.12))
    return rmax_x, rmax_y


def run_dry_preview(demo_actions: np.ndarray, demo_pose_xy: np.ndarray, preview_steps: int) -> None:
    n = min(preview_steps, demo_actions.shape[0])
    print(f"Dry-run preview: showing first {n} reconstructed actions")
    for i in range(n):
        print(
            f"idx={i:04d} pose=({demo_pose_xy[i, 0]: .4f}, {demo_pose_xy[i, 1]: .4f}) "
            f"action=({demo_actions[i, 0]: .4f}, {demo_actions[i, 1]: .4f})"
        )


def run_imitation(
    env: AirHockeyEnv,
    demo_pose_xy: np.ndarray,
    rmax_x: float,
    rmax_y: float,
    start_tol: float,
    final_tol: float,
    max_align_steps: int,
    max_imitation_steps: int,
    backtrack_window: int,
    lookahead_steps: int,
    nearest_recovery_threshold: float,
    verbose: bool,
) -> None:
    start_target = demo_pose_xy[0]
    final_target = demo_pose_xy[-1]
    upward_stop_x = -0.8
    final_stage_progress_ratio = 0.9
    upward_direction_ratio = 0.9
    final_stage_action_boost = 2.0
    post_stop_upward_steps = 7
    post_stop_upward_action_sequence = [
        np.array([-0.5, 0.0], dtype=np.float32),
        np.array([-1, 0.0], dtype=np.float32),
        np.array([-1, 0.0], dtype=np.float32),
    ]

    def run_post_stop_upward_burst(reason: str) -> None:
        simulator = env.simulator
        original_move_lims = tuple(simulator.move_lims)
        original_lims = tuple(simulator.lims)
        original_edge_lims = tuple(simulator.edge_lims)
        burst_move_lims = (1.0, 1.0) # 1-1 scale
        burst_lims = (-10.0, 10.0, -10.0, 10.0)
        burst_edge_lims = (0.0, 0.0, 1e6, 1e6)
        print(
            f"{reason} Executing {post_stop_upward_steps} additional upward steps "
        )
        if verbose:
            print(
                "Temporarily overriding simulator limits for burst: "
                f"move_lims={burst_move_lims}, lims={burst_lims}, edge_lims={burst_edge_lims}"
            )
        try:
            simulator.move_lims = burst_move_lims
            simulator.lims = burst_lims
            simulator.edge_lims = burst_edge_lims
            for burst_step in range(post_stop_upward_steps):
                if burst_step < len(post_stop_upward_action_sequence):
                    burst_action = post_stop_upward_action_sequence[burst_step]
                else:
                    burst_action = post_stop_upward_action_sequence[-1]
                env.step(burst_action)
        finally:
            simulator.move_lims = original_move_lims
            simulator.lims = original_lims
            simulator.edge_lims = original_edge_lims
            if verbose:
                print(
                    "Restored simulator limits after burst: "
                    f"move_lims={original_move_lims}, lims={original_lims}, edge_lims={original_edge_lims}"
                )

    print("Phase A: align to demo start position.")
    aligned = False
    align_steps = 0

    with NonBlockingConsole() as nbc:
        for step in range(max_align_steps):
            key = nbc.get_data()
            if key == "x":
                print("Emergency exit key received during alignment.")
                return

            live_xy = get_live_pose_xy(env)
            dist_to_start = float(np.linalg.norm(live_xy - start_target))
            if dist_to_start <= start_tol:
                aligned = True
                align_steps = step
                print(f"Aligned to demo start in {step} steps (dist={dist_to_start:.4f}m).")
                break

            action = target_position_to_action(live_xy, start_target, rmax_x, rmax_y)
            env.step(action)

            if verbose and step % 5 == 0:
                print(
                    f"[align] step={step:04d} "
                    f"live=({live_xy[0]: .4f}, {live_xy[1]: .4f}) "
                    f"start_target=({start_target[0]: .4f}, {start_target[1]: .4f}) "
                    f"dist_to_start={dist_to_start:.4f} "
                    f"action=({action[0]: .4f}, {action[1]: .4f})"
                )

        if not aligned:
            live_xy = get_live_pose_xy(env)
            dist_to_start = float(np.linalg.norm(live_xy - start_target))
            align_steps = max_align_steps
            print(
                f"Alignment step limit reached ({max_align_steps}). "
                f"Continuing imitation from dist={dist_to_start:.4f}m."
            )

        print("Phase B: position-target imitation with lookahead.")
        terminal_idx = demo_pose_xy.shape[0] - 1
        last_progress_idx = 0
        lookahead_steps = max(0, int(lookahead_steps))
        nearest_recovery_threshold = max(0.0, float(nearest_recovery_threshold))
        has_seen_mostly_upward_motion = False

        for step in range(max_imitation_steps):
            key = nbc.get_data()
            if key == "x":
                print("Emergency exit key received during imitation.")
                return

            live_xy = get_live_pose_xy(env)
            dist_to_final = float(np.linalg.norm(live_xy - final_target))

            if dist_to_final <= final_tol:
                run_post_stop_upward_burst(
                    f"Reached final pose tolerance in {step} imitation steps "
                    f"(dist={dist_to_final:.4f}m)."
                )
                return

            search_start = max(0, last_progress_idx - max(0, backtrack_window))
            search_pose = demo_pose_xy[search_start:]
            nearest_local_idx = int(np.argmin(np.linalg.norm(search_pose - live_xy, axis=1)))
            nearest_idx = search_start + nearest_local_idx
            last_progress_idx = max(last_progress_idx, nearest_idx)
            nearest_pose = demo_pose_xy[nearest_idx]
            nearest_dist = float(np.linalg.norm(nearest_pose - live_xy))

            if nearest_idx >= terminal_idx:
                run_post_stop_upward_burst(
                    f"Nearest demo index reached terminal frame ({terminal_idx})."
                )
                return

            # Normal case: aim a few steps ahead of the closest demo pose.
            # Recovery case: if we drift too far from the nearest pose, first
            # drive back to that nearest pose before resuming lookahead.
            if nearest_dist > nearest_recovery_threshold:
                target_idx = nearest_idx
                target_mode = "recover_nearest"
            else:
                target_idx = min(nearest_idx + lookahead_steps, terminal_idx)
                target_mode = "lookahead"
            target_pose = demo_pose_xy[target_idx]
            action = target_position_to_action(live_xy, target_pose, rmax_x, rmax_y)
            target_delta = target_pose - live_xy
            target_delta_norm = float(np.linalg.norm(target_delta))
            if target_delta_norm > 1e-8:
                upward_alignment = float(target_delta[0] / target_delta_norm)
            else:
                upward_alignment = 0.0
            if upward_alignment <= -upward_direction_ratio:
                has_seen_mostly_upward_motion = True
            in_final_stage = target_idx >= int(final_stage_progress_ratio * terminal_idx)
            should_boost = in_final_stage and upward_alignment <= -upward_direction_ratio
            if should_boost:
                action = np.clip(action * final_stage_action_boost, -1.0, 1.0).astype(np.float32)
            if (
                step >= 50
                and has_seen_mostly_upward_motion
                and live_xy[0] < upward_stop_x
            ):
                run_post_stop_upward_burst(
                    f"Reached upward stop threshold in x (x={live_xy[0]:.4f} < {upward_stop_x:.4f}) "
                    f"after mostly upward motion at step {step}."
                )
                return
            env.step(action)

            if verbose and step % 5 == 0:
                print(
                    f"[imit] step={step:04d} nearest_idx={nearest_idx:04d} "
                    f"live=({live_xy[0]: .4f}, {live_xy[1]: .4f}) "
                    f"nearest_demo_pose=({nearest_pose[0]: .4f}, {nearest_pose[1]: .4f}) "
                    f"target_idx={target_idx:04d} target_mode={target_mode} "
                    f"target_pose=({target_pose[0]: .4f}, {target_pose[1]: .4f}) "
                    f"final_target=({final_target[0]: .4f}, {final_target[1]: .4f}) "
                    f"nearest_dist={nearest_dist:.4f} dist_to_final={dist_to_final:.4f} "
                    f"upward_alignment={upward_alignment:.4f} boosted={should_boost} "
                    f"action=({action[0]: .4f}, {action[1]: .4f})"
                )

        print(
            f"Imitation step limit reached ({max_imitation_steps}) without terminal hit. "
            "Stopping."
        )
        return


def main() -> None:
    args = parse_args()
    vals = load_demo_vals(args.demo_hdf5)

    if args.dry_run:
        rmax_x, rmax_y = get_rmax_from_config(args.config_path)
        demo_pose_xy, demo_actions = extract_demo_pose_and_actions(vals, rmax_x=rmax_x, rmax_y=rmax_y)
        run_dry_preview(demo_actions, demo_pose_xy, preview_steps=args.dry_run_steps)
        return

    min_steps = args.max_align_steps + args.max_imitation_steps + 5
    env = build_eval_env(
        args.config_path,
        min_timesteps=min_steps,
        save_path_override=args.save_path,
    )

    simulator = env.simulator
    rmax_x = float(getattr(simulator, "rmax_x", 0.26))
    rmax_y = float(getattr(simulator, "rmax_y", 0.12))
    demo_pose_xy, _ = extract_demo_pose_and_actions(vals, rmax_x=rmax_x, rmax_y=rmax_y)

    print(f"Trajectory save path: {simulator.save_path}")
    print(f"Loaded demo frames: {demo_pose_xy.shape[0]}")
    print(f"Using rmax_x={rmax_x:.4f}, rmax_y={rmax_y:.4f}")
    print("Press 'x' any time for emergency stop.")

    run_imitation(
        env=env,
        demo_pose_xy=demo_pose_xy,
        rmax_x=rmax_x,
        rmax_y=rmax_y,
        start_tol=args.start_tol,
        final_tol=args.final_tol,
        max_align_steps=args.max_align_steps,
        max_imitation_steps=args.max_imitation_steps,
        backtrack_window=args.backtrack_window,
        lookahead_steps=args.lookahead_steps,
        nearest_recovery_threshold=args.nearest_recovery_threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
