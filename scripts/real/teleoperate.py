from __future__ import annotations

import os
import sys

# Ensure project root is first on sys.path so the local `scripts/` package
# takes precedence over the ROS `scripts` package from /opt/ros.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from airhockey import AirHockeyEnv
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from scripts.smooth_policy.amp_history.amp_training.td3.helper.episode_artifacts import (
    save_split_episode_hdf5,
)


# ---------------------------------------------------------------------------
# Split-schema row builder (mirrors _build_split_episode_row in async_td3_real)
# ---------------------------------------------------------------------------

def _vector_with_width(values, width: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros((int(width),), dtype=np.float64)
    copy_width = min(int(width), int(vector.shape[0]))
    if copy_width > 0:
        out[:copy_width] = vector[:copy_width]
    return out


def _build_teleop_row(
    env: AirHockeyEnv,
    action_xy: np.ndarray,
    episode_id: int,
    episode_step_idx: int,
) -> Dict[str, np.ndarray]:
    """Build a split-schema row from the env's current state.

    Uses the same field layout as _build_split_episode_row in async_td3_real
    so that downstream tools (visualization, system-ID, replay_real_in_sim)
    work identically.
    """
    state_info = env.simulator.get_current_state()
    paddle = state_info["paddles"]["paddle_ego"]
    puck_info = state_info["pucks"][0]

    paddle_pos = np.asarray(paddle.get("position", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    paddle_vel = np.asarray(paddle.get("velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    move_lims = np.asarray(getattr(env.simulator, "move_lims", (1.0, 1.0)), dtype=np.float64).reshape(-1)
    if move_lims.shape[0] < 2:
        move_lims = np.array([1.0, 1.0], dtype=np.float64)
    desired_xy = paddle_pos[:2] + np.asarray(action_xy[:2], dtype=np.float64) * move_lims[:2]
    puck_position = np.asarray(puck_info.get("position", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    puck_occluded = float(np.asarray(puck_info.get("occluded", [0.0]), dtype=np.float64).reshape(-1)[0])

    pose = _vector_with_width(np.concatenate([paddle_pos[:2], np.zeros(4)]), 6)
    speed = _vector_with_width(np.concatenate([paddle_vel[:2], np.zeros(4)]), 6)
    force = np.zeros((6,), dtype=np.float64)
    acc = np.zeros((3,), dtype=np.float64)
    desired_pose = _vector_with_width(np.concatenate([desired_xy, np.zeros(4)]), 6)
    puck = _vector_with_width(np.concatenate([puck_position[:2], np.array([puck_occluded])]), 3)

    return {
        "cur_time": np.array([time.time()], dtype=np.float64),
        "tidx": np.array([float(episode_id)], dtype=np.float64),
        "i": np.array([float(episode_step_idx)], dtype=np.float64),
        "estop": np.array([0.0], dtype=np.float64),
        "safety": np.array([1.0], dtype=np.float64),
        "pose": pose,
        "speed": speed,
        "force": force,
        "acc": acc,
        "desired_pose": desired_pose,
        "puck": puck,
    }


# ---------------------------------------------------------------------------
# Episode ID helper (same logic as async_td3_real)
# ---------------------------------------------------------------------------

def _next_episode_id(output_dir: str | Path) -> int:
    artifact_dir = Path(output_dir).expanduser().resolve()
    if not artifact_dir.exists():
        return 0
    max_seen = -1
    pattern = re.compile(r"^trajectory_data(\d+)\.hdf5$")
    for path in artifact_dir.rglob("trajectory_data*.hdf5"):
        match = pattern.match(path.name)
        if match is not None:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


# ---------------------------------------------------------------------------
# Main teleop loop
# ---------------------------------------------------------------------------

def run_teleop(air_hockey_cfg, use_split_schema: bool = True, policy_limits: bool = False):
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']

    if 'sac' == air_hockey_cfg['algorithm']:
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = True
        else:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    else:
        air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp['seed'] = 42
    air_hockey_params_cp['max_timesteps'] = 200

    eval_env = AirHockeyEnv(air_hockey_params_cp)
    sp = eval_env.simulator.save_path or ''
    abs_sp = os.path.abspath(os.path.expanduser(sp)) if sp else '(empty — trajectories may not be saved)'
    ts = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    schema_label = "split-schema" if use_split_schema else "legacy flat train_vals"
    print(f"[teleoperate] {ts}  Active trajectory directory: {abs_sp}")
    print(f"[teleoperate] Recording format: {schema_label}")
    if policy_limits:
        scale = getattr(eval_env.simulator, 'mouse_action_scale', None)
        print(f"[teleoperate] Policy-limits mode: action_scale={scale}")
    print("[teleoperate] On each 'y', the previous rollout is written as trajectory_data<N>.hdf5 in that directory.")

    episode_rows: List[Dict[str, np.ndarray]] = []
    episode_id = 0
    step_idx = 0

    if use_split_schema and sp:
        episode_id = _next_episode_id(sp)

    action = np.array([0.0, 0.0])

    with NonBlockingConsole() as nbc:
        print("Press 'y' to collect data (write trajectory), 'q' to reset without saving, 'x' to exit")
        while True:
            eval_env.step(action)

            if use_split_schema:
                if policy_limits:
                    recorded_action = getattr(eval_env.simulator, '_last_teleop_policy_action', action)
                else:
                    recorded_action = action
                row = _build_teleop_row(eval_env, recorded_action, episode_id, step_idx)
                episode_rows.append(row)
                step_idx += 1

            key = nbc.get_data()
            if key:
                print(f"Key pressed: {repr(key)}")
                if key == 'y':
                    if use_split_schema and episode_rows and sp:
                        out_dir = Path(sp).expanduser().resolve()
                        artifact_path = save_split_episode_hdf5(
                            output_dir=out_dir,
                            episode_id=episode_id,
                            episode_rows=episode_rows,
                        )
                        print(f"[teleoperate] Wrote {len(episode_rows)} steps -> {artifact_path}")
                        episode_id += 1
                        episode_rows = []
                        step_idx = 0
                        eval_env.reset(seed=None, write_traj=False)
                    else:
                        eval_env.reset(seed=None, write_traj=True)
                        episode_rows = []
                        step_idx = 0
                elif key == 'q':
                    print("Resetting without saving trajectory")
                    eval_env.reset(seed=None, write_traj=False)
                    episode_rows = []
                    step_idx = 0
                elif key == 'x':
                    print("Exiting...")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleoperate the air hockey robot and record trajectories.')
    parser.add_argument('--cfg', type=str, default='configs/real_configs/mouse_config.yaml', help='Path to the configuration file.')
    parser.add_argument(
        '--save-path', '-o', type=str, default=None,
        help='Override air_hockey.simulator_params.save_path (trajectory directory). '
             'Tilde and relative paths are expanded to an absolute path.',
    )
    parser.add_argument(
        '--legacy-schema', action='store_true',
        help='Use the legacy flat train_vals HDF5 format instead of split-schema.',
    )
    parser.add_argument(
        '--policy-limits', action='store_true',
        help='Constrain mouse movement to the same per-step magnitude as the policy '
             '(action_scale * move_lims) and record actions in policy convention.',
    )
    parser.add_argument(
        '--action-scale', type=float, default=1.0,
        help='Action scale used to clamp the per-step displacement '
             '(default 1.0, matching td3_online.yaml). Only effective with --policy-limits.',
    )
    args = parser.parse_args()

    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'configs/baseline_configs/paddle_pos_neg_regions_real_preset.yaml')
    else:
        air_hockey_cfg_fp = args.cfg

    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)

    if args.save_path is not None:
        air_hockey_cfg['air_hockey']['simulator_params']['save_path'] = os.path.abspath(
            os.path.expanduser(args.save_path)
        )

    if args.policy_limits:
        air_hockey_cfg['air_hockey']['simulator_params']['mouse_action_scale'] = args.action_scale

    raw_sp = air_hockey_cfg['air_hockey']['simulator_params'].get('save_path', '')
    abs_cfg_sp = os.path.abspath(os.path.expanduser(str(raw_sp))) if raw_sp else ''
    ts0 = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f"[teleoperate] {ts0}  Config trajectory directory (before env init): {abs_cfg_sp or '(not set)'}")

    run_teleop(
        air_hockey_cfg,
        use_split_schema=not args.legacy_schema,
        policy_limits=args.policy_limits,
    )
