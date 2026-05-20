"""Warm-start replay ingestion from split real trajectory HDF5 files.

See notes/docs/environments/real-world/episode-lifecycle.md for the warm-start flow.
"""

from __future__ import annotations

import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from airhockey import AirHockeyEnv
from scripts.td3.helper.real_stop_state import (
    _stop_state_from_saved_row,
)
from scripts.td3.helper.shared_replay import SharedTD3Replay
from scripts.td3.helper.td3_episode_collection import (
    EpisodeTrajectory,
)
from scripts.visualization.visualize_real_trajectory_split import (
    load_split_optional_data,
    load_split_trajectory_data,
)

_TRAIN_VALS_CUR_TIME = 0
_TRAIN_VALS_STEP_INDEX = 2
_TRAIN_VALS_POSE = slice(5, 11)
_TRAIN_VALS_SPEED = slice(11, 17)
_TRAIN_VALS_FORCE = slice(17, 23)
_TRAIN_VALS_ACC = slice(23, 26)
_TRAIN_VALS_DESIRED_POSE = slice(26, 32)
_TRAIN_VALS_PUCK = slice(32, 35)


def _list_warm_start_hdf5_files(
    input_roots: Sequence[str],
    *,
    recursive: bool,
    rng: np.random.Generator | None = None,
) -> list[Path]:
    if rng is None:
        rng = np.random.default_rng()
    seen_paths: set[Path] = set()
    shuffled_buckets: list[list[Path]] = []
    for input_root in input_roots:
        root_str = str(input_root).strip()
        if not root_str:
            continue
        root = Path(root_str).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Warm-start path does not exist: {root}")
        if root.is_file():
            if root.suffix.lower() != ".hdf5":
                raise ValueError(f"Warm-start file must end with .hdf5: {root}")
            if root not in seen_paths:
                seen_paths.add(root)
                shuffled_buckets.append([root])
            continue
        iterator = root.rglob("*.hdf5") if recursive else root.glob("*.hdf5")
        bucket: list[Path] = []
        for path in iterator:
            if path.is_file():
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                bucket.append(resolved)
        if bucket:
            rng.shuffle(bucket)
            shuffled_buckets.append(bucket)

    if not shuffled_buckets:
        return []

    # Interleave folder/file buckets so warm-start ingestion is mixed across sources.
    mixed_paths: list[Path] = []
    active_bucket_indices = [idx for idx, bucket in enumerate(shuffled_buckets) if bucket]
    while active_bucket_indices:
        visit_order = list(active_bucket_indices)
        rng.shuffle(visit_order)
        next_active_indices: list[int] = []
        for bucket_idx in visit_order:
            bucket = shuffled_buckets[bucket_idx]
            if not bucket:
                continue
            mixed_paths.append(bucket.pop())
            if bucket:
                next_active_indices.append(bucket_idx)
        active_bucket_indices = next_active_indices
    return mixed_paths


def _estimate_xy_derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 2:
        raise ValueError(f"Expected rank-2 derivative input, got shape {values_arr.shape}")
    derivative = np.zeros_like(values_arr, dtype=np.float64)
    if values_arr.shape[0] <= 1:
        return derivative

    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if times_arr.shape[0] != values_arr.shape[0]:
        raise ValueError(
            f"Expected one timestamp per row, got {times_arr.shape[0]} timestamps and "
            f"{values_arr.shape[0]} values"
        )

    for idx in range(1, values_arr.shape[0]):
        dt = float(times_arr[idx] - times_arr[idx - 1])
        if (not np.isfinite(dt)) or abs(dt) < 1e-6:
            dt = 1.0
        derivative[idx] = (values_arr[idx] - values_arr[idx - 1]) / dt
    derivative[0] = derivative[1]
    return derivative


def _padded_history_window(
    entries: Sequence[tuple[float, float, float]],
    *,
    default_entry: tuple[float, float, float],
    window_size: int = 5,
) -> list[tuple[float, float, float]]:
    if window_size <= 0:
        return []
    entries_list = list(entries)
    if len(entries_list) >= window_size:
        return entries_list[-window_size:]
    if not entries_list:
        return [default_entry for _ in range(window_size)]
    pad_entry = entries_list[0]
    return [pad_entry for _ in range(window_size - len(entries_list))] + entries_list


def _build_split_state_info(
    *,
    paddle_position_xy: np.ndarray,
    paddle_velocity_xy: np.ndarray,
    paddle_acceleration_xy: np.ndarray,
    paddle_force_xy: np.ndarray,
    paddle_jerk_xy: np.ndarray,
    puck_position_xy: np.ndarray,
    puck_velocity_xy: np.ndarray,
    puck_occluded: float,
    puck_history_entries: Sequence[tuple[float, float, float]],
) -> dict:
    return {
        "paddles": {
            "paddle_ego": {
                "position": np.asarray(paddle_position_xy, dtype=np.float64).copy(),
                "velocity": np.asarray(paddle_velocity_xy, dtype=np.float64).copy(),
                "acceleration": np.asarray(paddle_acceleration_xy, dtype=np.float64).copy(),
                "force": np.asarray(paddle_force_xy, dtype=np.float64).copy(),
                "jerk": np.asarray(paddle_jerk_xy, dtype=np.float64).copy(),
            }
        },
        "pucks": [
            {
                "position": np.asarray(puck_position_xy, dtype=np.float64).copy(),
                "velocity": np.asarray(puck_velocity_xy, dtype=np.float64).copy(),
                "occluded": np.array([float(puck_occluded)], dtype=np.float64),
                "history": [
                    (float(x), float(y), float(occluded))
                    for x, y, occluded in puck_history_entries
                ],
            }
        ],
    }


def _reset_warm_start_env_state(env: AirHockeyEnv, first_state: dict) -> None:
    env.current_state = first_state
    env.old_state = first_state
    env.current_timestep = 0
    env.success_in_ep = False
    env.max_reward_in_single_step = -np.inf
    env.min_reward_in_single_step = np.inf
    env.episode_return = 0.0
    env.episode_length = 0
    env._last_done_reasons = {"terminated": [], "truncated": []}
    env._puck_pass_paddle_score = 0
    if "pucks" in first_state and len(first_state["pucks"]) > 0:
        env.puck_initial_position = np.asarray(
            first_state["pucks"][0]["position"], dtype=np.float64
        ).copy()


def _recompute_warm_start_rewards(
    args: Any,
    env: AirHockeyEnv,
    *,
    prev_state: dict,
    next_state: dict,
    step_index: int,
    stop_state: Any,
    is_last_transition: bool,
) -> tuple[float, float]:
    del args, stop_state
    env.current_timestep = int(max(step_index, 0))
    env.old_state = prev_state
    env.current_state = next_state

    terminations, truncations, _, _, _, _ = env.has_finished(next_state)
    if not truncations:
        task_reward, success = env.get_base_reward(next_state)
        task_reward = float(task_reward) * float(env.base_reward_scaling)
        if (not env.success_in_ep) and bool(success):
            env.success_in_ep = True
    else:
        task_reward = float(env.truncate_rew)

    if env.enable_survival_bonus and (not terminations) and (not truncations):
        task_reward += float(env.survival_bonus_per_step)

    # E-stops are stored as truncations: `done=0` so the learner bootstraps
    # from V(s'). Only true env terminations or the final warm-start
    # transition mark the trajectory boundary. See
    # notes/docs/environments/real-world/episode-lifecycle.md.
    terminations_only = float(1.0 if (terminations or is_last_transition) else 0.0)
    return float(task_reward), terminations_only


def _load_warm_start_episode(
    episode_hdf5_path: Path,
    *,
    args: Any,
    env: AirHockeyEnv,
) -> EpisodeTrajectory:
    train_vals = np.asarray(load_split_trajectory_data(str(episode_hdf5_path)), dtype=np.float64)
    optional_data = load_split_optional_data(str(episode_hdf5_path))
    if train_vals.ndim != 2 or train_vals.shape[0] < 2:
        return EpisodeTrajectory.empty()

    if int(getattr(env, "num_pucks", 1)) != 1 or int(getattr(env, "num_paddles", 1)) != 1:
        raise NotImplementedError("Warm-start HDF5 loading currently supports one puck and one paddle.")
    if int(getattr(env, "num_blocks", 0)) != 0 or int(getattr(env, "num_targets", 0)) != 0:
        raise NotImplementedError("Warm-start HDF5 loading does not support block/target observations.")

    timestamps = train_vals[:, _TRAIN_VALS_CUR_TIME]
    step_indices = np.rint(train_vals[:, _TRAIN_VALS_STEP_INDEX]).astype(np.int64)
    pose_xy = np.asarray(train_vals[:, _TRAIN_VALS_POSE][:, :2], dtype=np.float64)
    speed_xy = np.asarray(train_vals[:, _TRAIN_VALS_SPEED][:, :2], dtype=np.float64)
    force_xy = np.asarray(train_vals[:, _TRAIN_VALS_FORCE][:, :2], dtype=np.float64)
    stored_acc_xy = np.asarray(train_vals[:, _TRAIN_VALS_ACC][:, :2], dtype=np.float64)
    desired_pose_xy = np.asarray(train_vals[:, _TRAIN_VALS_DESIRED_POSE][:, :2], dtype=np.float64)
    puck_xy = np.asarray(train_vals[:, _TRAIN_VALS_PUCK][:, :2], dtype=np.float64)
    puck_occluded = np.asarray(train_vals[:, _TRAIN_VALS_PUCK][:, 2], dtype=np.float64)

    paddle_acc_xy = stored_acc_xy
    if not np.any(np.abs(paddle_acc_xy) > 1e-8):
        paddle_acc_xy = _estimate_xy_derivative(speed_xy, timestamps)
    paddle_jerk_xy = _estimate_xy_derivative(paddle_acc_xy, timestamps)
    puck_vel_xy = _estimate_xy_derivative(puck_xy, timestamps)

    move_lims = np.asarray(getattr(env.simulator, "move_lims", (1.0, 1.0)), dtype=np.float64).reshape(-1)
    if move_lims.shape[0] < 2:
        move_lims = np.array([1.0, 1.0], dtype=np.float64)
    move_lims = move_lims[:2].copy()
    move_lims[np.abs(move_lims) < 1e-6] = 1.0
    actions_xy = np.clip((desired_pose_xy - pose_xy) / move_lims[None, :], -1.0, 1.0)

    full_puck_history: list[tuple[float, float, float]] = []
    full_paddle_history: list[tuple[float, float, float]] = []
    padded_puck_histories: list[list[tuple[float, float, float]]] = []
    padded_paddle_histories: list[list[tuple[float, float, float]]] = []
    state_infos: list[dict] = []
    default_puck_entry = (0.0, 0.0, 1.0)
    default_paddle_entry = (0.0, 0.0, 0.0)

    for row_idx in range(train_vals.shape[0]):
        puck_entry = (
            float(puck_xy[row_idx, 0]),
            float(puck_xy[row_idx, 1]),
            float(puck_occluded[row_idx]),
        )
        paddle_entry = (
            float(pose_xy[row_idx, 0]),
            float(pose_xy[row_idx, 1]),
            0.0,
        )
        full_puck_history.append(puck_entry)
        full_paddle_history.append(paddle_entry)
        padded_puck_histories.append(
            _padded_history_window(
                full_puck_history,
                default_entry=default_puck_entry,
            )
        )
        padded_paddle_histories.append(
            _padded_history_window(
                full_paddle_history,
                default_entry=default_paddle_entry,
            )
        )
        state_infos.append(
            _build_split_state_info(
                paddle_position_xy=pose_xy[row_idx],
                paddle_velocity_xy=speed_xy[row_idx],
                paddle_acceleration_xy=paddle_acc_xy[row_idx],
                paddle_force_xy=force_xy[row_idx],
                paddle_jerk_xy=paddle_jerk_xy[row_idx],
                puck_position_xy=puck_xy[row_idx],
                puck_velocity_xy=puck_vel_xy[row_idx],
                puck_occluded=puck_occluded[row_idx],
                puck_history_entries=full_puck_history,
            )
        )

    env.reset(seed=int(getattr(env, "rng", np.random.RandomState(0)).randint(0, int(1e8))))
    _reset_warm_start_env_state(env, state_infos[0])

    episode_trajectory = EpisodeTrajectory.empty()
    for row_idx in range(1, train_vals.shape[0]):
        prev_state = state_infos[row_idx - 1]
        next_state = state_infos[row_idx]
        prev_obs = env.get_observation(
            prev_state,
            obs_type=env.obs_type,
            puck_history=padded_puck_histories[row_idx - 1],
            paddle_history=padded_paddle_histories[row_idx - 1],
        )
        next_obs = env.get_observation(
            next_state,
            obs_type=env.obs_type,
            puck_history=padded_puck_histories[row_idx],
            paddle_history=padded_paddle_histories[row_idx],
        )
        stop_state = _stop_state_from_saved_row(train_vals, optional_data, row_idx)
        task_reward, terminations_only = _recompute_warm_start_rewards(
            args,
            env,
            prev_state=prev_state,
            next_state=next_state,
            step_index=int(step_indices[row_idx]),
            stop_state=stop_state,
            is_last_transition=bool(row_idx == (train_vals.shape[0] - 1)),
        )
        episode_trajectory.append_step(
            obs=torch.as_tensor(prev_obs, dtype=torch.float32),
            next_obs=torch.as_tensor(next_obs, dtype=torch.float32),
            action=torch.as_tensor(actions_xy[row_idx], dtype=torch.float32),
            reward=torch.tensor(task_reward, dtype=torch.float32),
            done=torch.tensor(terminations_only, dtype=torch.float32),
            prev_action=torch.as_tensor(actions_xy[row_idx - 1], dtype=torch.float32),
        )
    return episode_trajectory


AddEpisodeToSharedReplayFn = Callable[
    [SharedTD3Replay, EpisodeTrajectory, deque, float],
    tuple[str, float, float, int],
]


def _warm_start_replay_from_hdf5(
    *,
    args: Any,
    replay: SharedTD3Replay,
    env: AirHockeyEnv,
    add_episode_to_shared_replay: AddEpisodeToSharedReplayFn,
) -> dict[str, float]:
    warm_start_rng = np.random.default_rng(int(args.seed))
    episode_paths = _list_warm_start_hdf5_files(
        args.warm_start_hdf5_dirs,
        recursive=bool(args.warm_start_hdf5_recursive),
        rng=warm_start_rng,
    )
    if not episode_paths:
        raise ValueError("No warm-start HDF5 files found in the configured input directories.")

    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    loaded_files = 0
    skipped_files = 0
    loaded_transitions = 0
    last_threshold = 0.0

    print(f"[warm_start] loading {len(episode_paths)} HDF5 files into shared replay")
    for episode_path in episode_paths:
        try:
            episode_trajectory = _load_warm_start_episode(episode_path, args=args, env=env)
        except Exception:
            skipped_files += 1
            print(f"[warm_start] failed to load {episode_path}:\n{traceback.format_exc()}")
            continue
        if len(episode_trajectory.observations) <= 0:
            skipped_files += 1
            print(f"[warm_start] skipped {episode_path} (not enough transitions)")
            continue
        partition, episode_return, last_threshold, inserted_steps = add_episode_to_shared_replay(
            replay,
            episode_trajectory,
            recent_episode_returns,
            float(args.success_top_fraction),
        )
        loaded_files += 1
        loaded_transitions += int(inserted_steps)
        print(
            f"[warm_start] loaded {episode_path} partition={partition} "
            f"transitions={inserted_steps} episode_return={episode_return:.4f}"
        )

    snapshot = replay.state_snapshot()
    print(
        "[warm_start] "
        f"loaded_files={loaded_files} skipped_files={skipped_files} "
        f"loaded_transitions={loaded_transitions} "
        f"success_rb={snapshot['success']['size']} failure_rb={snapshot['failure']['size']}"
    )
    return {
        "files_loaded": float(loaded_files),
        "files_skipped": float(skipped_files),
        "transitions_loaded": float(loaded_transitions),
        "episode_return_success_threshold": float(last_threshold),
        "success_buffer_size": float(snapshot["success"]["size"]),
        "failure_buffer_size": float(snapshot["failure"]["size"]),
    }
