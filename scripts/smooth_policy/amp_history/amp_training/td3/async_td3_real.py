"""Asynchronous TD3 training for real-world adaptation.

This script starts:
1) a collector process that continuously rolls out in the environment and writes
   episode transitions to a shared replay service with success/failure partitions,
2) a learner process that continuously samples from that same replay service and
   performs TD3 updates at a configurable update rate.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import shutil
import time
import traceback
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.smooth_policy.amp_history.amp_training.td3.episode_artifacts import (
    clean_episode_hdf5,
    generate_episode_camera_video,
    generate_episode_gif,
    save_split_episode_hdf5,
)
from scripts.smooth_policy.amp_history.amp_training.td3.dual_head_q import TD3DualHeadQNetwork
from scripts.smooth_policy.amp_history.amp_training.td3.shared_replay import SharedTD3Replay
from scripts.smooth_policy.amp_history.amp_training.td3.td3_episode_collection import EpisodeTrajectory
from scripts.smooth_policy.amp_history.amp_training.td3.td3_replay_sampling import (
    critic_success_failure_counts,
)
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM
from scripts.smooth_policy.deterministic_agent import DeterministicAgent


def _prepare_air_hockey_config(config: dict, seed: int = 0, return_goal_obs: bool = False) -> dict:
    """Build an air_hockey config dict with required top-level fields merged in."""
    ah = dict(config["air_hockey"])
    if "seed" not in ah:
        seed_cfg = config.get("seed", seed)
        if isinstance(seed_cfg, (list, tuple)):
            seed_cfg = seed_cfg[0] if len(seed_cfg) > 0 else 0
        ah["seed"] = int(seed_cfg)
    if "n_training_steps" not in ah:
        ah["n_training_steps"] = config.get("n_training_steps", 1000000)
    if "return_goal_obs" not in ah:
        ah["return_goal_obs"] = return_goal_obs
    return ah


def h_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def h_inverse(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    abs_x = torch.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    sqrt_inner = torch.sqrt(inner)
    quotient = (sqrt_inner - 1) / (2 * eps)
    return torch.sign(x) * (quotient**2 - 1)


def augment_policy_observation(
    observation: torch.Tensor,
    last_action: torch.Tensor,
    use_last_action_in_policy_state: bool,
) -> torch.Tensor:
    if not use_last_action_in_policy_state:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def deterministic_actor_action(actor: DeterministicAgent, policy_obs: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "get_action"):
        return actor.get_action(policy_obs)
    raise TypeError(f"Unsupported actor type for deterministic action: {type(actor)}")


def extract_deterministic_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    deterministic_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor.") or key.startswith("actor_mean_head."):
            deterministic_state[key] = value
        if key in ("action_scale", "action_bias"):
            deterministic_state[key] = value
    if not deterministic_state:
        raise ValueError("No deterministic actor weights found in provided state dict.")
    return deterministic_state


def build_policy_env_view(obs_dim: int, act_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
    )


class SharedActorState:
    """Versioned shared actor parameter storage for collector sync at episode boundaries."""

    def __init__(self, initial_state_dict: Dict[str, torch.Tensor]) -> None:
        self.lock = mp.Lock()
        self.version = mp.Value("i", 0, lock=False)
        self.tensors: Dict[str, torch.Tensor] = {}
        for key, value in initial_state_dict.items():
            self.tensors[key] = value.detach().to("cpu").clone().share_memory_()

    def publish(self, state_dict: Dict[str, torch.Tensor]) -> int:
        with self.lock:
            for key, value in state_dict.items():
                if key not in self.tensors:
                    self.tensors[key] = value.detach().to("cpu").clone().share_memory_()
                else:
                    self.tensors[key].copy_(value.detach().to("cpu"))
            self.version.value += 1
            return int(self.version.value)

    def read(self) -> Tuple[int, Dict[str, torch.Tensor]]:
        with self.lock:
            version = int(self.version.value)
            state_dict = {key: tensor.clone() for key, tensor in self.tensors.items()}
        return version, state_dict


@dataclass
class Args:
    args_file: str | None = None
    config: str = "configs/real_configs/rollout_config.yaml"
    model_path: str | None = None
    collector_device: str = "cpu"
    learner_device: str = "cuda:0"
    seed: int = 0

    # Shared replay sizes
    success_buffer_size: int = int(2e5)
    failure_buffer_size: int = int(8e5)
    recent_episode_window_size: int = 500
    success_top_fraction: float = 0.2

    # TD3 core
    task_gamma: float = 0.975
    motion_gamma: float = 0.8
    tau: float = 0.005
    batch_size: int = 256
    min_replay_size_before_learning: int = 5000
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    policy_lr: float = 3e-4
    q_updates: int = 1
    actor_updates_per_iteration: int = 1
    target_network_frequency: int = 1
    updates_per_second: float = 10.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    h_transform_eps: float = 1e-3
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 1.0
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7

    # Actor/critic architecture
    action_scale: float = 0.02
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2
    use_last_action_in_policy_state: bool = False

    # Collector behavior
    exploration_noise: float = 0.1
    collector_policy_stand_still: bool = False
    actor_sync_check_every_episode: bool = True
    collector_log_interval_sec: float = 5.0
    learner_log_interval_sec: float = 5.0
    episode_artifact_dir: str = "runs/async_td3/episode_hdf5"
    episode_gif_dir: str = "runs/async_td3/episode_gifs"
    episode_min_timesteps: int = 30
    estop_episode_min_timesteps: int = 50
    enable_episode_gif: bool = True
    episode_gif_fps: int = 20
    episode_gif_subsample: int = 1
    # Set to 0 to disable cap.
    episode_gif_max_frames: int = 0
    episode_gif_require_puck: bool = False
    enable_episode_camera_video: bool = True
    episode_camera_video_fps: int = 20
    episode_camera_video_subsample: int = 1
    # Set to 0 to disable cap.
    episode_camera_video_max_frames: int = 0
    episode_camera_video_codec: str = "mp4v"
    log_parent_dir: str | None = None
    run_name: str = "async_td3_real"

    # Optional smoke-test mode (0 disables)
    smoke_test_seconds: float = 0.0


def _build_args_file_defaults(
    args_file_path: str,
) -> tuple[dict, list[str], list[str]]:
    with open(args_file_path, "r") as f:
        loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    if loaded_yaml is None:
        return {}, [], []
    if not isinstance(loaded_yaml, dict):
        raise ValueError(f"Expected args_file YAML to be a mapping, got {type(loaded_yaml)}")

    # TD3 args-file keys that map directly or approximately into async Args.
    key_map = {
        "config": "config",
        "model_path": "model_path",
        "log_parent_dir": "log_parent_dir",
        "run_name": "run_name",
        "seed": "seed",
        "success_buffer_size": "success_buffer_size",
        "failure_buffer_size": "failure_buffer_size",
        "recent_episode_window_size": "recent_episode_window_size",
        "success_top_fraction": "success_top_fraction",
        "task_gamma": "task_gamma",
        "motion_gamma": "motion_gamma",
        "tau": "tau",
        "batch_size": "batch_size",
        "learning_starts": "min_replay_size_before_learning",
        "policy_lr": "policy_lr",
        "q_lr": "q_lr",
        "q_weight_decay": "q_weight_decay",
        "q_updates": "q_updates",
        "target_network_frequency": "target_network_frequency",
        "actor_updates_per_iteration": "actor_updates_per_iteration",
        "policy_noise": "policy_noise",
        "noise_clip": "noise_clip",
        "h_transform_eps": "h_transform_eps",
        "task_reward_weight": "task_reward_weight",
        "motion_reward_weight": "motion_reward_weight",
        "critic_success_sample_fraction": "critic_success_sample_fraction",
        "critic_failure_sample_fraction": "critic_failure_sample_fraction",
        "action_scale": "action_scale",
        "agent_hidden_layer_size": "agent_hidden_layer_size",
        "agent_num_hidden_layers": "agent_num_hidden_layers",
        "q_hidden_layer_size": "q_hidden_layer_size",
        "q_num_hidden_layers": "q_num_hidden_layers",
        "use_last_action_in_policy_state": "use_last_action_in_policy_state",
        "exploration_noise": "exploration_noise",
        "collector_policy_stand_still": "collector_policy_stand_still",
        "enable_episode_gif": "enable_episode_gif",
        "episode_gif_dir": "episode_gif_dir",
        "episode_min_timesteps": "episode_min_timesteps",
        "estop_episode_min_timesteps": "estop_episode_min_timesteps",
        "episode_gif_fps": "episode_gif_fps",
        "episode_gif_subsample": "episode_gif_subsample",
        "episode_gif_max_frames": "episode_gif_max_frames",
        "episode_gif_require_puck": "episode_gif_require_puck",
        "enable_episode_camera_video": "enable_episode_camera_video",
        "episode_camera_video_fps": "episode_camera_video_fps",
        "episode_camera_video_subsample": "episode_camera_video_subsample",
        "episode_camera_video_max_frames": "episode_camera_video_max_frames",
        "episode_camera_video_codec": "episode_camera_video_codec",
        "device": "learner_device",
        "agent_hidden_size": "agent_hidden_layer_size",
        "q_hidden_size": "q_hidden_layer_size",
        # keep this self-mapped so explicit YAML args_file doesn't appear as ignored.
        "args_file": "args_file",
    }
    valid_async_keys = {field.name for field in fields(Args)}
    mapped_defaults: dict = {}
    applied_source_keys: list[str] = []
    ignored_source_keys: list[str] = []

    for source_key, source_value in loaded_yaml.items():
        target_key = key_map.get(source_key)
        if target_key is None or target_key not in valid_async_keys:
            ignored_source_keys.append(source_key)
            continue
        mapped_defaults[target_key] = source_value
        applied_source_keys.append(source_key)

    return mapped_defaults, sorted(applied_source_keys), sorted(ignored_source_keys)


def run_reset_fsm(env: AirHockeyEnv, rng: np.random.Generator) -> None:
    """Run the ResetPolicyFSM to get the puck back in play.

    Executes FSM actions through env.step() in a tight loop.
    These steps are NOT recorded in the replay buffer.
    """
    wait_logged = False
    while _detect_estop(env):
        if not wait_logged:
            print("[reset_fsm] protective stop active; waiting for clear...")
            wait_logged = True
        time.sleep(0.25)
    if wait_logged:
        print("[reset_fsm] protective stop cleared; resuming reset FSM.")

    fsm = ResetPolicyFSM(env, rng)
    print(f"[reset_fsm] starting (side={fsm.start_side})")
    while not fsm.done:
        state = env.simulator.get_current_state()
        action = fsm.step(state)
        env.step(action)
    print(f"[reset_fsm] done after {fsm.total_steps} steps (final phase={fsm.phase})")


def _episode_to_tensors(episode_trajectory: EpisodeTrajectory) -> Dict[str, torch.Tensor]:
    return {
        "observations": torch.stack(episode_trajectory.observations, dim=0),
        "next_observations": torch.stack(episode_trajectory.next_observations, dim=0),
        "actions": torch.stack(episode_trajectory.actions, dim=0),
        "prev_actions": torch.stack(episode_trajectory.prev_actions, dim=0),
        "task_rewards": torch.stack(episode_trajectory.task_rewards, dim=0).view(-1),
        "motion_rewards": torch.stack(episode_trajectory.motion_rewards, dim=0).view(-1),
        "dones": torch.stack(episode_trajectory.dones, dim=0).view(-1),
    }


def _vector_with_width(values: np.ndarray | list | tuple, width: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros((int(width),), dtype=np.float64)
    copy_width = min(int(width), int(vector.shape[0]))
    if copy_width > 0:
        out[:copy_width] = vector[:copy_width]
    return out


def _finite_or_default(value: float, default: float = -1.0) -> float:
    value_f = float(value)
    return value_f if np.isfinite(value_f) else float(default)


def _latest_camera_frame(env: AirHockeyEnv) -> np.ndarray | None:
    """Fetch the latest raw camera frame if available."""
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return None
    images = getattr(simulator, "images", None)
    if not isinstance(images, list) or len(images) == 0:
        return None
    latest = images[-1]
    if latest is None:
        return None
    frame = np.asarray(latest)
    if frame.ndim != 3:
        return None
    return np.array(frame, copy=True)


def _next_available_episode_id(output_dir: str | Path) -> int:
    """Return one plus the largest saved trajectory_data*.hdf5 index (recursive)."""
    artifact_dir = Path(output_dir).expanduser().resolve()
    if not artifact_dir.exists():
        return 0
    max_seen = -1
    pattern = re.compile(r"^trajectory_data(\d+)\.hdf5$")
    for path in artifact_dir.rglob("trajectory_data*.hdf5"):
        match = pattern.match(path.name)
        if match is None:
            continue
        max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def _episode_length_bucket_name(episode_steps: int) -> str:
    """Map episode step count to a quality bucket label."""
    if episode_steps <= 100:
        return "50-100"
    if episode_steps <= 200:
        return "100-200"
    return ">200"


def _bucketed_output_dir(base_dir: str | Path, episode_steps: int) -> str:
    """Return a length-bucketed output directory under the given parent."""
    return str(Path(base_dir).expanduser().resolve() / _episode_length_bucket_name(episode_steps))


def _estop_output_dir(base_dir: str | Path) -> str:
    """Return the additional e-stop-only output directory under the parent."""
    return str(Path(base_dir).expanduser().resolve() / "estop")


def _copy_to_estop_dir(file_path: str | Path, estop_root: str | Path) -> Path:
    """Copy one artifact file into the e-stop directory and return destination."""
    src = Path(file_path).expanduser().resolve()
    estop_root_path = Path(estop_root).expanduser().resolve()
    if src.suffix == ".hdf5":
        dst_dir = estop_root_path
    else:
        # Keep per-episode media grouping (trajectory_data{N}/...).
        dst_dir = estop_root_path / src.parent.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _build_split_episode_row(
    env: AirHockeyEnv,
    action_xy: np.ndarray,
    episode_id: int,
    episode_step_idx: int,
    estop_active: bool,
) -> Dict[str, np.ndarray]:
    state_info = getattr(env, "current_state", None)
    if not isinstance(state_info, dict):
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

    pose = _vector_with_width(np.concatenate([paddle_pos[:2], np.zeros(4, dtype=np.float64)]), 6)
    speed = _vector_with_width(np.concatenate([paddle_vel[:2], np.zeros(4, dtype=np.float64)]), 6)
    # TODO: Source force/acc/safety/estop directly from robot telemetry for real deployment.
    force = np.zeros((6,), dtype=np.float64)
    acc = np.zeros((3,), dtype=np.float64)
    desired_pose = _vector_with_width(np.concatenate([desired_xy, np.zeros(4, dtype=np.float64)]), 6)
    puck = _vector_with_width(np.concatenate([puck_position[:2], np.array([puck_occluded])]), 3)
    timing_info = state_info.get("timing", {}) if isinstance(state_info, dict) else {}
    paddle_actual_pose = np.asarray(state_info.get("paddle_actual_pose", np.zeros(6)), dtype=np.float64).reshape(-1)
    paddle_actual_speed = np.asarray(state_info.get("paddle_actual_speed", np.zeros(6)), dtype=np.float64).reshape(-1)
    paddle_target_pre = np.asarray(
        state_info.get("paddle_target_pose_pre_filter", np.zeros(6)), dtype=np.float64
    ).reshape(-1)
    paddle_target_post = np.asarray(
        state_info.get("paddle_target_pose_post_filter", np.zeros(6)), dtype=np.float64
    ).reshape(-1)
    paddle_actual = _vector_with_width(np.concatenate([paddle_actual_pose[:2], paddle_actual_speed[:2], np.zeros(2)]), 6)
    paddle_cmd = _vector_with_width(np.concatenate([paddle_target_pre[:6], paddle_target_post[:6]]), 12)
    timing = _vector_with_width(
        np.array(
            [
                time.time(),
                _finite_or_default(timing_info.get("step_start_s", -1.0)),
                _finite_or_default(timing_info.get("telemetry_read_s", -1.0)),
                # Post-homography puck timestamp used by delay analysis.
                _finite_or_default(timing_info.get("puck_detection_done_s", -1.0)),
                _finite_or_default(timing_info.get("command_sent_s", -1.0)),
                _finite_or_default(timing_info.get("step_end_s", -1.0)),
                _finite_or_default(timing_info.get("sleep_before_step_s", -1.0)),
                _finite_or_default(timing_info.get("loop_runtime_before_sleep_s", -1.0)),
                _finite_or_default(timing_info.get("camera_frame_received_s", -1.0)),
            ],
            dtype=np.float64,
        ),
        9,
    )
    puck_meta = _vector_with_width(
        np.array(
            [
                puck_occluded,
                1.0 if bool(state_info.get("puck_detector_used_fallback", puck_occluded > 0.5)) else 0.0,
            ],
            dtype=np.float64,
        ),
        2,
    )

    return {
        "cur_time": np.array([time.time()], dtype=np.float64),
        "tidx": np.array([float(episode_id)], dtype=np.float64),
        "i": np.array([float(episode_step_idx)], dtype=np.float64),
        "estop": np.array([1.0 if estop_active else 0.0], dtype=np.float64),
        "safety": np.array([1.0], dtype=np.float64),
        "pose": pose,
        "speed": speed,
        "force": force,
        "acc": acc,
        "desired_pose": desired_pose,
        "puck": puck,
        "timing": timing,
        "paddle_actual": paddle_actual,
        "paddle_cmd": paddle_cmd,
        "puck_meta": puck_meta,
    }


def _mixed_sample_from_shared(
    replay: SharedTD3Replay,
    batch_size: int,
    success_fraction: float,
    device: str,
) -> Dict[str, torch.Tensor] | None:
    success_count, failure_count = critic_success_failure_counts(
        batch_size=batch_size,
        success_fraction=success_fraction,
        success_available=replay.len("success") > 0,
        failure_available=replay.len("failure") > 0,
    )
    if success_count + failure_count == 0:
        return None

    chunks = []
    if success_count > 0:
        chunks.append(replay.sample("success", success_count, device=device))
    if failure_count > 0:
        chunks.append(replay.sample("failure", failure_count, device=device))
    if len(chunks) == 1:
        batch = chunks[0]
    else:
        batch = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in (
                "observations",
                "next_observations",
                "actions",
                "prev_actions",
                "task_rewards",
                "motion_rewards",
                "dones",
            )
        }
    return batch


def _detect_estop(env: AirHockeyEnv, step_info: dict | None = None) -> bool:
    """Best-effort e-stop/protective-stop detection from available real-world signals."""
    if isinstance(step_info, dict):
        if "estop" in step_info:
            estop_value = np.asarray(step_info["estop"], dtype=np.float64).reshape(-1)
            if estop_value.size > 0:
                return bool(estop_value[0] > 0.5)
        if "protective_stop" in step_info:
            return bool(step_info["protective_stop"])

    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return False

    rcv = getattr(simulator, "rcv", None)
    if rcv is not None and hasattr(rcv, "isProtectiveStopped"):
        try:
            return bool(rcv.isProtectiveStopped())
        except Exception:
            pass

    # Fallback: real simulator appends canonical state arrays where index 3 is estop.
    vals = getattr(simulator, "vals", None)
    if isinstance(vals, list) and len(vals) > 0:
        try:
            latest = np.asarray(vals[-1], dtype=np.float64).reshape(-1)
            if latest.shape[0] > 3:
                return bool(latest[3] > 0.5)
        except Exception:
            return False
    return False


def collector_process(
    args: Args,
    replay: SharedTD3Replay,
    actor_state: SharedActorState,
    stop_event: mp.Event,
    stats: Dict[str, float],
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
) -> None:
    post_reset_stand_still_steps = 5
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.collector_device)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    collector_config = _prepare_air_hockey_config(config, seed=args.seed)
    sim_params = collector_config.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
    env = AirHockeyEnv(collector_config)
    writer = SummaryWriter(tb_log_dir)

    policy_obs_dim = obs_dim + act_dim if args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(device)
    actor.eval()

    action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0)
    action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0)

    applied_version, latest_state = actor_state.read()
    actor.load_state_dict(latest_state, strict=False)

    reset_rng = np.random.default_rng(args.seed)
    # Commit to one startup behavior: run reset FSM once before policy takeover.
    # This avoids an immediate policy->reset mode flip in the first episode.
    run_reset_fsm(env, reset_rng)
    obs, _ = env.soft_reset()
    post_reset_steps_remaining = int(post_reset_stand_still_steps)
    episode_trajectory = EpisodeTrajectory.empty()
    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0
    last_action_for_policy = torch.zeros((1, act_dim), dtype=torch.float32, device=device)

    total_steps = 0
    total_episodes = 0
    next_episode_file_id = _next_available_episode_id(args.episode_artifact_dir)
    if next_episode_file_id > 0:
        print(
            f"[collector] continuing episode artifact ids from {next_episode_file_id} "
            f"(existing data found in {args.episode_artifact_dir})"
        )
    last_log_time = time.time()
    episode_rows = []
    episode_images: list[np.ndarray] = []
    episode_camera_null_frames = 0
    episodes_saved = 0
    episodes_removed_short = 0
    episodes_removed_invalid = 0
    episodes_gif_generated = 0
    episodes_gif_failed = 0
    episodes_camera_video_generated = 0
    episodes_camera_video_failed = 0
    estop_episodes = 0
    estop_steps = 0
    estop_penalty_applied_this_episode = False
    episode_had_estop = False
    collector_start_time = time.time()
    episodic_returns: list[float] = []
    episodic_lengths: list[float] = []
    success_rates: list[float] = []

    while not stop_event.is_set():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        policy_obs = augment_policy_observation(
            obs_tensor,
            last_action_for_policy,
            args.use_last_action_in_policy_state,
        )
        with torch.no_grad():
            action_tensor = deterministic_actor_action(actor, policy_obs)
            if args.exploration_noise > 0:
                action_tensor = action_tensor + torch.randn_like(action_tensor) * float(args.exploration_noise)
            action_tensor = torch.clamp(action_tensor, action_low, action_high)
            if post_reset_steps_remaining > 0:
                action_tensor = torch.zeros_like(action_tensor)
                post_reset_steps_remaining -= 1
            if args.collector_policy_stand_still:
                action_tensor = torch.zeros_like(action_tensor)

        env_action = action_tensor.squeeze(0).detach().cpu().numpy()
        prev_action = last_action_for_policy.clone()
        next_obs, task_reward, terminated, truncated, step_info = env.step(env_action)
        camera_frame = _latest_camera_frame(env)
        if camera_frame is not None:
            episode_images.append(camera_frame)
        else:
            episode_camera_null_frames += 1
        base_done = bool(terminated or truncated)
        estop_now = _detect_estop(env, step_info=step_info)
        done = bool(base_done or estop_now)
        if estop_now:
            estop_steps += 1
            episode_had_estop = True

        # TODO: Replace this placeholder with real motion reward decomposition if needed.
        motion_reward = 0.0
        if estop_now and not estop_penalty_applied_this_episode:
            motion_reward += -5.0
            estop_penalty_applied_this_episode = True

        episode_rows.append(
            _build_split_episode_row(
                env=env,
                action_xy=env_action,
                episode_id=next_episode_file_id,
                episode_step_idx=len(episode_rows),
                estop_active=estop_now,
            )
        )
        done_val = 1.0 if done else 0.0

        episode_trajectory.append_step(
            obs=obs_tensor[0],
            next_obs=torch.as_tensor(next_obs, dtype=torch.float32, device=device),
            action=action_tensor[0],
            task_reward=torch.tensor(float(task_reward), dtype=torch.float32, device=device),
            motion_reward=torch.tensor(float(motion_reward), dtype=torch.float32, device=device),
            done=torch.tensor(done_val, dtype=torch.float32, device=device),
            prev_action=prev_action[0],
        )
        total_steps += 1

        if args.use_last_action_in_policy_state:
            last_action_for_policy = action_tensor.detach().clone()
        obs = next_obs

        if done:
            episode_end_wall_time = time.time()
            total_episodes += 1
            if episode_had_estop:
                estop_episodes += 1
            episode_return = float(episode_trajectory.episode_return)
            episode_length = float(len(episode_trajectory.observations))
            recent_episode_returns.append(episode_return)
            episodic_returns.append(episode_return)
            episodic_lengths.append(episode_length)
            episode_success = bool(step_info.get("success", False)) if isinstance(step_info, dict) else False
            episode_end_type = (
                str(step_info.get("episode_end_type"))
                if isinstance(step_info, dict) and step_info.get("episode_end_type") is not None
                else None
            )
            episode_end_reasons = (
                list(step_info.get("episode_end_reasons", []))
                if isinstance(step_info, dict) and isinstance(step_info.get("episode_end_reasons", []), list)
                else []
            )
            episode_end_reason = (
                str(step_info.get("episode_end_reason"))
                if isinstance(step_info, dict) and step_info.get("episode_end_reason") is not None
                else None
            )
            if estop_now:
                # Collector can end an episode on protective stop even if env did not set done metadata.
                episode_end_type = "estop"
                episode_end_reasons = ["collector_estop"]
                episode_end_reason = "collector_estop"
            episode_ended_estop = (episode_end_type == "estop")
            success_rates.append(1.0 if episode_success else 0.0)
            writer.add_scalar("charts/episodic_return", episode_return, total_steps)
            writer.add_scalar("charts/episodic_length", episode_length, total_steps)
            writer.add_scalar("charts/episodic_success", float(1.0 if episode_success else 0.0), total_steps)
            if len(recent_episode_returns) > 0:
                quantile = 1.0 - float(args.success_top_fraction)
                episode_return_success_threshold = float(
                    np.quantile(np.asarray(recent_episode_returns, dtype=np.float32), quantile)
                )
            partition = "success" if episode_return >= episode_return_success_threshold else "failure"
            written = replay.add_episode(partition, _episode_to_tensors(episode_trajectory))
            episode_trajectory.reset()

            # Do all slow I/O and actor sync BEFORE the reset FSM,
            # so the transition from reset to policy is immediate.
            n_episode_steps = len(episode_rows)
            n_camera_frames = len(episode_images)
            has_camera_images = n_camera_frames > 0
            print(
                f"[collector] episode_id={next_episode_file_id} "
                f"steps={n_episode_steps} camera_frames={n_camera_frames} "
                f"null_frames={episode_camera_null_frames} "
                f"has_images={'yes' if has_camera_images else 'NO'} "
                f"end_type={episode_end_type} end_reason={episode_end_reason} "
                f"end_reasons={episode_end_reasons}"
            )
            if episode_camera_null_frames > 0 and n_camera_frames == 0:
                sim = getattr(env, "simulator", None)
                sim_images_len = len(getattr(sim, "images", [])) if sim else -1
                sim_cap = getattr(sim, "cap", "N/A")
                print(
                    f"[collector] WARNING: zero camera frames captured this episode. "
                    f"simulator.images len={sim_images_len} simulator.cap={sim_cap}"
                )
            artifact_path = save_split_episode_hdf5(
                output_dir=_bucketed_output_dir(args.episode_artifact_dir, n_episode_steps),
                episode_id=next_episode_file_id,
                episode_rows=episode_rows,
                episode_images=episode_images if has_camera_images else None,
            )
            episodes_saved += 1
            min_timesteps = int(args.episode_min_timesteps)
            if episode_had_estop:
                # Keep e-stop episodes with a lower floor so safety events are retained.
                min_timesteps = int(args.estop_episode_min_timesteps)
            clean_result = clean_episode_hdf5(
                artifact_path,
                min_timesteps=min_timesteps,
            )
            if not clean_result.kept:
                print(
                    f"[collector] episode_id={next_episode_file_id} "
                    f"removed: reason={clean_result.reason} timesteps={clean_result.timesteps}"
                )
                if clean_result.reason == "short_episode":
                    episodes_removed_short += 1
                else:
                    episodes_removed_invalid += 1
            else:
                if episode_ended_estop:
                    estop_copy_path = _copy_to_estop_dir(
                        clean_result.path,
                        _estop_output_dir(args.episode_artifact_dir),
                    )
                    print(
                        f"[collector] episode_id={next_episode_file_id} "
                        f"e-stop HDF5 copied to {estop_copy_path}"
                    )
                if args.enable_episode_gif:
                    try:
                        gif_path = generate_episode_gif(
                            episode_hdf5_path=clean_result.path,
                            gif_root=_bucketed_output_dir(args.episode_gif_dir, n_episode_steps),
                            fps=args.episode_gif_fps,
                            max_frames=(
                                args.episode_gif_max_frames if args.episode_gif_max_frames > 0 else None
                            ),
                            subsample=args.episode_gif_subsample,
                            require_puck=args.episode_gif_require_puck,
                        )
                        episodes_gif_generated += 1
                        if episode_ended_estop:
                            estop_gif_path = _copy_to_estop_dir(
                                gif_path,
                                _estop_output_dir(args.episode_gif_dir),
                            )
                            print(
                                f"[collector] episode_id={next_episode_file_id} "
                                f"e-stop GIF copied to {estop_gif_path}"
                            )
                    except Exception:
                        episodes_gif_failed += 1
                        print(
                            f"[collector] episode_id={next_episode_file_id} "
                            f"GIF generation FAILED:\n{traceback.format_exc()}"
                        )
                if args.enable_episode_camera_video:
                    try:
                        camera_video_path = generate_episode_camera_video(
                            episode_hdf5_path=clean_result.path,
                            video_root=_bucketed_output_dir(args.episode_gif_dir, n_episode_steps),
                            fps=args.episode_camera_video_fps,
                            max_frames=(
                                args.episode_camera_video_max_frames
                                if args.episode_camera_video_max_frames > 0
                                else None
                            ),
                            subsample=args.episode_camera_video_subsample,
                            codec=args.episode_camera_video_codec,
                        )
                        episodes_camera_video_generated += 1
                        if episode_ended_estop:
                            estop_camera_video_path = _copy_to_estop_dir(
                                camera_video_path,
                                _estop_output_dir(args.episode_gif_dir),
                            )
                            print(
                                f"[collector] episode_id={next_episode_file_id} "
                                f"e-stop camera video copied to {estop_camera_video_path}"
                            )
                        print(
                            f"[collector] episode_id={next_episode_file_id} "
                            f"camera video OK"
                        )
                    except Exception:
                        episodes_camera_video_failed += 1
                        print(
                            f"[collector] episode_id={next_episode_file_id} "
                            f"camera video FAILED:\n{traceback.format_exc()}"
                        )
                else:
                    print(
                        f"[collector] episode_id={next_episode_file_id} "
                        f"camera video SKIPPED (enable_episode_camera_video=False)"
                    )
            episode_rows = []
            episode_images = []
            episode_camera_null_frames = 0
            next_episode_file_id += 1

            if args.actor_sync_check_every_episode:
                version, maybe_new_state = actor_state.read()
                if version != applied_version:
                    actor.load_state_dict(maybe_new_state, strict=False)
                    applied_version = version

            min_reset_delay_s = 3.0
            processing_elapsed_s = time.time() - episode_end_wall_time
            artificial_delay_s = max(0.0, min_reset_delay_s - processing_elapsed_s)
            if artificial_delay_s > 0.0:
                time.sleep(artificial_delay_s)
            print(
                "[collector] "
                f"episode_id={next_episode_file_id - 1} "
                f"post_episode_processing_s={processing_elapsed_s:.3f} "
                f"artificial_delay_s={artificial_delay_s:.3f} "
                f"min_reset_delay_s={min_reset_delay_s:.3f}"
            )

            # Reset FSM + soft_reset after minimum end-of-episode delay.
            run_reset_fsm(env, reset_rng)
            obs, _ = env.soft_reset()
            post_reset_steps_remaining = int(post_reset_stand_still_steps)
            last_action_for_policy.zero_()
            estop_penalty_applied_this_episode = False
            episode_had_estop = False

        now = time.time()
        if now - last_log_time >= float(args.collector_log_interval_sec):
            snapshot = replay.state_snapshot()
            stats["collector_steps"] = float(total_steps)
            stats["collector_episodes"] = float(total_episodes)
            stats["collector_actor_version"] = float(applied_version)
            stats["replay_success_size"] = float(snapshot["success"]["size"])
            stats["replay_failure_size"] = float(snapshot["failure"]["size"])
            stats["episodes_saved"] = float(episodes_saved)
            stats["episodes_removed_short"] = float(episodes_removed_short)
            stats["episodes_removed_invalid"] = float(episodes_removed_invalid)
            stats["episodes_gif_generated"] = float(episodes_gif_generated)
            stats["episodes_gif_failed"] = float(episodes_gif_failed)
            stats["episodes_camera_video_generated"] = float(episodes_camera_video_generated)
            stats["episodes_camera_video_failed"] = float(episodes_camera_video_failed)
            stats["estop_steps"] = float(estop_steps)
            stats["estop_episodes"] = float(estop_episodes)
            writer.add_scalar("replay/success_buffer_size", float(snapshot["success"]["size"]), total_steps)
            writer.add_scalar("replay/failure_buffer_size", float(snapshot["failure"]["size"]), total_steps)
            writer.add_scalar(
                "replay/episode_return_success_threshold",
                float(episode_return_success_threshold),
                total_steps,
            )
            writer.add_scalar(
                "replay/recent_episode_window_count",
                float(len(recent_episode_returns)),
                total_steps,
            )
            writer.add_scalar("artifacts/episodes_saved", float(episodes_saved), total_steps)
            writer.add_scalar(
                "artifacts/episodes_removed_short",
                float(episodes_removed_short),
                total_steps,
            )
            writer.add_scalar(
                "artifacts/episodes_removed_invalid",
                float(episodes_removed_invalid),
                total_steps,
            )
            writer.add_scalar(
                "artifacts/episodes_gif_generated",
                float(episodes_gif_generated),
                total_steps,
            )
            writer.add_scalar("artifacts/episodes_gif_failed", float(episodes_gif_failed), total_steps)
            writer.add_scalar(
                "artifacts/episodes_camera_video_generated",
                float(episodes_camera_video_generated),
                total_steps,
            )
            writer.add_scalar(
                "artifacts/episodes_camera_video_failed",
                float(episodes_camera_video_failed),
                total_steps,
            )
            writer.add_scalar("safety/estop_steps", float(estop_steps), total_steps)
            writer.add_scalar("safety/estop_episodes", float(estop_episodes), total_steps)
            writer.add_scalar(
                "charts/SPS",
                float(total_steps) / max(now - collector_start_time, 1e-6),
                total_steps,
            )
            if episodic_returns:
                writer.add_scalar("charts/avg_episodic_return", float(np.mean(episodic_returns)), total_steps)
                writer.add_scalar("charts/min_episodic_return", float(np.min(episodic_returns)), total_steps)
                writer.add_scalar("charts/max_episodic_return", float(np.max(episodic_returns)), total_steps)
                writer.add_scalar(
                    "charts/avg_success_rate",
                    float(np.mean(success_rates)) if success_rates else 0.0,
                    total_steps,
                )
                writer.add_scalar(
                    "charts/avg_episodic_length",
                    float(np.mean(episodic_lengths)),
                    total_steps,
                )
                episodic_returns.clear()
                episodic_lengths.clear()
                success_rates.clear()
            print(
                "[collector] "
                f"steps={total_steps} episodes={total_episodes} "
                f"actor_version={applied_version} "
                f"success_rb={snapshot['success']['size']} failure_rb={snapshot['failure']['size']} "
                f"saved={episodes_saved} short_removed={episodes_removed_short} "
                f"invalid_removed={episodes_removed_invalid} gif_ok={episodes_gif_generated} gif_fail={episodes_gif_failed} "
                f"cam_video_ok={episodes_camera_video_generated} cam_video_fail={episodes_camera_video_failed} "
                f"estop_steps={estop_steps} estop_episodes={estop_episodes}"
            )
            last_log_time = now

    env.close()
    writer.close()


def learner_process(
    args: Args,
    replay: SharedTD3Replay,
    actor_state: SharedActorState,
    stop_event: mp.Event,
    stats: Dict[str, float],
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.learner_device)
    writer = SummaryWriter(tb_log_dir)

    policy_obs_dim = obs_dim + act_dim if args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)

    actor = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    qf1 = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(device)
    qf2 = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(device)
    qf1_target = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(device)
    qf2_target = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    if args.model_path is not None and os.path.exists(args.model_path):
        loaded_obj = torch.load(args.model_path, map_location=device, weights_only=False)
        if isinstance(loaded_obj, dict) and "actor" in loaded_obj:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj["actor"]), strict=False)
            actor_target.load_state_dict(actor.state_dict())
            if "qf1" in loaded_obj and "qf2" in loaded_obj:
                qf1.load_state_dict(loaded_obj["qf1"])
                qf2.load_state_dict(loaded_obj["qf2"])
                if "qf1_target" in loaded_obj and "qf2_target" in loaded_obj:
                    qf1_target.load_state_dict(loaded_obj["qf1_target"])
                    qf2_target.load_state_dict(loaded_obj["qf2_target"])
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
            actor_target.load_state_dict(actor.state_dict())

    actor_state.publish({key: value.detach().cpu() for key, value in actor.state_dict().items()})

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr,
        weight_decay=args.q_weight_decay,
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0)
    action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0)

    total_updates = 0
    total_actor_updates = 0
    last_log_time = time.time()
    learner_start_time = time.time()
    latest_train_metrics: Dict[str, float] = {}

    while not stop_event.is_set():
        total_replay_size = replay.len("success") + replay.len("failure")
        if total_replay_size < int(args.min_replay_size_before_learning):
            time.sleep(0.05)
            continue

        iter_start = time.time()
        actor_updated = False

        for q_update_idx in range(args.q_updates):
            success_batch_count, failure_batch_count = critic_success_failure_counts(
                batch_size=args.batch_size,
                success_fraction=args.critic_success_sample_fraction,
                success_available=replay.len("success") > 0,
                failure_available=replay.len("failure") > 0,
            )
            batch = _mixed_sample_from_shared(
                replay=replay,
                batch_size=args.batch_size,
                success_fraction=args.critic_success_sample_fraction,
                device=args.learner_device,
            )
            if batch is None:
                continue

            sampled_observations = batch["observations"]
            sampled_next_observations = batch["next_observations"]
            sampled_actions = batch["actions"]
            sampled_task_rewards = batch["task_rewards"]
            sampled_motion_rewards = batch["motion_rewards"]
            sampled_dones = batch["dones"]

            sampled_next_prev_actions = sampled_actions * (1.0 - sampled_dones.unsqueeze(-1))
            sampled_next_policy_observations = augment_policy_observation(
                sampled_next_observations,
                sampled_next_prev_actions,
                args.use_last_action_in_policy_state,
            )

            with torch.no_grad():
                target_next_action = deterministic_actor_action(actor_target, sampled_next_policy_observations)
                noise = torch.randn_like(target_next_action) * float(args.policy_noise)
                noise = torch.clamp(noise, -float(args.noise_clip), float(args.noise_clip))
                target_next_action = torch.clamp(target_next_action + noise, action_low, action_high)

                q1_next_task_h, q1_next_motion_h = qf1_target(sampled_next_observations, target_next_action)
                q2_next_task_h, q2_next_motion_h = qf2_target(sampled_next_observations, target_next_action)
                min_next_task = h_inverse(
                    torch.min(q1_next_task_h, q2_next_task_h),
                    eps=float(args.h_transform_eps),
                ).view(-1)
                min_next_motion = h_inverse(
                    torch.min(q1_next_motion_h, q2_next_motion_h),
                    eps=float(args.h_transform_eps),
                ).view(-1)

                bellman_task = sampled_task_rewards + (1.0 - sampled_dones) * float(args.task_gamma) * min_next_task
                bellman_motion = sampled_motion_rewards + (1.0 - sampled_dones) * float(args.motion_gamma) * min_next_motion
                target_task_h = h_transform(bellman_task, eps=float(args.h_transform_eps))
                target_motion_h = h_transform(bellman_motion, eps=float(args.h_transform_eps))

            q1_task_h, q1_motion_h = qf1(sampled_observations, sampled_actions)
            q2_task_h, q2_motion_h = qf2(sampled_observations, sampled_actions)
            q1_task_loss = torch.nn.functional.mse_loss(q1_task_h.view(-1), target_task_h)
            q2_task_loss = torch.nn.functional.mse_loss(q2_task_h.view(-1), target_task_h)
            q1_motion_loss = torch.nn.functional.mse_loss(q1_motion_h.view(-1), target_motion_h)
            q2_motion_loss = torch.nn.functional.mse_loss(q2_motion_h.view(-1), target_motion_h)

            q_loss = (
                q1_task_loss
                + q2_task_loss
                + q1_motion_loss
                + q2_motion_loss
            )
            q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            q_optimizer.step()
            total_updates += 1
            positive_task_reward_mask = sampled_task_rewards > 0
            positive_task_reward_count = float(positive_task_reward_mask.sum().item())
            minibatch_size = max(int(sampled_task_rewards.numel()), 1)
            positive_task_rewards = sampled_task_rewards[positive_task_reward_mask]
            latest_train_metrics.update(
                {
                    "losses/q_task_loss": float(((q1_task_loss + q2_task_loss) / 2.0).item()),
                    "losses/q_motion_loss": float(((q1_motion_loss + q2_motion_loss) / 2.0).item()),
                    "losses/q_total_loss": float(q_loss.item()),
                    "losses/q1_task_mean": float(q1_task_h.mean().item()),
                    "losses/q1_motion_mean": float(q1_motion_h.mean().item()),
                    "losses/q1_total_mean": float((q1_task_h + q1_motion_h).mean().item()),
                    "rewards/sampled_task_reward_mean": float(sampled_task_rewards.mean().item()),
                    "rewards/sampled_task_reward_min": float(sampled_task_rewards.min().item()),
                    "rewards/sampled_task_reward_std": float(sampled_task_rewards.std(unbiased=False).item()),
                    "rewards/sampled_task_reward_positive_count": positive_task_reward_count,
                    "rewards/sampled_task_reward_positive_fraction": (
                        positive_task_reward_count / float(minibatch_size)
                    ),
                    "rewards/sampled_task_reward_positive_mean": (
                        float(positive_task_rewards.mean().item()) if positive_task_rewards.numel() > 0 else 0.0
                    ),
                    "rewards/sampled_task_reward_positive_std": (
                        float(positive_task_rewards.std(unbiased=False).item())
                        if positive_task_rewards.numel() > 0
                        else 0.0
                    ),
                    "rewards/sampled_motion_reward_mean": float(sampled_motion_rewards.mean().item()),
                    "rewards/sampled_combined_reward_mean": float(
                        (sampled_task_rewards + sampled_motion_rewards).mean().item()
                    ),
                    "replay/success_buffer_size": float(replay.len("success")),
                    "replay/failure_buffer_size": float(replay.len("failure")),
                    "replay/critic_success_sample_count": float(success_batch_count),
                    "replay/critic_failure_sample_count": float(failure_batch_count),
                    "replay/critic_success_sample_fraction": (
                        float(success_batch_count) / float(max(args.batch_size, 1))
                    ),
                    "replay/critic_failure_sample_fraction": (
                        float(failure_batch_count) / float(max(args.batch_size, 1))
                    ),
                }
            )

            if (q_update_idx + 1) % int(args.target_network_frequency) == 0:
                with torch.no_grad():
                    for source, target in (
                        (qf1, qf1_target),
                        (qf2, qf2_target),
                        (actor, actor_target),
                    ):
                        for param, target_param in zip(source.parameters(), target.parameters()):
                            target_param.data.copy_(
                                float(args.tau) * param.data + (1.0 - float(args.tau)) * target_param.data
                            )

        for _ in range(args.actor_updates_per_iteration):
            actor_batch = _mixed_sample_from_shared(
                replay=replay,
                batch_size=args.batch_size,
                success_fraction=args.critic_success_sample_fraction,
                device=args.learner_device,
            )
            if actor_batch is None:
                continue
            actor_obs = actor_batch["observations"]
            actor_prev_actions = actor_batch["prev_actions"]
            actor_policy_obs = augment_policy_observation(
                actor_obs,
                actor_prev_actions,
                args.use_last_action_in_policy_state,
            )
            policy_actions = deterministic_actor_action(actor, actor_policy_obs)
            q1_task_h, q1_motion_h = qf1(actor_obs, policy_actions)
            q1_task = h_inverse(q1_task_h, eps=float(args.h_transform_eps)).view(-1)
            q1_motion = h_inverse(q1_motion_h, eps=float(args.h_transform_eps)).view(-1)
            actor_objective = float(args.task_reward_weight) * q1_task + float(args.motion_reward_weight) * q1_motion
            actor_loss = -actor_objective.mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()
            total_actor_updates += 1
            actor_updated = True
            norm_task = (1.0 - float(args.task_gamma)) * q1_task
            norm_motion = (1.0 - float(args.motion_gamma)) * q1_motion
            latest_train_metrics.update(
                {
                    "losses/actor_loss": float(actor_loss.item()),
                    "losses/actor_norm_task_mean": float(norm_task.mean().item()),
                    "losses/actor_norm_motion_mean": float(norm_motion.mean().item()),
                }
            )

        if actor_updated:
            published_version = actor_state.publish(
                {key: value.detach().cpu() for key, value in actor.state_dict().items()}
            )
            stats["learner_actor_version"] = float(published_version)

        elapsed = time.time() - iter_start
        if args.updates_per_second > 0:
            sleep_time = max(0.0, (1.0 / float(args.updates_per_second)) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        now = time.time()
        if now - last_log_time >= float(args.learner_log_interval_sec):
            stats["learner_q_updates"] = float(total_updates)
            stats["learner_actor_updates"] = float(total_actor_updates)
            stats["learner_replay_size"] = float(total_replay_size)
            step_index = max(total_updates, 1)
            for metric_name, metric_value in latest_train_metrics.items():
                writer.add_scalar(metric_name, float(metric_value), step_index)
            writer.add_scalar("charts/SPS", float(total_updates) / max(now - learner_start_time, 1e-6), step_index)
            writer.add_scalar("replay/success_buffer_size", float(replay.len("success")), step_index)
            writer.add_scalar("replay/failure_buffer_size", float(replay.len("failure")), step_index)
            print(
                "[learner] "
                f"q_updates={total_updates} actor_updates={total_actor_updates} replay_size={total_replay_size}"
            )
            last_log_time = now
    writer.close()


def _load_initial_actor_state(args: Args, policy_obs_dim: int, act_dim: int) -> Dict[str, torch.Tensor]:
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    )
    if args.model_path is not None and os.path.exists(args.model_path):
        loaded_obj = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if isinstance(loaded_obj, dict) and "actor" in loaded_obj:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj["actor"]), strict=False)
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
    return {key: value.detach().cpu() for key, value in actor.state_dict().items()}


def main(args: Args) -> None:
    if not (0.0 < args.success_top_fraction < 1.0):
        raise ValueError("success_top_fraction must be in (0, 1).")
    if args.q_updates <= 0:
        raise ValueError("q_updates must be > 0.")
    if args.actor_updates_per_iteration <= 0:
        raise ValueError("actor_updates_per_iteration must be > 0.")
    if args.target_network_frequency <= 0:
        raise ValueError("target_network_frequency must be > 0.")
    if abs(float(args.critic_success_sample_fraction + args.critic_failure_sample_fraction) - 1.0) > 1e-6:
        raise ValueError("critic_success_sample_fraction + critic_failure_sample_fraction must equal 1.0.")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    probe_config = _prepare_air_hockey_config(config, seed=args.seed, return_goal_obs=False)
    probe_config["simulator"] = "box2d"
    probe_sim_params = dict(probe_config.get("simulator_params", {}))
    for key in ("control_mode", "wait_for_space_to_start", "save_path",
                "debug_control", "debug_control_every"):
        probe_sim_params.pop(key, None)
    probe_config["simulator_params"] = probe_sim_params
    probe_env = AirHockeyEnv(probe_config)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    action_low_np = np.asarray(probe_env.action_space.low, dtype=np.float32)
    action_high_np = np.asarray(probe_env.action_space.high, dtype=np.float32)
    probe_env.close()

    policy_obs_dim = obs_dim + act_dim if args.use_last_action_in_policy_state else obs_dim
    initial_actor_state = _load_initial_actor_state(args, policy_obs_dim=policy_obs_dim, act_dim=act_dim)

    replay = SharedTD3Replay(
        success_capacity=args.success_buffer_size,
        failure_capacity=args.failure_buffer_size,
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
    )
    actor_state = SharedActorState(initial_actor_state)
    stop_event = mp.Event()
    manager = mp.Manager()
    stats = manager.dict()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_log_dir = args.log_parent_dir or f"runs/async_td3/{args.run_name}_{timestamp}"
    collector_tb_dir = os.path.join(base_log_dir, "collector_tb")
    learner_tb_dir = os.path.join(base_log_dir, "learner_tb")
    os.makedirs(collector_tb_dir, exist_ok=True)
    os.makedirs(learner_tb_dir, exist_ok=True)
    print(f"TensorBoard logs: {base_log_dir}")

    collector = mp.Process(
        target=collector_process,
        args=(
            args,
            replay,
            actor_state,
            stop_event,
            stats,
            obs_dim,
            act_dim,
            action_low_np,
            action_high_np,
            collector_tb_dir,
        ),
        name="td3_collector",
    )
    learner = mp.Process(
        target=learner_process,
        args=(
            args,
            replay,
            actor_state,
            stop_event,
            stats,
            obs_dim,
            act_dim,
            action_low_np,
            action_high_np,
            learner_tb_dir,
        ),
        name="td3_learner",
    )

    collector.start()
    learner.start()

    try:
        if args.smoke_test_seconds > 0:
            print(f"Running smoke test for {args.smoke_test_seconds:.1f} seconds...")
            time.sleep(float(args.smoke_test_seconds))
            stop_event.set()
        while collector.is_alive() and learner.is_alive() and not stop_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        collector.join(timeout=15.0)
        learner.join(timeout=15.0)
        if collector.is_alive():
            collector.terminate()
        if learner.is_alive():
            learner.terminate()
        print("Final stats:", dict(stats))


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        mapped_defaults, applied_keys, ignored_keys = _build_args_file_defaults(temp_args.args_file)
        default_args = Args(**mapped_defaults)
    else:
        mapped_defaults, applied_keys, ignored_keys = {}, [], []
        default_args = Args()

    args = tyro.cli(Args, default=default_args)
    if args.args_file is not None:
        print(f"[args_file] loaded defaults from: {args.args_file}")
        if applied_keys:
            print("[args_file] applied keys:", ", ".join(applied_keys))
        else:
            print("[args_file] applied keys: none")
        if ignored_keys:
            print("[args_file] ignored unsupported keys:", ", ".join(ignored_keys))
    main(args)
