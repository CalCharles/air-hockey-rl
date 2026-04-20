"""Synchronous TD3 training for real-world adaptation.

This script runs collection and learning in a single process. One learner
iteration is executed at each policy-episode boundary when learning is active.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import inspect
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.smooth_policy.amp_history.amp_training.td3.helper.episode_artifacts import (
    clean_episode_hdf5,
    generate_episode_camera_video,
    generate_episode_gif,
    save_split_episode_hdf5,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.dual_head_q import TD3DualHeadQNetwork
from scripts.smooth_policy.amp_history.amp_training.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.shared_replay import SharedTD3Replay
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_collector_factories import (
    build_primitive_exploration_selector_for_real_collector,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_collector_reset import (
    merge_reset_fsm_artifact_into_pending,
    soft_reset_prime_paddle_and_extract_previous_puck,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_collector_metrics import (
    compute_rolling50_metrics,
    rolling_mean,
    update_stats_dict_rolling50,
    write_rolling50_tensorboard_scalars,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_episode_buffers import (
    truncate_collector_episode_for_readiness_fail,
    vector_with_width,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_motion_rewards import (
    MotionRewardState,
    _compute_motion_reward_components,
    _extract_motion_magnitudes_from_state_info,
    _extract_motion_magnitudes_from_step_info,
    _extract_motion_positions_from_state_info,
    _init_motion_reward_state,
    _reset_motion_reward_state,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_stop_state import _classify_stop_event
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_warm_start import _warm_start_replay_from_hdf5
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_episode_collection import EpisodeTrajectory
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_replay_sampling import (
    critic_success_failure_counts,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_checkpointing import (
    get_rng_states,
    set_rng_states,
)
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM
from scripts.smooth_policy.deterministic_agent import DeterministicAgent

ROLLING_PERF_WINDOW_EPISODES = 50


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


def linear_anneal(start: float, end: float, step: int, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return end
    progress = min(max(step, 0) / float(anneal_steps), 1.0)
    return start + progress * (end - start)


def primitive_exploration_chance_for_step(args: "Args", step: int) -> float:
    return linear_anneal(
        args.exploration_primitive_chance_start,
        args.exploration_primitive_chance,
        step,
        args.exploration_primitive_chance_anneal_steps,
    )


def _primitive_state_tensor(values: object, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(1, -1)[:, :2]


def _extract_primitive_state_tensors(
    env: AirHockeyEnv,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((1, 2), dtype=torch.float32, device=device)
    state_info = None
    simulator = getattr(env, "simulator", None)
    if simulator is not None and hasattr(simulator, "get_current_state"):
        try:
            state_info = simulator.get_current_state()
        except Exception:
            state_info = None
    if state_info is None:
        state_info = getattr(env, "current_state", None)
    if not isinstance(state_info, dict):
        return zeros.clone(), zeros.clone(), zeros.clone()
    try:
        paddle_position = _primitive_state_tensor(
            state_info["paddles"]["paddle_ego"]["position"],
            device=device,
        )
        puck_position = _primitive_state_tensor(
            state_info["pucks"][0]["position"],
            device=device,
        )
        puck_velocity = _primitive_state_tensor(
            state_info["pucks"][0]["velocity"],
            device=device,
        )
        return paddle_position, puck_position, puck_velocity
    except Exception:
        return zeros.clone(), zeros.clone(), zeros.clone()


def _reset_primitive_rollout_state(
    primitive_selector: PrimitiveExplorationSelector | None,
) -> None:
    if primitive_selector is not None:
        done_mask = torch.ones(primitive_selector.num_envs, dtype=torch.bool, device=primitive_selector.device)
        primitive_selector.reset(done_mask)


def _normalize_replay_source_priority(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"warmstart_only", "checkpoint_only", "checkpoint_then_append"}:
        return normalized
    return "warmstart_only"


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _checkpoint_root_from_tb(tb_log_dir: str, override_root: str | None) -> Path:
    if override_root is not None and str(override_root).strip():
        return Path(override_root).expanduser().resolve()
    # learner_tb and collector_tb are children of the same run root.
    return Path(tb_log_dir).expanduser().resolve().parent


def _coerce_float_list(values: object, *, max_items: int | None = None) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    coerced = [float(item) for item in values]
    if max_items is not None and max_items > 0 and len(coerced) > max_items:
        coerced = coerced[-int(max_items) :]
    return coerced


def _build_async_training_state(
    *,
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    actor,
    actor_target,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    q_optimizer,
    actor_optimizer,
    total_updates: int,
    total_actor_updates: int,
    latest_train_metrics: Dict[str, float],
    collector_total_steps: int,
    run_elapsed_total_s: float,
    rolling50_task_reward_values: Sequence[float],
    rolling50_motion_reward_values: Sequence[float],
    rolling50_episode_length_values: Sequence[float],
    rolling50_estop_episode_flags: Sequence[float],
    include_non_vital_training_state_fields: bool,
) -> Dict[str, object]:
    replay_state = replay.state_dict()
    payload: Dict[str, object] = {
        "checkpoint_version": 2,
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "success_replay_buffer": replay_state["success"],
        "failure_replay_buffer": replay_state["failure"],
        "rng_states": get_rng_states(),
    }
    if include_non_vital_training_state_fields:
        payload.update(
            {
                "global_step": int(total_updates),
                "iteration": int(total_updates),
                "q_optimizer": q_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "train_metrics": dict(latest_train_metrics),
                "learner_q_updates": int(total_updates),
                "learner_actor_updates": int(total_actor_updates),
                "collector_total_steps": int(collector_total_steps),
                "run_elapsed_total_s": float(run_elapsed_total_s),
                "rolling_window_size": int(ROLLING_PERF_WINDOW_EPISODES),
                "rolling50_task_reward_values": list(rolling50_task_reward_values),
                "rolling50_motion_reward_values": list(rolling50_motion_reward_values),
                "rolling50_episode_length_values": list(rolling50_episode_length_values),
                "rolling50_estop_episode_flags": list(rolling50_estop_episode_flags),
                # Metadata only; runtime args always come from external CLI/args_file.
                "args": {**asdict(args), **asdict(train_args)},
            }
        )
    return payload


def _save_async_checkpoint(
    *,
    checkpoint_root: Path,
    checkpoint_tag: str,
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    actor,
    actor_target,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    q_optimizer,
    actor_optimizer,
    total_updates: int,
    total_actor_updates: int,
    latest_train_metrics: Dict[str, float],
    stats: Dict[str, object],
) -> Path:
    checkpoint_dir = checkpoint_root / f"checkpoint_{checkpoint_tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.config, "r") as f:
            checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(checkpoint_dir / "config.yaml", "w") as f:
            yaml.dump(checkpoint_config, f)
    except Exception as exc:
        print(f"[learner_checkpoint] failed to save config.yaml: {exc}")
    try:
        # Merge TrainArgs into args.yaml so the saved file is itself a valid
        # --train-args source for downstream rollouts.
        with open(checkpoint_dir / "args.yaml", "w") as f:
            yaml.dump({**asdict(args), **asdict(train_args)}, f)
    except Exception as exc:
        print(f"[learner_checkpoint] failed to save args.yaml: {exc}")
    torch.save(actor.state_dict(), checkpoint_dir / "model.pth")
    torch.save(actor_target.state_dict(), checkpoint_dir / "actor_target.pth")
    torch.save(qf1.state_dict(), checkpoint_dir / "qf1.pth")
    torch.save(qf2.state_dict(), checkpoint_dir / "qf2.pth")
    torch.save(qf1_target.state_dict(), checkpoint_dir / "qf1_target.pth")
    torch.save(qf2_target.state_dict(), checkpoint_dir / "qf2_target.pth")
    training_state = _build_async_training_state(
        args=args,
        train_args=train_args,
        replay=replay,
        actor=actor,
        actor_target=actor_target,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        total_updates=total_updates,
        total_actor_updates=total_actor_updates,
        latest_train_metrics=latest_train_metrics,
        collector_total_steps=int(stats.get("collector_total_steps", stats.get("collector_steps", 0.0))),
        run_elapsed_total_s=float(stats.get("run_elapsed_total_s", 0.0)),
        rolling50_task_reward_values=_coerce_float_list(
            stats.get("rolling50_task_reward_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_motion_reward_values=_coerce_float_list(
            stats.get("rolling50_motion_reward_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_episode_length_values=_coerce_float_list(
            stats.get("rolling50_episode_length_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_estop_episode_flags=_coerce_float_list(
            stats.get("rolling50_estop_episode_flags", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        include_non_vital_training_state_fields=args.include_non_vital_training_state_fields,
    )
    _atomic_torch_save(training_state, checkpoint_dir / "training_state.pth")
    return checkpoint_dir


def _save_checkpoint_from_learner_state(
    *,
    state: LearnerRuntimeState,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    checkpoint_tag: str,
    args: Args,
    train_args: TrainArgs,
) -> Path:
    return _save_async_checkpoint(
        checkpoint_root=state.checkpoint_root,
        checkpoint_tag=checkpoint_tag,
        args=args,
        train_args=train_args,
        replay=replay,
        actor=state.actor,
        actor_target=state.actor_target,
        qf1=state.qf1,
        qf2=state.qf2,
        qf1_target=state.qf1_target,
        qf2_target=state.qf2_target,
        q_optimizer=state.q_optimizer,
        actor_optimizer=state.actor_optimizer,
        total_updates=state.total_updates,
        total_actor_updates=state.total_actor_updates,
        latest_train_metrics=state.latest_train_metrics,
        stats=stats,
    )


@dataclass
class TrainArgs:
    """Architecture spec sourced from a td3_training.py-style args.yaml.

    These six fields describe the actor/critic network shape and policy-state
    contract used during training. They must be read from the training run's
    args.yaml — NOT from the online `--args-file` or CLI — so the rebuilt
    actor/critic layers match the saved checkpoint exactly.
    """

    action_scale: float
    agent_hidden_layer_size: int
    agent_num_hidden_layers: int
    q_hidden_layer_size: int
    q_num_hidden_layers: int
    use_last_action_in_policy_state: bool


TRAIN_ARGS_FIELD_NAMES: Tuple[str, ...] = tuple(f.name for f in fields(TrainArgs))


def _load_train_args(train_args_path: str) -> TrainArgs:
    """Load architecture fields from a td3_training.py-style args.yaml.

    Only canonical field names are accepted; deprecated aliases
    (`agent_hidden_size`, `q_hidden_size`, ...) are not remapped.
    Extra keys in the file are ignored.
    """
    if not os.path.exists(train_args_path):
        raise FileNotFoundError(f"--train-args file does not exist: {train_args_path}")
    with open(train_args_path, "r") as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected --train-args to contain a YAML mapping, got: {type(loaded)}"
        )
    missing = [name for name in TRAIN_ARGS_FIELD_NAMES if name not in loaded]
    if missing:
        raise KeyError(
            f"--train-args file {train_args_path} is missing required fields: {missing}. "
            f"Expected canonical td3_training.py args.yaml field names."
        )
    return TrainArgs(
        action_scale=float(loaded["action_scale"]),
        agent_hidden_layer_size=int(loaded["agent_hidden_layer_size"]),
        agent_num_hidden_layers=int(loaded["agent_num_hidden_layers"]),
        q_hidden_layer_size=int(loaded["q_hidden_layer_size"]),
        q_num_hidden_layers=int(loaded["q_num_hidden_layers"]),
        use_last_action_in_policy_state=bool(loaded["use_last_action_in_policy_state"]),
    )


@dataclass
class Args:
    # Required: training args.yaml (architecture source; see TrainArgs).
    train_args: str | None = None
    # Optional: online-behavior defaults YAML (e.g. td3_online.yaml).
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
    warm_start_hdf5_dirs: Tuple[str, ...] = ()
    warm_start_hdf5_recursive: bool = True

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
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    h_transform_eps: float = 1e-3
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 1.0
    stand_still_reward_weight: float = 0.5
    temporal_alignment_reward_weight: float = 0.5
    axis_alignment_reward_weight: float = 0.5
    velocity_reward_weight: float = 0.5
    jerk_reward_weight: float = 0.5
    stand_still_threshold: float = 0.015
    temporal_alignment_horizon: int = 4
    velocity_at_one: float = 0.3
    velocity_at_zero: float = 0.6
    jerk_at_one: float = 10.0
    jerk_at_zero: float = 23.0
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7

    # Collector behavior
    exploration_noise: float = 0.1
    exploration_primitive_chance: float = 0.0
    exploration_primitive_chance_start: float = 0.0
    exploration_primitive_chance_anneal_steps: int = 200000
    exploration_primitive_steps: int = 5
    exploration_primitive_weight_stand_still: float = 1.0
    exploration_primitive_weight_same_direction: float = 1.0
    exploration_primitive_weight_y_aligned: float = 1.0
    exploration_primitive_weight_target_position_directional: float = 1.0
    # Legacy action-space skew knob kept for compatibility with older configs.
    exploration_direction_y_component_weight: float = 2.0
    # Simulator-space per-step displacement ranges for directional primitives.
    exploration_same_direction_min_angle_deg: float = -180.0
    exploration_same_direction_max_angle_deg: float = 180.0
    exploration_same_direction_min_magnitude: float = 0.012
    exploration_same_direction_max_magnitude: float = 0.26
    exploration_y_aligned_min_angle_deg: float = 45.0
    exploration_y_aligned_max_angle_deg: float = 135.0
    exploration_y_aligned_min_magnitude: float = 0.012
    exploration_y_aligned_max_magnitude: float = 0.12
    exploration_target_position_directional_min_angle_deg: float = -180.0
    exploration_target_position_directional_max_angle_deg: float = 180.0
    exploration_target_position_directional_min_magnitude: float = 0.2
    exploration_target_position_directional_max_magnitude: float = 0.5
    exploration_target_position_min_distance: float = 0.2
    exploration_target_position_max_distance: float = 0.5
    exploration_target_position_delta_x: float = 0.26
    exploration_target_position_delta_y: float = 0.12
    exploration_target_position_steps: int = 5
    collector_policy_stand_still: bool = False
    transition_hold_steps_post_reset: int = 8
    transition_hold_steps_post_estop_enter: int = 0
    transition_hold_steps_post_estop_clear: int = 8
    transition_hold_steps_post_actor_sync: int = 3
    transition_hold_steps_post_safety_rearm: int = 3
    transition_disable_exploration_noise: bool = True
    transition_last_action_mode: str = "zero"
    transition_hold_log_every_step: bool = False
    actor_sync_check_every_episode: bool = True
    collector_log_interval_sec: float = 60.0
    learner_log_interval_sec: float = 60.0
    episode_artifact_dir: str = "runs/async_td3/episode_hdf5"
    reset_artifact_dir: str = "runs/async_td3/reset_hdf5"
    episode_gif_dir: str = "runs/async_td3/episode_gifs"
    episode_camera_video_dir: str | None = None
    # Kept for config compatibility; short-episode filtering is disabled.
    episode_min_timesteps: int = 1
    estop_episode_min_timesteps: int = 1
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
    enable_periodic_checkpointing: bool = True
    checkpoint_every_successful_online_episodes: int = 10
    checkpoint_root_dir: str | None = None
    load_replay_from_checkpoint: bool = True
    replay_source_priority: str = "warmstart_only"
    include_non_vital_training_state_fields: bool = False

    # Optional smoke-test mode (0 disables)
    smoke_test_seconds: float = 0.0
    enable_latency_profiling: bool = False
    latency_profile_output_dir: str | None = None
    latency_profile_hist_bins: int = 40


def _build_args_file_defaults(
    args_file_path: str,
) -> tuple[dict, list[str], list[str]]:
    """Load defaults from a td3_training.py-style args.yaml.

    Only canonical Args field names are accepted; any other keys in the YAML
    are returned as `ignored_source_keys` and not applied. Deprecated legacy
    aliases (e.g. `agent_hidden_size`, `q_hidden_size`, `learning_starts`,
    `device`) are intentionally NOT remapped — use the canonical names.
    """
    with open(args_file_path, "r") as f:
        loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    if loaded_yaml is None:
        return {}, [], []
    if not isinstance(loaded_yaml, dict):
        raise ValueError(f"Expected args_file YAML to be a mapping, got {type(loaded_yaml)}")

    valid_async_keys = {field.name for field in fields(Args)}
    mapped_defaults: dict = {}
    applied_source_keys: list[str] = []
    ignored_source_keys: list[str] = []

    for source_key, source_value in loaded_yaml.items():
        if source_key in valid_async_keys:
            mapped_defaults[source_key] = source_value
            applied_source_keys.append(source_key)
        else:
            ignored_source_keys.append(source_key)

    return mapped_defaults, sorted(applied_source_keys), sorted(ignored_source_keys)


@dataclass(frozen=True)
class PendingResetArtifact:
    episode_id: int
    partition: str
    done_reason: str
    step_count: int
    rows: list[Dict[str, np.ndarray]]
    images: list[np.ndarray]
    camera_null_frames: int


@dataclass(frozen=True)
class ResetFSMRunResult:
    total_steps: int
    done_reason: str
    artifact: PendingResetArtifact | None = None


def _reset_stage_id_from_phase(phase: str) -> int:
    """Map reset FSM phase to a coarse stage id.

    Stage ids:
      0: before first upward motion is completed
      1: after first upward motion is completed
     -1: unknown/unmapped phase
    """
    phase_name = str(phase)
    if phase_name in ("goto_start", "edge_loop", "upward_burst", "post_first_upward_check"):
        return 0
    if phase_name in ("wait_for_puck", "strike", "post_second_upward_check"):
        return 1
    return -1


def _reset_artifact_partition(done_reason: str) -> str:
    return "success" if str(done_reason) == "success" else "failure"


def run_reset_fsm(
    env: AirHockeyEnv,
    rng: np.random.Generator,
    artifact_episode_id: int,
) -> ResetFSMRunResult:
    """Run the ResetPolicyFSM to get the puck back in play.

    Executes FSM actions through env.step() in a tight loop.
    These steps are NOT recorded in the replay buffer.
    """
    wait_logged = False
    stop_state = _classify_stop_event(env)
    while stop_state.active:
        if not wait_logged:
            print(
                "[reset_fsm] "
                f"stop active; waiting for clear (reason={stop_state.reason})..."
            )
            wait_logged = True
        time.sleep(0.25)
        stop_state = _classify_stop_event(env)
    if wait_logged:
        print("[reset_fsm] stop cleared; resuming reset FSM.")

    fsm = ResetPolicyFSM(env, rng)
    reset_rows: list[Dict[str, np.ndarray]] = []
    reset_images: list[np.ndarray] = []
    reset_camera_null_frames = 0
    print(f"[reset_fsm] starting (side={fsm.start_side})")
    try:
        while not fsm.done:
            state = env.simulator.get_current_state()
            action = fsm.step(state)
            reset_stage_id = _reset_stage_id_from_phase(getattr(fsm, "phase", "unknown"))
            _, _, _, _, step_info = env.step(action)
            stop_state = _classify_stop_event(env, step_info=step_info)
            camera_frame = _latest_camera_frame(env)
            if camera_frame is not None:
                reset_images.append(camera_frame)
            else:
                reset_camera_null_frames += 1
            reset_rows.append(
                _build_split_episode_row(
                    env=env,
                    action_xy=action,
                    episode_id=artifact_episode_id,
                    episode_step_idx=len(reset_rows),
                    protective_stop_active=stop_state.protective_stop,
                    controller_disconnected=stop_state.controller_disconnected,
                    reset_stage_id=reset_stage_id,
                )
            )
    finally:
        fsm.close()
    done_reason = getattr(fsm, "done_reason", "unknown")
    artifact = None
    if reset_rows:
        artifact = PendingResetArtifact(
            episode_id=int(artifact_episode_id),
            partition=_reset_artifact_partition(done_reason),
            done_reason=str(done_reason),
            step_count=len(reset_rows),
            rows=reset_rows,
            images=reset_images,
            camera_null_frames=int(reset_camera_null_frames),
        )
    print(
        f"[reset_fsm] done after {fsm.total_steps} steps "
        f"(final phase={fsm.phase}, reason={done_reason})"
    )
    if done_reason == "hard_reset_required":
        _hard_reset_with_pause(env, reason="reset_fsm_stage2_max_retries", pause_s=0.0)
    return ResetFSMRunResult(
        total_steps=int(fsm.total_steps),
        done_reason=str(done_reason),
        artifact=artifact,
    )


def _episode_to_tensors(episode_trajectory: EpisodeTrajectory) -> Dict[str, torch.Tensor]:
    return {
        "observations": torch.stack(episode_trajectory.observations, dim=0),
        "next_observations": torch.stack(episode_trajectory.next_observations, dim=0),
        "actions": torch.stack(episode_trajectory.actions, dim=0),
        "prev_actions": torch.stack(episode_trajectory.prev_actions, dim=0),
        "task_rewards": torch.stack(episode_trajectory.task_rewards, dim=0).view(-1),
        "motion_rewards": torch.stack(episode_trajectory.motion_rewards, dim=0).view(-1),
        # Same semantics as td3_training replay `dones`: env terminations (+ stop / last-step
        # signals), not time-limit truncation alone.
        "dones": torch.stack(episode_trajectory.dones, dim=0).view(-1),
    }


def _compute_episode_return_success_threshold(
    recent_episode_returns: deque[float],
    success_top_fraction: float,
) -> float:
    if len(recent_episode_returns) <= 0:
        return 0.0
    quantile = 1.0 - float(success_top_fraction)
    return float(np.quantile(np.asarray(recent_episode_returns, dtype=np.float32), quantile))


def _add_episode_to_shared_replay(
    replay: SharedTD3Replay,
    episode_trajectory: EpisodeTrajectory,
    recent_episode_returns: deque[float],
    success_top_fraction: float,
) -> tuple[str, float, float, int]:
    episode_return = float(episode_trajectory.episode_return)
    recent_episode_returns.append(episode_return)
    episode_return_success_threshold = _compute_episode_return_success_threshold(
        recent_episode_returns=recent_episode_returns,
        success_top_fraction=success_top_fraction,
    )
    partition = "success" if episode_return >= episode_return_success_threshold else "failure"
    inserted_steps = replay.add_episode(partition, _episode_to_tensors(episode_trajectory))
    return partition, episode_return, episode_return_success_threshold, inserted_steps


def _load_replay_from_checkpoint_file(
    *,
    model_path: str,
    replay: SharedTD3Replay,
) -> bool:
    if not os.path.exists(model_path):
        return False
    loaded_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded_obj, dict):
        return False
    if "success_replay_buffer" not in loaded_obj or "failure_replay_buffer" not in loaded_obj:
        return False
    replay.load_state_dict(
        {
            "success": loaded_obj["success_replay_buffer"],
            "failure": loaded_obj["failure_replay_buffer"],
        }
    )
    snapshot = replay.state_snapshot()
    print(
        "[resume_replay] loaded from checkpoint "
        f"success_rb={snapshot['success']['size']} failure_rb={snapshot['failure']['size']}"
    )
    return True


def _load_runtime_perf_from_checkpoint_file(model_path: str) -> Dict[str, object]:
    runtime_state: Dict[str, object] = {
        "collector_total_steps": 0.0,
        "run_elapsed_total_s": 0.0,
        "rolling50_task_reward_values": [],
        "rolling50_motion_reward_values": [],
        "rolling50_episode_length_values": [],
        "rolling50_estop_episode_flags": [],
    }
    if not model_path or not os.path.exists(model_path):
        return runtime_state
    try:
        loaded_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return runtime_state
    if not isinstance(loaded_obj, dict):
        return runtime_state
    runtime_state["collector_total_steps"] = float(loaded_obj.get("collector_total_steps", 0.0))
    runtime_state["run_elapsed_total_s"] = float(loaded_obj.get("run_elapsed_total_s", 0.0))
    runtime_state["rolling50_task_reward_values"] = _coerce_float_list(
        loaded_obj.get("rolling50_task_reward_values", []),
        max_items=ROLLING_PERF_WINDOW_EPISODES,
    )
    runtime_state["rolling50_motion_reward_values"] = _coerce_float_list(
        loaded_obj.get("rolling50_motion_reward_values", []),
        max_items=ROLLING_PERF_WINDOW_EPISODES,
    )
    runtime_state["rolling50_episode_length_values"] = _coerce_float_list(
        loaded_obj.get("rolling50_episode_length_values", []),
        max_items=ROLLING_PERF_WINDOW_EPISODES,
    )
    runtime_state["rolling50_estop_episode_flags"] = _coerce_float_list(
        loaded_obj.get("rolling50_estop_episode_flags", []),
        max_items=ROLLING_PERF_WINDOW_EPISODES,
    )
    return runtime_state


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
    if episode_steps < 50:
        return "<50"
    if episode_steps <= 100:
        return "50-100"
    if episode_steps <= 200:
        return "100-200"
    return ">200"


def _bucketed_output_dir(base_dir: str | Path, episode_steps: int) -> str:
    """Return a length-bucketed output directory under the given parent."""
    return str(Path(base_dir).expanduser().resolve() / _episode_length_bucket_name(episode_steps))


def _stop_output_dir(base_dir: str | Path, stop_label: str) -> str:
    """Return the additional stop-specific output directory under the parent."""
    return str(Path(base_dir).expanduser().resolve() / str(stop_label))


def _reset_output_dir(base_dir: str | Path, partition: str, episode_steps: int) -> str:
    """Return the reset artifact directory under success/failure partition buckets."""
    partition_root = Path(base_dir).expanduser().resolve() / str(partition)
    return _bucketed_output_dir(partition_root, episode_steps)


def _copy_to_stop_dir(file_path: str | Path, stop_root: str | Path) -> Path:
    """Copy one artifact file into the stop-specific directory and return destination."""
    src = Path(file_path).expanduser().resolve()
    stop_root_path = Path(stop_root).expanduser().resolve()
    if src.suffix == ".hdf5":
        dst_dir = stop_root_path
    else:
        # Keep per-episode media grouping (trajectory_data{N}/...).
        dst_dir = stop_root_path / src.parent.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _safe_nonnegative_ms(value: float) -> float:
    value_f = float(value)
    if not np.isfinite(value_f):
        return 0.0
    return max(0.0, value_f)


def _env_timing_info(env: AirHockeyEnv) -> dict:
    state_info = getattr(env, "current_state", None)
    if not isinstance(state_info, dict):
        return {}
    timing_info = state_info.get("timing", {})
    if not isinstance(timing_info, dict):
        return {}
    return timing_info


def _latency_bucket_stats(values_ms: list[float]) -> dict:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p90_ms": float("nan"),
            "p99_ms": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _write_latency_profile_episode(
    output_dir: str | Path,
    episode_id: int,
    puck_detection_ms: list[float],
    model_inference_ms: list[float],
    block_sleep_ms: list[float],
    other_ms: list[float],
    hist_bins: int,
) -> tuple[Path, Path, dict]:
    # Keep matplotlib import local so normal training path has no extra import cost.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    bucket_arrays = {
        "puck_detection_ms": np.asarray(puck_detection_ms, dtype=np.float64),
        "model_inference_ms": np.asarray(model_inference_ms, dtype=np.float64),
        "block_sleep_ms": np.asarray(block_sleep_ms, dtype=np.float64),
        "other_ms": np.asarray(other_ms, dtype=np.float64),
    }
    summary = {
        "episode_id": int(episode_id),
        "step_count": int(max((arr.size for arr in bucket_arrays.values()), default=0)),
        "buckets": {name: _latency_bucket_stats(arr.tolist()) for name, arr in bucket_arrays.items()},
        "per_step_ms": {name: arr.tolist() for name, arr in bucket_arrays.items()},
    }
    json_path = output_path / f"episode_{int(episode_id):06d}_latency.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Episode {int(episode_id)} latency (ms)")
    ordered_names = ("puck_detection_ms", "model_inference_ms", "block_sleep_ms", "other_ms")
    bins = max(5, int(hist_bins))
    for axis, name in zip(axes.flatten(), ordered_names):
        values = bucket_arrays[name]
        if values.size > 0:
            axis.hist(values, bins=bins)
        axis.set_title(name)
        axis.set_xlabel("ms")
        axis.set_ylabel("count")
    fig.tight_layout()
    png_path = output_path / f"episode_{int(episode_id):06d}_latency.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return json_path, png_path, summary


def _build_split_episode_row(
    env: AirHockeyEnv,
    action_xy: np.ndarray,
    episode_id: int,
    episode_step_idx: int,
    protective_stop_active: bool,
    controller_disconnected: bool,
    reset_stage_id: int | None = None,
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

    pose = vector_with_width(np.concatenate([paddle_pos[:2], np.zeros(4, dtype=np.float64)]), 6)
    speed = vector_with_width(np.concatenate([paddle_vel[:2], np.zeros(4, dtype=np.float64)]), 6)
    # TODO: Source force/acc/safety/estop directly from robot telemetry for real deployment.
    force = np.zeros((6,), dtype=np.float64)
    acc = np.zeros((3,), dtype=np.float64)
    desired_pose = vector_with_width(np.concatenate([desired_xy, np.zeros(4, dtype=np.float64)]), 6)
    puck = vector_with_width(np.concatenate([puck_position[:2], np.array([puck_occluded])]), 3)
    timing_info = state_info.get("timing", {}) if isinstance(state_info, dict) else {}
    paddle_actual_pose = np.asarray(state_info.get("paddle_actual_pose", np.zeros(6)), dtype=np.float64).reshape(-1)
    paddle_actual_speed = np.asarray(state_info.get("paddle_actual_speed", np.zeros(6)), dtype=np.float64).reshape(-1)
    paddle_target_pre = np.asarray(
        state_info.get("paddle_target_pose_pre_filter", np.zeros(6)), dtype=np.float64
    ).reshape(-1)
    paddle_target_post = np.asarray(
        state_info.get("paddle_target_pose_post_filter", np.zeros(6)), dtype=np.float64
    ).reshape(-1)
    paddle_actual = vector_with_width(np.concatenate([paddle_actual_pose[:2], paddle_actual_speed[:2], np.zeros(2)]), 6)
    paddle_cmd = vector_with_width(np.concatenate([paddle_target_pre[:6], paddle_target_post[:6]]), 12)
    timing = vector_with_width(
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
    puck_meta = vector_with_width(
        np.array(
            [
                puck_occluded,
                1.0 if bool(state_info.get("puck_detector_used_fallback", puck_occluded > 0.5)) else 0.0,
            ],
            dtype=np.float64,
        ),
        2,
    )
    stop_flags = vector_with_width(
        np.array(
            [
                1.0 if protective_stop_active else 0.0,
                1.0 if controller_disconnected else 0.0,
                1.0 if (protective_stop_active or controller_disconnected) else 0.0,
            ],
            dtype=np.float64,
        ),
        3,
    )

    row = {
        "cur_time": np.array([time.time()], dtype=np.float64),
        "tidx": np.array([float(episode_id)], dtype=np.float64),
        "i": np.array([float(episode_step_idx)], dtype=np.float64),
        "estop": np.array([1.0 if protective_stop_active else 0.0], dtype=np.float64),
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
        "stop_flags": stop_flags,
    }
    if reset_stage_id is not None:
        row["reset_stage_id"] = np.array([float(reset_stage_id)], dtype=np.float64)
    return row


def _mixed_sample_from_shared(
    replay: SharedTD3Replay,
    batch_size: int,
    success_fraction: float,
    device: str,
) -> tuple[Dict[str, torch.Tensor] | None, int, int]:
    success_count, failure_count = critic_success_failure_counts(
        batch_size=batch_size,
        success_fraction=success_fraction,
        success_available=replay.len("success") > 0,
        failure_available=replay.len("failure") > 0,
    )
    if success_count + failure_count == 0:
        return None, success_count, failure_count

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
    return batch, success_count, failure_count


def _detect_estop(env: AirHockeyEnv, step_info: dict | None = None) -> bool:
    """Detect protective stops without conflating them with command-readiness metadata."""
    return _classify_stop_event(env, step_info=step_info).protective_stop


def _should_run_reset_policy_at_episode_start(
    state_info: dict | None,
    table_x_bot: float | None,
    bottom_margin: float,
    bottom_fail_count: int,
    occluded_fail_count: int,
    counters: dict[str, int],
) -> bool:
    """Decide whether to enter reset-policy mode at episode start."""
    if not isinstance(state_info, dict) or table_x_bot is None or not np.isfinite(float(table_x_bot)):
        return False
    try:
        puck = state_info["pucks"][0]
        puck_x = float(puck["position"][0])
        puck_occ = int(np.asarray(puck.get("occluded", 0)).reshape(-1)[0]) > 0
    except Exception:
        return False

    if puck_x >= (float(table_x_bot) - float(bottom_margin)):
        counters["bottom"] = int(counters.get("bottom", 0)) + 1
    else:
        counters["bottom"] = 0

    if puck_occ:
        counters["occ"] = int(counters.get("occ", 0)) + 1
    else:
        counters["occ"] = 0

    return bool(
        counters["bottom"] >= int(bottom_fail_count) or counters["occ"] >= int(occluded_fail_count)
    )


def _hard_reset_with_pause(env: AirHockeyEnv, reason: str, pause_s: float = 3.0) -> tuple[np.ndarray, dict]:
    """Force physical env reset, then wait before returning to policy collection."""
    print(f"[collector_fallback_reset] reason={reason} -> hard env reset")
    simulator = getattr(env, "simulator", None)
    if simulator is not None:
        if hasattr(simulator, "wait_for_space_to_start"):
            try:
                simulator.wait_for_space_to_start = False
            except Exception:
                pass
        real_env = getattr(simulator, "air_hockey_env", None)
        if real_env is not None and hasattr(real_env, "wait_for_space_to_start"):
            try:
                real_env.wait_for_space_to_start = False
            except Exception:
                pass
    supports_write_traj = False
    try:
        reset_signature = inspect.signature(env.reset)
        supports_write_traj = "write_traj" in reset_signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in reset_signature.parameters.values()
        )
    except (TypeError, ValueError):
        supports_write_traj = False
    if supports_write_traj:
        obs, info = env.reset(seed=None, write_traj=False)
    else:
        obs, info = env.reset(seed=None)
    print(f"[collector_fallback_reset] sleeping {pause_s:.1f}s before resume")
    time.sleep(float(pause_s))
    return obs, info


def _prime_paddle_history_stand_still_non_occluded(env: AirHockeyEnv) -> np.ndarray:
    """Fill paddle history with stationary non-occluded entries and rebuild observation."""
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        obs, _ = env.get_current_state()
        return np.asarray(obs, dtype=np.float32)
    state_info = simulator.get_current_state()
    try:
        paddle_position = np.asarray(
            state_info["paddles"]["paddle_ego"]["position"],
            dtype=np.float64,
        ).reshape(-1)
        paddle_x = float(paddle_position[0])
        paddle_y = float(paddle_position[1])
    except Exception:
        return env.get_observation(
            state_info,
            obs_type=env.obs_type,
            puck_history=simulator.puck_history,
            paddle_history=simulator.paddle_history,
        )
    history_len = int(getattr(simulator, "paddle_history_len", 5))
    simulator.paddle_history = [(paddle_x, paddle_y, 0) for _ in range(max(1, history_len))]
    return env.get_observation(
        state_info,
        obs_type=env.obs_type,
        puck_history=simulator.puck_history,
        paddle_history=simulator.paddle_history,
    )


def _normalize_transition_last_action_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"zero", "executed", "keep"}:
        return mode_norm
    print(f"[collector_transition] unsupported transition_last_action_mode='{mode}', defaulting to 'zero'")
    return "zero"


def _request_sim_transition_hold(env: AirHockeyEnv, steps: int, reason: str) -> bool:
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return False
    begin_fn = getattr(simulator, "begin_transition_hold", None)
    if not callable(begin_fn):
        return False
    try:
        begin_fn(int(steps), reason=str(reason))
        return True
    except Exception as exc:
        print(f"[collector_transition] begin_transition_hold failed for reason={reason}: {exc}")
        return False


def _simulator_step_readiness(env: AirHockeyEnv) -> tuple[bool, str]:
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return True, "no_simulator"
    readiness_fn = getattr(simulator, "robot_command_readiness", None)
    if callable(readiness_fn):
        try:
            readiness = readiness_fn()
            step_ready = bool(readiness.get("step_ready", True))
            reason = str(readiness.get("reason", "ready"))
            return step_ready, reason
        except Exception as exc:
            return False, f"readiness_exception:{exc.__class__.__name__}"
    return (not _detect_estop(env)), "legacy_estop_fallback"


def collector_process(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    learner_state: LearnerRuntimeState,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.collector_device)
    episode_start_reset_bottom_margin = 0.25
    episode_start_reset_bottom_fail_count = 2
    episode_start_reset_occluded_fail_count = 6
    episode_start_reset_counters = {"bottom": 0, "occ": 0}

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    collector_config = _prepare_air_hockey_config(config, seed=args.seed)
    sim_params = collector_config.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
        sim_params["transition_hold_steps_on_estop_enter"] = int(args.transition_hold_steps_post_estop_enter)
        sim_params["transition_hold_steps_on_estop_clear"] = int(args.transition_hold_steps_post_estop_clear)
        sim_params["transition_hold_steps_on_safety_rearm"] = int(args.transition_hold_steps_post_safety_rearm)
    env = AirHockeyEnv(collector_config)
    writer = SummaryWriter(tb_log_dir)
    latency_output_dir: Path | None = None
    if args.enable_latency_profiling:
        if args.latency_profile_output_dir is not None:
            latency_output_dir = Path(args.latency_profile_output_dir).expanduser().resolve()
        else:
            latency_output_dir = Path(tb_log_dir).resolve().parent / "latency_profiles"
        latency_output_dir.mkdir(parents=True, exist_ok=True)

    policy_obs_dim = obs_dim + act_dim if train_args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    actor.eval()

    action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0)
    action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0)
    primitive_selector = build_primitive_exploration_selector_for_real_collector(
        args, device, initial_total_steps=0
    )
    primitive_selector.set_primitive_weights(
        stand_still=float(args.exploration_primitive_weight_stand_still),
        same_direction=float(args.exploration_primitive_weight_same_direction),
        y_aligned=float(args.exploration_primitive_weight_y_aligned),
        policy_takeover=0.0,
        target_position_directional=float(args.exploration_primitive_weight_target_position_directional),
        pre_contact_hit_variant=0.0,
    )
    actor.load_state_dict(
        {key: value.detach().cpu() for key, value in learner_state.actor.state_dict().items()},
        strict=False,
    )

    next_reset_file_id = _next_available_episode_id(args.reset_artifact_dir)
    if next_reset_file_id > 0:
        print(
            f"[collector] continuing reset artifact ids from {next_reset_file_id} "
            f"(existing data found in {args.reset_artifact_dir})"
        )
    pending_reset_artifact: PendingResetArtifact | None = None
    reset_rng = np.random.default_rng(args.seed)
    # Commit to one startup behavior: run reset FSM once before policy takeover.
    # This avoids an immediate policy->reset mode flip in the first episode.
    startup_reset_result = run_reset_fsm(
        env,
        reset_rng,
        artifact_episode_id=next_reset_file_id,
    )
    reset_fsm_steps_total = startup_reset_result.total_steps
    pending_reset_artifact, next_reset_file_id = merge_reset_fsm_artifact_into_pending(
        startup_reset_result.artifact,
        pending_reset_artifact,
        next_reset_file_id,
        startup_buffered_message=True,
    )
    obs, previous_puck_position_for_primitive = soft_reset_prime_paddle_and_extract_previous_puck(
        env,
        device=device,
        prime_paddle_history_stand_still_non_occluded=_prime_paddle_history_stand_still_non_occluded,
        extract_primitive_state_tensors=_extract_primitive_state_tensors,
    )
    episode_trajectory = EpisodeTrajectory.empty()
    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0
    last_action_for_policy = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
    transition_last_action_mode = _normalize_transition_last_action_mode(args.transition_last_action_mode)
    transition_hold_steps_remaining = 0
    transition_hold_reason = "none"
    transition_hold_events_total = 0
    transition_hold_reason_counts: dict[str, int] = {}
    last_executed_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
    interval_primitive_env_steps = 0
    interval_primitive_horizontal_env_steps = 0
    interval_target_position_directional_env_steps = 0

    def begin_transition_hold(
        reason: str,
        hold_steps: int,
        *,
        request_sim_hold: bool = True,
    ) -> None:
        nonlocal transition_hold_steps_remaining
        nonlocal transition_hold_reason
        nonlocal transition_hold_events_total
        nonlocal last_action_for_policy
        nonlocal previous_puck_position_for_primitive
        hold_steps = max(int(hold_steps), 0)
        transition_hold_events_total += 1
        transition_hold_reason_counts[reason] = int(transition_hold_reason_counts.get(reason, 0)) + 1
        transition_hold_reason = str(reason)
        transition_hold_steps_remaining = max(int(transition_hold_steps_remaining), hold_steps)
        _reset_primitive_rollout_state(primitive_selector)
        _, previous_puck_position_for_primitive, _ = _extract_primitive_state_tensors(env, device=device)
        if train_args.use_last_action_in_policy_state:
            if transition_last_action_mode == "zero":
                last_action_for_policy.zero_()
            elif transition_last_action_mode == "executed":
                last_action_for_policy = last_executed_action.detach().clone()
        sim_hold_started = False
        if request_sim_hold and hold_steps > 0:
            sim_hold_started = _request_sim_transition_hold(env, steps=hold_steps, reason=reason)
        print(
            "[collector_transition] "
            f"reason={reason} hold_steps={hold_steps} "
            f"collector_hold_remaining={transition_hold_steps_remaining} "
            f"sim_hold_started={sim_hold_started} last_action_mode={transition_last_action_mode}"
        )

    begin_transition_hold(
        reason="startup_reset_to_policy",
        hold_steps=int(args.transition_hold_steps_post_reset),
        request_sim_hold=True,
    )

    total_steps = int(stats.get("collector_total_steps", stats.get("collector_steps", 0.0)))
    total_episodes = 0
    next_episode_file_id = _next_available_episode_id(args.episode_artifact_dir)
    if next_episode_file_id > 0:
        print(
            f"[collector] continuing episode artifact ids from {next_episode_file_id} "
            f"(existing data found in {args.episode_artifact_dir})"
        )
    last_log_time = time.time()
    episode_rows = []
    episode_puck_detection_latency_ms: list[float] = []
    episode_model_inference_latency_ms: list[float] = []
    episode_block_sleep_latency_ms: list[float] = []
    episode_other_latency_ms: list[float] = []
    episode_images: list[np.ndarray] = []
    episode_camera_null_frames = 0
    episodes_saved = 0
    episodes_removed_short = 0
    episodes_removed_invalid = 0
    episodes_gif_generated = 0
    episodes_gif_failed = 0
    episodes_camera_video_generated = 0
    episodes_camera_video_failed = 0
    successful_online_episodes_kept = int(stats.get("successful_online_episodes_kept", 0))
    checkpoint_save_request_id = int(stats.get("checkpoint_save_request_id", 0))
    protective_stop_episodes = 0
    protective_stop_steps = 0
    controller_disconnect_episodes = 0
    controller_disconnect_steps = 0
    transition_hold_steps_total = 0
    stop_penalty_applied_this_episode = False
    episode_had_stop = False
    episode_had_protective_stop = False
    episode_had_controller_disconnect = False
    episode_had_readiness_fail_estop = False
    episode_readiness_first_fail_step_idx: int | None = None
    episode_readiness_first_fail_reason: str | None = None
    motion_metric_names = (
        "temporal_valid_fraction",
        "stand_still_reward_raw",
        "temporal_alignment_reward_raw",
        "axis_alignment_reward_raw",
        "velocity_reward_raw",
        "jerk_reward_raw",
        "stand_still_reward_weighted",
        "temporal_alignment_reward_weighted",
        "axis_alignment_reward_weighted",
        "velocity_reward_weighted",
        "jerk_reward_weighted",
    )
    episode_motion_metric_sums = {name: 0.0 for name in motion_metric_names}
    episode_motion_metric_count = 0
    initial_state_info = getattr(env, "current_state", None)
    if not isinstance(initial_state_info, dict):
        simulator = getattr(env, "simulator", None)
        if simulator is not None and hasattr(simulator, "get_current_state"):
            try:
                initial_state_info = simulator.get_current_state()
            except Exception:
                initial_state_info = None
    initial_paddle_xy, initial_puck_xy = _extract_motion_positions_from_state_info(initial_state_info)
    motion_reward_state = _init_motion_reward_state(
        int(args.temporal_alignment_horizon),
        anchor_paddle_xy=initial_paddle_xy,
        anchor_puck_xy=initial_puck_xy,
    )
    rolling50_task_reward_values = deque(
        _coerce_float_list(
            stats.get("rolling50_task_reward_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        maxlen=ROLLING_PERF_WINDOW_EPISODES,
    )
    rolling50_motion_reward_values = deque(
        _coerce_float_list(
            stats.get("rolling50_motion_reward_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        maxlen=ROLLING_PERF_WINDOW_EPISODES,
    )
    rolling50_episode_length_values = deque(
        _coerce_float_list(
            stats.get("rolling50_episode_length_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        maxlen=ROLLING_PERF_WINDOW_EPISODES,
    )
    rolling50_estop_episode_flags = deque(
        _coerce_float_list(
            stats.get("rolling50_estop_episode_flags", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        maxlen=ROLLING_PERF_WINDOW_EPISODES,
    )
    collector_elapsed_resume_offset_s = float(stats.get("run_elapsed_total_s", 0.0))
    collector_start_time = time.time()
    episodic_returns: list[float] = []
    episodic_lengths: list[float] = []
    success_rates: list[float] = []
    readiness_fail_streak = 0
    readiness_fail_first_episode_step_idx: int | None = None
    readiness_fail_first_total_step: int | None = None
    readiness_fail_window = 5
    readiness_fail_steps_total = 0
    readiness_fail_estop_episodes = 0
    readiness_fail_estop_dropped_steps_total = 0
    readiness_fail_prev = False
    readiness_fail_prev_reason = "none"

    while True:
        if args.smoke_test_seconds > 0 and (time.time() - collector_start_time) >= float(args.smoke_test_seconds):
            print(f"[collector] smoke-test duration reached ({args.smoke_test_seconds:.1f}s), stopping.")
            break
        step_ready, step_ready_reason = _simulator_step_readiness(env)
        if not step_ready:
            readiness_fail_steps_total += 1
            if readiness_fail_streak == 0:
                readiness_fail_first_episode_step_idx = int(len(episode_rows))
                readiness_fail_first_total_step = int(total_steps)
                episode_readiness_first_fail_step_idx = readiness_fail_first_episode_step_idx
                episode_readiness_first_fail_reason = str(step_ready_reason)
            readiness_fail_streak += 1
            if (not readiness_fail_prev) or (step_ready_reason != readiness_fail_prev_reason):
                print(
                    "[collector_safety] "
                    f"robot_step_ready=False reason={step_ready_reason}; continuing collection "
                    f"(consecutive_failures={readiness_fail_streak}/{readiness_fail_window})"
                )
            elif readiness_fail_streak <= readiness_fail_window:
                print(
                    "[collector_safety] "
                    f"robot_step_ready still false reason={step_ready_reason}; "
                    f"consecutive_failures={readiness_fail_streak}/{readiness_fail_window}"
                )
            readiness_fail_prev = True
            readiness_fail_prev_reason = str(step_ready_reason)
        else:
            if readiness_fail_prev:
                recovered_from_reason = str(readiness_fail_prev_reason)
                recovered_streak = int(readiness_fail_streak)
                had_triggered_window = recovered_streak >= readiness_fail_window
                print(
                    "[collector_safety] "
                    f"robot step readiness restored after reason={recovered_from_reason}; "
                    f"consecutive_failures={recovered_streak} "
                    f"window_triggered={int(had_triggered_window)}"
                )
            readiness_fail_streak = 0
            readiness_fail_first_episode_step_idx = None
            readiness_fail_first_total_step = None
            episode_readiness_first_fail_step_idx = None
            episode_readiness_first_fail_reason = None
            readiness_fail_prev = False
            readiness_fail_prev_reason = "none"
        if readiness_fail_prev and readiness_fail_streak == readiness_fail_window:
            print(
                "[collector_safety] "
                f"readiness failure window reached ({readiness_fail_window} consecutive); "
                f"will terminate episode at first failure step "
                f"(episode_step_idx={readiness_fail_first_episode_step_idx}, "
                f"total_step={readiness_fail_first_total_step})"
            )

        collector_step_start_s = time.perf_counter() if args.enable_latency_profiling else 0.0
        transition_hold_active = bool(transition_hold_steps_remaining > 0)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        policy_obs = augment_policy_observation(
            obs_tensor,
            last_action_for_policy,
            train_args.use_last_action_in_policy_state,
        )
        model_inference_ms = 0.0
        primitive_step_stats = {
            "primitive_applied_count": 0,
            "primitive_horizontal_dominant_count": 0,
            "target_position_directional_applied_count": 0,
        }
        with torch.no_grad():
            inference_start_s = time.perf_counter() if args.enable_latency_profiling else 0.0
            action_tensor = deterministic_actor_action(actor, policy_obs)
            disable_noise_for_transition = bool(
                transition_hold_active and args.transition_disable_exploration_noise
            )
            if args.exploration_noise > 0 and not disable_noise_for_transition:
                action_tensor = action_tensor + torch.randn_like(action_tensor) * float(args.exploration_noise)
            action_tensor = torch.clamp(action_tensor, action_low, action_high)
            if not transition_hold_active and not args.collector_policy_stand_still:
                primitive_selector.chance = float(primitive_exploration_chance_for_step(args, total_steps))
                current_paddle_pos, current_puck_pos, current_puck_vel = _extract_primitive_state_tensors(
                    env,
                    device=device,
                )
                if torch.all(current_puck_vel == 0):
                    current_puck_vel = current_puck_pos - previous_puck_position_for_primitive
                y_alignment_sign = torch.sign(current_puck_pos[:, 1] - current_paddle_pos[:, 1])
                action_tensor, primitive_step_stats = primitive_selector.apply(
                    action_tensor,
                    action_low=action_low,
                    action_high=action_high,
                    y_alignment_sign=y_alignment_sign,
                    current_paddle_position=current_paddle_pos,
                    current_puck_position=current_puck_pos,
                    current_puck_velocity=current_puck_vel,
                    return_stats=True,
                )
            if transition_hold_active:
                action_tensor = torch.zeros_like(action_tensor)
            if args.collector_policy_stand_still:
                action_tensor = torch.zeros_like(action_tensor)

        env_action = action_tensor.squeeze(0).detach().cpu().numpy()
        if args.enable_latency_profiling: # includes transferring between CPU and GPU
                model_inference_ms = _safe_nonnegative_ms((time.perf_counter() - inference_start_s) * 1000.0)

        prev_action = last_action_for_policy.clone()
        next_obs, task_reward, terminations, truncations, step_info = env.step(env_action)
        if args.enable_latency_profiling:
            collector_step_end_s = time.perf_counter()
            step_total_ms = _safe_nonnegative_ms((collector_step_end_s - collector_step_start_s) * 1000.0)
            timing_info = _env_timing_info(env)
            camera_received_s = float(timing_info.get("camera_frame_received_s", float("nan")))
            puck_done_s = float(timing_info.get("puck_detection_done_s", float("nan")))
            if np.isfinite(camera_received_s) and np.isfinite(puck_done_s):
                puck_detection_ms = _safe_nonnegative_ms((puck_done_s - camera_received_s) * 1000.0)
            else:
                puck_detection_ms = 0.0
            block_sleep_ms = _safe_nonnegative_ms(float(timing_info.get("sleep_before_step_s", 0.0)) * 1000.0)
            other_ms = _safe_nonnegative_ms(
                step_total_ms - model_inference_ms - puck_detection_ms - block_sleep_ms
            )
            episode_model_inference_latency_ms.append(model_inference_ms)
            episode_puck_detection_latency_ms.append(puck_detection_ms)
            episode_block_sleep_latency_ms.append(block_sleep_ms)
            episode_other_latency_ms.append(other_ms)
        camera_frame = _latest_camera_frame(env)
        if camera_frame is not None:
            episode_images.append(camera_frame)
        else:
            episode_camera_null_frames += 1
        stop_state = _classify_stop_event(env, step_info=step_info)
        readiness_fail_stop_now = bool(
            readiness_fail_streak >= readiness_fail_window
            and episode_readiness_first_fail_step_idx is not None
        )
        # Keep collector stop behavior for normal states, but while readiness is
        # currently failing we defer termination to the 5-step readiness rule.
        stop_now = bool(stop_state.active and step_ready)
        if readiness_fail_stop_now:
            stop_now = True
        dones = bool(np.logical_or(terminations, truncations) or stop_now)
        # td3_training replay `dones`: env terminations (+ collector stop), not truncation-only.
        terminations_tensor = torch.tensor(
            float(bool(terminations or stop_now)),
            dtype=torch.float32,
            device=device,
        )
        if stop_state.protective_stop:
            protective_stop_steps += 1
            episode_had_protective_stop = True
        if stop_state.controller_disconnected:
            controller_disconnect_steps += 1
            episode_had_controller_disconnect = True
        if readiness_fail_stop_now:
            episode_had_readiness_fail_estop = True
        if stop_now:
            episode_had_stop = True

        next_state_info = getattr(env, "current_state", None)
        next_paddle_xy, next_puck_xy = _extract_motion_positions_from_state_info(next_state_info)
        velocity_mag, _, jerk_mag = _extract_motion_magnitudes_from_step_info(step_info, motion_reward_state)
        motion_components = _compute_motion_reward_components(
            args=args,
            motion_state=motion_reward_state,
            paddle_xy=next_paddle_xy,
            puck_xy=next_puck_xy,
            velocity_mag=velocity_mag,
            jerk_mag=jerk_mag,
        )
        motion_reward = float(motion_components["motion_reward_total"])
        for metric_name in motion_metric_names:
            episode_motion_metric_sums[metric_name] += float(motion_components[metric_name])
        episode_motion_metric_count += 1
        if stop_now and not stop_penalty_applied_this_episode:
            motion_reward += -5.0
            stop_penalty_applied_this_episode = True

        episode_rows.append(
            _build_split_episode_row(
                env=env,
                action_xy=env_action,
                episode_id=next_episode_file_id,
                episode_step_idx=len(episode_rows),
                protective_stop_active=stop_state.protective_stop,
                controller_disconnected=stop_state.controller_disconnected,
            )
        )
        interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
        interval_primitive_horizontal_env_steps += int(
            primitive_step_stats["primitive_horizontal_dominant_count"]
        )
        interval_target_position_directional_env_steps += int(
            primitive_step_stats["target_position_directional_applied_count"]
        )

        episode_trajectory.append_step(
            obs=obs_tensor[0],
            next_obs=torch.as_tensor(next_obs, dtype=torch.float32, device=device),
            action=action_tensor[0],
            task_reward=torch.tensor(float(task_reward), dtype=torch.float32, device=device),
            motion_reward=torch.tensor(float(motion_reward), dtype=torch.float32, device=device),
            done=terminations_tensor,
            prev_action=prev_action[0],
        )
        total_steps += 1
        if transition_hold_active:
            transition_hold_steps_total += 1
        last_executed_action = action_tensor.detach().clone()
        previous_puck_position_for_primitive = _extract_primitive_state_tensors(env, device=device)[1]
        primitive_selector.reset(torch.tensor([dones], dtype=torch.bool, device=device))

        if train_args.use_last_action_in_policy_state:
            if not (transition_hold_active and transition_last_action_mode == "keep"):
                last_action_for_policy = last_executed_action.clone()
        obs = next_obs
        if transition_hold_active:
            transition_hold_steps_remaining = max(0, int(transition_hold_steps_remaining) - 1)
            if args.transition_hold_log_every_step:
                print(
                    "[collector_transition] "
                    f"hold_step reason={transition_hold_reason} remaining={transition_hold_steps_remaining}"
                )
            elif transition_hold_steps_remaining == 0:
                print(f"[collector_transition] hold_complete reason={transition_hold_reason}")

        if dones:
            episode_end_wall_time = time.time()
            total_episodes += 1
            if episode_had_protective_stop:
                protective_stop_episodes += 1
            if episode_had_controller_disconnect:
                controller_disconnect_episodes += 1
            readiness_fail_dropped_steps = 0
            if episode_had_readiness_fail_estop and episode_readiness_first_fail_step_idx is not None:
                (
                    readiness_fail_dropped_steps,
                    episode_rows,
                    episode_images,
                    episode_puck_detection_latency_ms,
                    episode_model_inference_latency_ms,
                    episode_block_sleep_latency_ms,
                    episode_other_latency_ms,
                    episode_camera_null_frames,
                ) = truncate_collector_episode_for_readiness_fail(
                    episode_trajectory=episode_trajectory,
                    episode_readiness_first_fail_step_idx=episode_readiness_first_fail_step_idx,
                    episode_rows=episode_rows,
                    episode_images=episode_images,
                    episode_puck_detection_latency_ms=episode_puck_detection_latency_ms,
                    episode_model_inference_latency_ms=episode_model_inference_latency_ms,
                    episode_block_sleep_latency_ms=episode_block_sleep_latency_ms,
                    episode_other_latency_ms=episode_other_latency_ms,
                    episode_camera_null_frames=episode_camera_null_frames,
                    device=device,
                )
                readiness_fail_estop_dropped_steps_total += int(readiness_fail_dropped_steps)
                print(
                    "[collector_safety] "
                    f"episode_id={next_episode_file_id} readiness_fail_estop=1 "
                    f"first_fail_step_idx={episode_readiness_first_fail_step_idx} "
                    f"dropped_post_fail_steps={readiness_fail_dropped_steps} "
                    f"reason={episode_readiness_first_fail_reason}"
                )
            episode_return = float(episode_trajectory.episode_return)
            episode_length = float(len(episode_trajectory.observations))
            episode_task_reward = float(
                torch.stack(episode_trajectory.task_rewards, dim=0).sum().item()
            )
            episode_motion_reward = float(
                torch.stack(episode_trajectory.motion_rewards, dim=0).sum().item()
            )
            episode_estop_flag = 1.0 if (episode_had_protective_stop or episode_had_readiness_fail_estop) else 0.0
            rolling50_task_reward_values.append(episode_task_reward)
            rolling50_motion_reward_values.append(episode_motion_reward)
            rolling50_episode_length_values.append(episode_length)
            rolling50_estop_episode_flags.append(episode_estop_flag)
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
            if stop_now:
                # Collector can end an episode on stop conditions even if env did not set done metadata.
                episode_end_type = stop_state.episode_end_type
                episode_end_reasons = [str(stop_state.episode_end_reason)]
                episode_end_reason = str(stop_state.episode_end_reason)
            episode_stop_artifact_label = stop_state.artifact_label if stop_now else None
            if episode_had_readiness_fail_estop:
                episode_end_type = "estop"
                episode_end_reasons = ["collector_readiness_fail_5steps"]
                episode_end_reason = "collector_readiness_fail_5steps"
                episode_stop_artifact_label = "estop"
                readiness_fail_estop_episodes += 1
            success_rates.append(1.0 if episode_success else 0.0)
            writer.add_scalar("charts/episodic_return", episode_return, total_steps)
            writer.add_scalar("charts/episodic_length", episode_length, total_steps)
            writer.add_scalar("charts/episodic_success", float(1.0 if episode_success else 0.0), total_steps)
            if episode_motion_metric_count > 0:
                for metric_name in motion_metric_names:
                    metric_mean = float(episode_motion_metric_sums[metric_name] / float(episode_motion_metric_count))
                    stats[f"rewards/{metric_name}_mean"] = metric_mean
                    writer.add_scalar(f"rewards/{metric_name}_mean", metric_mean, total_steps)
            _, _, episode_return_success_threshold, _ = _add_episode_to_shared_replay(
                replay=replay,
                episode_trajectory=episode_trajectory,
                recent_episode_returns=recent_episode_returns,
                success_top_fraction=args.success_top_fraction,
            )
            actor_updated = _run_sync_learner_iteration(
                args=args,
                train_args=train_args,
                replay=replay,
                stats=stats,
                state=learner_state,
            )
            if actor_updated:
                actor.load_state_dict(
                    {key: value.detach().cpu() for key, value in learner_state.actor.state_dict().items()},
                    strict=False,
                )
                begin_transition_hold(
                    reason="actor_sync_update",
                    hold_steps=int(args.transition_hold_steps_post_actor_sync),
                    request_sim_hold=False,
                )
            episode_trajectory.reset()
            _reset_primitive_rollout_state(primitive_selector)

            # Do all slow I/O and actor sync BEFORE the reset FSM,
            # so the transition from reset to policy is immediate.
            n_episode_steps = len(episode_rows)
            n_camera_frames = len(episode_images)
            has_camera_images = n_camera_frames > 0
            elapsed_s = max(0.0, collector_elapsed_resume_offset_s + (time.time() - collector_start_time))
            elapsed_min = elapsed_s / 60.0
            elapsed_hr = elapsed_s / 3600.0
            rolling50_m = compute_rolling50_metrics(
                rolling50_task_reward_values,
                rolling50_motion_reward_values,
                rolling50_episode_length_values,
                rolling50_estop_episode_flags,
            )
            stats["run_elapsed_total_s"] = float(elapsed_s)
            stats["collector_steps"] = float(total_steps)
            stats["collector_total_steps"] = float(total_steps)
            update_stats_dict_rolling50(
                stats,
                rolling50_m,
                window_size=ROLLING_PERF_WINDOW_EPISODES,
                rolling50_task_reward_values=rolling50_task_reward_values,
                rolling50_motion_reward_values=rolling50_motion_reward_values,
                rolling50_episode_length_values=rolling50_episode_length_values,
                rolling50_estop_episode_flags=rolling50_estop_episode_flags,
            )
            write_rolling50_tensorboard_scalars(writer, rolling50_m, total_steps)
            print(
                f"[collector] episode_id={next_episode_file_id} "
                f"steps={n_episode_steps} camera_frames={n_camera_frames} "
                f"null_frames={episode_camera_null_frames} "
                f"has_images={'yes' if has_camera_images else 'NO'} "
                f"end_type={episode_end_type} end_reason={episode_end_reason} "
                f"stop_reason={stop_state.reason if stop_now else 'none'} "
                f"protective_stop={int(stop_state.protective_stop)} "
                f"controller_disconnected={int(stop_state.controller_disconnected)} "
                f"readiness_fail_estop={int(episode_had_readiness_fail_estop)} "
                f"end_reasons={episode_end_reasons}"
            )
            print(
                "[collector_progress] "
                f"episode_policy_steps={n_episode_steps} "
                f"policy_steps={total_steps} "
                f"reset_fsm_steps={reset_fsm_steps_total} "
                f"transition_hold_steps={transition_hold_steps_total} "
                f"estop_steps={protective_stop_steps} "
                f"disconnect_steps={controller_disconnect_steps} "
                f"readiness_fail_steps={readiness_fail_steps_total} "
                f"readiness_fail_estop_episodes={readiness_fail_estop_episodes} "
                f"readiness_fail_dropped_steps={readiness_fail_estop_dropped_steps_total} "
                f"episodes={total_episodes} "
                f"elapsed_s={elapsed_s:.1f} "
                f"elapsed_min={elapsed_min:.2f} "
                f"elapsed_hr={elapsed_hr:.3f} "
                f"rolling50_task_avg={rolling50_m.task_reward_avg:.4f} "
                f"rolling50_motion_avg={rolling50_m.motion_reward_avg:.4f} "
                f"rolling50_len_avg={rolling50_m.episode_length_avg:.2f} "
                f"rolling50_estops={rolling50_m.estop_episode_count:.0f}"
            )
            if episode_camera_null_frames > 0 and n_camera_frames == 0:
                sim = getattr(env, "simulator", None)
                sim_images_len = len(getattr(sim, "images", [])) if sim else -1
                sim_cap = getattr(sim, "cap", "N/A")
                print(
                    f"[collector] WARNING: zero camera frames captured this episode. "
                    f"simulator.images len={sim_images_len} simulator.cap={sim_cap}"
                )
            if args.enable_latency_profiling and latency_output_dir is not None:
                try:
                    latency_json_path, latency_plot_path, latency_summary = _write_latency_profile_episode(
                        output_dir=latency_output_dir,
                        episode_id=next_episode_file_id,
                        puck_detection_ms=episode_puck_detection_latency_ms,
                        model_inference_ms=episode_model_inference_latency_ms,
                        block_sleep_ms=episode_block_sleep_latency_ms,
                        other_ms=episode_other_latency_ms,
                        hist_bins=args.latency_profile_hist_bins,
                    )
                    bucket_summary = latency_summary["buckets"]
                    print(
                        "[latency] "
                        f"episode_id={next_episode_file_id} "
                        f"puck_p50={bucket_summary['puck_detection_ms']['median_ms']:.3f} "
                        f"model_p50={bucket_summary['model_inference_ms']['median_ms']:.3f} "
                        f"sleep_p50={bucket_summary['block_sleep_ms']['median_ms']:.3f} "
                        f"other_p50={bucket_summary['other_ms']['median_ms']:.3f} "
                        f"json={latency_json_path} plot={latency_plot_path}"
                    )
                except Exception:
                    print(
                        f"[latency] episode_id={next_episode_file_id} "
                        f"latency output FAILED:\n{traceback.format_exc()}"
                    )
            artifact_path = save_split_episode_hdf5(
                output_dir=_bucketed_output_dir(args.episode_artifact_dir, n_episode_steps),
                episode_id=next_episode_file_id,
                episode_rows=episode_rows,
                episode_images=episode_images if has_camera_images else None,
            )
            episodes_saved += 1
            clean_result = clean_episode_hdf5(
                artifact_path,
                # Always keep non-empty episodes regardless of length.
                min_timesteps=1,
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
                successful_online_episodes_kept += 1
                stats["successful_online_episodes_kept"] = float(successful_online_episodes_kept)
                if (
                    args.enable_periodic_checkpointing
                    and int(args.checkpoint_every_successful_online_episodes) > 0
                    and (successful_online_episodes_kept % int(args.checkpoint_every_successful_online_episodes) == 0)
                ):
                    checkpoint_save_request_id += 1
                    stats["checkpoint_save_request_id"] = float(checkpoint_save_request_id)
                    stats["checkpoint_reason"] = "periodic_successful_online_episode"
                    stats["checkpoint_trigger_episode_id"] = float(next_episode_file_id)
                    stats["checkpoint_trigger_successful_online_episodes_kept"] = float(
                        successful_online_episodes_kept
                    )
                if episode_stop_artifact_label is not None:
                    stop_copy_path = _copy_to_stop_dir(
                        clean_result.path,
                        _stop_output_dir(args.episode_artifact_dir, episode_stop_artifact_label),
                    )
                    print(
                        f"[collector] episode_id={next_episode_file_id} "
                        f"{episode_stop_artifact_label} HDF5 copied to {stop_copy_path}"
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
                        if episode_stop_artifact_label is not None:
                            stop_gif_path = _copy_to_stop_dir(
                                gif_path,
                                _stop_output_dir(args.episode_gif_dir, episode_stop_artifact_label),
                            )
                            print(
                                f"[collector] episode_id={next_episode_file_id} "
                                f"{episode_stop_artifact_label} GIF copied to {stop_gif_path}"
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
                            video_root=_bucketed_output_dir(
                                args.episode_camera_video_dir or args.episode_gif_dir,
                                n_episode_steps,
                            ),
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
                        if episode_stop_artifact_label is not None:
                            stop_camera_video_path = _copy_to_stop_dir(
                                camera_video_path,
                                _stop_output_dir(
                                    args.episode_camera_video_dir or args.episode_gif_dir,
                                    episode_stop_artifact_label,
                                ),
                            )
                            print(
                                f"[collector] episode_id={next_episode_file_id} "
                                f"{episode_stop_artifact_label} camera video copied to "
                                f"{stop_camera_video_path}"
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
            if pending_reset_artifact is not None:
                reset_camera_frames = len(pending_reset_artifact.images)
                has_reset_images = reset_camera_frames > 0
                print(
                    "[collector_reset_artifact] "
                    f"episode_id={pending_reset_artifact.episode_id} "
                    f"partition={pending_reset_artifact.partition} "
                    f"reason={pending_reset_artifact.done_reason} "
                    f"steps={pending_reset_artifact.step_count} "
                    f"camera_frames={reset_camera_frames} "
                    f"null_frames={pending_reset_artifact.camera_null_frames} "
                    f"flush_after_policy_episode={next_episode_file_id}"
                )
                reset_artifact_path = save_split_episode_hdf5(
                    output_dir=_reset_output_dir(
                        args.reset_artifact_dir,
                        pending_reset_artifact.partition,
                        pending_reset_artifact.step_count,
                    ),
                    episode_id=pending_reset_artifact.episode_id,
                    episode_rows=pending_reset_artifact.rows,
                    episode_images=pending_reset_artifact.images if has_reset_images else None,
                )
                reset_clean_result = clean_episode_hdf5(
                    reset_artifact_path,
                    min_timesteps=1,
                )
                if not reset_clean_result.kept:
                    print(
                        "[collector_reset_artifact] "
                        f"episode_id={pending_reset_artifact.episode_id} "
                        f"removed: reason={reset_clean_result.reason} "
                        f"timesteps={reset_clean_result.timesteps}"
                    )
                else:
                    print(
                        "[collector_reset_artifact] "
                        f"episode_id={pending_reset_artifact.episode_id} "
                        f"saved to {reset_clean_result.path}"
                    )
                pending_reset_artifact = None
            episode_rows = []
            episode_puck_detection_latency_ms = []
            episode_model_inference_latency_ms = []
            episode_block_sleep_latency_ms = []
            episode_other_latency_ms = []
            episode_images = []
            episode_camera_null_frames = 0
            next_episode_file_id += 1

            min_reset_delay_s = 3.0
            periodic_hard_reset = (total_episodes % 3) == 0
            stop_hard_reset = bool(episode_had_stop)
            if not periodic_hard_reset and not stop_hard_reset:
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
                reset_result = run_reset_fsm(
                    env,
                    reset_rng,
                    artifact_episode_id=next_reset_file_id,
                )
                reset_fsm_steps_total += reset_result.total_steps
                pending_reset_artifact, next_reset_file_id = merge_reset_fsm_artifact_into_pending(
                    reset_result.artifact,
                    pending_reset_artifact,
                    next_reset_file_id,
                    startup_buffered_message=False,
                )
                obs, previous_puck_position_for_primitive = soft_reset_prime_paddle_and_extract_previous_puck(
                    env,
                    device=device,
                    prime_paddle_history_stand_still_non_occluded=_prime_paddle_history_stand_still_non_occluded,
                    extract_primitive_state_tensors=_extract_primitive_state_tensors,
                )
                begin_transition_hold(
                    reason="reset_fsm_to_policy",
                    hold_steps=int(args.transition_hold_steps_post_reset),
                    request_sim_hold=True,
                )
            else:
                if episode_had_protective_stop:
                    hard_reset_reason = "collector_estop_next_step"
                elif episode_had_controller_disconnect:
                    hard_reset_reason = "collector_controller_disconnected_next_step"
                else:
                    hard_reset_reason = "periodic_every_3_episodes"
                print(
                    "[collector] "
                    f"episode_id={next_episode_file_id - 1} "
                    f"using hard reset path reason={hard_reset_reason}"
                )
                obs, _ = _hard_reset_with_pause(
                    env=env,
                    reason=hard_reset_reason,
                    pause_s=min_reset_delay_s,
                )
                hard_reset_state = env.simulator.get_current_state()
                run_reset_policy = _should_run_reset_policy_at_episode_start(
                    state_info=hard_reset_state,
                    table_x_bot=getattr(env, "table_x_bot", None),
                    bottom_margin=episode_start_reset_bottom_margin,
                    bottom_fail_count=episode_start_reset_bottom_fail_count,
                    occluded_fail_count=episode_start_reset_occluded_fail_count,
                    counters=episode_start_reset_counters,
                )
                decision = "reset_policy" if run_reset_policy else "policy"
                print(
                    "[collector] "
                    f"episode_id={next_episode_file_id - 1} "
                    f"hard_reset_start_decision={decision} "
                    f"bottom_counter={episode_start_reset_counters['bottom']} "
                    f"occ_counter={episode_start_reset_counters['occ']}"
                )
                if run_reset_policy:
                    reset_result = run_reset_fsm(
                        env,
                        reset_rng,
                        artifact_episode_id=next_reset_file_id,
                    )
                    reset_fsm_steps_total += reset_result.total_steps
                    pending_reset_artifact, next_reset_file_id = merge_reset_fsm_artifact_into_pending(
                        reset_result.artifact,
                        pending_reset_artifact,
                        next_reset_file_id,
                        startup_buffered_message=False,
                    )
                    obs, previous_puck_position_for_primitive = soft_reset_prime_paddle_and_extract_previous_puck(
                        env,
                        device=device,
                        prime_paddle_history_stand_still_non_occluded=_prime_paddle_history_stand_still_non_occluded,
                        extract_primitive_state_tensors=_extract_primitive_state_tensors,
                    )
                    begin_transition_hold(
                        reason="hard_reset_reset_fsm_to_policy",
                        hold_steps=int(args.transition_hold_steps_post_reset),
                        request_sim_hold=True,
                    )
                    episode_start_reset_counters["bottom"] = 0
                    episode_start_reset_counters["occ"] = 0
                else:
                    _, previous_puck_position_for_primitive, _ = _extract_primitive_state_tensors(
                        env,
                        device=device,
                    )
                    begin_transition_hold(
                        reason="hard_reset_to_policy",
                        hold_steps=int(args.transition_hold_steps_post_reset),
                        request_sim_hold=True,
                    )
            stop_penalty_applied_this_episode = False
            episode_had_stop = False
            episode_had_protective_stop = False
            episode_had_controller_disconnect = False
            episode_had_readiness_fail_estop = False
            episode_readiness_first_fail_step_idx = None
            episode_readiness_first_fail_reason = None
            readiness_fail_streak = 0
            readiness_fail_first_episode_step_idx = None
            readiness_fail_first_total_step = None
            readiness_fail_prev = False
            readiness_fail_prev_reason = "none"
            episode_motion_metric_sums = {name: 0.0 for name in motion_metric_names}
            episode_motion_metric_count = 0
            current_state_info = getattr(env, "current_state", None)
            if not isinstance(current_state_info, dict):
                simulator = getattr(env, "simulator", None)
                if simulator is not None and hasattr(simulator, "get_current_state"):
                    try:
                        current_state_info = simulator.get_current_state()
                    except Exception:
                        current_state_info = None
            current_paddle_xy, current_puck_xy = _extract_motion_positions_from_state_info(current_state_info)
            _reset_motion_reward_state(
                motion_reward_state,
                anchor_paddle_xy=current_paddle_xy,
                anchor_puck_xy=current_puck_xy,
            )

        now = time.time()
        if now - last_log_time >= float(args.collector_log_interval_sec):
            snapshot = replay.state_snapshot()
            elapsed_s = max(0.0, collector_elapsed_resume_offset_s + (now - collector_start_time))
            rolling50_m = compute_rolling50_metrics(
                rolling50_task_reward_values,
                rolling50_motion_reward_values,
                rolling50_episode_length_values,
                rolling50_estop_episode_flags,
            )
            stats["collector_steps"] = float(total_steps)
            stats["collector_total_steps"] = float(total_steps)
            stats["collector_episodes"] = float(total_episodes)
            stats["collector_actor_version"] = float(learner_state.total_actor_updates)
            stats["transition_hold_events_total"] = float(transition_hold_events_total)
            stats["transition_hold_active"] = float(1.0 if transition_hold_steps_remaining > 0 else 0.0)
            stats["transition_hold_steps_remaining"] = float(transition_hold_steps_remaining)
            stats["replay_success_size"] = float(snapshot["success"]["size"])
            stats["replay_failure_size"] = float(snapshot["failure"]["size"])
            stats["episodes_saved"] = float(episodes_saved)
            stats["episodes_removed_short"] = float(episodes_removed_short)
            stats["episodes_removed_invalid"] = float(episodes_removed_invalid)
            stats["episodes_gif_generated"] = float(episodes_gif_generated)
            stats["episodes_gif_failed"] = float(episodes_gif_failed)
            stats["episodes_camera_video_generated"] = float(episodes_camera_video_generated)
            stats["episodes_camera_video_failed"] = float(episodes_camera_video_failed)
            stats["successful_online_episodes_kept"] = float(successful_online_episodes_kept)
            stats["estop_steps"] = float(protective_stop_steps)
            stats["estop_episodes"] = float(protective_stop_episodes)
            stats["controller_disconnect_steps"] = float(controller_disconnect_steps)
            stats["controller_disconnect_episodes"] = float(controller_disconnect_episodes)
            stats["reset_fsm_steps"] = float(reset_fsm_steps_total)
            stats["transition_hold_steps"] = float(transition_hold_steps_total)
            stats["primitive_chance"] = float(primitive_selector.chance)
            stats["interval_primitive_env_steps"] = float(interval_primitive_env_steps)
            stats["interval_target_position_directional_env_steps"] = float(
                interval_target_position_directional_env_steps
            )
            stats["run_elapsed_total_s"] = float(elapsed_s)
            update_stats_dict_rolling50(
                stats,
                rolling50_m,
                window_size=ROLLING_PERF_WINDOW_EPISODES,
                rolling50_task_reward_values=rolling50_task_reward_values,
                rolling50_motion_reward_values=rolling50_motion_reward_values,
                rolling50_episode_length_values=rolling50_episode_length_values,
                rolling50_estop_episode_flags=rolling50_estop_episode_flags,
            )
            writer.add_scalar("replay/success_buffer_size", float(snapshot["success"]["size"]), total_steps)
            writer.add_scalar("replay/failure_buffer_size", float(snapshot["failure"]["size"]), total_steps)
            writer.add_scalar("exploration/primitive_chance", float(primitive_selector.chance), total_steps)
            writer.add_scalar("exploration/primitive_env_steps", float(interval_primitive_env_steps), total_steps)
            writer.add_scalar(
                "exploration/primitive_horizontal_env_steps",
                float(interval_primitive_horizontal_env_steps),
                total_steps,
            )
            writer.add_scalar(
                "exploration/target_position_directional_env_steps",
                float(interval_target_position_directional_env_steps),
                total_steps,
            )
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
            writer.add_scalar("safety/estop_steps", float(protective_stop_steps), total_steps)
            writer.add_scalar("safety/estop_episodes", float(protective_stop_episodes), total_steps)
            writer.add_scalar(
                "safety/controller_disconnect_steps",
                float(controller_disconnect_steps),
                total_steps,
            )
            writer.add_scalar(
                "safety/controller_disconnect_episodes",
                float(controller_disconnect_episodes),
                total_steps,
            )
            writer.add_scalar(
                "transitions/hold_active",
                float(1.0 if transition_hold_steps_remaining > 0 else 0.0),
                total_steps,
            )
            writer.add_scalar(
                "transitions/hold_steps_remaining",
                float(transition_hold_steps_remaining),
                total_steps,
            )
            writer.add_scalar(
                "transitions/hold_events_total",
                float(transition_hold_events_total),
                total_steps,
            )
            writer.add_scalar(
                "charts/SPS",
                float(total_steps) / max((now - collector_start_time), 1e-6),
                total_steps,
            )
            writer.add_scalar("runtime/elapsed_total_s", float(elapsed_s), total_steps)
            write_rolling50_tensorboard_scalars(writer, rolling50_m, total_steps)
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
                f"actor_version={learner_state.total_actor_updates} "
                f"success_rb={snapshot['success']['size']} failure_rb={snapshot['failure']['size']} "
                f"saved={episodes_saved} short_removed={episodes_removed_short} "
                f"invalid_removed={episodes_removed_invalid} gif_ok={episodes_gif_generated} gif_fail={episodes_gif_failed} "
                f"cam_video_ok={episodes_camera_video_generated} cam_video_fail={episodes_camera_video_failed} "
                f"estop_steps={protective_stop_steps} estop_episodes={protective_stop_episodes} "
                f"disconnect_steps={controller_disconnect_steps} "
                f"disconnect_episodes={controller_disconnect_episodes} "
                f"readiness_fail_steps={readiness_fail_steps_total} "
                f"readiness_fail_estop_episodes={readiness_fail_estop_episodes} "
                f"readiness_fail_dropped_steps={readiness_fail_estop_dropped_steps_total} "
                f"reset_fsm_steps={reset_fsm_steps_total} "
                f"transition_hold_steps={transition_hold_steps_total} "
                f"primitive_chance={primitive_selector.chance:.4f} "
                f"primitive_steps={interval_primitive_env_steps} "
                f"target_position_steps={interval_target_position_directional_env_steps} "
                f"transition_hold_active={int(transition_hold_steps_remaining > 0)} "
                f"transition_hold_remaining={transition_hold_steps_remaining} "
                f"transition_events_total={transition_hold_events_total} "
                f"transition_reason={transition_hold_reason} "
                f"elapsed_total_s={elapsed_s:.1f} "
                f"rolling50_task_avg={rolling50_m.task_reward_avg:.4f} "
                f"rolling50_motion_avg={rolling50_m.motion_reward_avg:.4f} "
                f"rolling50_len_avg={rolling50_m.episode_length_avg:.2f} "
                f"rolling50_estops={rolling50_m.estop_episode_count:.0f}"
            )
            if transition_hold_reason_counts:
                print(f"[collector_transition] reason_counts={dict(sorted(transition_hold_reason_counts.items()))}")
            interval_primitive_env_steps = 0
            interval_primitive_horizontal_env_steps = 0
            interval_target_position_directional_env_steps = 0
            last_log_time = now

    env.close()
    writer.close()


@dataclass
class LearnerRuntimeState:
    actor: DeterministicAgent
    actor_target: DeterministicAgent
    qf1: TD3DualHeadQNetwork
    qf2: TD3DualHeadQNetwork
    qf1_target: TD3DualHeadQNetwork
    qf2_target: TD3DualHeadQNetwork
    q_optimizer: optim.Optimizer
    actor_optimizer: optim.Optimizer
    action_low: torch.Tensor
    action_high: torch.Tensor
    writer: SummaryWriter
    checkpoint_root: Path
    last_log_time: float
    learner_start_time: float
    total_updates: int
    total_actor_updates: int
    last_handled_checkpoint_request_id: int
    latest_train_metrics: Dict[str, float]


def _make_qf(
    obs_dim: int,
    act_dim: int,
    hidden_layer_size: int,
    num_hidden_layers: int,
    device: torch.device,
) -> TD3DualHeadQNetwork:
    return TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=hidden_layer_size,
        num_hidden_layers=num_hidden_layers,
    ).to(device)


def _init_sync_learner_state(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
) -> LearnerRuntimeState:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.learner_device)
    writer = SummaryWriter(tb_log_dir)

    policy_obs_dim = obs_dim + act_dim if train_args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())
    qf1 = _make_qf(
        obs_dim,
        act_dim,
        train_args.q_hidden_layer_size,
        train_args.q_num_hidden_layers,
        device,
    )
    qf2 = _make_qf(
        obs_dim,
        act_dim,
        train_args.q_hidden_layer_size,
        train_args.q_num_hidden_layers,
        device,
    )
    qf1_target = _make_qf(
        obs_dim,
        act_dim,
        train_args.q_hidden_layer_size,
        train_args.q_num_hidden_layers,
        device,
    )
    qf2_target = _make_qf(
        obs_dim,
        act_dim,
        train_args.q_hidden_layer_size,
        train_args.q_num_hidden_layers,
        device,
    )
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    resume_checkpoint: Dict[str, object] | None = None
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
            if "q_optimizer" in loaded_obj and "actor_optimizer" in loaded_obj:
                resume_checkpoint = loaded_obj
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
            actor_target.load_state_dict(actor.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr,
        weight_decay=args.q_weight_decay,
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    total_updates = 0
    total_actor_updates = 0
    latest_train_metrics: Dict[str, float] = {}
    if resume_checkpoint is not None:
        q_optimizer.load_state_dict(resume_checkpoint["q_optimizer"])
        actor_optimizer.load_state_dict(resume_checkpoint["actor_optimizer"])
        total_updates = int(resume_checkpoint.get("global_step", resume_checkpoint.get("learner_q_updates", 0)))
        total_actor_updates = int(resume_checkpoint.get("learner_actor_updates", 0))
        if isinstance(resume_checkpoint.get("train_metrics"), dict):
            latest_train_metrics = {
                str(key): float(value)
                for key, value in resume_checkpoint["train_metrics"].items()
            }
        if "rng_states" in resume_checkpoint:
            set_rng_states(resume_checkpoint["rng_states"])
        print(
            "[learner] resumed optimizer state "
            f"q_updates={total_updates} actor_updates={total_actor_updates}"
        )
    return LearnerRuntimeState(
        actor=actor,
        actor_target=actor_target,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        action_low=torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0),
        action_high=torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0),
        writer=writer,
        checkpoint_root=_checkpoint_root_from_tb(tb_log_dir, args.checkpoint_root_dir),
        last_log_time=time.time(),
        learner_start_time=time.time(),
        total_updates=total_updates,
        total_actor_updates=total_actor_updates,
        last_handled_checkpoint_request_id=int(stats.get("checkpoint_save_request_id", 0)),
        latest_train_metrics=latest_train_metrics,
    )


def _run_sync_learner_iteration(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    state: LearnerRuntimeState,
) -> bool:
    current_checkpoint_request_id = int(stats.get("checkpoint_save_request_id", 0))
    if (
        args.enable_periodic_checkpointing
        and current_checkpoint_request_id > state.last_handled_checkpoint_request_id
    ):
        successful_kept = int(stats.get("checkpoint_trigger_successful_online_episodes_kept", 0))
        checkpoint_tag = f"successeps_{successful_kept}_qupdates_{state.total_updates}"
        try:
            checkpoint_dir = _save_checkpoint_from_learner_state(
                state=state,
                replay=replay,
                stats=stats,
                checkpoint_tag=checkpoint_tag,
                args=args,
                train_args=train_args,
            )
            stats["last_checkpoint_dir"] = str(checkpoint_dir)
            stats["last_checkpoint_success_episode_count"] = float(successful_kept)
            stats["last_checkpoint_q_updates"] = float(state.total_updates)
            stats["last_checkpoint_request_id"] = float(current_checkpoint_request_id)
            print(
                "[learner_checkpoint] "
                f"request_id={current_checkpoint_request_id} "
                f"successful_kept={successful_kept} path={checkpoint_dir}"
            )
        except Exception:
            print(f"[learner_checkpoint] save FAILED:\n{traceback.format_exc()}")
        state.last_handled_checkpoint_request_id = current_checkpoint_request_id

    total_replay_size = replay.len("success") + replay.len("failure")
    if total_replay_size < int(args.min_replay_size_before_learning):
        return False

    actor_updated = False
    for q_update_idx in range(args.q_updates):
        batch, success_batch_count, failure_batch_count = _mixed_sample_from_shared(
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
            train_args.use_last_action_in_policy_state,
        )
        with torch.no_grad():
            target_next_action = deterministic_actor_action(state.actor_target, sampled_next_policy_observations)
            noise = torch.randn_like(target_next_action) * float(args.policy_noise)
            noise = torch.clamp(noise, -float(args.noise_clip), float(args.noise_clip))
            target_next_action = torch.clamp(target_next_action + noise, state.action_low, state.action_high)
            q1_next_task_h, q1_next_motion_h = state.qf1_target(sampled_next_observations, target_next_action)
            q2_next_task_h, q2_next_motion_h = state.qf2_target(sampled_next_observations, target_next_action)
            min_next_task = h_inverse(
                torch.min(q1_next_task_h, q2_next_task_h),
                eps=float(args.h_transform_eps),
            ).view(-1)
            min_next_motion = h_inverse(
                torch.min(q1_next_motion_h, q2_next_motion_h),
                eps=float(args.h_transform_eps),
            ).view(-1)
            bellman_task = (
                sampled_task_rewards
                + (1.0 - sampled_dones) * float(args.task_gamma) * min_next_task
            )
            bellman_motion = (
                sampled_motion_rewards
                + (1.0 - sampled_dones) * float(args.motion_gamma) * min_next_motion
            )
            target_task_h = h_transform(bellman_task, eps=float(args.h_transform_eps))
            target_motion_h = h_transform(bellman_motion, eps=float(args.h_transform_eps))
        q1_task_h, q1_motion_h = state.qf1(sampled_observations, sampled_actions)
        q2_task_h, q2_motion_h = state.qf2(sampled_observations, sampled_actions)
        q1_task_loss = torch.nn.functional.mse_loss(q1_task_h.view(-1), target_task_h)
        q2_task_loss = torch.nn.functional.mse_loss(q2_task_h.view(-1), target_task_h)
        q1_motion_loss = torch.nn.functional.mse_loss(q1_motion_h.view(-1), target_motion_h)
        q2_motion_loss = torch.nn.functional.mse_loss(q2_motion_h.view(-1), target_motion_h)
        q_loss = q1_task_loss + q2_task_loss + q1_motion_loss + q2_motion_loss
        state.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        state.q_optimizer.step()
        state.total_updates += 1
        positive_task_reward_mask = sampled_task_rewards > 0
        positive_task_reward_count = float(positive_task_reward_mask.sum().item())
        minibatch_size = max(int(sampled_task_rewards.numel()), 1)
        positive_task_rewards = sampled_task_rewards[positive_task_reward_mask]
        state.latest_train_metrics.update(
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
                    (state.qf1, state.qf1_target),
                    (state.qf2, state.qf2_target),
                    (state.actor, state.actor_target),
                ):
                    for param, target_param in zip(source.parameters(), target.parameters()):
                        target_param.data.copy_(
                            float(args.tau) * param.data + (1.0 - float(args.tau)) * target_param.data
                        )
    for _ in range(args.actor_updates_per_iteration):
        actor_batch, _, _ = _mixed_sample_from_shared(
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
            train_args.use_last_action_in_policy_state,
        )
        policy_actions = deterministic_actor_action(state.actor, actor_policy_obs)
        q1_task_h, q1_motion_h = state.qf1(actor_obs, policy_actions)
        q1_task = h_inverse(q1_task_h, eps=float(args.h_transform_eps)).view(-1)
        q1_motion = h_inverse(q1_motion_h, eps=float(args.h_transform_eps)).view(-1)
        actor_objective = float(args.task_reward_weight) * q1_task + float(args.motion_reward_weight) * q1_motion
        actor_loss = -actor_objective.mean()
        state.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        state.actor_optimizer.step()
        state.total_actor_updates += 1
        actor_updated = True
        norm_task = (1.0 - float(args.task_gamma)) * q1_task
        norm_motion = (1.0 - float(args.motion_gamma)) * q1_motion
        state.latest_train_metrics.update(
            {
                "losses/actor_loss": float(actor_loss.item()),
                "losses/actor_norm_task_mean": float(norm_task.mean().item()),
                "losses/actor_norm_motion_mean": float(norm_motion.mean().item()),
            }
        )
    now = time.time()
    if now - state.last_log_time >= float(args.learner_log_interval_sec):
        stats["learner_q_updates"] = float(state.total_updates)
        stats["learner_actor_updates"] = float(state.total_actor_updates)
        stats["learner_replay_size"] = float(total_replay_size)
        step_index = max(state.total_updates, 1)
        for metric_name, metric_value in state.latest_train_metrics.items():
            state.writer.add_scalar(metric_name, float(metric_value), step_index)
        state.writer.add_scalar(
            "charts/SPS",
            float(state.total_updates) / max(now - state.learner_start_time, 1e-6),
            step_index,
        )
        state.writer.add_scalar("replay/success_buffer_size", float(replay.len("success")), step_index)
        state.writer.add_scalar("replay/failure_buffer_size", float(replay.len("failure")), step_index)
        print(
            "[learner] "
            f"q_updates={state.total_updates} actor_updates={state.total_actor_updates} replay_size={total_replay_size}"
        )
        state.last_log_time = now
    return actor_updated


def _prompt_optional_run_note() -> str:
    """Prompt the user for an optional note describing this run.

    Returns the entered note (stripped). Returns empty string if the user
    skips it or the prompt is run in a non-interactive context.
    """
    try:
        note = input("Optional run note (press Enter to skip): ").strip()
    except EOFError:
        note = ""
    return note


def _setup_run_data_dir(args: Args, run_note: str) -> Path:
    """Create a new timestamped folder under the log parent dir for collected
    data, redirect all data output paths into it, and write the run note.

    Returns the path of the created folder.
    """
    if args.log_parent_dir is not None and str(args.log_parent_dir).strip():
        log_parent_base = Path(args.log_parent_dir).expanduser().resolve()
    elif args.checkpoint_root_dir is not None and str(args.checkpoint_root_dir).strip():
        log_parent_base = Path(args.checkpoint_root_dir).expanduser().resolve()
    else:
        log_parent_base = Path(args.episode_artifact_dir).expanduser().resolve().parent

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_data_dir = log_parent_base / f"data_{timestamp}"
    run_data_dir.mkdir(parents=True, exist_ok=True)

    args.episode_artifact_dir = str(run_data_dir / "episode_hdf5")
    args.reset_artifact_dir = str(run_data_dir / "reset_hdf5")
    args.episode_gif_dir = str(run_data_dir / "episode_gifs")
    if args.episode_camera_video_dir is not None:
        args.episode_camera_video_dir = str(run_data_dir / "episode_camera_videos")

    note_path = run_data_dir / "run_note.txt"
    if run_note:
        note_path.write_text(run_note + "\n", encoding="utf-8")

    print(f"[run_data] collected data dir: {run_data_dir}")
    if run_note:
        print(f"[run_data] note: {run_note}")

    return run_data_dir


def _finalize_sync_learner_state(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    state: LearnerRuntimeState,
) -> None:
    if args.enable_periodic_checkpointing:
        final_tag = f"final_qupdates_{state.total_updates}"
        try:
            final_checkpoint_dir = _save_checkpoint_from_learner_state(
                state=state,
                replay=replay,
                stats=stats,
                checkpoint_tag=final_tag,
                args=args,
                train_args=train_args,
            )
            stats["last_checkpoint_dir"] = str(final_checkpoint_dir)
            stats["last_checkpoint_q_updates"] = float(state.total_updates)
            print(f"[learner_checkpoint] final path={final_checkpoint_dir}")
        except Exception:
            print(f"[learner_checkpoint] final save FAILED:\n{traceback.format_exc()}")
    state.writer.close()
def main(args: Args, train_args: TrainArgs) -> None:
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
    if args.enable_periodic_checkpointing and int(args.checkpoint_every_successful_online_episodes) <= 0:
        raise ValueError("checkpoint_every_successful_online_episodes must be > 0 when checkpointing is enabled.")
    normalized_replay_priority = _normalize_replay_source_priority(args.replay_source_priority)
    if normalized_replay_priority != str(args.replay_source_priority).strip().lower():
        print("[main] replay_source_priority normalized to 'warmstart_only' due to invalid input.")
    if _normalize_transition_last_action_mode(args.transition_last_action_mode) != str(
        args.transition_last_action_mode
    ).strip().lower():
        print("[main] transition_last_action_mode normalized to 'zero' due to invalid input.")

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

    replay = SharedTD3Replay(
        success_capacity=args.success_buffer_size,
        failure_capacity=args.failure_buffer_size,
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
    )
    stats: Dict[str, object] = {}
    stats["successful_online_episodes_kept"] = float(0)
    stats["checkpoint_save_request_id"] = float(0)
    stats["checkpoint_trigger_successful_online_episodes_kept"] = float(0)
    stats["collector_total_steps"] = float(0.0)
    stats["run_elapsed_total_s"] = float(0.0)
    stats["rolling50_window_size"] = float(ROLLING_PERF_WINDOW_EPISODES)
    stats["rolling50_window_count"] = float(0.0)
    stats["rolling50_task_reward_avg"] = float(0.0)
    stats["rolling50_motion_reward_avg"] = float(0.0)
    stats["rolling50_episode_length_avg"] = float(0.0)
    stats["rolling50_estop_episode_count"] = float(0.0)
    stats["rolling50_task_reward_values"] = []
    stats["rolling50_motion_reward_values"] = []
    stats["rolling50_episode_length_values"] = []
    stats["rolling50_estop_episode_flags"] = []
    if args.model_path is not None:
        loaded_runtime_state = _load_runtime_perf_from_checkpoint_file(args.model_path)
        stats["collector_total_steps"] = float(loaded_runtime_state.get("collector_total_steps", 0.0))
        loaded_task_values = _coerce_float_list(
            loaded_runtime_state.get("rolling50_task_reward_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        )
        loaded_motion_values = _coerce_float_list(
            loaded_runtime_state.get("rolling50_motion_reward_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        )
        loaded_length_values = _coerce_float_list(
            loaded_runtime_state.get("rolling50_episode_length_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        )
        loaded_estop_flags = _coerce_float_list(
            loaded_runtime_state.get("rolling50_estop_episode_flags", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        )
        stats["run_elapsed_total_s"] = float(loaded_runtime_state.get("run_elapsed_total_s", 0.0))
        stats["rolling50_task_reward_values"] = loaded_task_values
        stats["rolling50_motion_reward_values"] = loaded_motion_values
        stats["rolling50_episode_length_values"] = loaded_length_values
        stats["rolling50_estop_episode_flags"] = loaded_estop_flags
        stats["rolling50_window_count"] = float(
            max(
                len(loaded_task_values),
                len(loaded_motion_values),
                len(loaded_length_values),
                len(loaded_estop_flags),
            )
        )
        stats["rolling50_task_reward_avg"] = float(rolling_mean(loaded_task_values))
        stats["rolling50_motion_reward_avg"] = float(rolling_mean(loaded_motion_values))
        stats["rolling50_episode_length_avg"] = float(rolling_mean(loaded_length_values))
        stats["rolling50_estop_episode_count"] = float(sum(loaded_estop_flags))

    warm_start_requested = len(args.warm_start_hdf5_dirs) > 0
    checkpoint_replay_loaded = False
    if args.load_replay_from_checkpoint and args.model_path is not None:
        should_load_checkpoint_replay = True
        if warm_start_requested and normalized_replay_priority == "warmstart_only":
            should_load_checkpoint_replay = False
            print("[resume_replay] skipping checkpoint replay because warmstart_only is active.")
        if should_load_checkpoint_replay:
            checkpoint_replay_loaded = _load_replay_from_checkpoint_file(
                model_path=args.model_path,
                replay=replay,
            )
            if checkpoint_replay_loaded:
                stats["resume_replay_loaded"] = float(1)
    try:
        if warm_start_requested:
            if checkpoint_replay_loaded and normalized_replay_priority == "checkpoint_only":
                print("[warm_start] skipped because replay_source_priority=checkpoint_only")
            else:
                if checkpoint_replay_loaded and normalized_replay_priority == "checkpoint_then_append":
                    print("[warm_start] appending warm-start data on top of checkpoint replay.")
                warm_start_summary = _warm_start_replay_from_hdf5(
                    args=args,
                    replay=replay,
                    env=probe_env,
                    add_episode_to_shared_replay=_add_episode_to_shared_replay,
                )
                for key, value in warm_start_summary.items():
                    stats[f"warm_start/{key}"] = float(value)
    finally:
        probe_env.close()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Keep TensorBoard logs co-located with checkpoint parent when an explicit
    # checkpoint root is configured, so run artifacts stay under one root.
    if args.checkpoint_root_dir is not None and str(args.checkpoint_root_dir).strip():
        base_log_dir = str(Path(args.checkpoint_root_dir).expanduser().resolve())
        if args.log_parent_dir is not None and str(args.log_parent_dir).strip():
            requested_log_dir = str(Path(args.log_parent_dir).expanduser().resolve())
            if requested_log_dir != base_log_dir:
                print(
                    "[main] log_parent_dir differs from checkpoint_root_dir; "
                    "using checkpoint_root_dir for TensorBoard logs."
                )
    else:
        base_log_dir = args.log_parent_dir or f"runs/async_td3/{args.run_name}_{timestamp}"
    collector_tb_dir = os.path.join(base_log_dir, "collector_tb")
    learner_tb_dir = os.path.join(base_log_dir, "learner_tb")
    os.makedirs(collector_tb_dir, exist_ok=True)
    os.makedirs(learner_tb_dir, exist_ok=True)
    print(f"TensorBoard logs: {base_log_dir}")

    learner_state = _init_sync_learner_state(
        args=args,
        train_args=train_args,
        replay=replay,
        stats=stats,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        tb_log_dir=learner_tb_dir,
    )
    try:
        collector_process(
            args,
            train_args,
            replay,
            stats,
            learner_state,
            obs_dim,
            act_dim,
            action_low_np,
            action_high_np,
            collector_tb_dir,
        )
    except KeyboardInterrupt:
        print("[main] interrupted by user; shutting down.")
    finally:
        _finalize_sync_learner_state(
            args=args,
            train_args=train_args,
            replay=replay,
            stats=stats,
            state=learner_state,
        )
        print("Final stats:", dict(stats))


if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.train_args is None:
        raise SystemExit(
            "async_td3_real.py requires --train-args pointing to the training run's "
            "args.yaml (produced by td3_training.py). It supplies the actor/critic "
            "architecture and use_last_action_in_policy_state flag that must match "
            "the saved checkpoint."
        )
    if temp_args.args_file is None:
        raise SystemExit(
            "async_td3_real.py requires --args-file pointing to an online-behavior "
            "YAML (e.g. td3_online.yaml). Architecture comes from --train-args; this "
            "file supplies online training/collection defaults only."
        )
    train_args = _load_train_args(temp_args.train_args)
    mapped_defaults, applied_keys, ignored_keys = _build_args_file_defaults(temp_args.args_file)
    # Carry the CLI-provided paths through so the final Args records which files were used.
    mapped_defaults["args_file"] = temp_args.args_file
    mapped_defaults["train_args"] = temp_args.train_args
    default_args = Args(**mapped_defaults)

    args = tyro.cli(Args, default=default_args)
    print(f"[train_args] loaded architecture from: {args.train_args}")
    print(
        f"[train_args] "
        f"action_scale={train_args.action_scale} "
        f"agent_hidden_layer_size={train_args.agent_hidden_layer_size} "
        f"agent_num_hidden_layers={train_args.agent_num_hidden_layers} "
        f"q_hidden_layer_size={train_args.q_hidden_layer_size} "
        f"q_num_hidden_layers={train_args.q_num_hidden_layers} "
        f"use_last_action_in_policy_state={train_args.use_last_action_in_policy_state}"
    )
    print(f"[args_file] loaded defaults from: {args.args_file}")
    if applied_keys:
        print("[args_file] applied keys:", ", ".join(applied_keys))
    else:
        print("[args_file] applied keys: none")
    if ignored_keys:
        print("[args_file] ignored unsupported keys:", ", ".join(ignored_keys))
    run_note = _prompt_optional_run_note()
    _setup_run_data_dir(args, run_note)
    main(args, train_args)
