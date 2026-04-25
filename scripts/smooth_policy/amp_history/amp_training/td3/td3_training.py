"""
TD3 training with transformed Bellman targets and dual-head critics.

Compared to SAC+AMP:
- no discriminator
- no entropy term / alpha tuning
- deterministic actor updates (TD3)
- twin critics with separate task and motion heads in transformed space
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Literal, Tuple

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.smooth_policy.residual_agent import ResidualActor, zero_init_residual_head
from scripts.smooth_policy.amp_history.amp_training.td3.helper.dual_head_q import (
    TD3DualHeadQNetwork,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.replay_buffer import TD3ReplayBuffer
from scripts.smooth_policy.amp_history.amp_training.td3.helper.prioritized_replay_buffer import (
    TD3PrioritizedReplayBuffer,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_checkpointing import (
    build_training_state,
    load_fine_tune_optimizer_state,
    load_resume_training_state,
    seed_fine_tune_replay_from_source,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_episode_collection import (
    EpisodeTrajectory,
    finalize_episode_if_done,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_metrics import (
    initialize_train_metrics,
    log_scalar_metrics,
    tensor_mean_items,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.td3_replay_sampling import (
    concat_replay_samples,
    critic_success_failure_counts,
    sample_actor_source_chunk,
    sample_critic_source_chunk,
)
from scripts.smooth_policy.amp_history.amp_training.amp_training import (
    parse_motion_magnitudes_from_infos,
)
from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.utils import save_tensorboard_plots

ROLLING_STATS_WINDOW_STEPS = 2000


def h_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def h_inverse(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    abs_x = torch.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    sqrt_inner = torch.sqrt(inner)
    quotient = (sqrt_inner - 1) / (2 * eps)
    return torch.sign(x) * (quotient**2 - 1)


def augment_policy_observation(observation, last_action, use_last_action):
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def deterministic_actor_action(actor, policy_obs):
    if hasattr(actor, "get_action_mean_and_logstd"):
        action_mean, _ = actor.get_action_mean_and_logstd(policy_obs)
        return torch.tanh(action_mean) * actor.action_scale + actor.action_bias
    if hasattr(actor, "get_action"):
        return actor.get_action(policy_obs)
    raise TypeError(f"Unsupported actor type for deterministic action: {type(actor)}")


def save_training_data_like_rollout_gif(
    save_path: str,
    env_params: Dict[str, object],
    actor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    use_last_action_in_policy_state: bool,
    exploration_noise: float,
    primitive_selector: PrimitiveExplorationSelector,
    warm_policy_actor,
    warm_policy_use_last_action: bool,
    max_timesteps: int,
    fps: int,
    device: str,
) -> None:
    viz_env = AirHockeyEnv(dict(env_params))
    viz_env.max_timesteps = int(max_timesteps)
    renderer = AirHockeyRenderer(viz_env, show_target_position=True, show_acceleration_arrow=False)

    obs, _ = viz_env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    act_dim = int(np.prod(viz_env.action_space.shape))
    last_action_for_policy = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
    warm_policy_last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
    action_low_single = action_low.unsqueeze(0)
    action_high_single = action_high.unsqueeze(0)

    rollout_selector = PrimitiveExplorationSelector(
        num_envs=1,
        chance=float(primitive_selector.chance),
        takeover_steps=int(primitive_selector.takeover_steps),
        device=device,
        dtype=torch.float32,
        direction_y_component_weight=float(primitive_selector.direction_y_component_weight),
        target_min_distance=float(primitive_selector.target_min_distance),
        target_max_distance=float(primitive_selector.target_max_distance),
        target_action_delta_x=float(primitive_selector.target_action_delta_x),
        target_action_delta_y=float(primitive_selector.target_action_delta_y),
        target_takeover_steps=int(primitive_selector.target_takeover_steps),
        pre_contact_hit_variant_chance=float(primitive_selector.pre_contact_hit_variant_chance),
        pre_contact_hit_variant_steps=int(primitive_selector.pre_contact_hit_variant_steps),
        pre_contact_hit_variant_distance_threshold=float(
            primitive_selector.pre_contact_hit_variant_distance_threshold
        ),
        pre_contact_hit_variant_scale_min=float(primitive_selector.pre_contact_hit_variant_scale_min),
        pre_contact_hit_variant_scale_max=float(primitive_selector.pre_contact_hit_variant_scale_max),
        pre_contact_hit_variant_min_upward_displacement_x=float(
            primitive_selector.pre_contact_hit_variant_min_upward_displacement_x
        ),
    )
    primitive_weights = primitive_selector.primitive_weights.detach().cpu().numpy()
    rollout_selector.set_primitive_weights(
        stand_still=float(primitive_weights[0]),
        same_direction=float(primitive_weights[1]),
        y_aligned=float(primitive_weights[2]),
        policy_takeover=float(primitive_weights[3]),
        target_position_directional=(
            float(primitive_weights[4]) if primitive_weights.shape[0] > 4 else 0.0
        ),
    )

    frames = []
    rew = 0.0
    cum_rew = 0.0
    done = False
    previous_puck_position = extract_current_puck_position(obs_tensor).clone()
    while not done and len(frames) < int(max_timesteps):
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
        cv2.putText(
            frame,
            f"Reward: {rew:.2f}",
            (frame.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Return: {cum_rew:.2f}",
            (frame.shape[1] - 150, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        frames.append(frame)

        with torch.no_grad():
            policy_obs = augment_policy_observation(
                obs_tensor, last_action_for_policy, use_last_action_in_policy_state
            )
            action_tensor = deterministic_actor_action(actor, policy_obs)
            if exploration_noise > 0:
                action_tensor = action_tensor + torch.randn_like(action_tensor) * float(exploration_noise)
            action_tensor = torch.clamp(action_tensor, action_low_single, action_high_single)

            current_paddle_pos = extract_current_paddle_position(obs_tensor)
            current_puck_pos = extract_current_puck_position(obs_tensor)
            current_puck_vel = extract_current_puck_velocity(obs_tensor)
            if torch.all(current_puck_vel == 0):
                current_puck_vel = current_puck_pos - previous_puck_position
            y_alignment_sign = torch.sign(current_puck_pos[:, 1] - current_paddle_pos[:, 1])
            action_tensor, _ = rollout_selector.apply(
                action_tensor,
                action_low=action_low_single,
                action_high=action_high_single,
                y_alignment_sign=y_alignment_sign,
                current_paddle_position=current_paddle_pos,
                current_puck_position=current_puck_pos,
                current_puck_velocity=current_puck_vel,
                return_stats=True,
            )
            if warm_policy_actor is not None:
                policy_takeover_mask = rollout_selector.active & (
                    rollout_selector.primitive_id == rollout_selector.primitive_ids.POLICY_TAKEOVER
                )
                if torch.any(policy_takeover_mask):
                    warm_policy_obs = augment_policy_observation(
                        obs_tensor,
                        warm_policy_last_action,
                        warm_policy_use_last_action,
                    )
                    warm_policy_actions = deterministic_actor_action(warm_policy_actor, warm_policy_obs)
                    warm_policy_actions = torch.clamp(
                        warm_policy_actions,
                        action_low_single,
                        action_high_single,
                    )
                    action_tensor[policy_takeover_mask] = warm_policy_actions[policy_takeover_mask]

        action_np = action_tensor.squeeze(0).detach().cpu().numpy()
        next_obs, rew, terminated, truncated, _ = viz_env.step(action_np)
        done = bool(terminated or truncated)
        cum_rew += float(rew)
        obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        previous_puck_position = extract_current_puck_position(obs_tensor).clone()
        done_tensor = torch.tensor([done], dtype=torch.bool, device=device)
        rollout_selector.reset(done_tensor)
        last_action_for_policy = action_tensor.clone()
        warm_policy_last_action = action_tensor.clone()
        if done:
            last_action_for_policy.zero_()
            warm_policy_last_action.zero_()

    duration_ms = int(1000 * 1 / max(int(fps), 1))
    imageio.mimsave(save_path, frames, format="GIF", loop=0, duration=duration_ms)
    viz_env.close()


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


def resolve_path_relative_to_dir(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def create_warm_start_policy(
    warm_model_path: str,
    warm_args_path: str,
    warm_config_path: str,
    env_action_space: gym.spaces.Box,
    raw_obs_dim: int,
    act_dim: int,
    device: str,
):
    with open(warm_args_path, "r") as f:
        warm_args = yaml.load(f, Loader=yaml.FullLoader)
    with open(warm_config_path, "r") as f:
        warm_config = yaml.load(f, Loader=yaml.FullLoader)

    warm_use_last_action = bool(warm_args.get("use_last_action_in_policy_state", False))
    warm_hidden_layer_size = int(warm_args.get("agent_hidden_layer_size", 64))
    warm_num_hidden_layers = int(warm_args.get("agent_num_hidden_layers", 2))
    warm_action_scale = float(warm_args.get("action_scale", 0.02))
    if warm_config.get("air_hockey", {}).get("use_pid", False):
        warm_action_scale = 1.0

    warm_policy_obs_dim = raw_obs_dim + act_dim if warm_use_last_action else raw_obs_dim
    warm_policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(warm_policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=env_action_space,
    )
    warm_policy = DeterministicAgent(
        warm_policy_env_view,
        action_scale=warm_action_scale,
        action_bias=0.0,
        hidden_layer_size=warm_hidden_layer_size,
        num_hidden_layers=warm_num_hidden_layers,
    ).to(device)
    warm_state_dict = torch.load(warm_model_path, map_location=device, weights_only=False)
    warm_policy.load_state_dict(warm_state_dict)
    warm_policy.eval()
    return warm_policy, warm_use_last_action


def extract_current_paddle_position(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        return observation[:, 12:14]
    return observation[:, 0:2]


def extract_current_puck_position(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        return observation[:, 27:29]
    if obs_dim >= 8:
        return observation[:, 4:6]
    if obs_dim >= 4:
        return observation[:, 2:4]
    raise ValueError(f"Observation dim {obs_dim} is too small to extract puck position.")


def extract_current_puck_velocity(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        # History obs stores 5 puck positions at indices [15:30], so estimate velocity
        # using window displacement (last - first) to reduce one-step noise.
        return observation[:, 27:29] - observation[:, 15:17]
    if obs_dim >= 8 and obs_dim < 30:
        return observation[:, 6:8]
    return torch.zeros((observation.shape[0], 2), dtype=observation.dtype, device=observation.device)


def velocity_reward_from_magnitude(
    velocity_mag: torch.Tensor,
    velocity_at_one: float,
    velocity_at_zero: float,
) -> torch.Tensor:
    denom = max(velocity_at_zero - velocity_at_one, 1e-6)
    reward = 1.0 - (velocity_mag - velocity_at_one) / denom
    return torch.clamp_max(reward, 1.0)


def jerk_reward_from_magnitude(
    jerk_mag: torch.Tensor,
    jerk_at_one: float,
    jerk_at_zero: float,
) -> torch.Tensor:
    denom = max(jerk_at_zero - jerk_at_one, 1e-6)
    return 1.0 - (jerk_mag - jerk_at_one) / denom


def linear_anneal(start: float, end: float, step: int, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return end
    progress = min(max(step, 0) / float(anneal_steps), 1.0)
    return start + progress * (end - start)


def primitive_exploration_chance_for_step(args, step: int) -> float:
    # Allow an explicit pre-learning takeover chance while the replay buffer is bootstrapped.
    if (
        args.exploration_primitive_chance_pre_learning_starts is not None
        and step < args.learning_starts
    ):
        return float(args.exploration_primitive_chance_pre_learning_starts)
    return linear_anneal(
        args.exploration_primitive_chance_start,
        args.exploration_primitive_chance,
        step,
        args.exploration_primitive_chance_anneal_steps,
    )


def validate_optional_exploration_range(
    *,
    primitive_name: str,
    min_angle_deg: float | None,
    max_angle_deg: float | None,
    min_magnitude: float | None,
    max_magnitude: float | None,
) -> None:
    values = (min_angle_deg, max_angle_deg, min_magnitude, max_magnitude)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            f"{primitive_name} exploration range requires all four fields: "
            "min_angle_deg, max_angle_deg, min_magnitude, max_magnitude."
        )


def sum_info_metric(infos: dict, metric_name: str) -> float:
    metric_values = infos.get(metric_name)
    if metric_values is None:
        return 0.0
    try:
        return float(np.asarray(metric_values, dtype=np.float32).sum())
    except Exception:
        return 0.0


def sum_info_bool_metric(infos: dict, metric_name: str) -> float:
    metric_values = infos.get(metric_name)
    if metric_values is None:
        return 0.0
    try:
        return float(np.asarray(metric_values, dtype=np.bool_).sum())
    except Exception:
        return 0.0


@dataclass
class Args:
    # Evaluation-only mode (no exploration, no replay writes, no updates).
    # In this mode, total_timesteps is the rollout horizon in env-steps.
    eval_mode: bool = False
    total_timesteps: int = 1000000
    num_envs: int = 1

    # TD3 core
    buffer_size: int = int(1e6)
    task_gamma: float = 0.975
    motion_gamma: float = 0.8
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    q_frequency: int = 1
    q_updates: int = 1
    policy_frequency: int = 2
    target_network_frequency: int = 1
    actor_updates_per_iteration: int = 1
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    per_enabled: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 200000
    per_eps: float = 1e-6
    critic_per_fraction: float = 0.7
    critic_uniform_fraction: float = 0.3
    success_buffer_size: int = int(2e5)
    failure_buffer_size: int = int(8e5)
    success_top_fraction: float = 0.2
    recent_episode_window_size: int = 500
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7

    # Primitive exploration takeover
    exploration_primitive_chance: float = 0.05
    exploration_primitive_chance_start: float = 0.5
    exploration_primitive_chance_pre_learning_starts: float | None = None
    exploration_pre_learning_action_source: Literal["random", "policy"] = "random"
    exploration_primitive_chance_anneal_steps: int = 50000
    exploration_primitive_steps: int = 3
    exploration_primitive_weight_stand_still: float = 1.0 / 3.0
    exploration_primitive_weight_same_direction: float = 1.0 / 3.0
    exploration_primitive_weight_y_aligned: float = 1.0 / 3.0
    exploration_primitive_weight_policy_takeover: float = 0.4
    exploration_primitive_weight_target_position_directional: float = 0.0
    exploration_primitive_weight_anneal_stand_still: float = 0.3
    exploration_primitive_weight_anneal_same_direction: float = 0.1
    exploration_primitive_weight_anneal_y_aligned: float = 0.6
    exploration_primitive_weight_anneal_policy_takeover: float = 0.4
    exploration_primitive_weight_anneal_target_position_directional: float = 0.0
    exploration_direction_y_component_weight: float = 1.5
    exploration_target_position_min_distance: float = 0.2
    exploration_target_position_max_distance: float = 0.5
    exploration_target_position_delta_x: float = 0.26
    exploration_target_position_delta_y: float = 0.12
    exploration_target_position_steps: int = 5
    exploration_same_direction_min_angle_deg: float | None = None
    exploration_same_direction_max_angle_deg: float | None = None
    exploration_same_direction_min_magnitude: float | None = None
    exploration_same_direction_max_magnitude: float | None = None
    exploration_y_aligned_min_angle_deg: float | None = None
    exploration_y_aligned_max_angle_deg: float | None = None
    exploration_y_aligned_min_magnitude: float | None = None
    exploration_y_aligned_max_magnitude: float | None = None
    exploration_target_position_directional_min_angle_deg: float | None = None
    exploration_target_position_directional_max_angle_deg: float | None = None
    exploration_target_position_directional_min_magnitude: float | None = None
    exploration_target_position_directional_max_magnitude: float | None = None
    exploration_pre_contact_hit_variant_chance: float = 0.15
    exploration_pre_contact_hit_variant_steps: int = 5
    exploration_pre_contact_hit_variant_distance_threshold: float = 0.25
    exploration_pre_contact_hit_variant_scale_min: float = 0.5
    exploration_pre_contact_hit_variant_scale_max: float = 1.5
    exploration_pre_contact_hit_variant_min_upward_displacement_x: float = 0.12
    exploration_policy_takeover_enabled: bool = True
    exploration_policy_takeover_model_path: str = "model1.pth"
    exploration_policy_takeover_args_path: str = "args.yaml"
    exploration_policy_takeover_config_path: str = "config.yaml"

    # Dual-head reward decomposition
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 1.0
    estop_motion_reward_penalty: float = -5.0

    # Motion reward component weights
    stand_still_reward_weight: float = 0.5
    temporal_alignment_reward_weight: float = 0.5
    axis_alignment_reward_weight: float = 0.5
    velocity_reward_weight: float = 0.5
    jerk_reward_weight: float = 0.5

    # Motion reward component calibration
    stand_still_threshold: float = 0.015
    temporal_alignment_horizon: int = 4
    velocity_at_one: float = 0.3
    velocity_at_zero: float = 0.6
    jerk_at_one: float = 10.0
    jerk_at_zero: float = 23.0

    # Checkpointing
    checkpoint_interval: int = 25000
    save_replay_buffer: bool = True

    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str | None = None
    model_path: str | None = None
    # Full checkpoint load behavior when model_path points to a training-state dict.
    # - "full_resume": restore full training runtime state (legacy/default behavior)
    # - "weights_only": restore actor/Q networks only, keep runtime fresh
    # - "fine_tune": restore actor/Q networks + optimizer states, keep runtime fresh
    # - "residual": load source actor as frozen base, build fresh residual + fresh critic
    # Note: optimizer state restore may overwrite optimizer param-group values such as lr.
    full_checkpoint_load: Literal["full_resume", "weights_only", "fine_tune", "residual"] = "full_resume"
    # Residual RL: max magnitude of the residual action component (used when
    # full_checkpoint_load=="residual"). Combined action is clipped to the env
    # action bounds, so residual_scale > 0 caps |residual|_inf via tanh.
    residual_scale: float = 0.25
    # Fine-tune replay seeding: when full_checkpoint_load=="fine_tune", how many
    # samples (total, split proportionally between success/failure) to subsample
    # from the source replay buffer into the fresh target buffers. None or 0
    # disables seeding (current default behavior).
    fine_tune_replay_keep: int | None = None
    log_parent_dir: str | None = None
    run_name: str = "default"

    # Runtime
    device: str = "cuda:0"
    seed: int = 0
    action_scale: float = 0.02
    h_transform_eps: float = 1e-3

    # Agent/critic architecture
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    agent_hidden_size: int | None = None
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2
    q_hidden_size: int | None = None

    # Policy state options
    use_last_action_in_policy_state: bool = False

    # Optional physics overrides (applied to air_hockey config when not None)
    puck_density: float | None = None
    paddle_density: float | None = None
    gravity: float | None = None
    puck_damping: float | None = None
    paddle_damping: float | None = None
    puck_restitution: float | None = None
    paddle_restitution: float | None = None

    # Puck delay interpolation (timing jitter simulation)
    enable_puck_delay_interpolation: bool = False
    puck_delay_interpolation_min: float | None = None
    puck_delay_interpolation_max: float | None = None

    # Stochastic force attenuation
    enable_action_force_attenuation: bool = False
    action_force_attenuation_prob: float | None = None
    action_force_attenuation_min: float | None = None
    action_force_attenuation_max: float | None = None

    # Live episode GIF recording
    watch_ring_size: int = 10
    watch_episode_interval: int = 50
    sample_gif_interval: int = 10000
    sample_gif_max_storage_mb: float = 50.0


def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env

    return _thunk


def enforce_sample_storage_cap(samples_dir: str, max_mb: float) -> None:
    files = sorted(
        [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith(".gif")],
        key=os.path.getmtime,
    )
    total = sum(os.path.getsize(f) for f in files)
    max_bytes = max_mb * 1024 * 1024
    while total > max_bytes and files:
        oldest = files.pop(0)
        total -= os.path.getsize(oldest)
        os.remove(oldest)


if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()

    args = tyro.cli(Args, default=default_args)
    if args.num_envs != 1:
        raise ValueError(
            "This training script currently supports only single-environment collection. "
            f"Set num_envs=1, got {args.num_envs}."
        )
    if args.agent_hidden_size is not None:
        args.agent_hidden_layer_size = int(args.agent_hidden_size)
    if args.q_hidden_size is not None:
        args.q_hidden_layer_size = int(args.q_hidden_size)
    if not (0.0 <= args.critic_per_fraction <= 1.0):
        raise ValueError("critic_per_fraction must be between 0 and 1.")
    if not (0.0 <= args.critic_uniform_fraction <= 1.0):
        raise ValueError("critic_uniform_fraction must be between 0 and 1.")
    critic_mix_sum = float(args.critic_per_fraction + args.critic_uniform_fraction)
    if abs(critic_mix_sum - 1.0) > 1e-6:
        raise ValueError(
            f"critic_per_fraction + critic_uniform_fraction must equal 1.0, got {critic_mix_sum:.6f}."
        )
    if args.success_buffer_size <= 0:
        raise ValueError("success_buffer_size must be > 0.")
    if args.failure_buffer_size <= 0:
        raise ValueError("failure_buffer_size must be > 0.")
    if args.recent_episode_window_size <= 0:
        raise ValueError("recent_episode_window_size must be > 0.")
    if not (0.0 < args.success_top_fraction < 1.0):
        raise ValueError("success_top_fraction must be between 0 and 1 (exclusive).")
    if not (0.0 <= args.critic_success_sample_fraction <= 1.0):
        raise ValueError("critic_success_sample_fraction must be between 0 and 1.")
    if not (0.0 <= args.critic_failure_sample_fraction <= 1.0):
        raise ValueError("critic_failure_sample_fraction must be between 0 and 1.")
    success_failure_mix_sum = float(
        args.critic_success_sample_fraction + args.critic_failure_sample_fraction
    )
    if abs(success_failure_mix_sum - 1.0) > 1e-6:
        raise ValueError(
            "critic_success_sample_fraction + critic_failure_sample_fraction must equal 1.0, "
            f"got {success_failure_mix_sum:.6f}."
        )
    if args.q_updates <= 0:
        raise ValueError("q_updates must be > 0.")
    if args.target_network_frequency <= 0:
        raise ValueError("target_network_frequency must be > 0.")
    if args.actor_updates_per_iteration <= 0:
        raise ValueError("actor_updates_per_iteration must be > 0.")
    validate_optional_exploration_range(
        primitive_name="same_direction",
        min_angle_deg=args.exploration_same_direction_min_angle_deg,
        max_angle_deg=args.exploration_same_direction_max_angle_deg,
        min_magnitude=args.exploration_same_direction_min_magnitude,
        max_magnitude=args.exploration_same_direction_max_magnitude,
    )
    validate_optional_exploration_range(
        primitive_name="y_aligned",
        min_angle_deg=args.exploration_y_aligned_min_angle_deg,
        max_angle_deg=args.exploration_y_aligned_max_angle_deg,
        min_magnitude=args.exploration_y_aligned_min_magnitude,
        max_magnitude=args.exploration_y_aligned_max_magnitude,
    )
    validate_optional_exploration_range(
        primitive_name="target_position_directional",
        min_angle_deg=args.exploration_target_position_directional_min_angle_deg,
        max_angle_deg=args.exploration_target_position_directional_max_angle_deg,
        min_magnitude=args.exploration_target_position_directional_min_magnitude,
        max_magnitude=args.exploration_target_position_directional_max_magnitude,
    )

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sim_params_overrides = {
        "puck_density": args.puck_density,
        "paddle_density": args.paddle_density,
        "gravity": args.gravity,
        "puck_damping": args.puck_damping,
        "paddle_damping": args.paddle_damping,
        "puck_restitution": args.puck_restitution,
        "paddle_restitution": args.paddle_restitution,
        "enable_puck_delay_interpolation": True if args.enable_puck_delay_interpolation else None,
        "puck_delay_interpolation_min": args.puck_delay_interpolation_min,
        "puck_delay_interpolation_max": args.puck_delay_interpolation_max,
        "enable_action_force_attenuation": True if args.enable_action_force_attenuation else None,
        "action_force_attenuation_prob": args.action_force_attenuation_prob,
        "action_force_attenuation_min": args.action_force_attenuation_min,
        "action_force_attenuation_max": args.action_force_attenuation_max,
    }
    for key, value in sim_params_overrides.items():
        if value is not None:
            config["air_hockey"].setdefault("simulator_params", {})[key] = value
            print(f"Physics override: simulator_params.{key} = {value}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    run_name = args.run_name
    log_parent_dir = args.log_parent_dir or f"runs/default_training/{task_name}/{run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        base_log_parent_dir = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base_log_parent_dir}r{i}"
            i += 1
        print(f"Log directory exists. Saving to alternate log directory: {log_parent_dir}")
    os.makedirs(log_parent_dir, exist_ok=True)

    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    if "use_pid" in config["air_hockey"] and config["air_hockey"]["use_pid"]:
        action_scale = 1
    else:
        action_scale = args.action_scale

    raw_obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    policy_obs_dim = raw_obs_dim + act_dim if args.use_last_action_in_policy_state else raw_obs_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32
        ),
        single_action_space=envs.single_action_space,
    )

    actor = DeterministicAgent(
        policy_env_view,
        action_scale=action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(args.device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(args.device)
    actor_target.load_state_dict(actor.state_dict())

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    qf1 = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(args.device)
    qf2 = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(args.device)
    qf1_target = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(args.device)
    qf2_target = TD3DualHeadQNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=args.q_hidden_layer_size,
        num_hidden_layers=args.q_num_hidden_layers,
    ).to(args.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    resume_checkpoint = None
    checkpoint_load_mode = args.full_checkpoint_load

    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading model/checkpoint from {args.model_path}")
        loaded_obj = torch.load(args.model_path, map_location=args.device, weights_only=False)
        is_full_state = (
            isinstance(loaded_obj, dict) and "actor" in loaded_obj and "qf1" in loaded_obj
        )
        if checkpoint_load_mode == "residual":
            if args.eval_mode:
                raise ValueError(
                    "full_checkpoint_load='residual' is incompatible with eval_mode=True."
                )
            base_state = loaded_obj["actor"] if is_full_state else loaded_obj
            actor.load_state_dict(extract_deterministic_state_dict(base_state), strict=False)
            residual_online = DeterministicAgent(
                policy_env_view,
                action_scale=args.residual_scale,
                action_bias=0.0,
                hidden_layer_size=args.agent_hidden_layer_size,
                num_hidden_layers=args.agent_num_hidden_layers,
            ).to(args.device)
            residual_target = DeterministicAgent(
                policy_env_view,
                action_scale=args.residual_scale,
                action_bias=0.0,
                hidden_layer_size=args.agent_hidden_layer_size,
                num_hidden_layers=args.agent_num_hidden_layers,
            ).to(args.device)
            zero_init_residual_head(residual_online)
            zero_init_residual_head(residual_target)
            residual_target.load_state_dict(residual_online.state_dict())
            residual_action_low = torch.as_tensor(
                envs.single_action_space.low, dtype=torch.float32, device=args.device
            )
            residual_action_high = torch.as_tensor(
                envs.single_action_space.high, dtype=torch.float32, device=args.device
            )
            actor = ResidualActor(
                base_actor=actor,
                residual_actor=residual_online,
                action_low=residual_action_low,
                action_high=residual_action_high,
            ).to(args.device)
            actor_target = ResidualActor(
                base_actor=actor.base,
                residual_actor=residual_target,
                action_low=residual_action_low,
                action_high=residual_action_high,
            ).to(args.device)
            print(
                f"Residual mode: base frozen, residual_scale={args.residual_scale},"
                " critic from scratch."
            )
        elif is_full_state:
            resume_checkpoint = loaded_obj
            if args.eval_mode:
                if checkpoint_load_mode == "fine_tune":
                    raise ValueError(
                        "full_checkpoint_load='fine_tune' is incompatible with eval_mode=True."
                    )
                if checkpoint_load_mode == "full_resume":
                    print(
                        "Deprecation warning: eval_mode with full_checkpoint_load='full_resume' "
                        "will load as weights_only. Set full_checkpoint_load='weights_only' "
                        "explicitly to silence this warning."
                    )
                checkpoint_load_mode = "weights_only"
            actor.load_state_dict(extract_deterministic_state_dict(resume_checkpoint["actor"]), strict=False)
            if "actor_target" in resume_checkpoint:
                actor_target.load_state_dict(
                    extract_deterministic_state_dict(resume_checkpoint["actor_target"]),
                    strict=False,
                )
            else:
                actor_target.load_state_dict(actor.state_dict())
            qf1.load_state_dict(resume_checkpoint["qf1"])
            qf2.load_state_dict(resume_checkpoint["qf2"])
            qf1_target.load_state_dict(resume_checkpoint["qf1_target"])
            qf2_target.load_state_dict(resume_checkpoint["qf2_target"])
            print("Full training checkpoint loaded (network weights).")
            if checkpoint_load_mode == "weights_only":
                # Weights-only mode: keep networks, skip optimizer/replay/runtime restore.
                resume_checkpoint = None
                print("Weights-only load enabled: skipping resume of optimizer/replay/runtime state.")
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
            actor_target.load_state_dict(actor.state_dict())
            print("Actor-only model loaded successfully.")

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.q_weight_decay
    )
    if checkpoint_load_mode == "residual":
        actor_optimizer = optim.Adam(actor.residual.parameters(), lr=args.policy_lr)
    else:
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    pending_fine_tune_source_replay = None
    if resume_checkpoint is not None and checkpoint_load_mode == "fine_tune":
        load_fine_tune_optimizer_state(
            resume_checkpoint,
            q_optimizer=q_optimizer,
            actor_optimizer=actor_optimizer,
        )
        if args.fine_tune_replay_keep:
            pending_fine_tune_source_replay = {
                "success": resume_checkpoint.get("success_replay_buffer"),
                "failure": resume_checkpoint.get("failure_replay_buffer"),
            }
        resume_checkpoint = None
        print("Fine-tune load enabled: restored optimizer state, skipping replay/runtime resume.")

    if args.per_enabled:
        success_rb = TD3PrioritizedReplayBuffer(
            buffer_size=args.success_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            device=args.device,
            n_envs=args.num_envs,
            alpha=args.per_alpha,
            priority_eps=args.per_eps,
        )
        failure_rb = TD3PrioritizedReplayBuffer(
            buffer_size=args.failure_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            device=args.device,
            n_envs=args.num_envs,
            alpha=args.per_alpha,
            priority_eps=args.per_eps,
        )
        print(
            "✓ TD3 prioritized replay buffers initialized "
            f"(success_capacity={args.success_buffer_size:,}, "
            f"failure_capacity={args.failure_buffer_size:,}, alpha={args.per_alpha:.3f})\n"
        )
    else:
        success_rb = TD3ReplayBuffer(
            buffer_size=args.success_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            device=args.device,
            n_envs=args.num_envs,
        )
        failure_rb = TD3ReplayBuffer(
            buffer_size=args.failure_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            device=args.device,
            n_envs=args.num_envs,
        )
        print(
            "✓ TD3 replay buffers initialized "
            f"(success_capacity={args.success_buffer_size:,}, "
            f"failure_capacity={args.failure_buffer_size:,})\n"
        )

    if pending_fine_tune_source_replay is not None and args.fine_tune_replay_keep:
        kept = seed_fine_tune_replay_from_source(
            success_rb=success_rb,
            failure_rb=failure_rb,
            source_success=pending_fine_tune_source_replay["success"],
            source_failure=pending_fine_tune_source_replay["failure"],
            keep_total=int(args.fine_tune_replay_keep),
            seed=args.seed,
        )
        print(
            f"Fine-tune replay seeded: success_kept={kept['success_kept']}, "
            f"failure_kept={kept['failure_kept']} "
            f"(target keep_total={int(args.fine_tune_replay_keep)})"
        )

    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []

    obs, _ = envs.reset(seed=args.seed)
    last_action_for_policy = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=args.device)
    interval_paddle_puck_collisions = 0.0
    interval_env_steps = 0
    interval_primitive_env_steps = 0
    interval_primitive_horizontal_env_steps = 0
    interval_policy_takeover_env_steps = 0
    interval_target_position_directional_env_steps = 0
    rolling_step_stats_window = deque()
    rolling_episode_stats_window = deque()
    prev_protective_stop_flags = np.zeros(args.num_envs, dtype=bool)
    episode_trajectory = EpisodeTrajectory.empty()
    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0

    if args.temporal_alignment_horizon <= 0:
        raise ValueError("temporal_alignment_horizon must be > 0 for stand-still and alignment rewards.")

    temporal_horizon = args.temporal_alignment_horizon
    temporal_paddle_history = torch.zeros((args.num_envs, temporal_horizon + 1, 2), device=args.device)
    temporal_puck_history = torch.zeros((args.num_envs, temporal_horizon + 1, 2), device=args.device)
    # Consecutive non-terminal transitions since the most recent done.
    # This replaces done-window + fill-count bookkeeping for temporal_valid.
    steps_since_done = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    initial_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
    temporal_paddle_history[:, -1, :] = extract_current_paddle_position(initial_obs_tensor)
    temporal_puck_history[:, -1, :] = extract_current_puck_position(initial_obs_tensor)
    previous_puck_position_for_trigger = extract_current_puck_position(initial_obs_tensor).clone()

    current_velocity_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_acceleration_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_jerk_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)

    # --- Live episode GIF recording ---
    watch_dir = os.path.join(log_parent_dir, "watch")
    samples_dir = os.path.join(log_parent_dir, "samples")
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    renderer_env = AirHockeyEnv(config["air_hockey"])
    train_renderer = AirHockeyRenderer(
        renderer_env, show_target_position=True, show_acceleration_arrow=False
    )
    recording_frames: list = []
    recording_episode = False
    recording_cum_rew = 0.0
    recording_last_rew = 0.0
    completed_episode_count = 0
    watch_ring_idx = 0
    last_sample_gif_step = 0

    start_time = time.time()
    global_step = 0
    iteration = 0

    train_metrics = initialize_train_metrics()

    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=args.device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=args.device)
    warm_start_base_dir = os.path.join(os.path.dirname(__file__), "extras", "warm_start")
    warm_model_path = resolve_path_relative_to_dir(
        args.exploration_policy_takeover_model_path,
        warm_start_base_dir,
    )
    warm_args_path = resolve_path_relative_to_dir(
        args.exploration_policy_takeover_args_path,
        warm_start_base_dir,
    )
    warm_config_path = resolve_path_relative_to_dir(
        args.exploration_policy_takeover_config_path,
        warm_start_base_dir,
    )
    warm_policy_actor = None
    warm_policy_use_last_action = False
    warm_policy_last_action = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=args.device)
    if args.exploration_policy_takeover_enabled and not args.eval_mode:
        warm_policy_actor, warm_policy_use_last_action = create_warm_start_policy(
            warm_model_path=warm_model_path,
            warm_args_path=warm_args_path,
            warm_config_path=warm_config_path,
            env_action_space=envs.single_action_space,
            raw_obs_dim=raw_obs_dim,
            act_dim=act_dim,
            device=args.device,
        )
        print(f"Warm-start takeover policy loaded from {warm_model_path}")

    primitive_selector = PrimitiveExplorationSelector(
        num_envs=args.num_envs,
        chance=primitive_exploration_chance_for_step(args, global_step),
        takeover_steps=args.exploration_primitive_steps,
        device=args.device,
        dtype=torch.float32,
        direction_y_component_weight=args.exploration_direction_y_component_weight,
        target_min_distance=args.exploration_target_position_min_distance,
        target_max_distance=args.exploration_target_position_max_distance,
        target_action_delta_x=args.exploration_target_position_delta_x,
        target_action_delta_y=args.exploration_target_position_delta_y,
        same_direction_min_angle_deg=args.exploration_same_direction_min_angle_deg,
        same_direction_max_angle_deg=args.exploration_same_direction_max_angle_deg,
        same_direction_min_magnitude=args.exploration_same_direction_min_magnitude,
        same_direction_max_magnitude=args.exploration_same_direction_max_magnitude,
        y_aligned_min_angle_deg=args.exploration_y_aligned_min_angle_deg,
        y_aligned_max_angle_deg=args.exploration_y_aligned_max_angle_deg,
        y_aligned_min_magnitude=args.exploration_y_aligned_min_magnitude,
        y_aligned_max_magnitude=args.exploration_y_aligned_max_magnitude,
        target_position_directional_min_angle_deg=(
            args.exploration_target_position_directional_min_angle_deg
        ),
        target_position_directional_max_angle_deg=(
            args.exploration_target_position_directional_max_angle_deg
        ),
        target_position_directional_min_magnitude=(
            args.exploration_target_position_directional_min_magnitude
        ),
        target_position_directional_max_magnitude=(
            args.exploration_target_position_directional_max_magnitude
        ),
        target_takeover_steps=args.exploration_target_position_steps,
        pre_contact_hit_variant_chance=args.exploration_pre_contact_hit_variant_chance,
        pre_contact_hit_variant_steps=args.exploration_pre_contact_hit_variant_steps,
        pre_contact_hit_variant_distance_threshold=(
            args.exploration_pre_contact_hit_variant_distance_threshold
        ),
        pre_contact_hit_variant_scale_min=args.exploration_pre_contact_hit_variant_scale_min,
        pre_contact_hit_variant_scale_max=args.exploration_pre_contact_hit_variant_scale_max,
        pre_contact_hit_variant_min_upward_displacement_x=(
            args.exploration_pre_contact_hit_variant_min_upward_displacement_x
        ),
    )
    primitive_selector.set_primitive_weights(
        stand_still=args.exploration_primitive_weight_stand_still,
        same_direction=args.exploration_primitive_weight_same_direction,
        y_aligned=args.exploration_primitive_weight_y_aligned,
        policy_takeover=(
            args.exploration_primitive_weight_policy_takeover
            if args.exploration_policy_takeover_enabled
            else 0.0
        ),
        target_position_directional=args.exploration_primitive_weight_target_position_directional,
    )
    if args.eval_mode:
        primitive_selector.chance = 0.0
        primitive_selector.pre_contact_hit_variant_chance = 0.0
    if resume_checkpoint is not None:
        restored_state = load_resume_training_state(
            resume_checkpoint,
            device=args.device,
            recent_episode_window_size=args.recent_episode_window_size,
            success_rb=success_rb,
            failure_rb=failure_rb,
            primitive_selector=primitive_selector,
            q_optimizer=q_optimizer,
            actor_optimizer=actor_optimizer,
            defaults={
                "warm_policy_last_action": warm_policy_last_action,
                "train_metrics": train_metrics,
                "velocity_magnitudes": velocity_magnitudes,
                "acceleration_magnitudes": acceleration_magnitudes,
                "jerk_magnitudes": jerk_magnitudes,
                "interval_paddle_puck_collisions": interval_paddle_puck_collisions,
                "interval_env_steps": interval_env_steps,
                "interval_primitive_env_steps": interval_primitive_env_steps,
                "interval_primitive_horizontal_env_steps": interval_primitive_horizontal_env_steps,
                "interval_policy_takeover_env_steps": interval_policy_takeover_env_steps,
                "interval_target_position_directional_env_steps": (
                    interval_target_position_directional_env_steps
                ),
                "steps_since_done": steps_since_done,
                "recent_episode_returns": recent_episode_returns,
                "episode_return_success_threshold": episode_return_success_threshold,
                "rolling_step_stats_window": rolling_step_stats_window,
                "rolling_episode_stats_window": rolling_episode_stats_window,
            },
        )
        global_step = restored_state["global_step"]
        iteration = restored_state["iteration"]
        obs = restored_state["obs"]
        previous_puck_position_for_trigger = extract_current_puck_position(
            torch.tensor(obs, dtype=torch.float32, device=args.device)
        ).clone()
        last_action_for_policy = restored_state["last_action_for_policy"]
        warm_policy_last_action = restored_state["warm_policy_last_action"]
        temporal_paddle_history = restored_state["temporal_paddle_history"]
        temporal_puck_history = restored_state["temporal_puck_history"]
        steps_since_done = restored_state["steps_since_done"]
        current_velocity_mag = restored_state["current_velocity_mag"]
        current_acceleration_mag = restored_state["current_acceleration_mag"]
        current_jerk_mag = restored_state["current_jerk_mag"]
        train_metrics = restored_state["train_metrics"]
        velocity_magnitudes = restored_state["velocity_magnitudes"]
        acceleration_magnitudes = restored_state["acceleration_magnitudes"]
        jerk_magnitudes = restored_state["jerk_magnitudes"]
        interval_paddle_puck_collisions = restored_state["interval_paddle_puck_collisions"]
        interval_env_steps = restored_state["interval_env_steps"]
        interval_primitive_env_steps = restored_state["interval_primitive_env_steps"]
        interval_primitive_horizontal_env_steps = restored_state["interval_primitive_horizontal_env_steps"]
        interval_policy_takeover_env_steps = restored_state["interval_policy_takeover_env_steps"]
        interval_target_position_directional_env_steps = restored_state[
            "interval_target_position_directional_env_steps"
        ]
        episode_trajectory = restored_state["episode_trajectory"]
        recent_episode_returns = restored_state["recent_episode_returns"]
        episode_return_success_threshold = restored_state["episode_return_success_threshold"]
        rolling_step_stats_window = restored_state["rolling_step_stats_window"]
        rolling_episode_stats_window = restored_state["rolling_episode_stats_window"]
        print(f"Resuming training from global_step={global_step}, iteration={iteration}")

    while global_step < args.total_timesteps:
        should_update_train_metrics = global_step > 0 and np.random.rand() < 0.1
        should_refresh_annealing = (not args.eval_mode) and (np.random.rand() < 0.1)
        
        if should_refresh_annealing:
            annealing_active = global_step < args.exploration_primitive_chance_anneal_steps
            primitive_selector.chance = primitive_exploration_chance_for_step(args, global_step)
            if annealing_active:
                primitive_selector.set_primitive_weights(
                    stand_still=args.exploration_primitive_weight_anneal_stand_still,
                    same_direction=args.exploration_primitive_weight_anneal_same_direction,
                    y_aligned=args.exploration_primitive_weight_anneal_y_aligned,
                    policy_takeover=(
                        args.exploration_primitive_weight_anneal_policy_takeover
                        if args.exploration_policy_takeover_enabled
                        else 0.0
                    ),
                    target_position_directional=(
                        args.exploration_primitive_weight_anneal_target_position_directional
                    ),
                )
            else:
                primitive_selector.set_primitive_weights(
                    stand_still=args.exploration_primitive_weight_stand_still,
                    same_direction=args.exploration_primitive_weight_same_direction,
                    y_aligned=args.exploration_primitive_weight_y_aligned,
                    policy_takeover=(
                        args.exploration_primitive_weight_policy_takeover
                        if args.exploration_policy_takeover_enabled
                        else 0.0
                    ),
                    target_position_directional=(
                        args.exploration_primitive_weight_target_position_directional
                    ),
                )
        prev_action_for_transition = last_action_for_policy.clone()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
        policy_obs_tensor = augment_policy_observation(
            obs_tensor, last_action_for_policy, args.use_last_action_in_policy_state
        )

        if global_step < args.learning_starts and not args.eval_mode:
            if args.exploration_pre_learning_action_source == "random":
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    deterministic_actions = deterministic_actor_action(actor, policy_obs_tensor)
                    exploration = torch.randn_like(deterministic_actions) * args.exploration_noise
                    deterministic_actions = torch.clamp(
                        deterministic_actions + exploration, action_low, action_high
                    )
                    actions = deterministic_actions.cpu().numpy()
        else:
            with torch.no_grad():
                deterministic_actions = deterministic_actor_action(actor, policy_obs_tensor)
                if not args.eval_mode:
                    exploration = torch.randn_like(deterministic_actions) * args.exploration_noise
                    deterministic_actions = torch.clamp(
                        deterministic_actions + exploration, action_low, action_high
                    )
                else:
                    deterministic_actions = torch.clamp(deterministic_actions, action_low, action_high)
                actions = deterministic_actions.cpu().numpy()

        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=args.device)
        if not args.eval_mode:
            current_paddle_pos_for_primitive = extract_current_paddle_position(obs_tensor)
            current_puck_pos_for_primitive = extract_current_puck_position(obs_tensor)
            current_puck_vel_for_primitive = extract_current_puck_velocity(obs_tensor)
            if torch.all(current_puck_vel_for_primitive == 0):
                current_puck_vel_for_primitive = (
                    current_puck_pos_for_primitive - previous_puck_position_for_trigger
                )
            y_alignment_sign = torch.sign(
                current_puck_pos_for_primitive[:, 1] - current_paddle_pos_for_primitive[:, 1]
            )
            action_tensor, primitive_step_stats = primitive_selector.apply(
                action_tensor,
                action_low=action_low,
                action_high=action_high,
                y_alignment_sign=y_alignment_sign,
                current_paddle_position=current_paddle_pos_for_primitive,
                current_puck_position=current_puck_pos_for_primitive,
                current_puck_velocity=current_puck_vel_for_primitive,
                return_stats=True,
            )
            if args.exploration_policy_takeover_enabled and warm_policy_actor is not None:
                policy_takeover_mask = primitive_selector.active & (
                    primitive_selector.primitive_id == primitive_selector.primitive_ids.POLICY_TAKEOVER
                )
                if torch.any(policy_takeover_mask):
                    with torch.no_grad():
                        warm_policy_obs = augment_policy_observation(
                            obs_tensor,
                            warm_policy_last_action,
                            warm_policy_use_last_action,
                        )
                        warm_policy_actions = deterministic_actor_action(warm_policy_actor, warm_policy_obs)
                        warm_policy_actions = torch.clamp(warm_policy_actions, action_low, action_high)
                        action_tensor[policy_takeover_mask] = warm_policy_actions[policy_takeover_mask]
        else:
            primitive_step_stats = {
                "primitive_applied_count": 0,
                "primitive_horizontal_dominant_count": 0,
                "policy_takeover_applied_count": 0,
                "target_position_directional_applied_count": 0,
            }
        actions = action_tensor.cpu().numpy()

        if recording_episode:
            frame = train_renderer.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
            cv2.putText(
                frame, f"R: {recording_last_rew:.2f}",
                (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
            )
            cv2.putText(
                frame, f"G: {recording_cum_rew:.2f}",
                (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
            )
            cv2.putText(
                frame, f"Step: {global_step}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1,
            )
            recording_frames.append(frame)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        step_puck_hits = sum_info_metric(infos, "paddle_puck_collision_count")
        interval_paddle_puck_collisions += step_puck_hits
        current_protective_stop_flags = np.asarray(
            infos.get("protective_stop", np.zeros(args.num_envs, dtype=bool)),
            dtype=np.bool_,
        )
        current_protective_stop_flags = np.atleast_1d(current_protective_stop_flags)
        estop_event_mask = np.zeros(args.num_envs, dtype=bool)
        if current_protective_stop_flags.size == args.num_envs:
            estop_event_mask = np.logical_and(current_protective_stop_flags, np.logical_not(prev_protective_stop_flags))
            step_estop_events = float(estop_event_mask.sum())
            prev_protective_stop_flags = current_protective_stop_flags.copy()
            prev_protective_stop_flags[dones] = False
        else:
            step_estop_events = sum_info_bool_metric(infos, "protective_stop")
            prev_protective_stop_flags = np.zeros(args.num_envs, dtype=bool)
        interval_env_steps += args.num_envs
        interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
        interval_primitive_horizontal_env_steps += int(
            primitive_step_stats["primitive_horizontal_dominant_count"]
        )
        interval_policy_takeover_env_steps += int(primitive_step_stats["policy_takeover_applied_count"])
        interval_target_position_directional_env_steps += int(
            primitive_step_stats["target_position_directional_applied_count"]
        )
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=args.device)
        current_paddle_pos = extract_current_paddle_position(next_obs_tensor)
        current_puck_pos = extract_current_puck_position(next_obs_tensor)
        previous_puck_position_for_trigger = current_puck_pos.clone()
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=args.device)
        primitive_selector.reset(done_tensor)
        last_action_for_policy = action_tensor.clone()
        last_action_for_policy[done_tensor] = 0
        warm_policy_last_action = action_tensor.clone()
        warm_policy_last_action[done_tensor] = 0

        task_rewards = torch.tensor(rewards, dtype=torch.float32, device=args.device)

        current_velocity_mag, current_acceleration_mag, current_jerk_mag = parse_motion_magnitudes_from_infos(
            infos=infos,
            num_envs=args.num_envs,
            device=args.device,
            fallback_velocity_mag=current_velocity_mag,
            fallback_acceleration_mag=current_acceleration_mag,
            fallback_jerk_mag=current_jerk_mag,
        )

        temporal_paddle_history = torch.roll(temporal_paddle_history, shifts=-1, dims=1)
        temporal_paddle_history[:, -1, :] = current_paddle_pos
        temporal_puck_history = torch.roll(temporal_puck_history, shifts=-1, dims=1)
        temporal_puck_history[:, -1, :] = current_puck_pos
        steps_since_done = torch.where(
            done_tensor,
            torch.zeros_like(steps_since_done),
            steps_since_done + 1,
        )

        realized_movement = temporal_paddle_history[:, -1, :] - temporal_paddle_history[:, 0, :]
        movement_norm = torch.norm(realized_movement, dim=-1)
        eps = 1e-8

        temporal_valid = steps_since_done >= temporal_horizon
        stand_still_reward_raw = ((movement_norm <= args.stand_still_threshold) & temporal_valid).float()

        target_direction = temporal_puck_history[:, 0, :] - temporal_paddle_history[:, 0, :]
        movement_norm_safe = movement_norm.clamp_min(eps)
        target_norm_safe = torch.norm(target_direction, dim=-1).clamp_min(eps)
        temporal_cosine = (realized_movement * target_direction).sum(dim=-1) / (
            movement_norm_safe * target_norm_safe
        )
        temporal_alignment_reward_raw = torch.clamp((temporal_cosine + 1.0) * 0.5, 0.0, 1.0)
        temporal_alignment_reward_raw = temporal_alignment_reward_raw * temporal_valid.float()
        temporal_alignment_reward_raw = torch.where(
            stand_still_reward_raw > 0.5, torch.ones_like(temporal_alignment_reward_raw), temporal_alignment_reward_raw
        )

        movement_unit = realized_movement / movement_norm_safe.unsqueeze(-1)
        max_axis_cosine = torch.max(torch.abs(movement_unit[:, 0]), torch.abs(movement_unit[:, 1]))
        min_axis_cosine = float(1.0 / np.sqrt(2.0))
        axis_alignment_reward_raw = (max_axis_cosine - min_axis_cosine) / (1.0 - min_axis_cosine + eps)
        axis_alignment_reward_raw = torch.clamp(axis_alignment_reward_raw, 0.0, 1.0) * temporal_valid.float()
        axis_alignment_reward_raw = torch.where(
            stand_still_reward_raw > 0.5, torch.ones_like(axis_alignment_reward_raw), axis_alignment_reward_raw
        )

        velocity_reward_raw = velocity_reward_from_magnitude(
            current_velocity_mag, velocity_at_one=args.velocity_at_one, velocity_at_zero=args.velocity_at_zero
        )
        jerk_reward_raw = jerk_reward_from_magnitude(
            current_jerk_mag, jerk_at_one=args.jerk_at_one, jerk_at_zero=args.jerk_at_zero
        )

        stand_still_reward_weighted = args.stand_still_reward_weight * stand_still_reward_raw
        temporal_alignment_reward_weighted = args.temporal_alignment_reward_weight * temporal_alignment_reward_raw
        axis_alignment_reward_weighted = args.axis_alignment_reward_weight * axis_alignment_reward_raw
        velocity_reward_weighted = args.velocity_reward_weight * velocity_reward_raw
        jerk_reward_weighted = args.jerk_reward_weight * jerk_reward_raw

        motion_rewards = (
            stand_still_reward_weighted
            + temporal_alignment_reward_weighted
            + axis_alignment_reward_weighted
            + velocity_reward_weighted
            + jerk_reward_weighted
        )
        if np.any(estop_event_mask):
            estop_event_mask_tensor = torch.as_tensor(estop_event_mask, dtype=torch.float32, device=args.device)
            motion_rewards = motion_rewards + args.estop_motion_reward_penalty * estop_event_mask_tensor

        if recording_episode:
            recording_last_rew = float(task_rewards[0].item())
            recording_cum_rew += recording_last_rew

        if should_update_train_metrics:
            motion_reward_mean_metrics = tensor_mean_items(
                {
                    "rewards/temporal_valid_fraction": temporal_valid.float(),
                    "rewards/stand_still_reward_raw_mean": stand_still_reward_raw,
                    "rewards/temporal_alignment_reward_raw_mean": temporal_alignment_reward_raw,
                    "rewards/axis_alignment_reward_raw_mean": axis_alignment_reward_raw,
                    "rewards/velocity_reward_raw_mean": velocity_reward_raw,
                    "rewards/jerk_reward_raw_mean": jerk_reward_raw,
                    "rewards/stand_still_reward_weighted_mean": stand_still_reward_weighted,
                    "rewards/temporal_alignment_reward_weighted_mean": temporal_alignment_reward_weighted,
                    "rewards/axis_alignment_reward_weighted_mean": axis_alignment_reward_weighted,
                    "rewards/velocity_reward_weighted_mean": velocity_reward_weighted,
                    "rewards/jerk_reward_weighted_mean": jerk_reward_weighted,
                }
            )
            train_metrics.update(motion_reward_mean_metrics)
            # Log this rollout diagnostic at collection time so it is not biased
            # by terminal-step-only training updates.
            writer.add_scalar(
                "rewards/temporal_valid_fraction",
                motion_reward_mean_metrics["rewards/temporal_valid_fraction"],
                global_step,
            )

        if dones.any():
            temporal_paddle_history[done_tensor] = 0
            temporal_paddle_history[done_tensor, -1, :] = current_paddle_pos[done_tensor]
            temporal_puck_history[done_tensor] = 0
            temporal_puck_history[done_tensor, -1, :] = current_puck_pos[done_tensor]

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode_return" in info:
                    writer.add_scalar("charts/episodic_return", info["episode_return"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode_length"], global_step)
                    rolling_episode_stats_window.append(
                        (
                            int(global_step + args.num_envs),
                            float(info["episode_return"]),
                            float(info["episode_length"]),
                            1.0 if info.get("success", False) else 0.0,
                        )
                    )
                    if "motion_data" in info:
                        velocity_magnitudes.extend(info["motion_data"]["velocity_mags"])
                        acceleration_magnitudes.extend(info["motion_data"]["acceleration_mags"])
                        jerk_magnitudes.extend(info["motion_data"]["jerk_mags"])
        rolling_step_stats_window.append(
            (
                int(global_step + args.num_envs),
                int(args.num_envs),
                float(step_puck_hits),
                float(step_estop_events),
            )
        )
        rolling_cutoff_step = int(global_step + args.num_envs - ROLLING_STATS_WINDOW_STEPS)
        while rolling_step_stats_window and int(rolling_step_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_step_stats_window.popleft()
        while rolling_episode_stats_window and int(rolling_episode_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_episode_stats_window.popleft()

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        real_next_obs_tensor = torch.as_tensor(real_next_obs, dtype=torch.float32, device=args.device)
        terminations_tensor = torch.as_tensor(terminations, dtype=torch.float32, device=args.device)
        if not args.eval_mode:
            episode_trajectory.append_step(
                obs=obs_tensor[0],
                next_obs=real_next_obs_tensor[0],
                action=action_tensor[0],
                task_reward=task_rewards[0],
                motion_reward=motion_rewards[0],
                done=terminations_tensor[0],
                prev_action=prev_action_for_transition[0],
            )
            episode_return_success_threshold = finalize_episode_if_done(
                episode_done=bool(dones[0]),
                episode_trajectory=episode_trajectory,
                recent_episode_returns=recent_episode_returns,
                success_top_fraction=args.success_top_fraction,
                episode_return_success_threshold=episode_return_success_threshold,
                success_rb=success_rb,
                failure_rb=failure_rb,
            )
        episode_finished = bool(dones[0])

        if recording_episode and episode_finished and len(recording_frames) > 0:
            watch_path = os.path.join(watch_dir, f"ep_{watch_ring_idx}.gif")
            imageio.mimsave(watch_path, recording_frames, format="GIF", loop=0, duration=50)
            watch_ring_idx = (watch_ring_idx + 1) % args.watch_ring_size

            if global_step - last_sample_gif_step >= args.sample_gif_interval:
                sample_path = os.path.join(samples_dir, f"step_{global_step}.gif")
                imageio.mimsave(sample_path, recording_frames, format="GIF", loop=0, duration=50)
                last_sample_gif_step = global_step
                enforce_sample_storage_cap(samples_dir, args.sample_gif_max_storage_mb)

            recording_frames.clear()
            recording_episode = False
            recording_cum_rew = 0.0
            recording_last_rew = 0.0

        if episode_finished:
            completed_episode_count += 1
            if completed_episode_count % args.watch_episode_interval == 0:
                recording_episode = True

        obs = next_obs

        if global_step > args.learning_starts and episode_finished and not args.eval_mode:
            for q_update_idx in range(args.q_updates):
                success_batch_count, failure_batch_count = critic_success_failure_counts(
                    batch_size=args.batch_size,
                    success_fraction=args.critic_success_sample_fraction,
                    success_available=len(success_rb) > 0,
                    failure_available=len(failure_rb) > 0,
                )
                if success_batch_count + failure_batch_count == 0:
                    continue
                if args.per_enabled:
                    per_beta = linear_anneal(
                        args.per_beta_start,
                        args.per_beta_end,
                        global_step,
                        args.per_beta_anneal_steps,
                    )
                else:
                    per_beta = 0.0

                replay_chunks: List[Dict[str, torch.Tensor]] = []
                per_priority_update_slices: List[Tuple[object, torch.Tensor, int, int]] = []
                per_sample_count = 0
                uniform_sample_count = 0
                running_offset = 0
                for source_buffer, source_count in (
                    (success_rb, success_batch_count),
                    (failure_rb, failure_batch_count),
                ):
                    if source_count <= 0:
                        continue
                    source_chunk = sample_critic_source_chunk(
                        replay_buffer=source_buffer,
                        sample_count=source_count,
                        per_enabled=args.per_enabled,
                        per_beta=per_beta,
                        critic_per_fraction=args.critic_per_fraction,
                    )
                    source_data = source_chunk["data"]
                    replay_chunks.append(source_data)
                    source_batch_size = int(source_data["task_rewards"].shape[0])
                    source_per_count = int(source_chunk["per_count"])
                    source_uniform_count = int(source_chunk["uniform_count"])
                    per_sample_count += source_per_count
                    uniform_sample_count += source_uniform_count
                    if (
                        args.per_enabled
                        and source_per_count > 0
                        and source_chunk["per_indices"] is not None
                    ):
                        per_priority_update_slices.append(
                            (
                                source_buffer,
                                source_chunk["per_indices"],
                                running_offset,
                                running_offset + source_per_count,
                            )
                        )
                    running_offset += source_batch_size

                if len(replay_chunks) == 1:
                    data = replay_chunks[0]
                else:
                    data = concat_replay_samples(replay_chunks)
                sampled_observations = data["observations"]
                sampled_next_observations = data["next_observations"]
                sampled_actions = data["actions"]
                sampled_task_rewards = data["task_rewards"]
                sampled_motion_rewards = data["motion_rewards"]
                sampled_dones = data["dones"]
                sampled_prev_actions = data["prev_actions"]
                sampled_weights = data.get("weights")
                if sampled_weights is None:
                    sampled_weights = torch.ones_like(sampled_task_rewards)
                sampled_weights = sampled_weights.view(-1)
                sampled_next_prev_actions = sampled_actions * (1.0 - sampled_dones.unsqueeze(-1))
                sampled_next_policy_observations = augment_policy_observation(
                    sampled_next_observations,
                    sampled_next_prev_actions,
                    args.use_last_action_in_policy_state,
                )

                with torch.no_grad():
                    target_next_action = deterministic_actor_action(
                        actor_target,
                        sampled_next_policy_observations,
                    )
                    noise = torch.randn_like(target_next_action) * args.policy_noise
                    noise = torch.clamp(noise, -args.noise_clip, args.noise_clip)
                    target_next_action = torch.clamp(target_next_action + noise, action_low, action_high)

                    q1_next_task_h, q1_next_motion_h = qf1_target(
                        sampled_next_observations,
                        target_next_action,
                    )
                    q2_next_task_h, q2_next_motion_h = qf2_target(
                        sampled_next_observations,
                        target_next_action,
                    )

                    min_next_task_h = torch.min(q1_next_task_h, q2_next_task_h)
                    min_next_motion_h = torch.min(q1_next_motion_h, q2_next_motion_h)

                    min_next_task = h_inverse(min_next_task_h, eps=args.h_transform_eps).view(-1)
                    min_next_motion = h_inverse(min_next_motion_h, eps=args.h_transform_eps).view(-1)

                    bellman_target_task_original = sampled_task_rewards + (
                        1 - sampled_dones
                    ) * args.task_gamma * min_next_task
                    bellman_target_motion_original = sampled_motion_rewards + (
                        1 - sampled_dones
                    ) * args.motion_gamma * min_next_motion

                    next_q_task_value_h = h_transform(
                        bellman_target_task_original, eps=args.h_transform_eps
                    )
                    next_q_motion_value_h = h_transform(
                        bellman_target_motion_original, eps=args.h_transform_eps
                    )

                    if should_update_train_metrics and q_update_idx == args.q_updates - 1:
                        train_metrics.update(
                            {
                                "debug/bellman_target_task_original_mean": (
                                    bellman_target_task_original.mean().item()
                                ),
                                "debug/bellman_target_motion_original_mean": (
                                    bellman_target_motion_original.mean().item()
                                ),
                                "debug/next_q_task_h_mean": next_q_task_value_h.mean().item(),
                                "debug/next_q_motion_h_mean": next_q_motion_value_h.mean().item(),
                            }
                        )

                q1_task_h, q1_motion_h = qf1(
                    sampled_observations,
                    sampled_actions,
                )
                q2_task_h, q2_motion_h = qf2(
                    sampled_observations,
                    sampled_actions,
                )

                q1_task_error_h = q1_task_h.view(-1) - next_q_task_value_h
                q2_task_error_h = q2_task_h.view(-1) - next_q_task_value_h
                q1_motion_error_h = q1_motion_h.view(-1) - next_q_motion_value_h
                q2_motion_error_h = q2_motion_h.view(-1) - next_q_motion_value_h

                q1_task_loss = (sampled_weights * q1_task_error_h.pow(2)).mean()
                q2_task_loss = (sampled_weights * q2_task_error_h.pow(2)).mean()
                q1_motion_loss = (sampled_weights * q1_motion_error_h.pow(2)).mean()
                q2_motion_loss = (sampled_weights * q2_motion_error_h.pow(2)).mean()
                q_total_loss = q1_task_loss + q2_task_loss + q1_motion_loss + q2_motion_loss

                q_optimizer.zero_grad()
                q_total_loss.backward()
                q_optimizer.step()

                priority_td_error = 0.25 * (
                    q1_task_error_h.abs()
                    + q2_task_error_h.abs()
                    + q1_motion_error_h.abs()
                    + q2_motion_error_h.abs()
                )
                if args.per_enabled and per_sample_count > 0:
                    for source_buffer, per_indices, start_idx, end_idx in per_priority_update_slices:
                        source_buffer.update_priorities(
                            per_indices,
                            priority_td_error[start_idx:end_idx].detach() + args.per_eps,
                        )

                if should_update_train_metrics and q_update_idx == args.q_updates - 1:
                    sampled_priorities = data.get("sampled_priorities")
                    if sampled_priorities is None:
                        sampled_priorities = torch.zeros_like(sampled_weights)
                    positive_task_reward_mask = sampled_task_rewards > 0.0
                    positive_task_reward_count = float(positive_task_reward_mask.sum().item())
                    minibatch_size = max(int(sampled_task_rewards.numel()), 1)
                    positive_task_rewards = sampled_task_rewards[positive_task_reward_mask]
                    priority_td_error_mean = (
                        priority_td_error.mean().item()
                        if args.per_enabled and per_sample_count > 0
                        else 0.0
                    )
                    train_metrics.update(
                        {
                            "losses/q_task_loss": (q1_task_loss + q2_task_loss).item() / 2.0,
                            "losses/q_motion_loss": (q1_motion_loss + q2_motion_loss).item() / 2.0,
                            "losses/q_total_loss": q_total_loss.item(),
                            "losses/q1_task_mean": q1_task_h.mean().item(),
                            "losses/q1_motion_mean": q1_motion_h.mean().item(),
                            "losses/q1_total_mean": (q1_task_h + q1_motion_h).mean().item(),
                            "rewards/sampled_task_reward_mean": sampled_task_rewards.mean().item(),
                            "rewards/sampled_task_reward_min": sampled_task_rewards.min().item(),
                            "rewards/sampled_task_reward_std": sampled_task_rewards.std(
                                unbiased=False
                            ).item(),
                            "rewards/sampled_task_reward_positive_count": positive_task_reward_count,
                            "rewards/sampled_task_reward_positive_fraction": (
                                positive_task_reward_count / float(minibatch_size)
                            ),
                            "rewards/sampled_task_reward_positive_mean": (
                                positive_task_rewards.mean().item()
                                if positive_task_rewards.numel() > 0
                                else 0.0
                            ),
                            "rewards/sampled_task_reward_positive_std": (
                                positive_task_rewards.std(unbiased=False).item()
                                if positive_task_rewards.numel() > 0
                                else 0.0
                            ),
                            "rewards/sampled_motion_reward_mean": sampled_motion_rewards.mean().item(),
                            "rewards/sampled_combined_reward_mean": (
                                sampled_task_rewards + sampled_motion_rewards
                            ).mean().item(),
                            "replay/per_beta": per_beta,
                            "replay/per_is_weight_mean": sampled_weights.mean().item(),
                            "replay/per_sampled_priority_mean": sampled_priorities.mean().item(),
                            "replay/per_priority_td_error_mean": priority_td_error_mean,
                            "replay/critic_per_sample_count": float(per_sample_count),
                            "replay/critic_uniform_sample_count": float(uniform_sample_count),
                            "replay/critic_per_sample_fraction": (
                                float(per_sample_count) / float(max(args.batch_size, 1))
                            ),
                            "replay/success_buffer_size": float(len(success_rb)),
                            "replay/failure_buffer_size": float(len(failure_rb)),
                            "replay/critic_success_sample_count": float(success_batch_count),
                            "replay/critic_failure_sample_count": float(failure_batch_count),
                            "replay/critic_success_sample_fraction": (
                                float(success_batch_count) / float(max(args.batch_size, 1))
                            ),
                            "replay/critic_failure_sample_fraction": (
                                float(failure_batch_count) / float(max(args.batch_size, 1))
                            ),
                            "replay/episode_return_success_threshold": (
                                episode_return_success_threshold
                            ),
                            "replay/recent_episode_window_count": float(len(recent_episode_returns)),
                        }
                    )

                if (q_update_idx + 1) % args.target_network_frequency == 0:
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for actor_update_idx in range(args.actor_updates_per_iteration):
                actor_success_count, actor_failure_count = critic_success_failure_counts(
                    batch_size=args.batch_size,
                    success_fraction=args.critic_success_sample_fraction,
                    success_available=len(success_rb) > 0,
                    failure_available=len(failure_rb) > 0,
                )
                if actor_success_count + actor_failure_count == 0:
                    continue
                actor_data_chunks: List[Dict[str, torch.Tensor]] = []
                if actor_success_count > 0:
                    actor_data_chunks.append(
                        sample_actor_source_chunk(success_rb, actor_success_count, args.per_enabled)
                    )
                if actor_failure_count > 0:
                    actor_data_chunks.append(
                        sample_actor_source_chunk(failure_rb, actor_failure_count, args.per_enabled)
                    )
                if len(actor_data_chunks) == 1:
                    data = actor_data_chunks[0]
                else:
                    data = {
                        "observations": torch.cat(
                            [chunk["observations"] for chunk in actor_data_chunks], dim=0
                        ),
                        "prev_actions": torch.cat(
                            [chunk["prev_actions"] for chunk in actor_data_chunks], dim=0
                        ),
                    }
                sampled_observations = data["observations"]
                sampled_prev_actions = data["prev_actions"]
                sampled_policy_observations = augment_policy_observation(
                    sampled_observations, sampled_prev_actions, args.use_last_action_in_policy_state
                )

                current_policy_actions = deterministic_actor_action(actor, sampled_policy_observations)
                q1_task_h, q1_motion_h = qf1(
                    sampled_observations,
                    current_policy_actions,
                )
                q1_task = h_inverse(q1_task_h, eps=args.h_transform_eps).view(-1)
                q1_motion = h_inverse(q1_motion_h, eps=args.h_transform_eps).view(-1)
                norm_task = (1.0 - args.task_gamma) * q1_task
                norm_motion = (1.0 - args.motion_gamma) * q1_motion
                actor_objective = (
                    args.task_reward_weight * norm_task
                    + args.motion_reward_weight * norm_motion
                )
                actor_loss = -actor_objective.mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                if should_update_train_metrics and actor_update_idx == args.actor_updates_per_iteration - 1:
                    train_metrics.update(
                        {
                            "losses/actor_loss": actor_loss.item(),
                            "losses/actor_norm_task_mean": norm_task.mean().item(),
                            "losses/actor_norm_motion_mean": norm_motion.mean().item(),
                        }
                    )

            if should_update_train_metrics:
                train_metrics.update(
                    {
                        "replay/success_buffer_size": float(len(success_rb)),
                        "replay/failure_buffer_size": float(len(failure_rb)),
                        "replay/episode_return_success_threshold": episode_return_success_threshold,
                        "replay/recent_episode_window_count": float(len(recent_episode_returns)),
                    }
                )
                log_scalar_metrics(writer, train_metrics, global_step)
                writer.add_scalar("charts/exploration_primitive_chance", primitive_selector.chance, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


        # LOGGING AND EVALUATION
        if global_step > 0 and global_step % 500 == 0:
            if rolling_episode_stats_window:
                rolling_returns = [item[1] for item in rolling_episode_stats_window]
                rolling_lengths = [item[2] for item in rolling_episode_stats_window]
                rolling_success = [item[3] for item in rolling_episode_stats_window]
                avg_return = float(np.mean(rolling_returns))
                min_return = float(np.min(rolling_returns))
                max_return = float(np.max(rolling_returns))
                avg_success = float(np.mean(rolling_success))
                avg_episode_length = float(np.mean(rolling_lengths))
                print(
                    f"Step {global_step}: Rolling(2k) Avg Return: {avg_return:.2f}, "
                    f"Min: {min_return:.2f}, Max: {max_return:.2f}, "
                    f"Success Rate: {avg_success:.2f}, Avg Episode Length: {avg_episode_length:.2f}, "
                    f"Episodes: {len(rolling_episode_stats_window)}"
                )
                writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
                writer.add_scalar("charts/min_episodic_return", min_return, global_step)
                writer.add_scalar("charts/max_episodic_return", max_return, global_step)
                writer.add_scalar("charts/avg_success_rate", avg_success, global_step)
                writer.add_scalar("charts/rolling2k_avg_episode_return", avg_return, global_step)
                writer.add_scalar("charts/rolling2k_avg_episode_length", avg_episode_length, global_step)
                writer.add_scalar("charts/rolling2k_episode_count", len(rolling_episode_stats_window), global_step)
            else:
                print(f"Step {global_step}: No episodes in rolling 2k-step window")

            rolling_window_env_steps = int(sum(item[1] for item in rolling_step_stats_window))
            rolling_window_puck_hits = float(sum(item[2] for item in rolling_step_stats_window))
            rolling_window_estop_events = float(sum(item[3] for item in rolling_step_stats_window))
            rolling_puck_hits_per_env_step = (
                rolling_window_puck_hits / float(rolling_window_env_steps)
                if rolling_window_env_steps > 0
                else 0.0
            )
            rolling_estop_rate = (
                rolling_window_estop_events / float(rolling_window_env_steps)
                if rolling_window_env_steps > 0
                else 0.0
            )
            print(
                f"Step {global_step}: Rolling(2k) Puck Hits: {int(rolling_window_puck_hits)}, "
                f"E-Stop Events: {int(rolling_window_estop_events)}, "
                f"Puck Hits/env-step: {rolling_puck_hits_per_env_step:.4f}, "
                f"E-Stop Rate: {rolling_estop_rate:.4f}"
            )
            writer.add_scalar("charts/rolling2k_puck_hits_total", rolling_window_puck_hits, global_step)
            writer.add_scalar("charts/rolling2k_estop_events_total", rolling_window_estop_events, global_step)
            writer.add_scalar(
                "charts/rolling2k_puck_hits_per_env_step",
                rolling_puck_hits_per_env_step,
                global_step,
            )
            writer.add_scalar("charts/rolling2k_estop_rate", rolling_estop_rate, global_step)

            if velocity_magnitudes:
                avg_vel_mag = np.mean(velocity_magnitudes)
                avg_acc_mag = np.mean(acceleration_magnitudes)
                avg_jerk_mag = np.mean(jerk_magnitudes)
                print(
                    f"Step {global_step}: Avg Velocity: {avg_vel_mag:.4f}, "
                    f"Avg Acceleration: {avg_acc_mag:.4f}, Avg Jerk: {avg_jerk_mag:.4f}"
                )
                writer.add_scalar("motion/avg_velocity_magnitude", avg_vel_mag, global_step)
                writer.add_scalar("motion/avg_acceleration_magnitude", avg_acc_mag, global_step)
                writer.add_scalar("motion/avg_jerk_magnitude", avg_jerk_mag, global_step)
                velocity_magnitudes.clear()
                acceleration_magnitudes.clear()
                jerk_magnitudes.clear()

            collisions_per_env_step = (
                interval_paddle_puck_collisions / max(interval_env_steps, 1) if interval_env_steps > 0 else 0.0
            )
            print(
                f"Step {global_step}: Paddle-Puck Collisions (last interval): "
                f"{int(interval_paddle_puck_collisions)} total, {collisions_per_env_step:.4f} per env-step"
            )
            writer.add_scalar(
                "contacts/interval_paddle_puck_collisions_total",
                interval_paddle_puck_collisions,
                global_step,
            )
            writer.add_scalar(
                "contacts/interval_paddle_puck_collisions_per_env_step",
                collisions_per_env_step,
                global_step,
            )
            primitive_fraction = interval_primitive_env_steps / max(interval_env_steps, 1) if interval_env_steps > 0 else 0.0
            primitive_directional_env_steps = max(
                interval_primitive_env_steps - interval_policy_takeover_env_steps, 0
            )
            primitive_horizontal_fraction = (
                interval_primitive_horizontal_env_steps / max(primitive_directional_env_steps, 1)
                if primitive_directional_env_steps > 0
                else 0.0
            )
            policy_takeover_fraction = (
                interval_policy_takeover_env_steps / max(interval_env_steps, 1) if interval_env_steps > 0 else 0.0
            )
            target_position_directional_fraction = (
                interval_target_position_directional_env_steps / max(interval_env_steps, 1)
                if interval_env_steps > 0
                else 0.0
            )
            print(
                f"Step {global_step}: Primitive Actions (last interval): "
                f"{interval_primitive_env_steps}/{interval_env_steps} env-steps ({primitive_fraction:.4f}), "
                f"horizontal-dominant: {interval_primitive_horizontal_env_steps}/{primitive_directional_env_steps} "
                f"({primitive_horizontal_fraction:.4f})"
            )
            print(
                f"Step {global_step}: Policy Takeover Actions (last interval): "
                f"{interval_policy_takeover_env_steps}/{interval_env_steps} env-steps "
                f"({policy_takeover_fraction:.4f})"
            )
            print(
                f"Step {global_step}: Target-Position Directional Actions (last interval): "
                f"{interval_target_position_directional_env_steps}/{interval_env_steps} env-steps "
                f"({target_position_directional_fraction:.4f})"
            )
            writer.add_scalar(
                "exploration/interval_primitive_env_steps",
                interval_primitive_env_steps,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_primitive_env_step_fraction",
                primitive_fraction,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_primitive_horizontal_env_steps",
                interval_primitive_horizontal_env_steps,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_primitive_horizontal_fraction",
                primitive_horizontal_fraction,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_policy_takeover_env_steps",
                interval_policy_takeover_env_steps,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_policy_takeover_fraction",
                policy_takeover_fraction,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_target_position_directional_env_steps",
                interval_target_position_directional_env_steps,
                global_step,
            )
            writer.add_scalar(
                "exploration/interval_target_position_directional_fraction",
                target_position_directional_fraction,
                global_step,
            )
            interval_paddle_puck_collisions = 0.0
            interval_env_steps = 0
            interval_primitive_env_steps = 0
            interval_primitive_horizontal_env_steps = 0
            interval_policy_takeover_env_steps = 0
            interval_target_position_directional_env_steps = 0

        if global_step > 0 and global_step % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(f"{checkpoint_dir}/config.yaml", "w") as f:
                yaml.dump(config, f)
            with open(f"{checkpoint_dir}/args.yaml", "w") as f:
                yaml.dump(vars(args), f)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(actor.state_dict(), model_path)
            torch.save(actor_target.state_dict(), f"{checkpoint_dir}/actor_target.pth")
            torch.save(qf1.state_dict(), f"{checkpoint_dir}/qf1.pth")
            torch.save(qf2.state_dict(), f"{checkpoint_dir}/qf2.pth")
            torch.save(qf1_target.state_dict(), f"{checkpoint_dir}/qf1_target.pth")
            torch.save(qf2_target.state_dict(), f"{checkpoint_dir}/qf2_target.pth")
            training_state = build_training_state(
                global_step=global_step,
                iteration=iteration,
                actor=actor,
                actor_target=actor_target,
                qf1=qf1,
                qf2=qf2,
                qf1_target=qf1_target,
                qf2_target=qf2_target,
                q_optimizer=q_optimizer,
                actor_optimizer=actor_optimizer,
                success_rb=success_rb,
                failure_rb=failure_rb,
                primitive_selector=primitive_selector,
                obs=obs,
                last_action_for_policy=last_action_for_policy,
                warm_policy_last_action=warm_policy_last_action,
                temporal_paddle_history=temporal_paddle_history,
                temporal_puck_history=temporal_puck_history,
                steps_since_done=steps_since_done,
                current_velocity_mag=current_velocity_mag,
                current_acceleration_mag=current_acceleration_mag,
                current_jerk_mag=current_jerk_mag,
                train_metrics=train_metrics,
                velocity_magnitudes=velocity_magnitudes,
                acceleration_magnitudes=acceleration_magnitudes,
                jerk_magnitudes=jerk_magnitudes,
                interval_paddle_puck_collisions=interval_paddle_puck_collisions,
                interval_env_steps=interval_env_steps,
                interval_primitive_env_steps=interval_primitive_env_steps,
                interval_primitive_horizontal_env_steps=interval_primitive_horizontal_env_steps,
                interval_policy_takeover_env_steps=interval_policy_takeover_env_steps,
                interval_target_position_directional_env_steps=(
                    interval_target_position_directional_env_steps
                ),
                episode_trajectory=episode_trajectory,
                recent_episode_returns=recent_episode_returns,
                episode_return_success_threshold=episode_return_success_threshold,
                rolling_step_stats_window=rolling_step_stats_window,
                rolling_episode_stats_window=rolling_episode_stats_window,
                args_dict=vars(args),
                include_replay_buffer=args.save_replay_buffer,
            )
            torch.save(training_state, f"{checkpoint_dir}/training_state.pth")
            print(f"\nCheckpoint saved at step {global_step}")
            try:
                evaluate_agent(
                    model_path,
                    checkpoint_dir,
                    config["air_hockey"],
                    n_eps=4,
                    n_gifs=1,
                    action_scale=action_scale,
                    agent_hidden_layer_size=args.agent_hidden_layer_size,
                    agent_num_hidden_layers=args.agent_num_hidden_layers,
                    use_last_action_in_policy_state=args.use_last_action_in_policy_state,
                )
                save_training_data_like_rollout_gif(
                    save_path=os.path.join(checkpoint_dir, "rollout_data_like_0.gif"),
                    env_params=config["air_hockey"],
                    actor=actor,
                    action_low=action_low,
                    action_high=action_high,
                    use_last_action_in_policy_state=args.use_last_action_in_policy_state,
                    exploration_noise=(0.0 if args.eval_mode else args.exploration_noise),
                    primitive_selector=primitive_selector,
                    warm_policy_actor=warm_policy_actor if args.exploration_policy_takeover_enabled else None,
                    warm_policy_use_last_action=warm_policy_use_last_action,
                    max_timesteps=200,
                    fps=20,
                    device=args.device,
                )
            except Exception as e:
                print(f"Evaluation failed: {e}")

        iteration += 1
        global_step += args.num_envs

    envs.close()

    if not args.eval_mode:
        torch.save(actor.state_dict(), f"{log_parent_dir}/model.pth")
        torch.save(actor_target.state_dict(), f"{log_parent_dir}/actor_target.pth")
        torch.save(qf1.state_dict(), f"{log_parent_dir}/qf1.pth")
        torch.save(qf2.state_dict(), f"{log_parent_dir}/qf2.pth")
        torch.save(qf1_target.state_dict(), f"{log_parent_dir}/qf1_target.pth")
        torch.save(qf2_target.state_dict(), f"{log_parent_dir}/qf2_target.pth")
        final_training_state = build_training_state(
            global_step=global_step,
            iteration=iteration,
            actor=actor,
            actor_target=actor_target,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1_target,
            qf2_target=qf2_target,
            q_optimizer=q_optimizer,
            actor_optimizer=actor_optimizer,
            success_rb=success_rb,
            failure_rb=failure_rb,
            primitive_selector=primitive_selector,
            obs=obs,
            last_action_for_policy=last_action_for_policy,
            warm_policy_last_action=warm_policy_last_action,
            temporal_paddle_history=temporal_paddle_history,
            temporal_puck_history=temporal_puck_history,
            steps_since_done=steps_since_done,
            current_velocity_mag=current_velocity_mag,
            current_acceleration_mag=current_acceleration_mag,
            current_jerk_mag=current_jerk_mag,
            train_metrics=train_metrics,
            velocity_magnitudes=velocity_magnitudes,
            acceleration_magnitudes=acceleration_magnitudes,
            jerk_magnitudes=jerk_magnitudes,
            interval_paddle_puck_collisions=interval_paddle_puck_collisions,
            interval_env_steps=interval_env_steps,
            interval_primitive_env_steps=interval_primitive_env_steps,
            interval_primitive_horizontal_env_steps=interval_primitive_horizontal_env_steps,
            interval_policy_takeover_env_steps=interval_policy_takeover_env_steps,
            interval_target_position_directional_env_steps=interval_target_position_directional_env_steps,
            episode_trajectory=episode_trajectory,
            recent_episode_returns=recent_episode_returns,
            episode_return_success_threshold=episode_return_success_threshold,
            rolling_step_stats_window=rolling_step_stats_window,
            rolling_episode_stats_window=rolling_episode_stats_window,
            args_dict=vars(args),
            include_replay_buffer=args.save_replay_buffer,
        )
        torch.save(final_training_state, f"{log_parent_dir}/training_state.pth")
    elif args.model_path is None:
        print("Eval mode is enabled without model_path; final evaluate_agent will be skipped.")

    try:
        if args.eval_mode:
            final_eval_model_path = os.path.join(log_parent_dir, "eval_mode_model.pth")
            torch.save(actor.state_dict(), final_eval_model_path)
        else:
            final_eval_model_path = f"{log_parent_dir}/model.pth"
        if final_eval_model_path is None:
            raise ValueError("No model path available for final evaluation.")
        evaluate_agent(
            final_eval_model_path,
            log_parent_dir,
            config["air_hockey"],
            action_scale=action_scale,
            agent_hidden_layer_size=args.agent_hidden_layer_size,
            agent_num_hidden_layers=args.agent_num_hidden_layers,
            use_last_action_in_policy_state=args.use_last_action_in_policy_state,
        )
    except Exception as e:
        print(f"Final evaluation failed: {e}")

    metrics = [
        "charts/episodic_return",
        "losses/q_task_loss",
        "losses/q_motion_loss",
        "losses/q_total_loss",
        "losses/actor_loss",
        "rewards/sampled_task_reward_mean",
        "rewards/sampled_motion_reward_mean",
        "rewards/sampled_combined_reward_mean",
        "replay/per_beta",
        "replay/per_is_weight_mean",
        "replay/per_sampled_priority_mean",
        "replay/per_priority_td_error_mean",
    ]
    save_tensorboard_plots(log_parent_dir, config, metrics=metrics)
    writer.close()

