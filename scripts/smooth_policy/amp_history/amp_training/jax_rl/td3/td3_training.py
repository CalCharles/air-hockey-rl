"""TD3 training script (JAX backend).

Drop-in replacement for amp_training/td3/td3_training.py.
Algorithm updates run in JAX; environment collection, exploration primitives,
and replay buffers are unchanged (NumPy / gymnasium).
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Tuple

import cv2
import gymnasium as gym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.amp_history.amp_training.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)
from scripts.smooth_policy.amp_history.amp_training.amp_training import (
    parse_motion_magnitudes_from_infos,
)
from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.utils import save_tensorboard_plots

from jax_rl.replay_buffer import (
    EpisodeBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    finalize_episode_if_done,
)
from jax_rl.td3.td3_algorithm import TD3
from jax_rl.td3.td3_config import TD3Config
from jax_rl.utils import linear_anneal, velocity_reward, jerk_reward

ROLLING_STATS_WINDOW_STEPS = 2000


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def extract_current_paddle_position(obs: np.ndarray) -> np.ndarray:
    if obs.shape[-1] >= 30:
        return obs[..., 12:14]
    return obs[..., 0:2]


def extract_current_puck_position(obs: np.ndarray) -> np.ndarray:
    d = obs.shape[-1]
    if d >= 30:
        return obs[..., 27:29]
    if d >= 8:
        return obs[..., 4:6]
    if d >= 4:
        return obs[..., 2:4]
    raise ValueError(f"Observation dim {d} too small to extract puck position.")


def extract_current_puck_velocity(obs: np.ndarray) -> np.ndarray:
    d = obs.shape[-1]
    if d >= 30:
        return obs[..., 27:29] - obs[..., 15:17]
    if 8 <= d < 30:
        return obs[..., 6:8]
    return np.zeros((*obs.shape[:-1], 2), dtype=np.float32)


def sum_info_metric(infos: dict, metric_name: str) -> float:
    total = 0.0
    vals = infos.get(metric_name)
    if vals is not None:
        try:
            total += float(np.asarray(vals, dtype=np.float32).sum())
        except Exception:
            pass
    for fi in (infos.get("final_info") or []):
        if isinstance(fi, dict):
            v = fi.get(metric_name)
            if v is not None:
                try:
                    total += float(v)
                except (TypeError, ValueError):
                    pass
    return total


def sum_info_bool_metric(infos: dict, metric_name: str) -> float:
    vals = infos.get(metric_name)
    if vals is None:
        return 0.0
    try:
        return float(np.asarray(vals, dtype=np.bool_).sum())
    except Exception:
        return 0.0


def primitive_exploration_chance_for_step(args, step: int) -> float:
    if args.exploration_primitive_chance_pre_learning_starts is not None and step < args.learning_starts:
        return float(args.exploration_primitive_chance_pre_learning_starts)
    return linear_anneal(
        args.exploration_primitive_chance_start,
        args.exploration_primitive_chance,
        step,
        args.exploration_primitive_chance_anneal_steps,
    )


def sample_batch(
    success_rb,
    failure_rb,
    batch_size: int,
    success_fraction: float,
    per_enabled: bool,
    per_beta: float,
    per_fraction: float,
) -> Tuple[Dict[str, np.ndarray], List[Tuple], int, int]:
    """Sample from success/failure buffers, return combined numpy batch + PER metadata."""
    success_available = len(success_rb) > 0
    failure_available = len(failure_rb) > 0
    if not success_available and not failure_available:
        return {}, [], 0, 0

    if not success_available:
        success_count, failure_count = 0, batch_size
    elif not failure_available:
        success_count, failure_count = batch_size, 0
    else:
        success_count = min(max(int(round(batch_size * success_fraction)), 0), batch_size)
        failure_count = batch_size - success_count

    chunks: List[Dict[str, np.ndarray]] = []
    per_updates: List[Tuple] = []
    per_count = 0
    offset = 0

    for buf, count in ((success_rb, success_count), (failure_rb, failure_count)):
        if count <= 0:
            continue
        if per_enabled:
            p_count = min(max(int(round(count * per_fraction)), 0), count)
            u_count = count - p_count
            buf_chunks = []
            if p_count > 0:
                p_data = buf.sample(p_count, beta=per_beta)
                buf_chunks.append(p_data)
                per_updates.append((buf, p_data["indices"], offset, offset + p_count))
                per_count += p_count
            if u_count > 0:
                buf_chunks.append(buf.sample_uniform(u_count))
            chunk = _concat_chunks(buf_chunks)
        else:
            chunk = buf.sample(count)
        chunks.append(chunk)
        offset += count

    data = chunks[0] if len(chunks) == 1 else _concat_chunks(chunks)
    uniform_count = batch_size - per_count
    return data, per_updates, per_count, uniform_count


def _concat_chunks(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ("observations", "next_observations", "actions", "prev_actions",
            "task_rewards", "motion_rewards", "dones", "weights", "sampled_priorities")
    return {k: np.concatenate([c[k] for c in chunks], axis=0) for k in keys if k in chunks[0]}


def to_jax(batch: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    return {k: jnp.array(v) for k, v in batch.items() if k not in ("indices",)}


# ------------------------------------------------------------------ #
# Args                                                                 #
# ------------------------------------------------------------------ #

@dataclass
class Args:
    eval_mode: bool = False
    total_timesteps: int = 1_000_000
    num_envs: int = 1
    learning_starts: int = 5000
    buffer_size: int = int(1e6)

    # TD3 algorithm (mirrors TD3Config)
    task_gamma: float = 0.975
    motion_gamma: float = 0.8
    tau: float = 0.005
    batch_size: int = 256
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    h_transform_eps: float = 1e-3
    q_updates: int = 1
    actor_updates_per_iteration: int = 1
    target_network_frequency: int = 1
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 1.0
    per_enabled: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 200_000
    per_eps: float = 1e-6
    critic_per_fraction: float = 0.7
    critic_uniform_fraction: float = 0.3
    success_buffer_size: int = int(2e5)
    failure_buffer_size: int = int(8e5)
    success_top_fraction: float = 0.2
    recent_episode_window_size: int = 500
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7
    action_scale: float = 0.02
    use_last_action_in_policy_state: bool = False
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2

    # Motion reward shaping
    estop_motion_reward_penalty: float = -5.0
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

    # Primitive exploration
    exploration_primitive_chance: float = 0.05
    exploration_primitive_chance_start: float = 0.5
    exploration_primitive_chance_pre_learning_starts: float | None = None
    exploration_pre_learning_action_source: Literal["random", "policy"] = "random"
    exploration_primitive_chance_anneal_steps: int = 50_000
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

    # Checkpointing
    checkpoint_interval: int = 25_000

    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str | None = None
    model_path: str | None = None
    log_parent_dir: str | None = None
    run_name: str = "default"

    # Runtime
    device: str = "cuda:0"
    seed: int = 0


def make_env(env_id, config):
    def _thunk():
        config["air_hockey"]["seed"] = random.randint(0, int(1e8))
        return AirHockeyEnv(config["air_hockey"])
    return _thunk


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file) as f:
            file_args = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    log_parent_dir = args.log_parent_dir or f"runs/default_training/{task_name}/{args.run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        base = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base}r{i}"
            i += 1
    os.makedirs(log_parent_dir, exist_ok=True)
    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # JAX device
    jax_device = jax.devices("gpu")[0] if "cuda" in args.device else jax.devices("cpu")[0]

    # Environment
    envs = gym.vector.AsyncVectorEnv([make_env(i, config) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    action_scale = 1.0 if config["air_hockey"].get("use_pid") else args.action_scale

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    action_low  = jnp.array(envs.single_action_space.low,  dtype=jnp.float32)
    action_high = jnp.array(envs.single_action_space.high, dtype=jnp.float32)

    # TD3 algorithm
    td3_config = TD3Config(
        task_gamma=args.task_gamma,
        motion_gamma=args.motion_gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        q_weight_decay=args.q_weight_decay,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        exploration_noise=args.exploration_noise,
        h_transform_eps=args.h_transform_eps,
        q_updates=args.q_updates,
        actor_updates_per_iteration=args.actor_updates_per_iteration,
        target_network_frequency=args.target_network_frequency,
        task_reward_weight=args.task_reward_weight,
        motion_reward_weight=args.motion_reward_weight,
        per_enabled=args.per_enabled,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_beta_anneal_steps=args.per_beta_anneal_steps,
        per_eps=args.per_eps,
        critic_per_fraction=args.critic_per_fraction,
        critic_uniform_fraction=args.critic_uniform_fraction,
        success_buffer_size=args.success_buffer_size,
        failure_buffer_size=args.failure_buffer_size,
        success_top_fraction=args.success_top_fraction,
        recent_episode_window_size=args.recent_episode_window_size,
        critic_success_sample_fraction=args.critic_success_sample_fraction,
        critic_failure_sample_fraction=args.critic_failure_sample_fraction,
        actor_hidden_dim=args.agent_hidden_layer_size,
        actor_num_blocks=args.agent_num_hidden_layers,
        q_hidden_dim=args.q_hidden_layer_size,
        q_num_blocks=args.q_num_hidden_layers,
        action_scale=action_scale,
        use_last_action_in_policy_state=args.use_last_action_in_policy_state,
    )
    algo = TD3(td3_config, action_low, action_high)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    train_state = algo.init_train_state(init_key, obs_dim, act_dim)

    # Replay buffers
    buf_cls = PrioritizedReplayBuffer if args.per_enabled else ReplayBuffer
    buf_kwargs = dict(
        obs_shape=envs.single_observation_space.shape,
        action_shape=envs.single_action_space.shape,
    )
    if args.per_enabled:
        buf_kwargs.update(alpha=args.per_alpha, priority_eps=args.per_eps)
    success_rb = buf_cls(buffer_size=args.success_buffer_size, **buf_kwargs)
    failure_rb = buf_cls(buffer_size=args.failure_buffer_size, **buf_kwargs)

    # Primitive exploration selector
    primitive_selector = PrimitiveExplorationSelector(
        num_envs=args.num_envs,
        chance=primitive_exploration_chance_for_step(args, 0),
        takeover_steps=args.exploration_primitive_steps,
        device="cpu",
        dtype=__import__("torch").float32,
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
        target_position_directional_min_angle_deg=args.exploration_target_position_directional_min_angle_deg,
        target_position_directional_max_angle_deg=args.exploration_target_position_directional_max_angle_deg,
        target_position_directional_min_magnitude=args.exploration_target_position_directional_min_magnitude,
        target_position_directional_max_magnitude=args.exploration_target_position_directional_max_magnitude,
        target_takeover_steps=args.exploration_target_position_steps,
        pre_contact_hit_variant_chance=args.exploration_pre_contact_hit_variant_chance,
        pre_contact_hit_variant_steps=args.exploration_pre_contact_hit_variant_steps,
        pre_contact_hit_variant_distance_threshold=args.exploration_pre_contact_hit_variant_distance_threshold,
        pre_contact_hit_variant_scale_min=args.exploration_pre_contact_hit_variant_scale_min,
        pre_contact_hit_variant_scale_max=args.exploration_pre_contact_hit_variant_scale_max,
        pre_contact_hit_variant_min_upward_displacement_x=args.exploration_pre_contact_hit_variant_min_upward_displacement_x,
    )
    primitive_selector.set_primitive_weights(
        stand_still=args.exploration_primitive_weight_stand_still,
        same_direction=args.exploration_primitive_weight_same_direction,
        y_aligned=args.exploration_primitive_weight_y_aligned,
        policy_takeover=args.exploration_primitive_weight_policy_takeover if args.exploration_policy_takeover_enabled else 0.0,
        target_position_directional=args.exploration_primitive_weight_target_position_directional,
    )
    if args.eval_mode:
        primitive_selector.chance = 0.0
        primitive_selector.pre_contact_hit_variant_chance = 0.0

    # Training state
    import torch
    obs, _ = envs.reset(seed=args.seed)
    last_action_np = np.zeros((args.num_envs, act_dim), dtype=np.float32)
    last_action_torch = torch.zeros((args.num_envs, act_dim))
    prev_protective_stop_flags = np.zeros(args.num_envs, dtype=bool)
    episode_buffer = EpisodeBuffer.empty()
    recent_episode_returns: deque = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0

    temporal_horizon = args.temporal_alignment_horizon
    temporal_paddle_history = np.zeros((args.num_envs, temporal_horizon + 1, 2), dtype=np.float32)
    temporal_puck_history   = np.zeros((args.num_envs, temporal_horizon + 1, 2), dtype=np.float32)
    steps_since_done = np.zeros(args.num_envs, dtype=np.int32)
    temporal_paddle_history[:, -1, :] = extract_current_paddle_position(obs)
    temporal_puck_history[:, -1, :]   = extract_current_puck_position(obs)
    previous_puck_pos = extract_current_puck_position(obs).copy()

    current_velocity_mag     = np.zeros(args.num_envs, dtype=np.float32)
    current_acceleration_mag = np.zeros(args.num_envs, dtype=np.float32)
    current_jerk_mag         = np.zeros(args.num_envs, dtype=np.float32)

    rolling_step_stats_window:    deque = deque()
    rolling_episode_stats_window: deque = deque()
    velocity_magnitudes:    list = []
    acceleration_magnitudes:list = []
    jerk_magnitudes:        list = []
    interval_paddle_puck_collisions = 0.0
    interval_env_steps = 0
    interval_primitive_env_steps = 0
    interval_primitive_horizontal_env_steps = 0
    interval_policy_takeover_env_steps = 0
    interval_target_position_directional_env_steps = 0

    global_step = 0
    start_time = time.time()
    train_metrics: Dict[str, float] = {}

    # ---------------------------------------------------------------- #
    # Training loop                                                      #
    # ---------------------------------------------------------------- #
    while global_step < args.total_timesteps:
        should_log = global_step > 0 and np.random.rand() < 0.1
        should_refresh_annealing = (not args.eval_mode) and (np.random.rand() < 0.1)

        if should_refresh_annealing:
            annealing_active = global_step < args.exploration_primitive_chance_anneal_steps
            primitive_selector.chance = primitive_exploration_chance_for_step(args, global_step)
            if annealing_active:
                primitive_selector.set_primitive_weights(
                    stand_still=args.exploration_primitive_weight_anneal_stand_still,
                    same_direction=args.exploration_primitive_weight_anneal_same_direction,
                    y_aligned=args.exploration_primitive_weight_anneal_y_aligned,
                    policy_takeover=args.exploration_primitive_weight_anneal_policy_takeover if args.exploration_policy_takeover_enabled else 0.0,
                    target_position_directional=args.exploration_primitive_weight_anneal_target_position_directional,
                )
            else:
                primitive_selector.set_primitive_weights(
                    stand_still=args.exploration_primitive_weight_stand_still,
                    same_direction=args.exploration_primitive_weight_same_direction,
                    y_aligned=args.exploration_primitive_weight_y_aligned,
                    policy_takeover=args.exploration_primitive_weight_policy_takeover if args.exploration_policy_takeover_enabled else 0.0,
                    target_position_directional=args.exploration_primitive_weight_target_position_directional,
                )

        prev_action_np = last_action_np.copy()
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        if args.use_last_action_in_policy_state:
            policy_obs_jax = jnp.concatenate([obs_jax, jnp.array(last_action_np)], axis=-1)
        else:
            policy_obs_jax = obs_jax

        # Action selection
        if global_step < args.learning_starts and not args.eval_mode:
            if args.exploration_pre_learning_action_source == "random":
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                key, noise_key = jax.random.split(key)
                actions = np.array(algo.select_action(train_state.actor.params, policy_obs_jax))
                actions = np.clip(actions + np.random.randn(*actions.shape) * args.exploration_noise,
                                  envs.single_action_space.low, envs.single_action_space.high)
        else:
            actions = np.array(algo.select_action(train_state.actor.params, policy_obs_jax))
            if not args.eval_mode:
                actions = np.clip(actions + np.random.randn(*actions.shape) * args.exploration_noise,
                                  envs.single_action_space.low, envs.single_action_space.high)
            else:
                actions = np.clip(actions, envs.single_action_space.low, envs.single_action_space.high)

        # Primitive exploration (PyTorch selector unchanged)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        if not args.eval_mode:
            paddle_pos = torch.as_tensor(extract_current_paddle_position(obs))
            puck_pos   = torch.as_tensor(extract_current_puck_position(obs))
            puck_vel   = torch.as_tensor(extract_current_puck_velocity(obs))
            if torch.all(puck_vel == 0):
                puck_vel = puck_pos - torch.as_tensor(previous_puck_pos)
            y_sign = torch.sign(puck_pos[:, 1] - paddle_pos[:, 1])
            action_low_t  = torch.as_tensor(envs.single_action_space.low)
            action_high_t = torch.as_tensor(envs.single_action_space.high)
            action_tensor, primitive_step_stats = primitive_selector.apply(
                action_tensor,
                action_low=action_low_t,
                action_high=action_high_t,
                y_alignment_sign=y_sign,
                current_paddle_position=paddle_pos,
                current_puck_position=puck_pos,
                current_puck_velocity=puck_vel,
                return_stats=True,
            )
        else:
            primitive_step_stats = {
                "primitive_applied_count": 0,
                "primitive_horizontal_dominant_count": 0,
                "policy_takeover_applied_count": 0,
                "target_position_directional_applied_count": 0,
            }

        actions = action_tensor.cpu().numpy()

        # Env step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Protective stop tracking
        current_protective_stop_flags = np.atleast_1d(
            np.asarray(infos.get("protective_stop", np.zeros(args.num_envs, bool)), dtype=bool)
        )
        if current_protective_stop_flags.size == args.num_envs:
            estop_event_mask = current_protective_stop_flags & ~prev_protective_stop_flags
            step_estop_events = float(estop_event_mask.sum())
            prev_protective_stop_flags = current_protective_stop_flags.copy()
            prev_protective_stop_flags[dones] = False
        else:
            estop_event_mask = np.zeros(args.num_envs, dtype=bool)
            step_estop_events = sum_info_bool_metric(infos, "protective_stop")
            prev_protective_stop_flags = np.zeros(args.num_envs, dtype=bool)

        step_puck_hits = sum_info_metric(infos, "paddle_puck_collision_count")
        interval_paddle_puck_collisions += step_puck_hits
        interval_env_steps += args.num_envs
        interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
        interval_primitive_horizontal_env_steps += int(primitive_step_stats["primitive_horizontal_dominant_count"])
        interval_policy_takeover_env_steps += int(primitive_step_stats["policy_takeover_applied_count"])
        interval_target_position_directional_env_steps += int(primitive_step_stats["target_position_directional_applied_count"])

        done_mask = dones.astype(bool)
        last_action_np = actions.copy()
        last_action_np[done_mask] = 0.0
        last_action_torch = torch.as_tensor(last_action_np)
        primitive_selector.reset(torch.as_tensor(dones, dtype=torch.bool))

        current_paddle_pos = extract_current_paddle_position(next_obs)
        current_puck_pos   = extract_current_puck_position(next_obs)
        previous_puck_pos  = current_puck_pos.copy()

        # Motion rewards (numpy)
        current_velocity_mag_t, current_acceleration_mag_t, current_jerk_mag_t = \
            parse_motion_magnitudes_from_infos(
                infos=infos,
                num_envs=args.num_envs,
                device="cpu",
                fallback_velocity_mag=torch.as_tensor(current_velocity_mag),
                fallback_acceleration_mag=torch.as_tensor(current_acceleration_mag),
                fallback_jerk_mag=torch.as_tensor(current_jerk_mag),
            )
        current_velocity_mag     = current_velocity_mag_t.numpy()
        current_acceleration_mag = current_acceleration_mag_t.numpy()
        current_jerk_mag         = current_jerk_mag_t.numpy()

        temporal_paddle_history = np.roll(temporal_paddle_history, -1, axis=1)
        temporal_paddle_history[:, -1, :] = current_paddle_pos
        temporal_puck_history   = np.roll(temporal_puck_history,   -1, axis=1)
        temporal_puck_history[:, -1, :]   = current_puck_pos
        steps_since_done = np.where(dones, 0, steps_since_done + 1)

        realized_movement = temporal_paddle_history[:, -1, :] - temporal_paddle_history[:, 0, :]
        movement_norm     = np.linalg.norm(realized_movement, axis=-1)
        eps = 1e-8
        temporal_valid = steps_since_done >= temporal_horizon

        stand_still_r = ((movement_norm <= args.stand_still_threshold) & temporal_valid).astype(np.float32)

        target_dir   = temporal_puck_history[:, 0, :] - temporal_paddle_history[:, 0, :]
        mv_safe      = np.maximum(movement_norm, eps)
        td_safe      = np.maximum(np.linalg.norm(target_dir, axis=-1), eps)
        cosine       = (realized_movement * target_dir).sum(-1) / (mv_safe * td_safe)
        temporal_align_r = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0) * temporal_valid
        temporal_align_r = np.where(stand_still_r > 0.5, 1.0, temporal_align_r)

        movement_unit   = realized_movement / mv_safe[:, None]
        max_axis_cosine = np.maximum(np.abs(movement_unit[:, 0]), np.abs(movement_unit[:, 1]))
        min_axis_cosine = float(1.0 / np.sqrt(2.0))
        axis_align_r    = np.clip((max_axis_cosine - min_axis_cosine) / (1.0 - min_axis_cosine + eps), 0.0, 1.0) * temporal_valid
        axis_align_r    = np.where(stand_still_r > 0.5, 1.0, axis_align_r)

        vel_r  = np.clip(1.0 - (current_velocity_mag - args.velocity_at_one) / max(args.velocity_at_zero - args.velocity_at_one, 1e-6), a_max=1.0)
        jerk_r = 1.0 - (current_jerk_mag - args.jerk_at_one) / max(args.jerk_at_zero - args.jerk_at_one, 1e-6)

        motion_rewards = (
            args.stand_still_reward_weight         * stand_still_r
            + args.temporal_alignment_reward_weight  * temporal_align_r
            + args.axis_alignment_reward_weight      * axis_align_r
            + args.velocity_reward_weight            * vel_r
            + args.jerk_reward_weight                * jerk_r
        )
        if np.any(estop_event_mask):
            motion_rewards += args.estop_motion_reward_penalty * estop_event_mask.astype(np.float32)

        if dones.any():
            temporal_paddle_history[done_mask] = 0.0
            temporal_paddle_history[done_mask, -1, :] = current_paddle_pos[done_mask]
            temporal_puck_history[done_mask] = 0.0
            temporal_puck_history[done_mask, -1, :] = current_puck_pos[done_mask]

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode_return" in info:
                    writer.add_scalar("charts/episodic_return", info["episode_return"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode_length"], global_step)
                    rolling_episode_stats_window.append((
                        int(global_step + args.num_envs),
                        float(info["episode_return"]),
                        float(info["episode_length"]),
                        1.0 if info.get("success") else 0.0,
                    ))
                    if "motion_data" in info:
                        velocity_magnitudes.extend(info["motion_data"]["velocity_mags"])
                        acceleration_magnitudes.extend(info["motion_data"]["acceleration_mags"])
                        jerk_magnitudes.extend(info["motion_data"]["jerk_mags"])

        rolling_step_stats_window.append((
            int(global_step + args.num_envs), args.num_envs,
            float(step_puck_hits), float(step_estop_events),
        ))
        cutoff = int(global_step + args.num_envs - ROLLING_STATS_WINDOW_STEPS)
        while rolling_step_stats_window    and rolling_step_stats_window[0][0]    <= cutoff:
            rolling_step_stats_window.popleft()
        while rolling_episode_stats_window and rolling_episode_stats_window[0][0] <= cutoff:
            rolling_episode_stats_window.popleft()

        # Store transition
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        if not args.eval_mode:
            episode_buffer.append_step(
                obs=obs[0],
                next_obs=real_next_obs[0],
                action=actions[0],
                prev_action=prev_action_np[0],
                task_reward=float(rewards[0]),
                motion_reward=float(motion_rewards[0]),
                done=float(terminations[0]),
            )
            episode_return_success_threshold = finalize_episode_if_done(
                episode_done=bool(dones[0]),
                episode_buffer=episode_buffer,
                recent_episode_returns=recent_episode_returns,
                success_top_fraction=args.success_top_fraction,
                episode_return_success_threshold=episode_return_success_threshold,
                success_rb=success_rb,
                failure_rb=failure_rb,
            )

        obs = next_obs
        episode_finished = bool(dones[0])
        global_step += args.num_envs

        # ------------------------------------------------------------ #
        # Updates                                                        #
        # ------------------------------------------------------------ #
        if global_step > args.learning_starts and episode_finished and not args.eval_mode:
            per_beta = linear_anneal(args.per_beta_start, args.per_beta_end, global_step, args.per_beta_anneal_steps)

            for q_idx in range(args.q_updates):
                batch_np, per_updates, per_count, uniform_count = sample_batch(
                    success_rb, failure_rb,
                    batch_size=args.batch_size,
                    success_fraction=args.critic_success_sample_fraction,
                    per_enabled=args.per_enabled,
                    per_beta=per_beta,
                    per_fraction=args.critic_per_fraction,
                )
                if not batch_np:
                    continue
                key, update_key = jax.random.split(key)
                batch_jax = to_jax(batch_np)
                train_state, critic_metrics, td_error = algo.update_critic(train_state, batch_jax, update_key)

                if args.per_enabled and per_count > 0:
                    td_error_np = np.array(td_error) + args.per_eps
                    for buf, indices, start, end in per_updates:
                        buf.update_priorities(indices, td_error_np[start:end])

                if should_log and q_idx == args.q_updates - 1:
                    train_metrics.update({k: float(v) for k, v in critic_metrics.items()})
                    train_metrics.update({
                        "replay/per_beta": per_beta,
                        "replay/per_is_weight_mean": float(batch_np.get("weights", np.ones(1)).mean()),
                        "replay/per_sampled_priority_mean": float(batch_np.get("sampled_priorities", np.zeros(1)).mean()),
                        "replay/per_priority_td_error_mean": float(np.array(td_error).mean()) if per_count > 0 else 0.0,
                        "replay/critic_per_sample_count": float(per_count),
                        "replay/critic_uniform_sample_count": float(uniform_count),
                        "replay/success_buffer_size": float(len(success_rb)),
                        "replay/failure_buffer_size": float(len(failure_rb)),
                        "replay/episode_return_success_threshold": episode_return_success_threshold,
                        "replay/recent_episode_window_count": float(len(recent_episode_returns)),
                    })

            for a_idx in range(args.actor_updates_per_iteration):
                batch_np, _, _, _ = sample_batch(
                    success_rb, failure_rb,
                    batch_size=args.batch_size,
                    success_fraction=args.critic_success_sample_fraction,
                    per_enabled=args.per_enabled,
                    per_beta=per_beta,
                    per_fraction=args.critic_per_fraction,
                )
                if not batch_np:
                    continue
                key, update_key = jax.random.split(key)
                batch_jax = to_jax(batch_np)
                train_state, actor_metrics = algo.update_actor(train_state, batch_jax, update_key)
                if should_log and a_idx == args.actor_updates_per_iteration - 1:
                    train_metrics.update({k: float(v) for k, v in actor_metrics.items()})

            if should_log:
                for name, val in train_metrics.items():
                    writer.add_scalar(name, val, global_step)
                writer.add_scalar("charts/exploration_primitive_chance", primitive_selector.chance, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # ------------------------------------------------------------ #
        # Periodic logging                                               #
        # ------------------------------------------------------------ #
        if global_step > 0 and global_step % 500 == 0:
            if rolling_episode_stats_window:
                returns  = [x[1] for x in rolling_episode_stats_window]
                lengths  = [x[2] for x in rolling_episode_stats_window]
                successes= [x[3] for x in rolling_episode_stats_window]
                print(
                    f"Step {global_step}: Avg Return {np.mean(returns):.2f}, "
                    f"Min {np.min(returns):.2f}, Max {np.max(returns):.2f}, "
                    f"Success {np.mean(successes):.2f}, Avg Len {np.mean(lengths):.2f}"
                )
                writer.add_scalar("charts/avg_episodic_return", np.mean(returns), global_step)
                writer.add_scalar("charts/min_episodic_return", np.min(returns),  global_step)
                writer.add_scalar("charts/max_episodic_return", np.max(returns),  global_step)
                writer.add_scalar("charts/avg_success_rate",    np.mean(successes), global_step)
                writer.add_scalar("charts/rolling2k_avg_episode_return", np.mean(returns), global_step)
                writer.add_scalar("charts/rolling2k_avg_episode_length", np.mean(lengths), global_step)
                writer.add_scalar("charts/rolling2k_episode_count", len(rolling_episode_stats_window), global_step)
            else:
                print(f"Step {global_step}: No episodes in rolling window")

            rolling_env_steps   = sum(x[1] for x in rolling_step_stats_window)
            rolling_puck_hits   = sum(x[2] for x in rolling_step_stats_window)
            rolling_estop_events= sum(x[3] for x in rolling_step_stats_window)
            puck_hit_rate  = rolling_puck_hits   / max(rolling_env_steps, 1)
            estop_rate     = rolling_estop_events/ max(rolling_env_steps, 1)
            print(
                f"Step {global_step}: Puck Hits {int(rolling_puck_hits)}, "
                f"E-Stops {int(rolling_estop_events)}, "
                f"Hit/step {puck_hit_rate:.4f}, E-Stop/step {estop_rate:.4f}"
            )
            writer.add_scalar("charts/rolling2k_puck_hits_per_env_step", puck_hit_rate,   global_step)
            writer.add_scalar("charts/rolling2k_estop_rate",             estop_rate,      global_step)
            writer.add_scalar("charts/rolling2k_puck_hits_total",        rolling_puck_hits, global_step)
            writer.add_scalar("charts/rolling2k_estop_events_total",     rolling_estop_events, global_step)

            if velocity_magnitudes:
                writer.add_scalar("motion/avg_velocity_magnitude",     np.mean(velocity_magnitudes),     global_step)
                writer.add_scalar("motion/avg_acceleration_magnitude", np.mean(acceleration_magnitudes), global_step)
                writer.add_scalar("motion/avg_jerk_magnitude",         np.mean(jerk_magnitudes),         global_step)
                velocity_magnitudes.clear()
                acceleration_magnitudes.clear()
                jerk_magnitudes.clear()

            collision_rate = interval_paddle_puck_collisions / max(interval_env_steps, 1)
            writer.add_scalar("contacts/interval_paddle_puck_collisions_per_env_step", collision_rate, global_step)
            interval_paddle_puck_collisions = 0.0
            interval_env_steps = 0
            interval_primitive_env_steps = 0
            interval_primitive_horizontal_env_steps = 0
            interval_policy_takeover_env_steps = 0
            interval_target_position_directional_env_steps = 0

        # ------------------------------------------------------------ #
        # Checkpointing                                                  #
        # ------------------------------------------------------------ #
        if not args.eval_mode and global_step % args.checkpoint_interval == 0 and global_step > 0:
            ckpt_path = os.path.join(log_parent_dir, f"checkpoint_{global_step}.npz")
            np.savez(
                ckpt_path,
                actor_params=jax.tree_util.tree_leaves(train_state.actor.params),
                qf1_params=jax.tree_util.tree_leaves(train_state.qf1.params),
                qf2_params=jax.tree_util.tree_leaves(train_state.qf2.params),
                global_step=global_step,
            )
            print(f"Checkpoint saved to {ckpt_path}")

    envs.close()
    writer.close()
