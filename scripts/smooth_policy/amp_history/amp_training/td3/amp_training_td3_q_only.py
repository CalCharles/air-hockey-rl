"""
TD3-style Q-only training with a fixed deterministic policy and exploration noise.

This keeps the same reward decomposition/returns procedure as amp_training_td3.py,
but removes actor optimization so only critic TD updates are performed.
"""

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.smooth_policy.amp_history.amp_training.td3.dual_head_q import (
    TD3DualHeadQNetwork,
)
from scripts.smooth_policy.amp_history.amp_training.td3.exploration_selector import (
    PrimitiveExplorationSelector,
)
from scripts.smooth_policy.amp_history.amp_training.td3.replay_buffer import TD3ReplayBuffer
from scripts.smooth_policy.amp_history.amp_training.amp_training_lsgan import (
    parse_motion_magnitudes_from_infos,
)
from scripts.utils import save_tensorboard_plots


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().cpu()


def get_rng_states() -> Dict[str, object]:
    states: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return states


def set_rng_states(states: Dict[str, object]) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


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
    return actor.get_action(policy_obs)


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


def log_scalar_metrics(writer: SummaryWriter, metrics: Dict[str, float], global_step: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(name, value, global_step)


def module_param_l2_norm(module: torch.nn.Module) -> float:
    total_sq = 0.0
    for param in module.parameters():
        param_l2 = param.detach().norm(2).item()
        total_sq += param_l2 * param_l2
    return float(np.sqrt(total_sq))


def module_grad_l2_norm(module: torch.nn.Module) -> float:
    total_sq = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_l2 = param.grad.detach().norm(2).item()
        total_sq += grad_l2 * grad_l2
    return float(np.sqrt(total_sq))


def initialize_train_metrics() -> Dict[str, float]:
    return {
        "losses/q_task_loss": 0.0,
        "losses/q_motion_loss": 0.0,
        "losses/q_total_loss": 0.0,
        "losses/q1_task_mean": 0.0,
        "losses/q2_task_mean": 0.0,
        "losses/q1_motion_mean": 0.0,
        "losses/q2_motion_mean": 0.0,
        "losses/q1_total_mean": 0.0,
        "losses/q2_total_mean": 0.0,
        "losses/q_task_td_abs_error_mean": 0.0,
        "losses/q_motion_td_abs_error_mean": 0.0,
        "debug/bellman_target_task_original_mean": 0.0,
        "debug/bellman_target_task_original_std": 0.0,
        "debug/bellman_target_motion_original_mean": 0.0,
        "debug/bellman_target_motion_original_std": 0.0,
        "debug/next_q_task_h_mean": 0.0,
        "debug/next_q_task_h_std": 0.0,
        "debug/next_q_motion_h_mean": 0.0,
        "debug/next_q_motion_h_std": 0.0,
        "debug/target_next_action_abs_mean": 0.0,
        "diagnostics/qf1_param_l2_norm": 0.0,
        "diagnostics/qf2_param_l2_norm": 0.0,
        "diagnostics/qf1_grad_l2_norm": 0.0,
        "diagnostics/qf2_grad_l2_norm": 0.0,
    }


@dataclass
class Args:
    total_timesteps: int = 1000000
    num_envs: int = 1

    # TD3 core
    buffer_size: int = int(1e6)
    task_gamma: float = 0.975
    motion_gamma: float = 0.8
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    # Kept for args-file compatibility; actor optimization is disabled in this script.
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    q_frequency: int = 1
    q_updates: int = 1
    # Kept for args-file compatibility; actor/target policy updates are disabled.
    policy_frequency: int = 2
    target_network_frequency: int = 1
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # Primitive exploration takeover
    exploration_primitive_chance: float = 0.05
    exploration_primitive_steps: int = 3

    # Dual-head reward decomposition
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 1.0

    # Motion reward component weights
    stand_still_reward_weight: float = 0.5
    temporal_alignment_reward_weight: float = 0.5
    axis_alignment_reward_weight: float = 0.5
    velocity_reward_weight: float = 0.5
    jerk_reward_weight: float = 0.5

    # Motion reward component calibration
    stand_still_threshold: float = 0.04
    temporal_alignment_horizon: int = 4
    velocity_at_one: float = 0.3
    velocity_at_zero: float = 0.6
    jerk_at_one: float = 10.0
    jerk_at_zero: float = 23.0

    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str | None = None
    model_path: str | None = None
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

def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env

    return _thunk


if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()

    args = tyro.cli(Args, default=default_args)
    if args.agent_hidden_size is not None:
        args.agent_hidden_layer_size = int(args.agent_hidden_size)
    if args.q_hidden_size is not None:
        args.q_hidden_layer_size = int(args.q_hidden_size)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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

    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading model/checkpoint from {args.model_path}")
        loaded_obj = torch.load(args.model_path, map_location=args.device)
        if isinstance(loaded_obj, dict) and "actor" in loaded_obj and "qf1" in loaded_obj:
            resume_checkpoint = loaded_obj
            actor.load_state_dict(extract_deterministic_state_dict(resume_checkpoint["actor"]), strict=False)
            qf1.load_state_dict(resume_checkpoint["qf1"])
            qf2.load_state_dict(resume_checkpoint["qf2"])
            qf1_target.load_state_dict(resume_checkpoint["qf1_target"])
            qf2_target.load_state_dict(resume_checkpoint["qf2_target"])
            print("Full training checkpoint loaded (network weights).")
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
            print("Actor-only model loaded successfully.")

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.q_weight_decay
    )
    rb = TD3ReplayBuffer(
        buffer_size=args.buffer_size,
        obs_shape=envs.single_observation_space.shape,
        action_shape=envs.single_action_space.shape,
        device=args.device,
        n_envs=args.num_envs,
    )
    print(f"✓ TD3 replay buffer initialized (capacity={args.buffer_size:,})\n")

    obs, _ = envs.reset(seed=args.seed)
    last_action_for_policy = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=args.device)
    episodic_returns = []

    if args.temporal_alignment_horizon <= 0:
        raise ValueError("temporal_alignment_horizon must be > 0 for stand-still and alignment rewards.")

    temporal_horizon = args.temporal_alignment_horizon
    temporal_paddle_history = torch.zeros((args.num_envs, temporal_horizon + 1, 2), device=args.device)
    temporal_puck_history = torch.zeros((args.num_envs, temporal_horizon + 1, 2), device=args.device)
    temporal_done_history = torch.zeros((args.num_envs, temporal_horizon), dtype=torch.bool, device=args.device)
    temporal_position_count = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    initial_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
    temporal_paddle_history[:, -1, :] = extract_current_paddle_position(initial_obs_tensor)
    temporal_puck_history[:, -1, :] = extract_current_puck_position(initial_obs_tensor)
    temporal_position_count[:] = 1

    current_velocity_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_acceleration_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_jerk_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)

    start_time = time.time()
    global_step = 0
    iteration = 0

    train_metrics = initialize_train_metrics()

    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=args.device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=args.device)
    primitive_selector = PrimitiveExplorationSelector(
        num_envs=args.num_envs,
        chance=args.exploration_primitive_chance,
        takeover_steps=args.exploration_primitive_steps,
        device=args.device,
        dtype=torch.float32,
    )
    if resume_checkpoint is not None:
        q_optimizer.load_state_dict(resume_checkpoint["q_optimizer"])
        rb.load_state_dict(resume_checkpoint["replay_buffer"])
        if "primitive_selector" in resume_checkpoint:
            primitive_selector.load_state_dict(resume_checkpoint["primitive_selector"])

        global_step = int(resume_checkpoint["global_step"])
        iteration = int(resume_checkpoint["iteration"])
        obs = np.asarray(resume_checkpoint["obs"], dtype=np.float32)
        last_action_for_policy = resume_checkpoint["last_action_for_policy"].to(args.device)
        temporal_paddle_history = resume_checkpoint["temporal_paddle_history"].to(args.device)
        temporal_puck_history = resume_checkpoint["temporal_puck_history"].to(args.device)
        temporal_done_history = resume_checkpoint["temporal_done_history"].to(args.device)
        temporal_position_count = resume_checkpoint["temporal_position_count"].to(args.device)
        current_velocity_mag = resume_checkpoint["current_velocity_mag"].to(args.device)
        current_acceleration_mag = resume_checkpoint["current_acceleration_mag"].to(args.device)
        current_jerk_mag = resume_checkpoint["current_jerk_mag"].to(args.device)
        train_metrics = dict(resume_checkpoint.get("train_metrics", train_metrics))
        episodic_returns = list(resume_checkpoint.get("episodic_returns", episodic_returns))
        if "rng_states" in resume_checkpoint:
            set_rng_states(resume_checkpoint["rng_states"])
        print(f"Resuming training from global_step={global_step}, iteration={iteration}")

    while global_step < args.total_timesteps:
        prev_action_for_transition = last_action_for_policy.clone()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
        policy_obs_tensor = augment_policy_observation(
            obs_tensor, last_action_for_policy, args.use_last_action_in_policy_state
        )

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                deterministic_actions = deterministic_actor_action(actor, policy_obs_tensor)
                exploration = torch.randn_like(deterministic_actions) * args.exploration_noise
                deterministic_actions = torch.clamp(deterministic_actions + exploration, action_low, action_high)
                actions = deterministic_actions.cpu().numpy()

        action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=args.device)
        current_paddle_pos_for_primitive = extract_current_paddle_position(obs_tensor)
        current_puck_pos_for_primitive = extract_current_puck_position(obs_tensor)
        y_alignment_sign = torch.sign(current_puck_pos_for_primitive[:, 1] - current_paddle_pos_for_primitive[:, 1])
        action_tensor = primitive_selector.apply(
            action_tensor,
            action_low=action_low,
            action_high=action_high,
            y_alignment_sign=y_alignment_sign,
        )
        actions = action_tensor.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=args.device)
        current_paddle_pos = extract_current_paddle_position(next_obs_tensor)
        current_puck_pos = extract_current_puck_position(next_obs_tensor)
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=args.device)
        primitive_selector.reset(done_tensor)
        last_action_for_policy = action_tensor.clone()
        last_action_for_policy[done_tensor] = 0

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
        temporal_done_history = torch.roll(temporal_done_history, shifts=-1, dims=1)
        temporal_done_history[:, -1] = done_tensor
        temporal_position_count = torch.clamp(temporal_position_count + 1, max=temporal_horizon + 1)

        realized_movement = temporal_paddle_history[:, -1, :] - temporal_paddle_history[:, 0, :]
        movement_norm = torch.norm(realized_movement, dim=-1)
        eps = 1e-8

        temporal_valid = (temporal_position_count >= temporal_horizon + 1) & (~temporal_done_history.any(dim=1))
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

        if dones.any():
            temporal_paddle_history[done_tensor] = 0
            temporal_paddle_history[done_tensor, -1, :] = current_paddle_pos[done_tensor]
            temporal_puck_history[done_tensor] = 0
            temporal_puck_history[done_tensor, -1, :] = current_puck_pos[done_tensor]
            temporal_done_history[done_tensor] = False
            temporal_position_count[done_tensor] = 1

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode_return" in info:
                    episodic_returns.append(info["episode_return"])
                    writer.add_scalar("charts/episodic_return", info["episode_return"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode_length"], global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(
            obs=obs,
            next_obs=real_next_obs,
            actions=actions,
            task_rewards=task_rewards,
            motion_rewards=motion_rewards,
            dones=terminations,
            prev_action=prev_action_for_transition.cpu().numpy(),
        )

        obs = next_obs

        if global_step > args.learning_starts:
            if iteration % args.q_frequency == 0:
                for _ in range(args.q_updates):
                    data = rb.sample(args.batch_size)
                    sampled_observations = data["observations"]
                    sampled_next_observations = data["next_observations"]
                    sampled_actions = data["actions"]
                    sampled_task_rewards = data["task_rewards"]
                    sampled_motion_rewards = data["motion_rewards"]
                    sampled_dones = data["dones"]
                    sampled_prev_actions = data["prev_actions"]
                    sampled_next_prev_actions = sampled_actions * (1.0 - sampled_dones.unsqueeze(-1))
                    sampled_next_policy_observations = augment_policy_observation(
                        sampled_next_observations,
                        sampled_next_prev_actions,
                        args.use_last_action_in_policy_state,
                    )

                    with torch.no_grad():
                        target_next_action = deterministic_actor_action(actor, sampled_next_policy_observations)
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

                        train_metrics.update(
                            {
                                "debug/bellman_target_task_original_mean": bellman_target_task_original.mean().item(),
                                "debug/bellman_target_task_original_std": bellman_target_task_original.std().item(),
                                "debug/bellman_target_motion_original_mean": bellman_target_motion_original.mean().item(),
                                "debug/bellman_target_motion_original_std": bellman_target_motion_original.std().item(),
                                "debug/next_q_task_h_mean": next_q_task_value_h.mean().item(),
                                "debug/next_q_task_h_std": next_q_task_value_h.std().item(),
                                "debug/next_q_motion_h_mean": next_q_motion_value_h.mean().item(),
                                "debug/next_q_motion_h_std": next_q_motion_value_h.std().item(),
                                "debug/target_next_action_abs_mean": target_next_action.abs().mean().item(),
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

                    q1_task_loss = F.mse_loss(q1_task_h.view(-1), next_q_task_value_h)
                    q2_task_loss = F.mse_loss(q2_task_h.view(-1), next_q_task_value_h)
                    q1_motion_loss = F.mse_loss(q1_motion_h.view(-1), next_q_motion_value_h)
                    q2_motion_loss = F.mse_loss(q2_motion_h.view(-1), next_q_motion_value_h)
                    q_total_loss = q1_task_loss + q2_task_loss + q1_motion_loss + q2_motion_loss

                    q_optimizer.zero_grad()
                    q_total_loss.backward()
                    qf1_grad_l2_norm = module_grad_l2_norm(qf1)
                    qf2_grad_l2_norm = module_grad_l2_norm(qf2)
                    q_optimizer.step()

                    task_td_abs_error = 0.5 * (
                        (q1_task_h.view(-1) - next_q_task_value_h).abs().mean()
                        + (q2_task_h.view(-1) - next_q_task_value_h).abs().mean()
                    )
                    motion_td_abs_error = 0.5 * (
                        (q1_motion_h.view(-1) - next_q_motion_value_h).abs().mean()
                        + (q2_motion_h.view(-1) - next_q_motion_value_h).abs().mean()
                    )

                    train_metrics.update(
                        {
                            "losses/q_task_loss": (q1_task_loss + q2_task_loss).item() / 2.0,
                            "losses/q_motion_loss": (q1_motion_loss + q2_motion_loss).item() / 2.0,
                            "losses/q_total_loss": q_total_loss.item(),
                            "losses/q1_task_mean": q1_task_h.mean().item(),
                            "losses/q2_task_mean": q2_task_h.mean().item(),
                            "losses/q1_motion_mean": q1_motion_h.mean().item(),
                            "losses/q2_motion_mean": q2_motion_h.mean().item(),
                            "losses/q1_total_mean": (q1_task_h + q1_motion_h).mean().item(),
                            "losses/q2_total_mean": (q2_task_h + q2_motion_h).mean().item(),
                            "losses/q_task_td_abs_error_mean": task_td_abs_error.item(),
                            "losses/q_motion_td_abs_error_mean": motion_td_abs_error.item(),
                            "diagnostics/qf1_param_l2_norm": module_param_l2_norm(qf1),
                            "diagnostics/qf2_param_l2_norm": module_param_l2_norm(qf2),
                            "diagnostics/qf1_grad_l2_norm": qf1_grad_l2_norm,
                            "diagnostics/qf2_grad_l2_norm": qf2_grad_l2_norm,
                        }
                    )
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                log_scalar_metrics(writer, train_metrics, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step > 0 and global_step % 500 == 0:
            if episodic_returns:
                avg_return = np.mean(episodic_returns)
                min_return = np.min(episodic_returns)
                max_return = np.max(episodic_returns)
                print(
                    f"Step {global_step}: Avg Return: {avg_return:.2f}, "
                    f"Min: {min_return:.2f}, Max: {max_return:.2f}, "
                    f"Episodes: {len(episodic_returns)}"
                )
                writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
                writer.add_scalar("charts/min_episodic_return", min_return, global_step)
                writer.add_scalar("charts/max_episodic_return", max_return, global_step)
                episodic_returns.clear()
            else:
                print(f"Step {global_step}: No episodes completed in last 500 steps")

        if global_step > 0 and global_step % 10000 == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(actor.state_dict(), model_path)
            torch.save(qf1.state_dict(), f"{checkpoint_dir}/qf1.pth")
            torch.save(qf2.state_dict(), f"{checkpoint_dir}/qf2.pth")
            torch.save(qf1_target.state_dict(), f"{checkpoint_dir}/qf1_target.pth")
            torch.save(qf2_target.state_dict(), f"{checkpoint_dir}/qf2_target.pth")
            training_state = {
                "checkpoint_version": 1,
                "global_step": global_step,
                "iteration": iteration,
                "actor": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "qf1_target": qf1_target.state_dict(),
                "qf2_target": qf2_target.state_dict(),
                "q_optimizer": q_optimizer.state_dict(),
                "replay_buffer": rb.state_dict(),
                "primitive_selector": primitive_selector.state_dict(),
                "obs": np.array(obs, copy=True),
                "last_action_for_policy": _cpu_tensor(last_action_for_policy),
                "temporal_paddle_history": _cpu_tensor(temporal_paddle_history),
                "temporal_puck_history": _cpu_tensor(temporal_puck_history),
                "temporal_done_history": _cpu_tensor(temporal_done_history),
                "temporal_position_count": _cpu_tensor(temporal_position_count),
                "current_velocity_mag": _cpu_tensor(current_velocity_mag),
                "current_acceleration_mag": _cpu_tensor(current_acceleration_mag),
                "current_jerk_mag": _cpu_tensor(current_jerk_mag),
                "train_metrics": dict(train_metrics),
                "episodic_returns": list(episodic_returns),
                "rng_states": get_rng_states(),
                "args": vars(args),
            }
            torch.save(training_state, f"{checkpoint_dir}/training_state.pth")
            print(f"\nCheckpoint saved at step {global_step}")

        iteration += 1
        global_step += args.num_envs

    envs.close()

    torch.save(actor.state_dict(), f"{log_parent_dir}/model.pth")
    torch.save(qf1.state_dict(), f"{log_parent_dir}/qf1.pth")
    torch.save(qf2.state_dict(), f"{log_parent_dir}/qf2.pth")
    torch.save(qf1_target.state_dict(), f"{log_parent_dir}/qf1_target.pth")
    torch.save(qf2_target.state_dict(), f"{log_parent_dir}/qf2_target.pth")
    final_training_state = {
        "checkpoint_version": 1,
        "global_step": global_step,
        "iteration": iteration,
        "actor": actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "q_optimizer": q_optimizer.state_dict(),
        "replay_buffer": rb.state_dict(),
        "primitive_selector": primitive_selector.state_dict(),
        "obs": np.array(obs, copy=True),
        "last_action_for_policy": _cpu_tensor(last_action_for_policy),
        "temporal_paddle_history": _cpu_tensor(temporal_paddle_history),
        "temporal_puck_history": _cpu_tensor(temporal_puck_history),
        "temporal_done_history": _cpu_tensor(temporal_done_history),
        "temporal_position_count": _cpu_tensor(temporal_position_count),
        "current_velocity_mag": _cpu_tensor(current_velocity_mag),
        "current_acceleration_mag": _cpu_tensor(current_acceleration_mag),
        "current_jerk_mag": _cpu_tensor(current_jerk_mag),
        "train_metrics": dict(train_metrics),
        "episodic_returns": list(episodic_returns),
        "rng_states": get_rng_states(),
        "args": vars(args),
    }
    torch.save(final_training_state, f"{log_parent_dir}/training_state.pth")

    metrics = [
        "charts/episodic_return",
        "losses/q_task_loss",
        "losses/q_motion_loss",
        "losses/q_total_loss",
        "losses/q1_task_mean",
        "losses/q2_task_mean",
        "losses/q1_motion_mean",
        "losses/q2_motion_mean",
        "losses/q1_total_mean",
        "losses/q2_total_mean",
        "losses/q_task_td_abs_error_mean",
        "losses/q_motion_td_abs_error_mean",
        "debug/bellman_target_task_original_mean",
        "debug/bellman_target_task_original_std",
        "debug/bellman_target_motion_original_mean",
        "debug/bellman_target_motion_original_std",
        "debug/next_q_task_h_mean",
        "debug/next_q_task_h_std",
        "debug/next_q_motion_h_mean",
        "debug/next_q_motion_h_std",
        "debug/target_next_action_abs_mean",
        "diagnostics/qf1_param_l2_norm",
        "diagnostics/qf2_param_l2_norm",
        "diagnostics/qf1_grad_l2_norm",
        "diagnostics/qf2_grad_l2_norm",
    ]
    save_tensorboard_plots(log_parent_dir, config, metrics=metrics)
    writer.close()

