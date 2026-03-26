"""
PPO training with TD3-style motion reward decomposition.

Replaces the AMP discriminator approach with the 5-component hand-crafted
motion reward from the TD3 pipeline:
  1. Stand-still reward (low movement = good when near idle)
  2. Temporal alignment (movement toward puck direction)
  3. Axis alignment (prefers horizontal/vertical movement)
  4. Velocity reward (smooth, controlled speed)
  5. Jerk reward (minimize acceleration changes)

Motion rewards are combined into the single PPO reward stream with
configurable weights. The critic learns a single value function over
the combined reward.

Run with:
  python scripts/smooth_policy/amp_history/amp_training/ppo/amp_training_ppo.py \
    --args-file <path-to-yaml>
"""

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.smooth_policy.amp_history.amp_training.amp_training_lsgan import (
    parse_motion_magnitudes_from_infos,
)
from scripts.utils import save_tensorboard_plots


# ---------------------------------------------------------------------------
# Motion reward helpers (ported from TD3 pipeline)
# ---------------------------------------------------------------------------

def velocity_reward_from_magnitude(
    velocity_mag: torch.Tensor,
    velocity_at_one: float,
    velocity_at_zero: float,
) -> torch.Tensor:
    """Linear ramp: reward=1 when vel<=velocity_at_one, reward=0 at velocity_at_zero."""
    denom = max(velocity_at_zero - velocity_at_one, 1e-6)
    reward = 1.0 - (velocity_mag - velocity_at_one) / denom
    return torch.clamp_max(reward, 1.0)


def jerk_reward_from_magnitude(
    jerk_mag: torch.Tensor,
    jerk_at_one: float,
    jerk_at_zero: float,
) -> torch.Tensor:
    """Linear ramp: reward=1 when jerk<=jerk_at_one, reward=0 at jerk_at_zero."""
    denom = max(jerk_at_zero - jerk_at_one, 1e-6)
    return 1.0 - (jerk_mag - jerk_at_one) / denom


def extract_current_paddle_position(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        return observation[..., 12:14]
    return observation[..., 0:2]


def extract_current_puck_position(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        return observation[..., 27:29]
    if obs_dim >= 8:
        return observation[..., 4:6]
    if obs_dim >= 4:
        return observation[..., 2:4]
    raise ValueError(f"Observation dim {obs_dim} too small to extract puck position.")


def augment_policy_observation(observation, last_action, use_last_action):
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def sum_info_metric(infos: dict, metric_name: str) -> float:
    metric_values = infos.get(metric_name)
    if metric_values is None:
        return 0.0
    try:
        return float(np.asarray(metric_values, dtype=np.float32).sum())
    except Exception:
        return 0.0


def compute_motion_rewards(
    paddle_positions_window: torch.Tensor,  # [num_envs, horizon+1, 2]
    puck_positions_window: torch.Tensor,    # [num_envs, horizon+1, 2]
    steps_since_done: torch.Tensor,         # [num_envs]
    temporal_horizon: int,
    velocity_mag: torch.Tensor,             # [num_envs]
    jerk_mag: torch.Tensor,                 # [num_envs]
    stand_still_threshold: float,
    velocity_at_one: float,
    velocity_at_zero: float,
    jerk_at_one: float,
    jerk_at_zero: float,
    stand_still_weight: float,
    temporal_alignment_weight: float,
    axis_alignment_weight: float,
    velocity_weight: float,
    jerk_weight: float,
) -> dict:
    """Compute the 5-component motion reward (same logic as TD3 pipeline)."""
    eps = 1e-8
    realized_movement = paddle_positions_window[:, -1, :] - paddle_positions_window[:, 0, :]
    movement_norm = torch.norm(realized_movement, dim=-1)
    temporal_valid = steps_since_done >= temporal_horizon

    # 1. Stand-still reward
    stand_still_raw = ((movement_norm <= stand_still_threshold) & temporal_valid).float()

    # 2. Temporal alignment reward (move toward puck)
    target_direction = puck_positions_window[:, 0, :] - paddle_positions_window[:, 0, :]
    movement_norm_safe = movement_norm.clamp_min(eps)
    target_norm_safe = torch.norm(target_direction, dim=-1).clamp_min(eps)
    temporal_cosine = (realized_movement * target_direction).sum(dim=-1) / (
        movement_norm_safe * target_norm_safe
    )
    temporal_alignment_raw = torch.clamp((temporal_cosine + 1.0) * 0.5, 0.0, 1.0)
    temporal_alignment_raw = temporal_alignment_raw * temporal_valid.float()
    # If standing still, temporal alignment is perfect
    temporal_alignment_raw = torch.where(
        stand_still_raw > 0.5, torch.ones_like(temporal_alignment_raw), temporal_alignment_raw
    )

    # 3. Axis alignment reward (prefer horizontal/vertical)
    movement_unit = realized_movement / movement_norm_safe.unsqueeze(-1)
    max_axis_cosine = torch.max(torch.abs(movement_unit[:, 0]), torch.abs(movement_unit[:, 1]))
    min_axis_cosine = float(1.0 / np.sqrt(2.0))
    axis_alignment_raw = (max_axis_cosine - min_axis_cosine) / (1.0 - min_axis_cosine + eps)
    axis_alignment_raw = torch.clamp(axis_alignment_raw, 0.0, 1.0) * temporal_valid.float()
    axis_alignment_raw = torch.where(
        stand_still_raw > 0.5, torch.ones_like(axis_alignment_raw), axis_alignment_raw
    )

    # 4. Velocity reward
    velocity_raw = velocity_reward_from_magnitude(velocity_mag, velocity_at_one, velocity_at_zero)

    # 5. Jerk reward
    jerk_raw = jerk_reward_from_magnitude(jerk_mag, jerk_at_one, jerk_at_zero)

    # Weighted components
    stand_still_weighted = stand_still_weight * stand_still_raw
    temporal_alignment_weighted = temporal_alignment_weight * temporal_alignment_raw
    axis_alignment_weighted = axis_alignment_weight * axis_alignment_raw
    velocity_weighted = velocity_weight * velocity_raw
    jerk_weighted = jerk_weight * jerk_raw

    motion_reward = (
        stand_still_weighted
        + temporal_alignment_weighted
        + axis_alignment_weighted
        + velocity_weighted
        + jerk_weighted
    )

    return {
        "motion_reward": motion_reward,
        "stand_still_raw": stand_still_raw,
        "temporal_alignment_raw": temporal_alignment_raw,
        "axis_alignment_raw": axis_alignment_raw,
        "velocity_raw": velocity_raw,
        "jerk_raw": jerk_raw,
        "stand_still_weighted": stand_still_weighted,
        "temporal_alignment_weighted": temporal_alignment_weighted,
        "axis_alignment_weighted": axis_alignment_weighted,
        "velocity_weighted": velocity_weighted,
        "jerk_weighted": jerk_weighted,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # PPO core
    num_envs: int = 8
    num_steps: int = 512
    learning_rate: float = 1e-4
    num_iterations: int = 5000
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    batch_size: int = 0  # computed at runtime
    norm_adv: bool = True

    # Reward weighting
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 0.0

    # Motion reward component weights (same as TD3)
    stand_still_reward_weight: float = 0.5
    temporal_alignment_reward_weight: float = 0.5
    axis_alignment_reward_weight: float = 0.5
    velocity_reward_weight: float = 0.5
    jerk_reward_weight: float = 0.5

    # Motion reward calibration
    stand_still_threshold: float = 0.015
    temporal_alignment_horizon: int = 4
    velocity_at_one: float = 0.3
    velocity_at_zero: float = 0.6
    jerk_at_one: float = 10.0
    jerk_at_zero: float = 23.0

    # Policy state
    use_last_action_in_policy_state: bool = True

    # Paths
    config: str = "scripts/smooth_policy/amp_history/configs/td3/puck_touch_verify/puck_touch_config.yaml"
    args_file: str = None
    model_path: str = None
    log_parent_dir: str = None
    run_name: str = "ppo_motion"

    # Runtime
    seed: int = 0
    device: str = "cuda:0"
    action_scale: float = 1

    # Agent architecture
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 5


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(env_id, config):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        cfg = dict(config)
        cfg["seed"] = curr_seed
        return AirHockeyEnv(cfg)
    return _thunk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)
    args.batch_size = args.num_envs * args.num_steps

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_config = config["air_hockey"]

    # Parallel envs
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, env_config) for i in range(args.num_envs)]
    )
    print("obs_space:", envs.single_observation_space.shape)

    # Logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = env_config.get("task")
    log_parent_dir = args.log_parent_dir or f"runs/ppo_training/{task_name}/{args.run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        base = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base}r{i}"
            i += 1
        print(f"Log directory exists. Using: {log_parent_dir}")
    os.makedirs(log_parent_dir, exist_ok=True)

    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(
            [f"|{k}|{v}|" for k, v in vars(args).items()]
        ),
    )
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    if env_config.get("use_pid", False):
        action_scale = 1
    else:
        action_scale = args.action_scale

    # Build agent
    policy_obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    if args.use_last_action_in_policy_state:
        policy_obs_dim += action_dim

    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32,
        ),
        single_action_space=envs.single_action_space,
    )
    agent = Agent(
        policy_env_view,
        action_scale=action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(args.device)

    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-6)

    use_motion = args.motion_reward_weight > 0.0
    temporal_horizon = args.temporal_alignment_horizon
    if temporal_horizon <= 0:
        raise ValueError("temporal_alignment_horizon must be > 0")

    # ---- Rollout storage ----
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=args.device,
    )
    policy_obs_buf = torch.zeros(
        (args.num_steps, args.num_envs, policy_obs_dim), device=args.device,
    )
    actions_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=args.device,
    )
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)

    # Motion reward per-step storage (for logging all 5 components)
    motion_rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    stand_still_raw_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    temporal_alignment_raw_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    axis_alignment_raw_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    velocity_raw_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    jerk_raw_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    stand_still_weighted_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    temporal_alignment_weighted_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    axis_alignment_weighted_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    velocity_weighted_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    jerk_weighted_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    velocity_mag_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    acceleration_mag_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    jerk_mag_buf = torch.zeros((args.num_steps, args.num_envs), device=args.device)

    # Temporal history for motion reward computation
    paddle_history = torch.zeros(
        (args.num_envs, temporal_horizon + 1, 2), device=args.device,
    )
    puck_history = torch.zeros(
        (args.num_envs, temporal_horizon + 1, 2), device=args.device,
    )
    steps_since_done = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)

    # Per-env motion mag tracking (fallback for info parsing)
    current_velocity_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_acceleration_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_jerk_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)

    # Interval statistics (match TD3 logging pattern)
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []
    interval_paddle_puck_collisions = 0.0
    interval_env_steps = 0

    # Initialize
    global_step = 0
    last_log_step = 0
    last_checkpoint_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=args.device)
    next_done = torch.zeros(args.num_envs, device=args.device)
    last_action_for_policy = torch.zeros((args.num_envs, action_dim), device=args.device)

    # Seed history with initial positions
    paddle_history[:, -1, :] = extract_current_paddle_position(next_obs)
    puck_history[:, -1, :] = extract_current_puck_position(next_obs)

    print(f"\n{'='*80}")
    print(f"Starting PPO training with motion rewards")
    print(f"  task_reward_weight={args.task_reward_weight}")
    print(f"  motion_reward_weight={args.motion_reward_weight}")
    print(f"  motion components: stand_still={args.stand_still_reward_weight}, "
          f"temporal_align={args.temporal_alignment_reward_weight}, "
          f"axis_align={args.axis_alignment_reward_weight}, "
          f"velocity={args.velocity_reward_weight}, jerk={args.jerk_reward_weight}")
    print(f"  num_envs={args.num_envs}, num_steps={args.num_steps}, "
          f"batch_size={args.batch_size}")
    print(f"  log_dir={log_parent_dir}")
    print(f"{'='*80}\n")

    for iteration in range(1, args.num_iterations + 1):
        episodic_returns = []
        success_rates = []

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---- Rollout collection ----
        for step in range(args.num_steps):
            global_step += args.num_envs
            interval_env_steps += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            policy_next_obs = augment_policy_observation(
                next_obs, last_action_for_policy, args.use_last_action_in_policy_state
            )
            policy_obs_buf[step] = policy_next_obs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(policy_next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done_np = np.logical_or(terminations, truncations)
            rewards_buf[step] = torch.tensor(reward, device=args.device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=args.device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=args.device)
            last_action_for_policy = action.detach().clone()
            last_action_for_policy[next_done.bool()] = 0

            # Track contact stats
            interval_paddle_puck_collisions += sum_info_metric(infos, "paddle_puck_collision_count")

            # Parse motion magnitudes from env infos
            current_velocity_mag, current_acceleration_mag, current_jerk_mag = (
                parse_motion_magnitudes_from_infos(
                    infos=infos,
                    num_envs=args.num_envs,
                    device=args.device,
                    fallback_velocity_mag=current_velocity_mag,
                    fallback_acceleration_mag=current_acceleration_mag,
                    fallback_jerk_mag=current_jerk_mag,
                )
            )
            velocity_mag_buf[step] = current_velocity_mag
            acceleration_mag_buf[step] = current_acceleration_mag
            jerk_mag_buf[step] = current_jerk_mag

            # Update temporal position histories
            current_paddle = extract_current_paddle_position(next_obs)
            current_puck = extract_current_puck_position(next_obs)
            paddle_history = torch.roll(paddle_history, shifts=-1, dims=1)
            paddle_history[:, -1, :] = current_paddle
            puck_history = torch.roll(puck_history, shifts=-1, dims=1)
            puck_history[:, -1, :] = current_puck

            done_mask = next_done.bool()
            steps_since_done = torch.where(
                done_mask,
                torch.zeros_like(steps_since_done),
                steps_since_done + 1,
            )

            # Compute per-step motion reward (all 5 components)
            motion_result = compute_motion_rewards(
                paddle_positions_window=paddle_history,
                puck_positions_window=puck_history,
                steps_since_done=steps_since_done,
                temporal_horizon=temporal_horizon,
                velocity_mag=current_velocity_mag,
                jerk_mag=current_jerk_mag,
                stand_still_threshold=args.stand_still_threshold,
                velocity_at_one=args.velocity_at_one,
                velocity_at_zero=args.velocity_at_zero,
                jerk_at_one=args.jerk_at_one,
                jerk_at_zero=args.jerk_at_zero,
                stand_still_weight=args.stand_still_reward_weight,
                temporal_alignment_weight=args.temporal_alignment_reward_weight,
                axis_alignment_weight=args.axis_alignment_reward_weight,
                velocity_weight=args.velocity_reward_weight,
                jerk_weight=args.jerk_reward_weight,
            )
            motion_rewards_buf[step] = motion_result["motion_reward"]
            stand_still_raw_buf[step] = motion_result["stand_still_raw"]
            temporal_alignment_raw_buf[step] = motion_result["temporal_alignment_raw"]
            axis_alignment_raw_buf[step] = motion_result["axis_alignment_raw"]
            velocity_raw_buf[step] = motion_result["velocity_raw"]
            jerk_raw_buf[step] = motion_result["jerk_raw"]
            stand_still_weighted_buf[step] = motion_result["stand_still_weighted"]
            temporal_alignment_weighted_buf[step] = motion_result["temporal_alignment_weighted"]
            axis_alignment_weighted_buf[step] = motion_result["axis_alignment_weighted"]
            velocity_weighted_buf[step] = motion_result["velocity_weighted"]
            jerk_weighted_buf[step] = motion_result["jerk_weighted"]

            # Reset histories for done envs
            if done_mask.any():
                paddle_history[done_mask] = 0
                paddle_history[done_mask, -1, :] = current_paddle[done_mask]
                puck_history[done_mask] = 0
                puck_history[done_mask, -1, :] = current_puck[done_mask]

            # Episode tracking
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        episodic_returns.append(info["episode_return"])
                        success_rates.append(1.0 if info.get("success", False) else 0.0)
                        writer.add_scalar(
                            "charts/episodic_return", info["episode_return"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode_length"], global_step
                        )
                        if "motion_data" in info:
                            velocity_magnitudes.extend(info["motion_data"]["velocity_mags"])
                            acceleration_magnitudes.extend(
                                info["motion_data"]["acceleration_mags"]
                            )
                            jerk_magnitudes.extend(info["motion_data"]["jerk_mags"])

        # ---- GAE + returns ----
        with torch.no_grad():
            next_policy_obs = augment_policy_observation(
                next_obs, last_action_for_policy, args.use_last_action_in_policy_state
            )
            next_value = agent.get_value(next_policy_obs).reshape(1, -1)

            # Combine reward streams
            task_r = args.task_reward_weight * rewards_buf
            motion_r = args.motion_reward_weight * motion_rewards_buf
            combined_rewards = task_r + motion_r

            advantages = torch.zeros_like(combined_rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = combined_rewards[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # ---- Log reward streams (matching TD3 metric names) ----
        # Temporal valid fraction
        temporal_valid_frac = (steps_since_done >= temporal_horizon).float().mean().item()
        writer.add_scalar("rewards/temporal_valid_fraction", temporal_valid_frac, global_step)

        # Task reward stats
        writer.add_scalar("rewards/task_reward_raw_mean", rewards_buf.mean().item(), global_step)
        writer.add_scalar("rewards/task_reward_raw_std", rewards_buf.std().item(), global_step)
        writer.add_scalar("rewards/task_reward_scaled_mean", task_r.mean().item(), global_step)

        # Positive task reward stats
        positive_task_mask = rewards_buf > 0.0
        positive_count = float(positive_task_mask.sum().item())
        total_count = max(float(rewards_buf.numel()), 1)
        writer.add_scalar("rewards/task_reward_positive_fraction", positive_count / total_count, global_step)
        if positive_count > 0:
            writer.add_scalar("rewards/task_reward_positive_mean", rewards_buf[positive_task_mask].mean().item(), global_step)

        # Motion reward component stats (raw)
        writer.add_scalar("rewards/stand_still_reward_raw_mean", stand_still_raw_buf.mean().item(), global_step)
        writer.add_scalar("rewards/temporal_alignment_reward_raw_mean", temporal_alignment_raw_buf.mean().item(), global_step)
        writer.add_scalar("rewards/axis_alignment_reward_raw_mean", axis_alignment_raw_buf.mean().item(), global_step)
        writer.add_scalar("rewards/velocity_reward_raw_mean", velocity_raw_buf.mean().item(), global_step)
        writer.add_scalar("rewards/jerk_reward_raw_mean", jerk_raw_buf.mean().item(), global_step)

        # Motion reward component stats (weighted)
        writer.add_scalar("rewards/stand_still_reward_weighted_mean", stand_still_weighted_buf.mean().item(), global_step)
        writer.add_scalar("rewards/temporal_alignment_reward_weighted_mean", temporal_alignment_weighted_buf.mean().item(), global_step)
        writer.add_scalar("rewards/axis_alignment_reward_weighted_mean", axis_alignment_weighted_buf.mean().item(), global_step)
        writer.add_scalar("rewards/velocity_reward_weighted_mean", velocity_weighted_buf.mean().item(), global_step)
        writer.add_scalar("rewards/jerk_reward_weighted_mean", jerk_weighted_buf.mean().item(), global_step)

        # Combined reward stats
        writer.add_scalar("rewards/motion_reward_scaled_mean", motion_r.mean().item(), global_step)
        writer.add_scalar("rewards/combined_reward_mean", combined_rewards.mean().item(), global_step)
        writer.add_scalar("rewards/combined_reward_std", combined_rewards.std().item(), global_step)

        # Motion magnitude stats (from rollout)
        writer.add_scalar("motion/velocity_mag_mean", velocity_mag_buf.mean().item(), global_step)
        writer.add_scalar("motion/velocity_mag_std", velocity_mag_buf.std().item(), global_step)
        writer.add_scalar("motion/acceleration_mag_mean", acceleration_mag_buf.mean().item(), global_step)
        writer.add_scalar("motion/acceleration_mag_std", acceleration_mag_buf.std().item(), global_step)
        writer.add_scalar("motion/jerk_mag_mean", jerk_mag_buf.mean().item(), global_step)
        writer.add_scalar("motion/jerk_mag_std", jerk_mag_buf.std().item(), global_step)

        # Value / advantage stats
        writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
        writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
        writer.add_scalar("charts/value_mean", values_buf.mean().item(), global_step)
        writer.add_scalar("charts/value_std", values_buf.std().item(), global_step)

        # ---- PPO update ----
        b_policy_obs = policy_obs_buf.reshape((-1, policy_obs_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, _, newvalue = agent.get_action_and_value(
                    b_policy_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = (-newlogprob).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- PPO loss logging ----
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        # ---- Periodic console + scalar logging every 500 steps (match TD3) ----
        steps_since_last_log = global_step - last_log_step
        if steps_since_last_log >= 500:
            last_log_step = global_step

            if episodic_returns:
                avg_return = np.mean(episodic_returns)
                min_return = np.min(episodic_returns)
                max_return = np.max(episodic_returns)
                avg_success = np.mean(success_rates) if success_rates else 0.0
                print(
                    f"Step {global_step}: Avg Return: {avg_return:.2f}, "
                    f"Min: {min_return:.2f}, Max: {max_return:.2f}, "
                    f"Success Rate: {avg_success:.2f}, Episodes: {len(episodic_returns)}"
                )
                writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
                writer.add_scalar("charts/min_episodic_return", min_return, global_step)
                writer.add_scalar("charts/max_episodic_return", max_return, global_step)
                writer.add_scalar("charts/avg_success_rate", avg_success, global_step)
                episodic_returns.clear()
                success_rates.clear()
            else:
                print(f"Step {global_step}: No episodes completed in last interval")

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

            # Contact stats
            collisions_per_env_step = (
                interval_paddle_puck_collisions / max(interval_env_steps, 1)
                if interval_env_steps > 0
                else 0.0
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
            interval_paddle_puck_collisions = 0.0
            interval_env_steps = 0

        # ---- Checkpointing every 5000 steps (match TD3) ----
        steps_since_last_ckpt = global_step - last_checkpoint_step
        if steps_since_last_ckpt >= 5000:
            last_checkpoint_step = global_step
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(agent.state_dict(), model_path)

            try:
                evaluate_agent(
                    model_path,
                    checkpoint_dir,
                    env_config,
                    n_eps=4,
                    n_gifs=1,
                    action_scale=action_scale,
                    agent_hidden_layer_size=args.agent_hidden_layer_size,
                    agent_num_hidden_layers=args.agent_num_hidden_layers,
                    use_last_action_in_policy_state=args.use_last_action_in_policy_state,
                )
            except Exception as e:
                print(f"Evaluation failed: {e}")

            print(f"\nCheckpoint saved at step {global_step}")

    # ---- Final save ----
    envs.close()

    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")
    try:
        evaluate_agent(
            f"{log_parent_dir}/model.pth",
            log_parent_dir,
            env_config,
            action_scale=action_scale,
            agent_hidden_layer_size=args.agent_hidden_layer_size,
            agent_num_hidden_layers=args.agent_num_hidden_layers,
            use_last_action_in_policy_state=args.use_last_action_in_policy_state,
        )
    except Exception as e:
        print(f"Final evaluation failed: {e}")

    # Save TensorBoard plots (matching TD3 pattern)
    metrics = [
        "charts/episodic_return",
        "charts/avg_episodic_return",
        "charts/avg_success_rate",
        "losses/policy_loss",
        "losses/value_loss",
        "losses/entropy",
        "losses/approx_kl",
        "losses/clipfrac",
        "losses/explained_variance",
        "rewards/task_reward_raw_mean",
        "rewards/motion_reward_scaled_mean",
        "rewards/combined_reward_mean",
        "rewards/stand_still_reward_raw_mean",
        "rewards/temporal_alignment_reward_raw_mean",
        "rewards/axis_alignment_reward_raw_mean",
        "rewards/velocity_reward_raw_mean",
        "rewards/jerk_reward_raw_mean",
        "motion/avg_velocity_magnitude",
        "motion/avg_acceleration_magnitude",
        "motion/avg_jerk_magnitude",
        "contacts/interval_paddle_puck_collisions_per_env_step",
    ]
    save_tensorboard_plots(log_parent_dir, config, metrics=metrics)
    writer.close()

    print(f"\nTraining complete. Results saved to {log_parent_dir}")
