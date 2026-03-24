"""
AMP training with self-supervised heads (PPO)

This script implements Adversarial Motion Priors (AMP) with a least-squares
discriminator objective (MSE on real/fake scores toward 1 / -1) instead of
binary cross-entropy.

Uses MSE loss: 0.5 * E[(D(x_real) - 1)^2] + 0.5 * E[(D(x_fake) - (-1))^2]
This variant uses [-1, 1] targets instead of [0, 1] for better gradient flow.
This often provides more stable gradients compared to standard GAN training.

Supports discriminator modes:
1. Position-only (default): 8D
2. Position + Action: 16D
3. Position (+ Action) + Puck features: +4D
   Enable with --use_action_discriminator / --use_puck_discriminator.
"""

import random
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import cv2
import imageio
import tqdm

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
import gymnasium as gym

from dataclasses import dataclass, field
import tyro

import os
from datetime import datetime
from types import SimpleNamespace

from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.amp_history.amp_training.self_supervised.ssl_modules import (
    ActionConditionedDynamicsHead,
    ActionConditionedRewardHead,
    SharedStateEncoder,
)

# AMP components
from scripts.smooth_policy.amp_history.amp_training.discriminator import Discriminator
from scripts.smooth_policy.amp_history.amp_training.replay_buffer import ReplayBuffer
from scripts.smooth_policy.amp_history.amp_training.normalizer import Normalizer
from scripts.smooth_policy.amp_history.amp_training.running_stats import RunningStatsNormalizer
from scripts.smooth_policy.amp_history.amp_training.demo_loader_position_history import DemoLoaderPositionHistory
from scripts.smooth_policy.amp_history.amp_training.feature_processing import (
    PUCK_FEATURE_DIM,
    build_puck_discriminator_features_torch,
    normalize_action_history_batch,
    normalize_position_history_batch,
)

# used for possible reference state initialization from demonstrations
class ReferenceStateWrapper(gym.Wrapper):
    """Wrapper that initializes paddle from demonstration states."""
    
    def __init__(self, env, reference_states):
        super().__init__(env)
        self.reference_states = reference_states  # numpy array [N, 4]
    
    def reset(self, **kwargs):
        # Sample random reference state [x, y, vx, vy]
        idx = np.random.randint(0, len(self.reference_states))
        ref_state = self.reference_states[idx]
        
        # Set reference state on underlying environment
        self.env.unwrapped._ref_paddle_state = (
            (float(ref_state[0]), float(ref_state[1])),  # pos
            (float(ref_state[2]), float(ref_state[3]))   # vel
        )
        
        return self.env.reset(**kwargs)


def validate_discriminator_arch(hidden_layer_size, num_hidden_layers):
    """Validate discriminator architecture in hidden-size + depth form."""
    hidden_layer_size = int(hidden_layer_size)
    num_hidden_layers = int(num_hidden_layers)
    if hidden_layer_size <= 0:
        raise ValueError(f"disc_hidden_layer_size must be positive, got {hidden_layer_size}")
    if num_hidden_layers < 1:
        raise ValueError(f"disc_num_hidden_layers must be >= 1, got {num_hidden_layers}")
    return hidden_layer_size, num_hidden_layers

def augment_policy_observation(observation, last_action, use_last_action):
    """Append last action to policy observation when enabled."""
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def build_policy_observation_from_latent(latent, last_action, use_last_action):
    """Build policy input from latent state and optional last action."""
    if not use_last_action:
        return latent
    return torch.cat([latent, last_action], dim=-1)


def extract_current_paddle_position(observation):
    """Extract current paddle (x, y) from common observation layouts."""
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        # history layout: paddle history [0:15], current paddle at indices 12:14
        return observation[:, 12:14]
    # pos/vel layouts both put paddle position in the first 2 dims
    return observation[:, 0:2]


def extract_current_puck_position(observation):
    """Extract current puck (x, y) from common observation layouts."""
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        # history layout: puck history [15:30], current puck at indices 27:29
        return observation[:, 27:29]
    if obs_dim >= 8:
        # vel layout: [paddle_pos, paddle_vel, puck_pos, puck_vel]
        return observation[:, 4:6]
    if obs_dim >= 4:
        # pos layout: [paddle_pos, puck_pos]
        return observation[:, 2:4]
    raise ValueError(f"Observation dim {obs_dim} is too small to extract puck position.")


def parse_scalar_info_from_infos(infos, key, num_envs, device, fallback_values):
    """Read vectorized scalar infos with mask-aware fallback handling."""
    if not (isinstance(infos, dict) and key in infos):
        return fallback_values

    raw = infos[key]
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    if values.shape[0] == num_envs:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    mask_key = f"_{key}"
    mask = infos.get(mask_key)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if values.shape[0] == int(mask.sum()):
            out = fallback_values.clone()
            out[torch.as_tensor(mask, dtype=torch.bool, device=device)] = torch.as_tensor(
                values, dtype=torch.float32, device=device
            )
            return out
    return fallback_values


def _extract_last_terminal_motion_value(info, motion_key):
    """Return last terminal motion sample from final_info.motion_data when available."""
    if not (isinstance(info, dict) and "motion_data" in info):
        return None
    values = info["motion_data"].get(motion_key)
    if values is None or len(values) == 0:
        return None
    return float(values[-1])


def parse_motion_magnitudes_from_infos(
    infos,
    num_envs,
    device,
    fallback_velocity_mag,
    fallback_acceleration_mag,
    fallback_jerk_mag,
):
    """Extract per-env motion magnitudes from infos, including terminal-step fallbacks."""
    velocity_mag = parse_scalar_info_from_infos(
        infos, "paddle_velocity_mag", num_envs, device, fallback_velocity_mag
    )
    acceleration_mag = parse_scalar_info_from_infos(
        infos, "paddle_acceleration_mag", num_envs, device, fallback_acceleration_mag
    )
    jerk_mag = parse_scalar_info_from_infos(
        infos, "paddle_jerk_mag", num_envs, device, fallback_jerk_mag
    )

    if isinstance(infos, dict) and "final_info" in infos:
        terminal_mask = infos.get("_final_info")
        if terminal_mask is None:
            terminal_mask = np.ones(num_envs, dtype=bool)
        else:
            terminal_mask = np.asarray(terminal_mask, dtype=bool).reshape(-1)

        final_infos = infos["final_info"]
        if isinstance(final_infos, np.ndarray):
            final_infos = final_infos.tolist()

        for env_idx, info in enumerate(final_infos):
            if env_idx >= num_envs or not terminal_mask[env_idx] or not info:
                continue
            terminal_velocity = _extract_last_terminal_motion_value(info, "velocity_mags")
            terminal_acceleration = _extract_last_terminal_motion_value(info, "acceleration_mags")
            terminal_jerk = _extract_last_terminal_motion_value(info, "jerk_mags")
            if terminal_velocity is not None:
                velocity_mag[env_idx] = terminal_velocity
            if terminal_acceleration is not None:
                acceleration_mag[env_idx] = terminal_acceleration
            if terminal_jerk is not None:
                jerk_mag[env_idx] = terminal_jerk

    return velocity_mag, acceleration_mag, jerk_mag


def warmup_motion_normalizer(envs, motion_normalizer, warmup_steps, seed, device):
    """Collect random-rollout motion statistics before training starts."""
    if warmup_steps <= 0:
        return

    num_envs = envs.num_envs
    fallback_velocity = torch.zeros(num_envs, dtype=torch.float32, device=device)
    fallback_acceleration = torch.zeros(num_envs, dtype=torch.float32, device=device)
    fallback_jerk = torch.zeros(num_envs, dtype=torch.float32, device=device)

    envs.reset(seed=seed)
    for _ in range(int(warmup_steps)):
        random_actions = np.stack(
            [envs.single_action_space.sample() for _ in range(num_envs)],
            axis=0,
        )
        _, _, _, _, infos = envs.step(random_actions)
        fallback_velocity, fallback_acceleration, fallback_jerk = parse_motion_magnitudes_from_infos(
            infos=infos,
            num_envs=num_envs,
            device=device,
            fallback_velocity_mag=fallback_velocity,
            fallback_acceleration_mag=fallback_acceleration,
            fallback_jerk_mag=fallback_jerk,
        )
        motion_batch = torch.stack(
            [fallback_velocity, fallback_acceleration, fallback_jerk], dim=-1
        )
        motion_normalizer.update(motion_batch)

def train_discriminator_step(
    demo_loader,
    disc_batch_size,
    b_disc_obs,
    b_valid,
    replay_buffer,
    replay_samples,
    disc_normalizer,
    discriminator,
    disc_optimizer,
    grad_penalty_weight,
    logit_reg_weight,
    max_grad_norm,
    device,
):
    """Run one discriminator update step (same least-squares objective as training)."""
    demo_disc_obs = demo_loader.sample(disc_batch_size)
    agent_samples = disc_batch_size // 2

    valid_disc_obs = b_disc_obs[b_valid]
    if len(valid_disc_obs) == 0 and len(replay_buffer) == 0:
        return None
    perm_indices = torch.randperm(len(valid_disc_obs), device=device)
    agent_disc_obs_current = valid_disc_obs[perm_indices[:agent_samples]]

    num_to_store = min(len(valid_disc_obs), replay_samples)
    replay_buffer.push(valid_disc_obs[perm_indices[:num_to_store]])

    if len(replay_buffer) > 0:
        agent_disc_obs_replay = replay_buffer.sample(agent_samples)
        agent_disc_obs = torch.cat([agent_disc_obs_current, agent_disc_obs_replay], dim=0)
    else:
        agent_disc_obs = agent_disc_obs_current

    norm_demo_disc_obs = disc_normalizer.normalize(demo_disc_obs)
    norm_agent_disc_obs = disc_normalizer.normalize(agent_disc_obs)
    norm_demo_disc_obs.requires_grad_(True)

    disc_demo_logit = discriminator(norm_demo_disc_obs).squeeze(-1)
    disc_agent_logit = discriminator(norm_agent_disc_obs).squeeze(-1)

    disc_loss_demo = 0.5 * torch.mean((disc_demo_logit - 1.0) ** 2)
    disc_loss_agent = 0.5 * torch.mean((disc_agent_logit - (-1.0)) ** 2)
    disc_loss = disc_loss_demo + disc_loss_agent

    disc_demo_grad = torch.autograd.grad(
        disc_demo_logit,
        norm_demo_disc_obs,
        grad_outputs=torch.ones_like(disc_demo_logit),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    disc_grad_penalty = torch.mean(torch.sum(disc_demo_grad ** 2, dim=-1))
    disc_loss = disc_loss + grad_penalty_weight * disc_grad_penalty

    if logit_reg_weight > 0:
        logit_weights = discriminator.get_logit_weights()
        disc_logit_reg_loss = torch.sum(logit_weights ** 2)
        disc_loss = disc_loss + logit_reg_weight * disc_logit_reg_loss

    disc_optimizer.zero_grad()
    disc_loss.backward()
    nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
    disc_optimizer.step()

    disc_normalizer.record(b_disc_obs[b_valid])
    disc_normalizer.record(demo_disc_obs)
    disc_normalizer.update()

    with torch.no_grad():
        disc_agent_acc = (disc_agent_logit < 0.0).float().mean().item()
        disc_demo_acc = (disc_demo_logit > 0.0).float().mean().item()

    return {
        "disc_loss": disc_loss,
        "disc_loss_demo": disc_loss_demo,
        "disc_loss_agent": disc_loss_agent,
        "disc_grad_penalty": disc_grad_penalty,
        "disc_agent_acc": disc_agent_acc,
        "disc_demo_acc": disc_demo_acc,
        "disc_agent_logit": disc_agent_logit,
        "disc_demo_logit": disc_demo_logit,
    }


def log_discriminator_metrics(writer, prefix, metrics, replay_buffer_size, global_step):
    writer.add_scalar(f"{prefix}/disc_loss", metrics["disc_loss"].item(), global_step)
    writer.add_scalar(f"{prefix}/disc_loss_demo", metrics["disc_loss_demo"].item(), global_step)
    writer.add_scalar(f"{prefix}/disc_loss_agent", metrics["disc_loss_agent"].item(), global_step)
    writer.add_scalar(f"{prefix}/disc_grad_penalty", metrics["disc_grad_penalty"].item(), global_step)
    writer.add_scalar(f"{prefix}/disc_agent_acc", metrics["disc_agent_acc"], global_step)
    writer.add_scalar(f"{prefix}/disc_demo_acc", metrics["disc_demo_acc"], global_step)
    writer.add_scalar(f"{prefix}/disc_agent_logit_mean", metrics["disc_agent_logit"].mean().item(), global_step)
    writer.add_scalar(f"{prefix}/disc_demo_logit_mean", metrics["disc_demo_logit"].mean().item(), global_step)
    writer.add_scalar(f"{prefix}/replay_buffer_size", replay_buffer_size, global_step)


def log_reward_stream_stats(
    writer,
    global_step,
    task_r_raw,
    task_r_scaled,
    temporal_alignment_reward_raw,
    temporal_alignment_reward_scaled,
    action_magnitude_reward_raw,
    action_magnitude_reward_scaled,
    combined_rewards,
    advantages,
    values,
    use_short_discriminator_reward,
    disc_r_raw,
    disc_r_scaled,
):
    if use_short_discriminator_reward:
        writer.add_scalar("amp/disc_reward_raw_mean", disc_r_raw.mean().item(), global_step)
        writer.add_scalar("amp/disc_reward_raw_std", disc_r_raw.std().item(), global_step)
        writer.add_scalar("amp/disc_reward_scaled_mean", disc_r_scaled.mean().item(), global_step)
        writer.add_scalar("amp/disc_reward_scaled_std", disc_r_scaled.std().item(), global_step)
    writer.add_scalar("amp/task_reward_raw_mean", task_r_raw.mean().item(), global_step)
    writer.add_scalar("amp/task_reward_raw_std", task_r_raw.std().item(), global_step)
    writer.add_scalar("amp/task_reward_scaled_mean", task_r_scaled.mean().item(), global_step)
    writer.add_scalar("amp/task_reward_scaled_std", task_r_scaled.std().item(), global_step)
    writer.add_scalar("amp/temporal_alignment_reward_raw_mean", temporal_alignment_reward_raw.mean().item(), global_step)
    writer.add_scalar("amp/temporal_alignment_reward_raw_std", temporal_alignment_reward_raw.std().item(), global_step)
    writer.add_scalar("amp/temporal_alignment_reward_scaled_mean", temporal_alignment_reward_scaled.mean().item(), global_step)
    writer.add_scalar("amp/temporal_alignment_reward_scaled_std", temporal_alignment_reward_scaled.std().item(), global_step)
    writer.add_scalar("amp/action_magnitude_reward_raw_mean", action_magnitude_reward_raw.mean().item(), global_step)
    writer.add_scalar("amp/action_magnitude_reward_raw_std", action_magnitude_reward_raw.std().item(), global_step)
    writer.add_scalar("amp/action_magnitude_reward_scaled_mean", action_magnitude_reward_scaled.mean().item(), global_step)
    writer.add_scalar("amp/action_magnitude_reward_scaled_std", action_magnitude_reward_scaled.std().item(), global_step)

    # Backward-compatible logs.
    if use_short_discriminator_reward:
        writer.add_scalar("amp/disc_reward_mean", disc_r_raw.mean().item(), global_step)
        writer.add_scalar("amp/disc_reward_std", disc_r_raw.std().item(), global_step)
    writer.add_scalar("amp/task_reward_mean", task_r_raw.mean().item(), global_step)
    writer.add_scalar("amp/combined_reward_mean", combined_rewards.mean().item(), global_step)

    writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
    writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
    writer.add_scalar("charts/value_mean", values.mean().item(), global_step)
    writer.add_scalar("charts/value_std", values.std().item(), global_step)


def log_motion_penalty_stats(
    writer,
    global_step,
    velocity_mag,
    acceleration_mag,
    jerk_mag,
    velocity_penalty_raw,
    acceleration_penalty_raw,
    jerk_penalty_raw,
    velocity_penalty_scaled,
    acceleration_penalty_scaled,
    jerk_penalty_scaled,
    motion_normalizer,
):
    writer.add_scalar("motion/velocity_mag_mean", velocity_mag.mean().item(), global_step)
    writer.add_scalar("motion/velocity_mag_std", velocity_mag.std().item(), global_step)
    writer.add_scalar("motion/acceleration_mag_mean", acceleration_mag.mean().item(), global_step)
    writer.add_scalar("motion/acceleration_mag_std", acceleration_mag.std().item(), global_step)
    writer.add_scalar("motion/jerk_mag_mean", jerk_mag.mean().item(), global_step)
    writer.add_scalar("motion/jerk_mag_std", jerk_mag.std().item(), global_step)

    writer.add_scalar("motion/velocity_penalty_raw_mean", velocity_penalty_raw.mean().item(), global_step)
    writer.add_scalar("motion/acceleration_penalty_raw_mean", acceleration_penalty_raw.mean().item(), global_step)
    writer.add_scalar("motion/jerk_penalty_raw_mean", jerk_penalty_raw.mean().item(), global_step)
    writer.add_scalar("motion/velocity_penalty_scaled_mean", velocity_penalty_scaled.mean().item(), global_step)
    writer.add_scalar("motion/acceleration_penalty_scaled_mean", acceleration_penalty_scaled.mean().item(), global_step)
    writer.add_scalar("motion/jerk_penalty_scaled_mean", jerk_penalty_scaled.mean().item(), global_step)

    motion_std = torch.sqrt(motion_normalizer.var).detach().cpu().numpy()
    writer.add_scalar("motion/normalizer_velocity_std", float(motion_std[0]), global_step)
    writer.add_scalar("motion/normalizer_acceleration_std", float(motion_std[1]), global_step)
    writer.add_scalar("motion/normalizer_jerk_std", float(motion_std[2]), global_step)
    writer.add_scalar("motion/normalizer_count", motion_normalizer.count.item(), global_step)


def save_amp_components(save_dir, discriminator, normalizer, replay_buffer):
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))
    torch.save(
        {
            "normalizer": normalizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
        },
        os.path.join(save_dir, "amp_components.pth"),
    )


def evaluate_ssl_agent(
    agent,
    state_encoder,
    save_dir,
    air_hockey_params,
    n_eps=5,
    n_gifs=1,
    reference_states=None,
    ref_max_episode_steps=None,
    use_last_action_in_policy_state=False,
    device="cuda:0",
):
    """Evaluate latent-conditioned policy, save GIFs, and write summary stats."""
    os.makedirs(save_dir, exist_ok=True)
    eval_air_hockey_params = air_hockey_params.copy()
    if ref_max_episode_steps is not None:
        eval_air_hockey_params["max_timesteps"] = ref_max_episode_steps

    def make_eval_env():
        env = AirHockeyEnv(eval_air_hockey_params)
        if reference_states is not None:
            env = ReferenceStateWrapper(env, reference_states)
        return env

    envs = gym.vector.SyncVectorEnv([make_eval_env])
    env = envs.envs[0]
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)
    action_dim = int(np.prod(envs.single_action_space.shape))
    returns = []
    n_eps = int(n_eps)
    n_gifs = max(1, int(n_gifs))
    fps = 20

    was_agent_training = bool(agent.training)
    was_encoder_training = bool(state_encoder.training)
    agent.eval()
    state_encoder.eval()
    with torch.no_grad():
        for gif_idx in range(n_gifs):
            frames = []
            for episode_idx in tqdm.tqdm(range(n_eps), desc=f"eval-gif-{gif_idx}", leave=False):
                obs_np, _ = envs.reset()
                done = np.array([False])
                episode_return = 0.0
                step_reward = 0.0
                last_action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
                while not done[0]:
                    frame = renderer.get_frame()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_color = (0, 0, 0)
                    line_type = 2
                    cv2.putText(
                        frame,
                        f"Ep {episode_idx + 1}/{n_eps}",
                        (10, 30),
                        font,
                        font_scale,
                        font_color,
                        line_type,
                    )
                    cv2.putText(
                        frame,
                        f"Reward: {step_reward:.2f}",
                        (frame.shape[1] - 150, 30),
                        font,
                        font_scale,
                        font_color,
                        line_type,
                    )
                    cv2.putText(
                        frame,
                        f"Return: {episode_return:.2f}",
                        (frame.shape[1] - 150, 60),
                        font,
                        font_scale,
                        font_color,
                        line_type,
                    )
                    frames.append(frame)

                    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                    latent = state_encoder(obs)
                    policy_obs = build_policy_observation_from_latent(
                        latent, last_action, use_last_action_in_policy_state
                    )
                    action = agent(policy_obs)
                    obs_np, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
                    done = np.logical_or(terminations, truncations)
                    step_reward = float(reward[0])
                    episode_return += step_reward
                    if done[0]:
                        last_action.zero_()
                    else:
                        last_action = action.detach().to(device=device)
                returns.append(episode_return)

            gif_savepath = os.path.join(save_dir, f"eval_{gif_idx}.gif")
            imageio.mimsave(
                gif_savepath,
                frames,
                format="GIF",
                loop=0,
                duration=int(1000 * 1 / fps),
            )

    if was_agent_training:
        agent.train()
    else:
        agent.eval()
    if was_encoder_training:
        state_encoder.train()
    else:
        state_encoder.eval()
    envs.close()

    summary = {
        "num_episodes": int(len(returns)),
        "num_gifs": int(n_gifs),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "return_min": float(np.min(returns)) if returns else 0.0,
        "return_max": float(np.max(returns)) if returns else 0.0,
    }
    with open(os.path.join(save_dir, "eval_summary_ssl.yaml"), "w") as f:
        yaml.safe_dump(summary, f)
    print(
        "[eval-ssl] "
        f"mean={summary['return_mean']:.2f}, min={summary['return_min']:.2f}, max={summary['return_max']:.2f}"
    )


@dataclass
class Args:
    num_envs: int = 8
    num_steps: int = 512
    learning_rate: float = 1e-4
    num_iterations: int = 100
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 64
    update_epochs: int = 10
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    batch_size: int = 0 # computed at runtime
    norm_adv: bool = True

    # SSL latent + auxiliary heads
    ssl_latent_dim: int = 10
    ssl_encoder_hidden_layer_size: int = 64
    ssl_encoder_num_hidden_layers: int = 5
    ssl_reward_head_hidden_layer_size: int = 64
    ssl_reward_head_num_hidden_layers: int = 10
    ssl_dynamics_head_hidden_layer_size: int = 64
    ssl_dynamics_head_num_hidden_layers: int = 10
    ssl_reward_loss_weight: float = 1.0
    ssl_dynamics_loss_weight: float = 1.0
    ssl_update_epochs: int = 25  # Number of SSL-only optimization epochs per PPO iteration
    ssl_minibatch_size: int = 0  # If <= 0, falls back to PPO minibatch_size
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 10
    weight_decay: float = 0.0001

    # AMP hyperparameters (always enabled)
    disc_replay_buffer_size: int = 100000
    disc_replay_samples: int = 1024
    disc_batch_size: int = 512
    disc_reward_warmup_iters: int = 10
    disc_reward_warmup_value: float = 0.4
    disc_learning_rate: float = 1e-5
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 5.0
    disc_weight_decay: float = 0.0001
    task_reward_weight: float = 0.5
    disc_reward_weight: float = 0.5
    num_discriminator_updates: int = 1
    disc_hidden_layer_size: int = 64
    disc_num_hidden_layers: int = 2
    # Optional auxiliary rewards (default disabled)
    temporal_alignment_reward_scale: float = 0.0
    action_magnitude_reward_scale: float = 0.0
    temporal_alignment_horizon: int = 4
    velocity_penalty_weight: float = 0.0
    acceleration_penalty_weight: float = 0.0
    jerk_penalty_weight: float = 0.0
    motion_norm_warmup_steps: int = 2048
    motion_norm_update_online: bool = True
    motion_norm_clip: float = 10.0
    motion_norm_eps: float = 1e-6
    exploration_reward_weight: float = 0.1
    exploration_error_deadzone: float = 0.1
    exploration_norm_update_online: bool = True
    exploration_norm_clip: float = 10.0
    exploration_norm_eps: float = 1e-6

    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str = None
    model_path: str = None  # Path to pre-trained model state dict
    bc_policy_path: str = None  # Path to frozen BC teacher policy (Agent state_dict)
    bc_kl_weight: float = 0.01  # Initial KL weight for BC regularization: KL(pi_BC || pi_theta)
    bc_kl_decay_iters: int = 200  # Linearly decay BC KL weight to 0 over this many iterations
    reset_actor_logstd_on_model_load: bool = False  # Override actor_logstd immediately after loading model_path
    actor_logstd_on_model_load: float = 0.0  # Value used when reset_actor_logstd_on_model_load=True
    discriminator_path: str = None  # Path to pre-trained discriminator state dict
    amp_components_path: str = None  # Path to AMP components (normalizer, replay buffer)
    state_encoder_path: str = None  # Path to pre-trained shared state encoder state dict
    reward_head_path: str = None  # Path to pre-trained reward head state dict
    dynamics_head_path: str = None  # Path to pre-trained dynamics head state dict
    log_parent_dir: str = None
    run_name: str = "default"
    demo_data_path: str = "scripts/smooth_policy/amp_data/amp_dataset.pt"

    # Others
    seed: int = 0
    device: str = "cuda:0"
    use_last_action_in_policy_state: bool = False  # Append previous action to policy input state
    ppo_stop_grad_state_encoder: bool = True  # Stop PPO/BC gradients from updating the shared state encoder

    # action scale for the agent (mostly deprecated, should default to 1)
    action_scale: float = 1
    
    # Reference state initialization
    use_reference_state_init: bool = False  # Enable/disable feature
    reference_data_path: str = None  # Path to raw demo data (defaults to demo_data_path)
    ref_max_episode_steps: int = 200  # Episode length when using reference init
    
    # Action-conditioned discriminator
    use_action_discriminator: bool = False  # If True, discriminator uses position + 4 transition actions (16D)
    use_puck_discriminator: bool = False  # If True, discriminator appends puck features (+4D)
    puck_vertical_axis: int = 0  # Canonical up/down axis (0=x, 1=y)
    puck_downward_positive_direction: float = 1.0  # +1 if increasing axis is downward, else -1
    puck_noise_std: float = 0.03  # Gaussian noise std for current puck position features
    puck_downward_speed_max: float = 0.75  # Max speed used for 3-level downward speed bins
    puck_speed_dt: float = 0.05  # Time delta for puck speed estimation
    puck_vertical_pos_min: float = -1.0  # Min vertical puck value mapped to 5-bin range
    puck_vertical_pos_max: float = 1.0  # Max vertical puck value mapped to 5-bin range
    disc_debug_interval: int = 5000  # Print discriminator feature samples every N env steps (<=0 disables)
    
    

def make_env(env_id, reference_states=None, max_episode_steps=20):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        
        # Override max_timesteps if using reference state initialization
        if reference_states is not None:
            config["air_hockey"]["max_timesteps"] = max_episode_steps
        
        env = AirHockeyEnv(config["air_hockey"])
        
        # Wrap with reference state initialization if provided
        if reference_states is not None:
            env = ReferenceStateWrapper(env, reference_states)
        
        return env
    return _thunk

# Example usage:
if __name__ == "__main__":

    temp_args = tyro.cli(Args) # checks for a passed in args file
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()  # Use class defaults

    # command line args override file args
    args = tyro.cli(Args, default=default_args)
    args.batch_size = args.num_envs * args.num_steps
    if args.agent_num_hidden_layers < 1:
        raise ValueError("agent_num_hidden_layers must be >= 1.")
    validate_discriminator_arch(
        hidden_layer_size=args.disc_hidden_layer_size,
        num_hidden_layers=args.disc_num_hidden_layers,
    )
    if args.ssl_encoder_hidden_layer_size <= 0:
        raise ValueError("ssl_encoder_hidden_layer_size must be > 0.")
    if args.ssl_encoder_num_hidden_layers < 1:
        raise ValueError("ssl_encoder_num_hidden_layers must be >= 1.")
    if args.ssl_reward_head_hidden_layer_size <= 0:
        raise ValueError("ssl_reward_head_hidden_layer_size must be > 0.")
    if args.ssl_reward_head_num_hidden_layers < 1:
        raise ValueError("ssl_reward_head_num_hidden_layers must be >= 1.")
    if args.ssl_dynamics_head_hidden_layer_size <= 0:
        raise ValueError("ssl_dynamics_head_hidden_layer_size must be > 0.")
    if args.ssl_dynamics_head_num_hidden_layers < 1:
        raise ValueError("ssl_dynamics_head_num_hidden_layers must be >= 1.")
    if args.ssl_update_epochs < 0:
        raise ValueError("ssl_update_epochs must be >= 0.")
    if args.ssl_minibatch_size < 0:
        raise ValueError("ssl_minibatch_size must be >= 0.")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load reference states if enabled
    reference_states = None
    if args.use_reference_state_init:
        ref_data_path = args.reference_data_path or args.demo_data_path
        print(f"\n{'='*80}")
        print(f"Loading reference states from: {ref_data_path}")
        raw_data = torch.load(ref_data_path, map_location='cpu')
        # Extract first position of each sequence: [N, 2] = [x, y]
        # Note: We only have position data, not velocity, so we'll set velocity to 0
        position_sequences = raw_data['position_sequences']  # Shape: [N, 5, 2]
        first_positions = position_sequences[:, 0, :].numpy()  # [N, 2]
        # Create reference states with zero velocity: [N, 4] = [x, y, 0, 0]
        reference_states = np.concatenate([
            first_positions,
            np.zeros_like(first_positions)  # Zero velocity
        ], axis=1)
        print(f"✓ Loaded {len(reference_states):,} reference states (position-based)")
        print(f"  Note: Velocities initialized to zero")
        print(f"  Episode length will be {args.ref_max_episode_steps} timesteps")
        print(f"{'='*80}\n")

    # should just create parallel envs for future use (can just use sync, async as placeholders)
    envs = gym.vector.AsyncVectorEnv([
        make_env(i, reference_states, args.ref_max_episode_steps) 
        for i in range(args.num_envs)
    ])

    print("envs.single_observation_space.shape:", envs.single_observation_space.shape)

    # Create folder with all results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    run_name = args.run_name

    log_parent_dir = args.log_parent_dir
    if log_parent_dir is None:
        log_parent_dir = f"runs/default_training/{task_name}/{run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        # Append r# to end of log_parent_dir, where # is the first unused integer
        base_log_parent_dir = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base_log_parent_dir}r{i}"
            i += 1
        print(f"Log directory exists. Saving to alternate log directory: {log_parent_dir}")
        # raise FileExistsError(f"Log directory {log_parent_dir} already exists.")
    os.makedirs(log_parent_dir, exist_ok=True)

    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # save yaml args and config into log_parent_dir
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    
    if 'use_pid' in config["air_hockey"] and config["air_hockey"]["use_pid"]:
        action_scale = 1
    else:
        action_scale = args.action_scale # use whatever action scale specified

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    if args.ssl_latent_dim <= 0:
        raise ValueError("ssl_latent_dim must be > 0.")
    policy_obs_dim = args.ssl_latent_dim + (action_dim if args.use_last_action_in_policy_state else 0)
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_obs_dim,),
            dtype=np.float32,
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
    state_encoder = SharedStateEncoder(
        obs_dim=obs_dim,
        latent_dim=args.ssl_latent_dim,
        hidden_layer_size=args.ssl_encoder_hidden_layer_size,
        num_hidden_layers=args.ssl_encoder_num_hidden_layers,
    ).to(args.device)
    reward_head = ActionConditionedRewardHead(
        latent_dim=args.ssl_latent_dim,
        action_dim=action_dim,
        hidden_layer_size=args.ssl_reward_head_hidden_layer_size,
        num_hidden_layers=args.ssl_reward_head_num_hidden_layers,
    ).to(args.device)
    dynamics_head = ActionConditionedDynamicsHead(
        latent_dim=args.ssl_latent_dim,
        action_dim=action_dim,
        position_dim=4,
        hidden_layer_size=args.ssl_dynamics_head_hidden_layer_size,
        num_hidden_layers=args.ssl_dynamics_head_num_hidden_layers,
    ).to(args.device)
    # Load pre-trained model if path is provided
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
        if args.reset_actor_logstd_on_model_load:
            with torch.no_grad():
                agent.actor_logstd.fill_(float(args.actor_logstd_on_model_load))
            print(
                "Applied actor_logstd override after model load: "
                f"{float(args.actor_logstd_on_model_load):.4f}"
            )
        print("Model loaded successfully")
    if args.state_encoder_path is not None:
        if not os.path.exists(args.state_encoder_path):
            raise FileNotFoundError(f"State encoder path {args.state_encoder_path} does not exist.")
        state_encoder.load_state_dict(torch.load(args.state_encoder_path, map_location=args.device))
        print(f"Loaded shared state encoder from {args.state_encoder_path}")
    if args.reward_head_path is not None:
        if not os.path.exists(args.reward_head_path):
            raise FileNotFoundError(f"Reward head path {args.reward_head_path} does not exist.")
        reward_head.load_state_dict(torch.load(args.reward_head_path, map_location=args.device))
        print(f"Loaded reward head from {args.reward_head_path}")
    if args.dynamics_head_path is not None:
        if not os.path.exists(args.dynamics_head_path):
            raise FileNotFoundError(f"Dynamics head path {args.dynamics_head_path} does not exist.")
        dynamics_head.load_state_dict(torch.load(args.dynamics_head_path, map_location=args.device))
        print(f"Loaded dynamics head from {args.dynamics_head_path}")

    bc_policy = None
    if args.bc_policy_path is not None:
        if not os.path.exists(args.bc_policy_path):
            raise FileNotFoundError(f"BC policy path {args.bc_policy_path} does not exist.")
        bc_policy = Agent(
            policy_env_view,
            action_scale=action_scale,
            action_bias=0.0,
            hidden_layer_size=args.agent_hidden_layer_size,
            num_hidden_layers=args.agent_num_hidden_layers,
        ).to(args.device)
        bc_policy.load_state_dict(torch.load(args.bc_policy_path, map_location=args.device))
        bc_policy.eval()
        bc_policy.requires_grad_(False)
        print(f"Loaded frozen BC policy from {args.bc_policy_path}")
    
    optimizer = torch.optim.Adam(
        list(agent.parameters())
        + list(state_encoder.parameters())
        + list(reward_head.parameters())
        + list(dynamics_head.parameters()),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        eps=1e-6,
    )
    use_short_discriminator_reward = args.disc_reward_weight > 0.0
    use_discriminator_reward = use_short_discriminator_reward
    use_motion_penalty = (
        args.velocity_penalty_weight > 0.0
        or args.acceleration_penalty_weight > 0.0
        or args.jerk_penalty_weight > 0.0
    )
    motion_normalizer = RunningStatsNormalizer(shape=(3,), device=args.device, clip=None)
    exploration_error_normalizer = RunningStatsNormalizer(shape=(1,), device=args.device, clip=None)

    # Initialize AMP components (always enabled)
    print("\n" + "="*80)
    print("Initializing AMP (Adversarial Motion Priors) components")
    print("="*80)

    history_len = 5  # Number of positions to track for short discriminator
    disc_obs_dim = 8
    use_action_disc = False
    use_puck_disc = False
    discriminator, disc_optimizer, disc_normalizer, replay_buffer, demo_loader = None, None, None, None, None
    
    if use_short_discriminator_reward:
        # Load demonstration data first to determine short observation dimension
        demo_loader = DemoLoaderPositionHistory(
            args.demo_data_path,
            device=args.device,
            use_actions=args.use_action_discriminator,
            use_puck=args.use_puck_discriminator,
            puck_vertical_axis=args.puck_vertical_axis,
            puck_downward_positive_direction=args.puck_downward_positive_direction,
            puck_downward_speed_max=args.puck_downward_speed_max,
            puck_speed_dt=args.puck_speed_dt,
            puck_noise_std=args.puck_noise_std,
            puck_vertical_pos_min=args.puck_vertical_pos_min,
            puck_vertical_pos_max=args.puck_vertical_pos_max,
        )
        print(f"✓ Short demo loader initialized ({len(demo_loader):,} demonstrations)")
        
        use_action_disc = demo_loader.use_actions
        use_puck_disc = demo_loader.use_puck
        disc_obs_dim = demo_loader.get_obs_dim()
        disc_hidden_layer_size, disc_num_hidden_layers = validate_discriminator_arch(
            hidden_layer_size=args.disc_hidden_layer_size,
            num_hidden_layers=args.disc_num_hidden_layers,
        )
        
        mode_parts = ["POSITION HISTORY (8D)"]
        if use_action_disc:
            mode_parts.append("ACTION HISTORY (8D)")
        if use_puck_disc:
            mode_parts.append(f"PUCK FEATURES ({PUCK_FEATURE_DIM}D)")
        print(f"  Mode: {' + '.join(mode_parts)}")
        print(f"  Discriminator input dim: {disc_obs_dim}")
        
        # Initialize short discriminator
        discriminator = Discriminator(
            disc_obs_dim,
            hidden_layer_size=disc_hidden_layer_size,
            num_hidden_layers=disc_num_hidden_layers,
            activation='leaky_relu'
        ).to(args.device)
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.disc_learning_rate,
            weight_decay=args.disc_weight_decay,
            eps=1e-6,
            betas=(0.5, 0.95),
        )
        print(
            "✓ Short discriminator initialized "
            f"(input_dim={disc_obs_dim}, hidden_layer_size={disc_hidden_layer_size}, "
            f"num_hidden_layers={disc_num_hidden_layers})"
        )
        
        # Initialize normalizer
        disc_normalizer = Normalizer(shape=(disc_obs_dim,), clip=10.0, device=args.device)
        print(f"✓ Normalizer initialized (clip=10.0)")
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(
            capacity=args.disc_replay_buffer_size,
            obs_shape=(disc_obs_dim,),
            device=args.device
        )
        print(f"✓ Replay buffer initialized (capacity={args.disc_replay_buffer_size:,})")
        
        # Load pre-trained short discriminator if path is provided
        if args.discriminator_path is not None:
            if not os.path.exists(args.discriminator_path):
                raise FileNotFoundError(f"Discriminator path {args.discriminator_path} does not exist.")
            print(f"Loading pre-trained discriminator from {args.discriminator_path}")
            discriminator.load_state_dict(torch.load(args.discriminator_path, map_location=args.device))
            print("✓ Discriminator loaded successfully")
        
        # Load short AMP components (normalizer, replay buffer) if path is provided
        if args.amp_components_path is not None:
            if not os.path.exists(args.amp_components_path):
                raise FileNotFoundError(f"AMP components path {args.amp_components_path} does not exist.")
            print(f"Loading AMP components from {args.amp_components_path}")
            amp_components = torch.load(args.amp_components_path, map_location=args.device)
            disc_normalizer.load_state_dict(amp_components['normalizer'])
            replay_buffer.load_state_dict(amp_components['replay_buffer'])
            print(f"✓ Short AMP components loaded successfully (replay buffer size: {len(replay_buffer):,})")
    else:
        print("  Short discriminator reward disabled (disc_reward_weight <= 0).")

    print("="*80 + "\n")


    # main training loop
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    policy_last_actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    ssl_current_positions = torch.zeros((args.num_steps, args.num_envs, 4), device=args.device)
    ssl_next_positions = torch.zeros((args.num_steps, args.num_envs, 4), device=args.device)
    
    # AMP: Storage for discriminator observations and position history
    disc_obs = torch.zeros((args.num_steps, args.num_envs, disc_obs_dim)).to(args.device) if use_short_discriminator_reward else None
    paddle_positions = torch.zeros((args.num_steps, args.num_envs, 2)).to(args.device)
    temporal_alignment_reward_raw = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    temporal_alignment_reward_scaled = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    action_magnitude_reward_raw = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    action_magnitude_reward_scaled = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    velocity_mag_rollout = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    acceleration_mag_rollout = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    jerk_mag_rollout = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    # Position history buffer: [num_envs, history_len, 2] for (x, y) positions
    position_history = torch.zeros((args.num_envs, history_len, 2)).to(args.device) if use_short_discriminator_reward else None
    puck_position_history = torch.zeros((args.num_envs, history_len, 2)).to(args.device) if (use_short_discriminator_reward and use_puck_disc) else None
    # Action history buffer: [num_envs, 4, 2] for transition actions between 5 states
    action_history_len = history_len - 1
    action_history = torch.zeros((args.num_envs, action_history_len, 2)).to(args.device) if (use_short_discriminator_reward and use_action_disc) else None
    # Track how many valid positions we have per environment (need history_len before valid)
    position_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device) if use_short_discriminator_reward else None
    action_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device) if (use_short_discriminator_reward and use_action_disc) else None
    puck_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device) if (use_short_discriminator_reward and use_puck_disc) else None
    valid_transition = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool).to(args.device) if use_short_discriminator_reward else None
    # Tracking lists for motion metrics
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []

    # Start
    global_step = 0
    start_time = time.time()
    if use_motion_penalty and args.motion_norm_warmup_steps > 0:
        print(
            f"Warming motion normalizer for {args.motion_norm_warmup_steps} steps "
            f"using random actions..."
        )
        warmup_motion_normalizer(
            envs=envs,
            motion_normalizer=motion_normalizer,
            warmup_steps=args.motion_norm_warmup_steps,
            seed=args.seed + 2026,
            device=args.device,
        )

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    just_reset = torch.zeros(args.num_envs, dtype=torch.bool).to(args.device)
    last_action_for_policy = torch.zeros((args.num_envs, action_dim), device=args.device)
    current_velocity_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_acceleration_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    current_jerk_mag = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    

    for iteration in range(1, args.num_iterations + 1):
        # Reset episodic return tracking for this iteration
        episodic_returns = []
        success_rates = []
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.bc_kl_decay_iters > 0:
            bc_anneal_frac = max(0.0, 1.0 - (iteration - 1.0) / float(args.bc_kl_decay_iters))
            current_bc_kl_weight = args.bc_kl_weight * bc_anneal_frac
        else:
            current_bc_kl_weight = args.bc_kl_weight

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            policy_last_actions[step] = last_action_for_policy

            with torch.no_grad():
                current_latent = state_encoder(next_obs)
                policy_next_obs = build_policy_observation_from_latent(
                    current_latent, last_action_for_policy, args.use_last_action_in_policy_state
                )
                action, logprob, _, value = agent.get_action_and_value(policy_next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # REWARD SCALING is done on the environment level, not here
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)
            last_action_for_policy = action.detach().clone()
            last_action_for_policy[next_done.bool()] = 0

            current_velocity_mag, current_acceleration_mag, current_jerk_mag = parse_motion_magnitudes_from_infos(
                infos=infos,
                num_envs=args.num_envs,
                device=args.device,
                fallback_velocity_mag=current_velocity_mag,
                fallback_acceleration_mag=current_acceleration_mag,
                fallback_jerk_mag=current_jerk_mag,
            )
            velocity_mag_rollout[step] = current_velocity_mag
            acceleration_mag_rollout[step] = current_acceleration_mag
            jerk_mag_rollout[step] = current_jerk_mag
            
            # AMP: Construct discriminator observations from trajectory history
            current_paddle_pos_before = extract_current_paddle_position(obs[step])
            current_puck_pos_before = extract_current_puck_position(obs[step])
            next_paddle_pos = extract_current_paddle_position(next_obs)
            next_puck_pos = extract_current_puck_position(next_obs)
            ssl_current_positions[step] = torch.cat([current_paddle_pos_before, current_puck_pos_before], dim=-1)
            ssl_next_positions[step] = torch.cat([next_paddle_pos, next_puck_pos], dim=-1)
            paddle_positions[step] = next_paddle_pos
            
            # Optional action magnitude reward (raw value is independent of scale)
            action_magnitude = actions[step].abs().sum(dim=-1)
            action_mag_raw = -action_magnitude + 1
            action_magnitude_reward_raw[step] = action_mag_raw
            action_magnitude_reward_scaled[step] = action_mag_raw * args.action_magnitude_reward_scale
            
            if use_short_discriminator_reward:
                # Update position history buffer (rolling buffer: shift left, add new position at end)
                position_history = torch.roll(position_history, shifts=-1, dims=1)
                position_history[:, -1, :] = next_paddle_pos

                if use_action_disc:
                    # Update action history buffer with current transition action (aligned with next_obs)
                    action_history = torch.roll(action_history, shifts=-1, dims=1)
                    action_history[:, -1, :] = action
                if use_puck_disc:
                    puck_position_history = torch.roll(puck_position_history, shifts=-1, dims=1)
                    puck_position_history[:, -1, :] = next_puck_pos
                
                # Increment position count (capped at history_len)
                position_count = torch.clamp(position_count + 1, max=history_len)
                if use_action_disc:
                    action_count = torch.clamp(action_count + 1, max=action_history_len)
                if use_puck_disc:
                    puck_count = torch.clamp(puck_count + 1, max=history_len)
                
                # Valid transition only if we have history_len positions AND not just reset
                has_enough_history = position_count >= history_len
                valid_transition[step] = has_enough_history
                if use_action_disc:
                    has_enough_actions = action_count >= action_history_len
                    valid_transition[step] = valid_transition[step] & has_enough_actions
                if use_puck_disc:
                    has_enough_puck = puck_count >= history_len
                    valid_transition[step] = valid_transition[step] & has_enough_puck
                
                # Invalidate for environments that just reset
                valid_transition[step, just_reset] = False
                
                # Normalize the position history to get relative positions
                # Result: [batch, 8] = 4 relative positions × 2 coords
                normalized_positions = normalize_position_history_batch(position_history)
                
                # Store the discriminator observations
                disc_feature_parts = [normalized_positions]
                if use_action_disc:
                    normalized_actions = normalize_action_history_batch(action_history)
                    disc_feature_parts.append(normalized_actions)
                if use_puck_disc:
                    puck_features = build_puck_discriminator_features_torch(
                        puck_position_history,
                        current_index=2,  # use puck position 2 steps before final state to match offline demos
                        vertical_axis=args.puck_vertical_axis,
                        downward_positive_direction=args.puck_downward_positive_direction,
                        downward_speed_max=args.puck_downward_speed_max,
                        speed_dt=args.puck_speed_dt,
                        noise_std=args.puck_noise_std,
                        vertical_pos_min=args.puck_vertical_pos_min,
                        vertical_pos_max=args.puck_vertical_pos_max,
                    )
                    disc_feature_parts.append(puck_features)
                disc_obs[step] = torch.cat(disc_feature_parts, dim=-1)

                # Occasional debug logging for discriminator feature formatting.
                if args.disc_debug_interval > 0 and global_step % args.disc_debug_interval == 0:
                    sample_idx = 0
                    sample_pos = normalized_positions[sample_idx].detach().cpu().numpy()
                    sample_disc = disc_obs[step, sample_idx].detach().cpu().numpy()
                    debug_msg = (
                        f"[disc-debug][step={global_step}] pos_shape={normalized_positions.shape}, "
                        f"disc_shape={disc_obs[step].shape}, valid={bool(valid_transition[step, sample_idx].item())}"
                    )
                    if use_action_disc:
                        debug_msg += f", action_shape={action_history.shape}"
                    if use_puck_disc:
                        debug_msg += f", puck_shape={puck_position_history.shape}"
                    print(debug_msg)
                    print(
                        "  sample pos[0:4]="
                        f"{np.array2string(sample_pos[:4], precision=4, suppress_small=True)}"
                    )
                    if use_action_disc:
                        sample_action = normalize_action_history_batch(action_history)[sample_idx].detach().cpu().numpy()
                        print(
                            "  sample action[0:4]="
                            f"{np.array2string(sample_action[:4], precision=4, suppress_small=True)}"
                        )
                    if use_puck_disc:
                        sample_puck = puck_features[sample_idx].detach().cpu().numpy()
                        print(
                            "  sample puck="
                            f"{np.array2string(sample_puck, precision=4, suppress_small=True)}"
                        )
                    print(
                        "  sample disc[0:8]="
                        f"{np.array2string(sample_disc[:8], precision=4, suppress_small=True)}"
                    )

            # Reset position history and count for environments that are done
            if use_discriminator_reward and next_done.any():
                done_mask = next_done.bool()
                if use_short_discriminator_reward:
                    position_history[done_mask] = 0
                    position_history[done_mask, -1, :] = next_paddle_pos[done_mask]
                    position_count[done_mask] = 1  # We have 1 position after reset
                    if use_action_disc:
                        action_history[done_mask] = 0
                        action_count[done_mask] = 0
                    if use_puck_disc:
                        puck_position_history[done_mask] = 0
                        puck_position_history[done_mask, -1, :] = next_puck_pos[done_mask]
                        puck_count[done_mask] = 1
            # Track which environments just reset for next step
            just_reset = next_done.bool()

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        episodic_returns.append(info['episode_return'])
                        success_rates.append(1.0 if info['success'] else 0.0)
                        # print(f"global_step={global_step}, episodic_return={info['episode_return']}")
                        writer.add_scalar("charts/episodic_return", info['episode_return'], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode_length'], global_step)
                        
                        # Extract motion data if available
                        if 'motion_data' in info:
                            velocity_magnitudes.extend(info['motion_data']['velocity_mags'])
                            acceleration_magnitudes.extend(info['motion_data']['acceleration_mags'])
                            jerk_magnitudes.extend(info['motion_data']['jerk_mags'])

        # bootstrap value if not done and compute advantages
        with torch.no_grad():
            next_latent = state_encoder(next_obs)
            next_policy_obs = build_policy_observation_from_latent(
                next_latent, last_action_for_policy, args.use_last_action_in_policy_state
            )
            next_value = agent.get_value(next_policy_obs).reshape(1, -1)
            
            # Optional temporal alignment reward:
            # compare realized movement over horizon vs commanded action direction from horizon steps ago.
            temporal_alignment_reward_raw.zero_()
            temporal_alignment_reward_scaled.zero_()
            horizon = args.temporal_alignment_horizon
            eps = 1e-8
            for t in range(horizon, args.num_steps):
                realized_movement = paddle_positions[t] - paddle_positions[t - horizon]
                target_direction = actions[t - horizon]

                movement_norm = torch.norm(realized_movement, dim=-1).clamp_min(eps)
                target_norm = torch.norm(target_direction, dim=-1).clamp_min(eps)
                cosine_sim = (realized_movement * target_direction).sum(dim=-1) / (movement_norm * target_norm)

                # Apply fallback reward per environment when target direction is near zero.
                small_target_mask = torch.norm(target_direction, dim=-1) < 0.025  # hard-coded threshold for now
                cosine_sim = torch.where(
                    small_target_mask,
                    torch.full_like(cosine_sim, 1.0),  # hard-coded reward
                    cosine_sim,
                )
                
                # Invalidate if episode reset happened between command and realized movement.
                temporal_valid = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
                for k in range(t - horizon + 1, t + 1):
                    temporal_valid = temporal_valid & (~dones[k].bool())
                
                temporal_alignment_reward_raw[t] = (cosine_sim * temporal_valid.float() + 1) / 2.0 # clamp to [0, 1]
            temporal_alignment_reward_scaled = temporal_alignment_reward_raw * args.temporal_alignment_reward_scale
            
            if use_short_discriminator_reward:
                b_disc_obs = disc_obs.reshape(-1, disc_obs_dim)
                if iteration <= args.disc_reward_warmup_iters:
                    # Keep discriminator reward fixed during early policy warmup.
                    disc_r_raw = torch.full(
                        (args.num_steps * args.num_envs,),
                        float(args.disc_reward_warmup_value),
                        device=args.device,
                    )
                else:
                    norm_disc_obs = disc_normalizer.normalize(b_disc_obs)
                    disc_scores = discriminator(norm_disc_obs).squeeze(-1)
                    disc_r_raw = torch.clamp(1 - 0.25 * (disc_scores - 1) ** 2, min=0)
                    b_valid = valid_transition.reshape(-1)
                    disc_r_raw = disc_r_raw * b_valid.float()
                disc_r_raw_shaped = disc_r_raw.reshape(args.num_steps, args.num_envs)
                disc_r_scaled = args.disc_reward_weight * disc_r_raw_shaped
            else:
                b_disc_obs = None
                disc_r_raw = torch.zeros(args.num_steps * args.num_envs, device=args.device)
                disc_r_scaled = torch.zeros((args.num_steps, args.num_envs), device=args.device)

            task_r_raw = rewards
            task_r_scaled = args.task_reward_weight * task_r_raw

            motion_raw = torch.stack(
                [velocity_mag_rollout, acceleration_mag_rollout, jerk_mag_rollout], dim=-1
            )
            if use_motion_penalty and args.motion_norm_update_online:
                motion_normalizer.update(motion_raw.reshape(-1, 3))

            motion_std = torch.sqrt(motion_normalizer.var).clamp_min(args.motion_norm_eps).view(1, 1, 3)
            motion_penalty_features = motion_raw / motion_std
            if args.motion_norm_clip is not None and args.motion_norm_clip > 0:
                motion_penalty_features = torch.clamp(
                    motion_penalty_features, -args.motion_norm_clip, args.motion_norm_clip
                )

            velocity_penalty_raw = -motion_penalty_features[:, :, 0]
            acceleration_penalty_raw = -motion_penalty_features[:, :, 1]
            jerk_penalty_raw = -motion_penalty_features[:, :, 2]
            velocity_penalty_scaled = args.velocity_penalty_weight * velocity_penalty_raw
            acceleration_penalty_scaled = args.acceleration_penalty_weight * acceleration_penalty_raw
            jerk_penalty_scaled = args.jerk_penalty_weight * jerk_penalty_raw
            
            # Base reward streams (used as reward-model prediction target).
            base_combined_rewards = (
                task_r_scaled
                + disc_r_scaled
                + temporal_alignment_reward_scaled
                + action_magnitude_reward_scaled
                + velocity_penalty_scaled
                + acceleration_penalty_scaled
                + jerk_penalty_scaled
            )

            # Exploration reward from prediction uncertainty (PPO-only, not reward-model target).
            b_obs_rollout = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_actions_rollout = actions.reshape((-1,) + envs.single_action_space.shape)
            b_ssl_current_positions_rollout = ssl_current_positions.reshape(-1, 4)
            b_ssl_next_positions_rollout = ssl_next_positions.reshape(-1, 4)
            b_base_rewards_rollout = base_combined_rewards.reshape(-1)

            rollout_latent = state_encoder(b_obs_rollout)
            rollout_pred_reward = reward_head(rollout_latent, b_actions_rollout)
            reward_pred_l1 = torch.abs(rollout_pred_reward - b_base_rewards_rollout)
            reward_pred_mse = (rollout_pred_reward - b_base_rewards_rollout) ** 2

            rollout_pred_delta = dynamics_head(
                rollout_latent,
                b_actions_rollout,
                b_ssl_current_positions_rollout,
            )
            rollout_pred_next_positions = b_ssl_current_positions_rollout + rollout_pred_delta
            position_abs_error = torch.abs(rollout_pred_next_positions - b_ssl_next_positions_rollout)
            position_pred_l1 = position_abs_error.mean(dim=-1)
            position_pred_mse = (rollout_pred_next_positions - b_ssl_next_positions_rollout).pow(2).mean(dim=-1)

            combined_prediction_error_l1 = reward_pred_l1 + position_pred_l1
            if args.exploration_error_deadzone > 0.0:
                combined_prediction_error_l1 = torch.where(
                    combined_prediction_error_l1 < args.exploration_error_deadzone,
                    torch.zeros_like(combined_prediction_error_l1),
                    combined_prediction_error_l1,
                )

            if args.exploration_norm_update_online:
                exploration_error_normalizer.update(combined_prediction_error_l1.unsqueeze(-1))
            exploration_std = torch.sqrt(exploration_error_normalizer.var).clamp_min(
                args.exploration_norm_eps
            )
            exploration_reward_raw = combined_prediction_error_l1 / exploration_std.squeeze(0)
            if args.exploration_norm_clip is not None and args.exploration_norm_clip > 0:
                exploration_reward_raw = torch.clamp(
                    exploration_reward_raw, -args.exploration_norm_clip, args.exploration_norm_clip
                )
            exploration_reward_raw = exploration_reward_raw.reshape(args.num_steps, args.num_envs)
            exploration_reward_scaled = args.exploration_reward_weight * exploration_reward_raw

            # PPO targets include exploration bonus.
            combined_rewards = base_combined_rewards + exploration_reward_scaled
            
            # Compute advantages with combined rewards
            advantages = torch.zeros_like(combined_rewards).to(args.device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = combined_rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            
            log_reward_stream_stats(
                writer=writer,
                global_step=global_step,
                task_r_raw=task_r_raw,
                task_r_scaled=task_r_scaled,
                temporal_alignment_reward_raw=temporal_alignment_reward_raw,
                temporal_alignment_reward_scaled=temporal_alignment_reward_scaled,
                action_magnitude_reward_raw=action_magnitude_reward_raw,
                action_magnitude_reward_scaled=action_magnitude_reward_scaled,
                combined_rewards=combined_rewards,
                advantages=advantages,
                values=values,
                use_short_discriminator_reward=use_short_discriminator_reward,
                disc_r_raw=disc_r_raw,
                disc_r_scaled=disc_r_scaled,
            )
            writer.add_scalar("ssl/reward_pred_l1_mean", reward_pred_l1.mean().item(), global_step)
            writer.add_scalar("ssl/reward_pred_mse_mean", reward_pred_mse.mean().item(), global_step)
            writer.add_scalar("ssl/position_pred_l1_mean", position_pred_l1.mean().item(), global_step)
            writer.add_scalar("ssl/position_pred_mse_mean", position_pred_mse.mean().item(), global_step)
            writer.add_scalar(
                "ssl/combined_prediction_error_l1_mean",
                combined_prediction_error_l1.mean().item(),
                global_step,
            )
            writer.add_scalar(
                "ssl/exploration_reward_raw_mean", exploration_reward_raw.mean().item(), global_step
            )
            writer.add_scalar(
                "ssl/exploration_reward_scaled_mean", exploration_reward_scaled.mean().item(), global_step
            )
            writer.add_scalar(
                "ssl/exploration_error_std",
                float(exploration_std.detach().cpu().numpy().reshape(-1)[0]),
                global_step,
            )
            writer.add_scalar(
                "ssl/exploration_error_count",
                exploration_error_normalizer.count.item(),
                global_step,
            )
            log_motion_penalty_stats(
                writer=writer,
                global_step=global_step,
                velocity_mag=velocity_mag_rollout,
                acceleration_mag=acceleration_mag_rollout,
                jerk_mag=jerk_mag_rollout,
                velocity_penalty_raw=velocity_penalty_raw,
                acceleration_penalty_raw=acceleration_penalty_raw,
                jerk_penalty_raw=jerk_penalty_raw,
                velocity_penalty_scaled=velocity_penalty_scaled,
                acceleration_penalty_scaled=acceleration_penalty_scaled,
                jerk_penalty_scaled=jerk_penalty_scaled,
                motion_normalizer=motion_normalizer,
            )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_policy_last_actions = policy_last_actions.reshape((-1, action_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_immediate_combined_rewards = base_combined_rewards.reshape(-1)
        b_ssl_current_positions = ssl_current_positions.reshape(-1, 4)
        b_ssl_next_positions = ssl_next_positions.reshape(-1, 4)
        b_ssl_delta_targets = b_ssl_next_positions - b_ssl_current_positions

        with torch.no_grad():
            latent_snapshot = state_encoder(b_obs)
            writer.add_scalar("ssl/latent_mean", latent_snapshot.mean().item(), global_step)
            writer.add_scalar("ssl/latent_std", latent_snapshot.std(unbiased=False).item(), global_step)
            writer.add_scalar("ssl/latent_norm_mean", torch.norm(latent_snapshot, dim=-1).mean().item(), global_step)
            

        # SSL-only optimization loop (separate from PPO updates).
        ssl_minibatch_size = (
            args.ssl_minibatch_size if args.ssl_minibatch_size > 0 else args.minibatch_size
        )
        if ssl_minibatch_size <= 0:
            raise ValueError("ssl_minibatch_size must be > 0 after fallback.")

        ssl_reward_loss_values = []
        ssl_dynamics_delta_loss_values = []
        ssl_dynamics_next_pos_loss_values = []
        for _ in range(args.ssl_update_epochs):
            ssl_inds = np.arange(args.batch_size)
            np.random.shuffle(ssl_inds)
            for start in range(0, args.batch_size, ssl_minibatch_size):
                end = start + ssl_minibatch_size
                mb_inds = ssl_inds[start:end]

                mb_latent = state_encoder(b_obs[mb_inds])
                pred_reward = reward_head(mb_latent, b_actions[mb_inds])
                mb_reward_pred_loss = torch.mean(
                    (pred_reward - b_immediate_combined_rewards[mb_inds]) ** 2
                )

                pred_delta = dynamics_head(
                    mb_latent, b_actions[mb_inds], b_ssl_current_positions[mb_inds]
                )
                mb_dynamics_delta_loss = torch.mean(
                    (pred_delta - b_ssl_delta_targets[mb_inds]) ** 2
                )
                pred_next_positions = b_ssl_current_positions[mb_inds] + pred_delta
                mb_dynamics_next_pos_loss = torch.mean(
                    (pred_next_positions - b_ssl_next_positions[mb_inds]) ** 2
                )

                ssl_loss = (
                    args.ssl_reward_loss_weight * mb_reward_pred_loss
                    + args.ssl_dynamics_loss_weight * mb_dynamics_delta_loss
                )
                optimizer.zero_grad()
                ssl_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(state_encoder.parameters())
                    + list(reward_head.parameters())
                    + list(dynamics_head.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

                ssl_reward_loss_values.append(mb_reward_pred_loss.detach().item())
                ssl_dynamics_delta_loss_values.append(mb_dynamics_delta_loss.detach().item())
                ssl_dynamics_next_pos_loss_values.append(mb_dynamics_next_pos_loss.detach().item())

        if ssl_reward_loss_values:
            reward_pred_loss = torch.tensor(
                float(np.mean(ssl_reward_loss_values)), device=args.device
            )
            dynamics_delta_loss = torch.tensor(
                float(np.mean(ssl_dynamics_delta_loss_values)), device=args.device
            )
            dynamics_next_pos_loss = torch.tensor(
                float(np.mean(ssl_dynamics_next_pos_loss_values)), device=args.device
            )


        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        bc_kl_loss = torch.zeros((), device=args.device)
        reward_pred_loss = torch.zeros((), device=args.device)
        dynamics_delta_loss = torch.zeros((), device=args.device)
        dynamics_next_pos_loss = torch.zeros((), device=args.device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_latent = state_encoder(b_obs[mb_inds])
                # Optional stop-gradient path: PPO/BC losses optimize only the policy/value heads,
                # while SSL heads can still update the shared encoder.
                mb_policy_latent = (
                    mb_latent.detach() if args.ppo_stop_grad_state_encoder else mb_latent
                )
                mb_policy_obs = build_policy_observation_from_latent(
                    mb_policy_latent,
                    b_policy_last_actions[mb_inds],
                    args.use_last_action_in_policy_state,
                )
                _, newlogprob, _, newvalue = agent.get_action_and_value(
                    mb_policy_obs, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = (-newlogprob).mean() # unbiased estimate of entropy
                # PPO + BC-KL objective:
                # L_total(theta) = L_PPO(theta) + lambda_BC * E_s[ KL(pi_BC(.|s) || pi_theta(.|s)) ].
                # Minibatch estimate uses sampled rollout states s_i:
                # (1/M) * sum_i KL(pi_BC(.|s_i) || pi_theta(.|s_i)).
                if bc_policy is not None and current_bc_kl_weight > 0.0:
                    with torch.no_grad():
                        bc_mean, bc_logstd = bc_policy.get_action_mean_and_logstd(mb_policy_obs)
                    student_mean, student_logstd = agent.get_action_mean_and_logstd(mb_policy_obs)
                    bc_var = torch.exp(2.0 * bc_logstd)
                    student_var = torch.exp(2.0 * student_logstd)
                    bc_kl_loss = (
                        student_logstd - bc_logstd
                        + (bc_var + (bc_mean - student_mean) ** 2) / (2.0 * student_var)
                        - 0.5
                    ).sum(dim=-1).mean()
                else:
                    bc_kl_loss = torch.zeros((), device=args.device)

                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + current_bc_kl_weight * bc_kl_loss
                )
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.parameters())
                    + list(state_encoder.parameters())
                    + list(reward_head.parameters())
                    + list(dynamics_head.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        
        
        # AMP: Train short discriminator
        if use_short_discriminator_reward:
            disc_metrics = None
            b_valid = valid_transition.reshape(-1)
            for i in range(args.num_discriminator_updates):
                step_metrics = train_discriminator_step(
                    demo_loader=demo_loader,
                    disc_batch_size=args.disc_batch_size,
                    b_disc_obs=b_disc_obs,
                    b_valid=b_valid,
                    replay_buffer=replay_buffer,
                    replay_samples=args.disc_replay_samples,
                    disc_normalizer=disc_normalizer,
                    discriminator=discriminator,
                    disc_optimizer=disc_optimizer,
                    grad_penalty_weight=args.disc_grad_penalty,
                    logit_reg_weight=args.disc_logit_reg,
                    max_grad_norm=args.max_grad_norm,
                    device=args.device,
                )
                if step_metrics is None:
                    break
                disc_metrics = step_metrics

            if disc_metrics is not None:
                # Log short discriminator metrics
                log_discriminator_metrics(
                    writer=writer,
                    prefix="amp",
                    metrics=disc_metrics,
                    replay_buffer_size=len(replay_buffer),
                    global_step=global_step,
                )
            else:
                writer.add_scalar("amp/disc_update_skipped", 1.0, global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/bc_kl_weight", current_bc_kl_weight, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/bc_kl_loss", bc_kl_loss.item(), global_step)
        writer.add_scalar("ssl/reward_pred_loss", reward_pred_loss.item(), global_step)
        writer.add_scalar("ssl/dynamics_delta_loss", dynamics_delta_loss.item(), global_step)
        writer.add_scalar("ssl/dynamics_next_pos_mse", dynamics_next_pos_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Calculate and log episodic return statistics for this iteration
        if episodic_returns:
            avg_return = np.mean(episodic_returns)
            min_return = np.min(episodic_returns)
            max_return = np.max(episodic_returns)
            
            print(f"Iteration {iteration}: Avg Return: {avg_return:.2f}, Min Return: {min_return:.2f}, Max Return: {max_return:.2f}")
            print(f"Iteration {iteration}: Avg Success Rate: {np.mean(success_rates):.2f}, Max Success Rate: {np.max(success_rates):.2f}")
            writer.add_scalar("charts/avg_episodic_return", avg_return, iteration)
            writer.add_scalar("charts/min_episodic_return", min_return, iteration)
            writer.add_scalar("charts/max_episodic_return", max_return, iteration)
            writer.add_scalar("charts/avg_success_rate", np.mean(success_rates), iteration)
            writer.add_scalar("charts/max_success_rate", np.max(success_rates), iteration)
            episodic_returns = []
            success_rates = []
        else:
            min_return = 0.0
            max_return = 0.0
            avg_return = 0.0
            print(f"Iteration {iteration}: No episodes completed")
        
        # Calculate and log motion statistics
        if velocity_magnitudes:
            avg_vel_mag = np.mean(velocity_magnitudes)
            avg_acc_mag = np.mean(acceleration_magnitudes) 
            avg_jerk_mag = np.mean(jerk_magnitudes)
            
            print(f"Iteration {iteration}: Avg Velocity Mag: {avg_vel_mag:.4f}, Avg Acceleration Mag: {avg_acc_mag:.4f}, Avg Jerk Mag: {avg_jerk_mag:.4f}")
            
            writer.add_scalar("motion/avg_velocity_magnitude", avg_vel_mag, iteration)
            writer.add_scalar("motion/avg_acceleration_magnitude", avg_acc_mag, iteration)
            writer.add_scalar("motion/avg_jerk_magnitude", avg_jerk_mag, iteration)
            
            # Clear lists for next iteration
            velocity_magnitudes.clear()
            acceleration_magnitudes.clear()
            jerk_magnitudes.clear()

        if iteration % 10 == 0 or min_return >= 5000: # start cherry-picking good policies
            # save a checkpoint of the model
            # create a subfolder for the checkpoint
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(agent.state_dict(), model_path)
            torch.save(state_encoder.state_dict(), f"{checkpoint_dir}/state_encoder.pth")
            torch.save(reward_head.state_dict(), f"{checkpoint_dir}/reward_head.pth")
            torch.save(dynamics_head.state_dict(), f"{checkpoint_dir}/dynamics_head.pth")
            
            # Save short AMP components in checkpoint when short discriminator reward is active
            if use_short_discriminator_reward:
                save_amp_components(checkpoint_dir, discriminator, disc_normalizer, replay_buffer)

            # evaluate the model with latent-conditioned policy input
            evaluate_ssl_agent(
                agent=agent,
                state_encoder=state_encoder,
                save_dir=checkpoint_dir,
                air_hockey_params=config["air_hockey"],
                n_eps=4,
                n_gifs=1,
                reference_states=reference_states,
                ref_max_episode_steps=args.ref_max_episode_steps if args.use_reference_state_init else None,
                use_last_action_in_policy_state=args.use_last_action_in_policy_state,
                device=args.device,
            )
            
            print(f"Iteration {iteration} complete")

    # save model
    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")
    torch.save(state_encoder.state_dict(), f"{log_parent_dir}/state_encoder.pth")
    torch.save(reward_head.state_dict(), f"{log_parent_dir}/reward_head.pth")
    torch.save(dynamics_head.state_dict(), f"{log_parent_dir}/dynamics_head.pth")
    
    # Save AMP components when discriminator reward is active
    if use_short_discriminator_reward:
        save_amp_components(log_parent_dir, discriminator, disc_normalizer, replay_buffer)
        print(f"✓ Saved short discriminator and AMP components")

    # evaluate the model and save results
    evaluate_ssl_agent(
        agent=agent,
        state_encoder=state_encoder,
        save_dir=log_parent_dir,
        air_hockey_params=config["air_hockey"],
        n_eps=5,
        n_gifs=3,
        reference_states=reference_states,
        ref_max_episode_steps=args.ref_max_episode_steps if args.use_reference_state_init else None,
        use_last_action_in_policy_state=args.use_last_action_in_policy_state,
        device=args.device,
    )
    
    # end of training