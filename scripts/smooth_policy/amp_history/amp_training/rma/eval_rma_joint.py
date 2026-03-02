import csv
import os
import random
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.amp_history.amp_training.rma.rma_adaptation import RMAAdaptationModule
from scripts.smooth_policy.encoder import EnvEncoder


def augment_policy_observation(observation, last_action, use_last_action):
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def concat_env_latent_to_policy_obs(policy_obs_base, env_latent):
    return torch.cat([policy_obs_base, env_latent], dim=-1)


def get_env_spec_ranges():
    return {
        "paddle_density": (2500 * 0.8, 2500 * 1.2),
        "paddle_damping": (3 * 0.8, 3 * 1.2),
        "puck_density": (250 * 0.8, 250 * 1.2),
        "puck_damping": (0.5 * 0.8, 0.5 * 1.2),
        "force_scaling": (1 * 0.8, 1 * 1.2),
    }


def get_env_spec_ordered_keys():
    return [
        "paddle_density",
        "paddle_damping",
        "puck_density",
        "puck_damping",
        "force_scaling",
    ]


def _load_yaml(path):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError:
            f.seek(0)
            return yaml.load(f, Loader=yaml.FullLoader)


def _coerce_env_spec_ranges(raw_ranges):
    if raw_ranges is None:
        return get_env_spec_ranges()

    ordered_keys = get_env_spec_ordered_keys()
    out = {}
    for key in ordered_keys:
        if key not in raw_ranges:
            raise ValueError(
                f"randomization_ranges missing key '{key}'. Expected keys: {ordered_keys}"
            )
        bounds = raw_ranges[key]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Invalid range for '{key}': {bounds}. Expected [low, high].")
        low = float(bounds[0])
        high = float(bounds[1])
        if not high > low:
            raise ValueError(f"Invalid range for '{key}': low={low}, high={high}.")
        out[key] = (low, high)
    return out


def load_randomization_setup(setup_path):
    input_path = os.path.abspath(setup_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"env_randomization_setup_path does not exist: '{setup_path}'")
    manifest = _load_yaml(input_path) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid randomization setup format in '{input_path}'.")
    ranges = _coerce_env_spec_ranges(manifest.get("randomization_ranges"))
    return manifest, ranges, input_path


def _validate_manifest_compatibility(manifest, args):
    env_repr = manifest.get("env_var_representation", {}) if isinstance(manifest, dict) else {}
    manifest_env_var_dim = env_repr.get("env_var_dim")
    manifest_env_latent_dim = env_repr.get("env_latent_dim")
    manifest_order = env_repr.get("ordered_keys")

    if manifest_env_var_dim is not None and int(manifest_env_var_dim) != int(args.env_var_dim):
        raise ValueError(
            f"env_var_dim mismatch: args={args.env_var_dim}, manifest={manifest_env_var_dim}. "
            "Use matching dimensions from stage-1 training."
        )
    if manifest_env_latent_dim is not None and int(manifest_env_latent_dim) != int(args.env_latent_dim):
        raise ValueError(
            f"env_latent_dim mismatch: args={args.env_latent_dim}, manifest={manifest_env_latent_dim}. "
            "Use matching dimensions from stage-1 training."
        )
    if manifest_order is not None and list(manifest_order) != get_env_spec_ordered_keys():
        raise ValueError(
            f"ordered_keys mismatch: manifest={manifest_order}, expected={get_env_spec_ordered_keys()}."
        )


def sample_env_spec_from_ranges(rng, env_id, env_spec_ranges):
    return {
        "env_id": int(env_id),
        "name": f"env_{int(env_id)}",
        "paddle_density": float(rng.uniform(*env_spec_ranges["paddle_density"])),
        "paddle_damping": float(rng.uniform(*env_spec_ranges["paddle_damping"])),
        "puck_density": float(rng.uniform(*env_spec_ranges["puck_density"])),
        "puck_damping": float(rng.uniform(*env_spec_ranges["puck_damping"])),
        "force_scaling": float(rng.uniform(*env_spec_ranges["force_scaling"])),
    }


def choose_eval_env_specs(args, env_spec_ranges):
    rng = np.random.default_rng(args.eval_env_seed)
    if args.num_eval_env_specs <= 0:
        raise ValueError("num_eval_env_specs must be > 0.")
    selected = [
        sample_env_spec_from_ranges(
            rng=rng,
            env_id=i,
            env_spec_ranges=env_spec_ranges,
        )
        for i in range(args.num_eval_env_specs)
    ]
    return selected, "randomized_from_ranges"


def extract_env_var_vector_from_spec(spec, env_var_dim, env_spec_ranges):
    ranges = _coerce_env_spec_ranges(env_spec_ranges)
    ordered_keys = get_env_spec_ordered_keys()
    normalized = []
    for key in ordered_keys:
        value = float(spec[key])
        low, high = ranges[key]
        mean = 0.5 * (low + high)
        std = (high - low) / np.sqrt(12.0)
        if std <= 1e-8:
            std = 1.0
        normalized.append((value - mean) / std)
    vec = np.zeros(env_var_dim, dtype=np.float32)
    base = np.asarray(normalized, dtype=np.float32)
    copy_len = min(env_var_dim, len(base))
    vec[:copy_len] = base[:copy_len]
    return vec


def apply_env_spec_to_unwrapped_env(env, env_spec):
    env.unwrapped.paddle_density = env_spec["paddle_density"]
    env.unwrapped.paddle_damping = env_spec["paddle_damping"]
    env.unwrapped.puck_density = env_spec["puck_density"]
    env.unwrapped.puck_damping = env_spec["puck_damping"]
    env.unwrapped.force_scaling = env_spec["force_scaling"]


def _summarize_array(values):
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _adaptation_window(hist_actions, hist_states, context_len, device):
    # Build left-padded fixed-length windows for adaptation conv stack.
    fixed_context_len = 100
    action_dim = hist_actions.shape[1]
    state_dim = hist_states.shape[1]
    ctx = min(context_len, fixed_context_len)

    action_seq = hist_actions[-ctx:]
    state_seq = hist_states[-ctx:]
    valid_mask = torch.ones(ctx, dtype=torch.float32, device=device)
    if ctx < fixed_context_len:
        pad = fixed_context_len - ctx
        action_pad = action_seq[:1].repeat(pad, 1)
        state_pad = state_seq[:1].repeat(pad, 1)
        action_seq = torch.cat([action_pad, action_seq], dim=0)
        state_seq = torch.cat([state_pad, state_seq], dim=0)
        valid_mask = torch.cat(
            [
                torch.zeros(pad, dtype=torch.float32, device=device),
                valid_mask,
            ],
            dim=0,
        )

    action_seq = action_seq.reshape(1, fixed_context_len, action_dim)
    state_seq = state_seq.reshape(1, fixed_context_len, state_dim)
    valid_mask = valid_mask.reshape(1, fixed_context_len)
    return action_seq, state_seq, valid_mask


def _render_eval_frame(renderer, step_reward, episode_return, label):
    frame = renderer.get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    aspect_ratio = frame.shape[1] / frame.shape[0]
    frame = cv2.resize(frame, (200, int(200 / max(1e-8, aspect_ratio))))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (8, 18), font, 0.45, (0, 0, 0), 1)
    cv2.putText(frame, f"r: {step_reward:.2f}", (8, 36), font, 0.42, (0, 0, 0), 1)
    cv2.putText(frame, f"R: {episode_return:.2f}", (8, 54), font, 0.42, (0, 0, 0), 1)
    return frame


def _save_gif(frames, gif_path, fps):
    if len(frames) == 0:
        return
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    duration_ms = int(1000 * (1.0 / max(1e-8, fps)))
    imageio.mimsave(gif_path, frames, format="GIF", loop=0, duration=duration_ms)


def _pad_frames(frames, target_len):
    if len(frames) == 0:
        return frames
    if len(frames) >= target_len:
        return frames
    return frames + [frames[-1]] * (target_len - len(frames))


def _combine_side_by_side(frames_left, frames_right):
    if len(frames_left) == 0 or len(frames_right) == 0:
        return []
    max_len = max(len(frames_left), len(frames_right))
    left = _pad_frames(frames_left, max_len)
    right = _pad_frames(frames_right, max_len)
    out = []
    for lf, rf in zip(left, right):
        if lf.shape[0] != rf.shape[0]:
            target_h = min(lf.shape[0], rf.shape[0])
            lf = cv2.resize(lf, (int(lf.shape[1] * target_h / lf.shape[0]), target_h))
            rf = cv2.resize(rf, (int(rf.shape[1] * target_h / rf.shape[0]), target_h))
        out.append(np.concatenate([lf, rf], axis=1))
    return out


def evaluate_single_env_spec_rma(
    args,
    env_spec,
    env_spec_ranges,
    config_air_hockey,
    agent,
    env_encoder,
    adaptation_module,
    device,
):
    env = AirHockeyEnv(dict(config_air_hockey))
    apply_env_spec_to_unwrapped_env(env, env_spec)
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False) if args.save_gifs else None
    action_dim = int(np.prod(env.action_space.shape))
    obs_dim = int(np.prod(env.observation_space.shape))

    env_var_np = extract_env_var_vector_from_spec(
        spec=env_spec,
        env_var_dim=args.env_var_dim,
        env_spec_ranges=env_spec_ranges,
    )
    env_var_t = torch.tensor(env_var_np, dtype=torch.float32, device=device).unsqueeze(0)

    episode_rows = []
    episode_returns = []
    episode_lengths = []
    episode_success = []
    vel_avgs = []
    acc_avgs = []
    jerk_avgs = []
    latent_encoder_fracs = []
    latent_adaptation_fracs = []
    gif_frames_by_episode = {}

    for episode_idx in range(args.episodes_per_env):
        obs_np, _ = env.reset(seed=args.seed + episode_idx)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        last_action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
        episode_frames = []

        hist_actions = torch.zeros((0, action_dim), dtype=torch.float32, device=device)
        hist_states = torch.zeros((0, obs_dim), dtype=torch.float32, device=device)

        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = 0.0
        encoder_steps = 0
        adaptation_steps = 0

        while not done:
            hist_states = torch.cat([hist_states, obs_t], dim=0)
            context_len = int(hist_actions.shape[0])
            use_encoder = (ep_len < args.warmstart_steps) or (context_len < args.adaptation_min_context)

            with torch.no_grad():
                if use_encoder:
                    latent = env_encoder(env_var_t)
                    encoder_steps += 1
                    latent_source = "enc"
                else:
                    a_seq, s_seq, v_mask = _adaptation_window(
                        hist_actions=hist_actions,
                        hist_states=hist_states[:-1],
                        context_len=context_len,
                        device=device,
                    )
                    latent = adaptation_module(a_seq, s_seq, valid_mask=v_mask)
                    adaptation_steps += 1
                    latent_source = "ada"

                policy_obs_base = augment_policy_observation(
                    obs_t, last_action, args.use_last_action_in_policy_state
                )
                policy_obs = concat_env_latent_to_policy_obs(policy_obs_base, latent)
                action, _, _, _ = agent.get_action_and_value(policy_obs)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1
            ep_success = max(ep_success, 1.0 if bool(info.get("success", False)) else 0.0)
            if args.save_gifs and episode_idx < args.gif_episodes_per_env and ep_len <= args.gif_max_steps:
                episode_frames.append(_render_eval_frame(renderer, float(reward), ep_return, f"RMA ({latent_source})"))

            ep_motion = info.get("motion_data")
            if done and isinstance(ep_motion, dict):
                if "velocity_mags" in ep_motion:
                    vel_avgs.append(float(np.mean(ep_motion["velocity_mags"])) if len(ep_motion["velocity_mags"]) > 0 else 0.0)
                if "acceleration_mags" in ep_motion:
                    acc_avgs.append(float(np.mean(ep_motion["acceleration_mags"])) if len(ep_motion["acceleration_mags"]) > 0 else 0.0)
                if "jerk_mags" in ep_motion:
                    jerk_avgs.append(float(np.mean(ep_motion["jerk_mags"])) if len(ep_motion["jerk_mags"]) > 0 else 0.0)

            hist_actions = torch.cat([hist_actions, action.detach()], dim=0)
            obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            last_action = action.detach()
            if done:
                last_action.zero_()

        total_steps = max(1, encoder_steps + adaptation_steps)
        encoder_frac = float(encoder_steps / total_steps)
        adaptation_frac = float(adaptation_steps / total_steps)

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_success.append(ep_success)
        latent_encoder_fracs.append(encoder_frac)
        latent_adaptation_fracs.append(adaptation_frac)

        episode_rows.append(
            {
                "agent": "rma",
                "env_id": int(env_spec["env_id"]),
                "env_name": env_spec.get("name", f"env_{int(env_spec['env_id'])}"),
                "episode_index": int(episode_idx),
                "seed": int(args.seed + episode_idx),
                "episode_return": float(ep_return),
                "episode_length": int(ep_len),
                "success": float(ep_success),
                "latent_encoder_fraction": encoder_frac,
                "latent_adaptation_fraction": adaptation_frac,
            }
        )
        if args.save_gifs and episode_idx < args.gif_episodes_per_env:
            gif_frames_by_episode[episode_idx] = episode_frames

    env.close()
    per_env = {
        "agent": "rma",
        "env_id": int(env_spec["env_id"]),
        "env_name": env_spec.get("name", f"env_{int(env_spec['env_id'])}"),
        "env_spec": {k: float(v) if isinstance(v, (int, float)) else v for k, v in env_spec.items()},
        "episodes": int(args.episodes_per_env),
        "return": _summarize_array(episode_returns),
        "episode_length": _summarize_array(episode_lengths),
        "success_rate": float(np.mean(episode_success)) if len(episode_success) > 0 else 0.0,
        "latent_encoder_fraction": float(np.mean(latent_encoder_fracs)) if len(latent_encoder_fracs) > 0 else 0.0,
        "latent_adaptation_fraction": float(np.mean(latent_adaptation_fracs)) if len(latent_adaptation_fracs) > 0 else 0.0,
        "motion": {
            "velocity_magnitude": _summarize_array(vel_avgs),
            "acceleration_magnitude": _summarize_array(acc_avgs),
            "jerk_magnitude": _summarize_array(jerk_avgs),
        },
    }
    return per_env, episode_rows, gif_frames_by_episode


def evaluate_single_env_spec_rma_gt(args, env_spec, env_spec_ranges, config_air_hockey, agent, env_encoder, device):
    """
    Evaluate RMA policy while always conditioning on ground-truth encoder latent.
    Adaptation module is not used in this mode.
    """
    env = AirHockeyEnv(dict(config_air_hockey))
    apply_env_spec_to_unwrapped_env(env, env_spec)
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False) if args.save_gifs else None
    action_dim = int(np.prod(env.action_space.shape))

    env_var_np = extract_env_var_vector_from_spec(
        spec=env_spec,
        env_var_dim=args.env_var_dim,
        env_spec_ranges=env_spec_ranges,
    )
    env_var_t = torch.tensor(env_var_np, dtype=torch.float32, device=device).unsqueeze(0)

    episode_rows = []
    episode_returns = []
    episode_lengths = []
    episode_success = []
    vel_avgs = []
    acc_avgs = []
    jerk_avgs = []
    gif_frames_by_episode = {}

    for episode_idx in range(args.episodes_per_env):
        obs_np, _ = env.reset(seed=args.seed + episode_idx)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        last_action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
        episode_frames = []

        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = 0.0

        while not done:
            with torch.no_grad():
                latent = env_encoder(env_var_t)
                policy_obs_base = augment_policy_observation(
                    obs_t, last_action, args.use_last_action_in_policy_state
                )
                policy_obs = concat_env_latent_to_policy_obs(policy_obs_base, latent)
                action, _, _, _ = agent.get_action_and_value(policy_obs)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1
            ep_success = max(ep_success, 1.0 if bool(info.get("success", False)) else 0.0)
            if args.save_gifs and episode_idx < args.gif_episodes_per_env and ep_len <= args.gif_max_steps:
                episode_frames.append(_render_eval_frame(renderer, float(reward), ep_return, "RMA-GT"))

            ep_motion = info.get("motion_data")
            if done and isinstance(ep_motion, dict):
                if "velocity_mags" in ep_motion:
                    vel_avgs.append(float(np.mean(ep_motion["velocity_mags"])) if len(ep_motion["velocity_mags"]) > 0 else 0.0)
                if "acceleration_mags" in ep_motion:
                    acc_avgs.append(float(np.mean(ep_motion["acceleration_mags"])) if len(ep_motion["acceleration_mags"]) > 0 else 0.0)
                if "jerk_mags" in ep_motion:
                    jerk_avgs.append(float(np.mean(ep_motion["jerk_mags"])) if len(ep_motion["jerk_mags"]) > 0 else 0.0)

            obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            last_action = action.detach()
            if done:
                last_action.zero_()

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_success.append(ep_success)
        episode_rows.append(
            {
                "agent": "rma_gt",
                "env_id": int(env_spec["env_id"]),
                "env_name": env_spec.get("name", f"env_{int(env_spec['env_id'])}"),
                "episode_index": int(episode_idx),
                "seed": int(args.seed + episode_idx),
                "episode_return": float(ep_return),
                "episode_length": int(ep_len),
                "success": float(ep_success),
                "latent_encoder_fraction": 1.0,
                "latent_adaptation_fraction": 0.0,
            }
        )
        if args.save_gifs and episode_idx < args.gif_episodes_per_env:
            gif_frames_by_episode[episode_idx] = episode_frames

    env.close()
    per_env = {
        "agent": "rma_gt",
        "env_id": int(env_spec["env_id"]),
        "env_name": env_spec.get("name", f"env_{int(env_spec['env_id'])}"),
        "env_spec": {k: float(v) if isinstance(v, (int, float)) else v for k, v in env_spec.items()},
        "episodes": int(args.episodes_per_env),
        "return": _summarize_array(episode_returns),
        "episode_length": _summarize_array(episode_lengths),
        "success_rate": float(np.mean(episode_success)) if len(episode_success) > 0 else 0.0,
        "latent_encoder_fraction": 1.0,
        "latent_adaptation_fraction": 0.0,
        "motion": {
            "velocity_magnitude": _summarize_array(vel_avgs),
            "acceleration_magnitude": _summarize_array(acc_avgs),
            "jerk_magnitude": _summarize_array(jerk_avgs),
        },
    }
    return per_env, episode_rows, gif_frames_by_episode


def evaluate_single_env_spec_ppo(args, env_spec, config_air_hockey, agent, device):
    env = AirHockeyEnv(dict(config_air_hockey))
    apply_env_spec_to_unwrapped_env(env, env_spec)
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False) if args.save_gifs else None
    action_dim = int(np.prod(env.action_space.shape))

    episode_rows = []
    episode_returns = []
    episode_lengths = []
    episode_success = []
    vel_avgs = []
    acc_avgs = []
    jerk_avgs = []
    gif_frames_by_episode = {}

    for episode_idx in range(args.episodes_per_env):
        obs_np, _ = env.reset(seed=args.seed + episode_idx)
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        last_action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
        episode_frames = []

        done = False
        ep_return = 0.0
        ep_len = 0
        ep_success = 0.0

        while not done:
            with torch.no_grad():
                policy_obs = augment_policy_observation(
                    obs_t, last_action, args.ppo_use_last_action_in_policy_state
                )
                action, _, _, _ = agent.get_action_and_value(policy_obs)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1
            ep_success = max(ep_success, 1.0 if bool(info.get("success", False)) else 0.0)
            if args.save_gifs and episode_idx < args.gif_episodes_per_env and ep_len <= args.gif_max_steps:
                episode_frames.append(_render_eval_frame(renderer, float(reward), ep_return, "PPO"))

            ep_motion = info.get("motion_data")
            if done and isinstance(ep_motion, dict):
                if "velocity_mags" in ep_motion:
                    vel_avgs.append(float(np.mean(ep_motion["velocity_mags"])) if len(ep_motion["velocity_mags"]) > 0 else 0.0)
                if "acceleration_mags" in ep_motion:
                    acc_avgs.append(float(np.mean(ep_motion["acceleration_mags"])) if len(ep_motion["acceleration_mags"]) > 0 else 0.0)
                if "jerk_mags" in ep_motion:
                    jerk_avgs.append(float(np.mean(ep_motion["jerk_mags"])) if len(ep_motion["jerk_mags"]) > 0 else 0.0)

            obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            last_action = action.detach()
            if done:
                last_action.zero_()

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_success.append(ep_success)
        episode_rows.append(
            {
                "agent": "ppo",
                "env_id": int(env_spec["env_id"]),
                "env_name": env_spec.get("name", f"env_{int(env_spec['env_id'])}"),
                "episode_index": int(episode_idx),
                "seed": int(args.seed + episode_idx),
                "episode_return": float(ep_return),
                "episode_length": int(ep_len),
                "success": float(ep_success),
                "latent_encoder_fraction": 0.0,
                "latent_adaptation_fraction": 0.0,
            }
        )
        if args.save_gifs and episode_idx < args.gif_episodes_per_env:
            gif_frames_by_episode[episode_idx] = episode_frames

    env.close()
    per_env = {
        "agent": "ppo",
        "env_id": int(env_spec["env_id"]),
        "env_name": env_spec.get("name", f"env_{int(env_spec['env_id'])}"),
        "env_spec": {k: float(v) if isinstance(v, (int, float)) else v for k, v in env_spec.items()},
        "episodes": int(args.episodes_per_env),
        "return": _summarize_array(episode_returns),
        "episode_length": _summarize_array(episode_lengths),
        "success_rate": float(np.mean(episode_success)) if len(episode_success) > 0 else 0.0,
        "latent_encoder_fraction": 0.0,
        "latent_adaptation_fraction": 0.0,
        "motion": {
            "velocity_magnitude": _summarize_array(vel_avgs),
            "acceleration_magnitude": _summarize_array(acc_avgs),
            "jerk_magnitude": _summarize_array(jerk_avgs),
        },
    }
    return per_env, episode_rows, gif_frames_by_episode


def aggregate_across_envs(per_env_metrics):
    env_avg_returns = [entry["return"]["mean"] for entry in per_env_metrics]
    env_success_rates = [entry["success_rate"] for entry in per_env_metrics]
    env_avg_lengths = [entry["episode_length"]["mean"] for entry in per_env_metrics]
    env_adapt_frac = [entry["latent_adaptation_fraction"] for entry in per_env_metrics]
    env_encoder_frac = [entry["latent_encoder_fraction"] for entry in per_env_metrics]
    return {
        "num_envs": int(len(per_env_metrics)),
        "macro_return": _summarize_array(env_avg_returns),
        "macro_success_rate": _summarize_array(env_success_rates),
        "macro_episode_length": _summarize_array(env_avg_lengths),
        "macro_latent_adaptation_fraction": _summarize_array(env_adapt_frac),
        "macro_latent_encoder_fraction": _summarize_array(env_encoder_frac),
    }


def print_summary(per_env_metrics, aggregate, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_comparison_report(rma_per_env, ppo_per_env):
    ppo_by_env = {int(x["env_id"]): x for x in ppo_per_env}
    per_env_deltas = []
    for rma in rma_per_env:
        env_id = int(rma["env_id"])
        if env_id not in ppo_by_env:
            continue
        ppo = ppo_by_env[env_id]
        per_env_deltas.append(
            {
                "env_id": env_id,
                "env_name": rma["env_name"],
                "return_mean_delta_rma_minus_ppo": float(rma["return"]["mean"] - ppo["return"]["mean"]),
                "success_rate_delta_rma_minus_ppo": float(rma["success_rate"] - ppo["success_rate"]),
                "length_mean_delta_rma_minus_ppo": float(rma["episode_length"]["mean"] - ppo["episode_length"]["mean"]),
            }
        )
    return {
        "num_compared_envs": int(len(per_env_deltas)),
        "return_mean_delta_summary_rma_minus_ppo": _summarize_array(
            [x["return_mean_delta_rma_minus_ppo"] for x in per_env_deltas]
        ),
        "success_rate_delta_summary_rma_minus_ppo": _summarize_array(
            [x["success_rate_delta_rma_minus_ppo"] for x in per_env_deltas]
        ),
        "episode_length_delta_summary_rma_minus_ppo": _summarize_array(
            [x["length_mean_delta_rma_minus_ppo"] for x in per_env_deltas]
        ),
        "per_env_deltas": per_env_deltas,
    }


def _metric_mean_std_from_per_env(per_env_metrics, key):
    if len(per_env_metrics) == 0:
        return 0.0, 0.0
    if key == "return":
        values = [entry["return"]["mean"] for entry in per_env_metrics]
    elif key == "success_rate":
        values = [entry["success_rate"] for entry in per_env_metrics]
    elif key == "episode_length":
        values = [entry["episode_length"]["mean"] for entry in per_env_metrics]
    else:
        raise ValueError(f"Unsupported metric key: {key}")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def save_performance_comparison_chart(log_parent_dir, rma_per_env, rma_gt_per_env, ppo_per_env):
    """
    Save a chart comparing aggregate performance across environments.
    Error bars show std across environments.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable, skipping performance chart save. Error: {exc}")
        return

    labels = ["RMA", "RMA-GT", "PPO"]
    groups = [rma_per_env, rma_gt_per_env, ppo_per_env]
    valid = [(label, data) for label, data in zip(labels, groups) if len(data) > 0]
    if len(valid) == 0:
        print("WARNING: no per-env metrics available for charting.")
        return

    metrics = [
        ("return", "Return Across Envs"),
        ("success_rate", "Success Rate Across Envs"),
        ("episode_length", "Episode Length Across Envs"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    if len(metrics) == 1:
        axes = [axes]

    x_labels = [x[0] for x in valid]
    x = np.arange(len(valid))

    for ax, (metric_key, title) in zip(axes, metrics):
        means = []
        stds = []
        for _, data in valid:
            mean_val, std_val = _metric_mean_std_from_per_env(data, metric_key)
            means.append(mean_val)
            stds.append(std_val)
        ax.bar(x, means, yerr=stds, capsize=6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Evaluation Performance Summary (Mean ± Std Across Environments)")
    fig.tight_layout()
    out_path = os.path.join(log_parent_dir, "performance_across_envs.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved performance comparison chart: {out_path}")


@dataclass
class Args:
    # Paths/configs.
    config: str = "scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_config.yaml"
    args_file: str | None = None
    model_path: str = ""
    encoder_path: str = ""
    adaptation_path: str = ""
    env_randomization_setup_path: str = ""
    # Deprecated legacy static-eval args. Kept only so older args_file configs still parse.
    env_spec_pool_path: str = ""
    explicit_env_specs_path: str = ""
    log_parent_dir: str | None = None
    run_name: str = "rma_joint_eval"

    # Runtime + architecture compatibility.
    seed: int = 0
    eval_env_seed: int = 123
    device: str = "cuda:0"
    action_scale: float = 1.0
    agent_hidden_size: int = 512
    use_last_action_in_policy_state: bool = True
    env_var_dim: int = 8
    env_latent_dim: int = 12
    env_encoder_hidden_size: int | tuple[int, ...] = (128, 128)
    adaptation_embed_dim: int = 16
    adaptation_conv_in_channels: int = 8
    adaptation_hidden_size: int = 64
    ppo_model_path: str = ""
    ppo_agent_hidden_size: int = 256
    ppo_action_scale: float = 1.0
    ppo_use_last_action_in_policy_state: bool = True
    include_rma_with_adaptation: bool = True

    # Eval setup.
    num_eval_env_specs: int = 50
    episodes_per_env: int = 5
    warmstart_steps: int = 75
    adaptation_min_context: int = 50
    use_tensorboard: bool = True
    save_gifs: bool = True
    gif_episodes_per_env: int = 1
    gif_max_steps: int = 300
    gif_fps: int = 20


if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)

    if not args.model_path:
        raise ValueError("model_path must be provided.")
    if not args.encoder_path:
        raise ValueError("encoder_path must be provided.")
    if args.include_rma_with_adaptation and not args.adaptation_path:
        raise ValueError("adaptation_path must be provided.")
    if args.explicit_env_specs_path:
        raise ValueError(
            "explicit_env_specs_path is no longer supported. "
            "Use env_randomization_setup_path (training_env_setup.yaml) and randomized evaluation."
        )
    if not args.env_randomization_setup_path and args.env_spec_pool_path:
        legacy_input = os.path.abspath(args.env_spec_pool_path)
        if os.path.isdir(legacy_input):
            candidate = os.path.join(legacy_input, "training_env_setup.yaml")
            if not os.path.exists(candidate):
                raise ValueError(
                    "Static env pool evaluation has been removed. "
                    f"Provided env_spec_pool_path '{args.env_spec_pool_path}' is a directory "
                    "without training_env_setup.yaml."
                )
            args.env_randomization_setup_path = candidate
            print(
                "WARNING: env_spec_pool_path is deprecated. "
                f"Using training manifest: {args.env_randomization_setup_path}"
            )
        elif os.path.basename(legacy_input) == "training_env_setup.yaml":
            args.env_randomization_setup_path = legacy_input
            print(
                "WARNING: env_spec_pool_path is deprecated. "
                f"Using training manifest: {args.env_randomization_setup_path}"
            )
        else:
            raise ValueError(
                "Static env pool evaluation has been removed. "
                "Use env_randomization_setup_path pointing to training_env_setup.yaml."
            )
    if not args.env_randomization_setup_path:
        raise ValueError("env_randomization_setup_path must be provided.")
    if args.warmstart_steps <= 0:
        raise ValueError("warmstart_steps must be > 0.")
    if args.adaptation_min_context <= 0:
        raise ValueError("adaptation_min_context must be > 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    air_hockey_config = config["air_hockey"]

    randomization_setup, env_spec_ranges, resolved_setup_path = load_randomization_setup(
        args.env_randomization_setup_path
    )
    _validate_manifest_compatibility(randomization_setup, args)
    eval_specs, selection_mode = choose_eval_env_specs(args, env_spec_ranges)
    print(f"Selected {len(eval_specs)} evaluation env specs via mode={selection_mode}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = air_hockey_config.get("task", "task")
    if args.log_parent_dir is None:
        log_parent_dir = f"runs/rma_joint_eval/{task_name}/{args.run_name}_{timestamp}"
    else:
        log_parent_dir = args.log_parent_dir
    if os.path.exists(log_parent_dir):
        base = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base}r{i}"
            i += 1
    os.makedirs(log_parent_dir, exist_ok=True)

    with open(os.path.join(log_parent_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    with open(os.path.join(log_parent_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)
    with open(os.path.join(log_parent_dir, "resolved_eval_env_specs.yaml"), "w") as f:
        yaml.dump(eval_specs, f, sort_keys=False)
    with open(os.path.join(log_parent_dir, "eval_env_selection.yaml"), "w") as f:
        yaml.dump(
            {
                "selection_mode": selection_mode,
                "env_randomization_setup_path": os.path.abspath(args.env_randomization_setup_path),
                "resolved_env_randomization_setup_path": resolved_setup_path,
                "env_spec_pool_source": randomization_setup.get("env_spec_pool_source"),
                "env_spec_pool_source_path": randomization_setup.get("env_spec_pool_source_path"),
                "randomization_ranges": {k: [float(v[0]), float(v[1])] for k, v in env_spec_ranges.items()},
                "num_eval_env_specs": int(len(eval_specs)),
                "eval_env_seed": int(args.eval_env_seed),
            },
            f,
            sort_keys=False,
        )

    device = torch.device(args.device)

    # Build model shapes from a probe environment.
    probe_env = AirHockeyEnv(dict(air_hockey_config))
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    action_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    base_policy_obs_dim = obs_dim + action_dim if args.use_last_action_in_policy_state else obs_dim
    policy_obs_dim = base_policy_obs_dim + args.env_latent_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(action_dim,),
            dtype=np.float32,
        ),
    )

    rma_agent = Agent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_size=args.agent_hidden_size,
    ).to(device)
    env_encoder = EnvEncoder(
        env_var_dim=args.env_var_dim,
        latent_dim=args.env_latent_dim,
        hidden_size=args.env_encoder_hidden_size,
    ).to(device)
    rma_agent.load_state_dict(torch.load(args.model_path, map_location=device))
    env_encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    rma_agent.eval()
    env_encoder.eval()
    adaptation_module = None
    if args.include_rma_with_adaptation:
        adaptation_module = RMAAdaptationModule(
            action_dim=action_dim,
            state_dim=obs_dim,
            embed_dim=args.adaptation_embed_dim,
            conv_in_channels=args.adaptation_conv_in_channels,
            latent_dim=args.env_latent_dim,
            hidden_size=args.adaptation_hidden_size,
        ).to(device)
        adaptation_module.load_state_dict(torch.load(args.adaptation_path, map_location=device))
        adaptation_module.eval()

    ppo_agent = None
    if args.ppo_model_path:
        ppo_policy_obs_dim = obs_dim + (action_dim if args.ppo_use_last_action_in_policy_state else 0)
        ppo_policy_env_view = SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(ppo_policy_obs_dim,),
                dtype=np.float32,
            ),
            single_action_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(action_dim,),
                dtype=np.float32,
            ),
        )
        ppo_agent = Agent(
            ppo_policy_env_view,
            action_scale=args.ppo_action_scale,
            action_bias=0.0,
            hidden_size=args.ppo_agent_hidden_size,
        ).to(device)
        ppo_agent.load_state_dict(torch.load(args.ppo_model_path, map_location=device))
        ppo_agent.eval()

    writer = SummaryWriter(log_parent_dir) if args.use_tensorboard else None
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
        )

    rma_per_env_metrics = []
    rma_episode_rows = []
    rma_gt_per_env_metrics = []
    rma_gt_episode_rows = []
    ppo_per_env_metrics = []
    ppo_episode_rows = []
    for idx, spec in enumerate(eval_specs):
        rma_per_env = None
        rma_gifs = {}
        if args.include_rma_with_adaptation:
            rma_per_env, rma_rows, rma_gifs = evaluate_single_env_spec_rma(
                args=args,
                env_spec=spec,
                env_spec_ranges=env_spec_ranges,
                config_air_hockey=air_hockey_config,
                agent=rma_agent,
                env_encoder=env_encoder,
                adaptation_module=adaptation_module,
                device=device,
            )
            rma_per_env_metrics.append(rma_per_env)
            rma_episode_rows.extend(rma_rows)

            if args.save_gifs:
                env_name = rma_per_env["env_name"]
                for ep_i, frames in rma_gifs.items():
                    _save_gif(
                        frames,
                        os.path.join(log_parent_dir, "gifs", "rma", f"env_{rma_per_env['env_id']}_{env_name}_ep{ep_i}.gif"),
                        args.gif_fps,
                    )

        rma_gt_per_env, rma_gt_rows, rma_gt_gifs = evaluate_single_env_spec_rma_gt(
            args=args,
            env_spec=spec,
            env_spec_ranges=env_spec_ranges,
            config_air_hockey=air_hockey_config,
            agent=rma_agent,
            env_encoder=env_encoder,
            device=device,
        )
        rma_gt_per_env_metrics.append(rma_gt_per_env)
        rma_gt_episode_rows.extend(rma_gt_rows)

        if args.save_gifs:
            env_name = rma_gt_per_env["env_name"]
            for ep_i, frames in rma_gt_gifs.items():
                _save_gif(
                    frames,
                    os.path.join(log_parent_dir, "gifs", "rma_gt", f"env_{rma_gt_per_env['env_id']}_{env_name}_ep{ep_i}.gif"),
                    args.gif_fps,
                )

        ppo_gifs = {}
        ppo_per_env = None
        if ppo_agent is not None:
            ppo_per_env, ppo_rows, ppo_gifs = evaluate_single_env_spec_ppo(
                args=args,
                env_spec=spec,
                config_air_hockey=air_hockey_config,
                agent=ppo_agent,
                device=device,
            )
            ppo_per_env_metrics.append(ppo_per_env)
            ppo_episode_rows.extend(ppo_rows)
            if args.save_gifs:
                env_name = ppo_per_env["env_name"]
                for ep_i, frames in ppo_gifs.items():
                    _save_gif(
                        frames,
                        os.path.join(log_parent_dir, "gifs", "ppo", f"env_{ppo_per_env['env_id']}_{env_name}_ep{ep_i}.gif"),
                        args.gif_fps,
                    )
                if rma_per_env is not None:
                    for ep_i in range(min(args.gif_episodes_per_env, args.episodes_per_env)):
                        if ep_i in rma_gifs and ep_i in ppo_gifs:
                            side_frames = _combine_side_by_side(rma_gifs[ep_i], ppo_gifs[ep_i])
                            _save_gif(
                                side_frames,
                                os.path.join(
                                    log_parent_dir,
                                    "gifs",
                                    "side_by_side",
                                    f"env_{rma_per_env['env_id']}_{rma_per_env['env_name']}_ep{ep_i}.gif",
                                ),
                                args.gif_fps,
                            )

        if writer is not None:
            if rma_per_env is not None:
                writer.add_scalar("eval_rma/per_env_return_mean", rma_per_env["return"]["mean"], idx)
                writer.add_scalar("eval_rma/per_env_success_rate", rma_per_env["success_rate"], idx)
                writer.add_scalar("eval_rma/per_env_adaptation_fraction", rma_per_env["latent_adaptation_fraction"], idx)
            writer.add_scalar("eval_rma_gt/per_env_return_mean", rma_gt_per_env["return"]["mean"], idx)
            writer.add_scalar("eval_rma_gt/per_env_success_rate", rma_gt_per_env["success_rate"], idx)
            if ppo_per_env is not None:
                writer.add_scalar("eval_ppo/per_env_return_mean", ppo_per_env["return"]["mean"], idx)
                writer.add_scalar("eval_ppo/per_env_success_rate", ppo_per_env["success_rate"], idx)
                if rma_per_env is not None:
                    writer.add_scalar(
                        "eval_compare/per_env_return_delta_rma_minus_ppo",
                        rma_per_env["return"]["mean"] - ppo_per_env["return"]["mean"],
                        idx,
                    )
                writer.add_scalar(
                    "eval_compare/per_env_return_delta_rma_gt_minus_ppo",
                    rma_gt_per_env["return"]["mean"] - ppo_per_env["return"]["mean"],
                    idx,
                )
        env_id = int(spec["env_id"])
        print(f"[env {idx + 1}/{len(eval_specs)}] env_id={env_id}")
        if rma_per_env is not None:
            print(
                f"                      RMA ret={rma_per_env['return']['mean']:.3f} "
                f"succ={rma_per_env['success_rate']:.3f} "
                f"adapt_frac={rma_per_env['latent_adaptation_fraction']:.3f}"
            )
        print(
            f"                      RMA-GT ret={rma_gt_per_env['return']['mean']:.3f} "
            f"succ={rma_gt_per_env['success_rate']:.3f}"
        )
        if ppo_per_env is not None:
            if rma_per_env is not None:
                print(
                    f"                      PPO ret={ppo_per_env['return']['mean']:.3f} "
                    f"succ={ppo_per_env['success_rate']:.3f} "
                    f"delta={rma_per_env['return']['mean'] - ppo_per_env['return']['mean']:.3f}"
                )
            else:
                print(
                    f"                      PPO ret={ppo_per_env['return']['mean']:.3f} "
                    f"succ={ppo_per_env['success_rate']:.3f}"
                )
            print(
                f"                      RMA-GT delta={rma_gt_per_env['return']['mean'] - ppo_per_env['return']['mean']:.3f}"
            )

    rma_aggregate = None
    if len(rma_per_env_metrics) > 0:
        rma_aggregate = aggregate_across_envs(rma_per_env_metrics)
        print_summary(rma_per_env_metrics, rma_aggregate, "Joint RMA evaluation summary")
    rma_gt_aggregate = aggregate_across_envs(rma_gt_per_env_metrics)
    print_summary(rma_gt_per_env_metrics, rma_gt_aggregate, "RMA with ground-truth latent evaluation summary")

    ppo_aggregate = None
    comparison_report = None
    comparison_report_rma_gt_vs_ppo = None
    if ppo_agent is not None:
        ppo_aggregate = aggregate_across_envs(ppo_per_env_metrics)
        print_summary(ppo_per_env_metrics, ppo_aggregate, "PPO baseline evaluation summary")
        if len(rma_per_env_metrics) > 0:
            comparison_report = build_comparison_report(rma_per_env_metrics, ppo_per_env_metrics)
        comparison_report_rma_gt_vs_ppo = build_comparison_report(rma_gt_per_env_metrics, ppo_per_env_metrics)

    if rma_aggregate is not None:
        with open(os.path.join(log_parent_dir, "per_env_metrics_rma.yaml"), "w") as f:
            yaml.dump(rma_per_env_metrics, f, sort_keys=False)
        with open(os.path.join(log_parent_dir, "aggregate_metrics_rma.yaml"), "w") as f:
            yaml.dump(rma_aggregate, f, sort_keys=False)
    with open(os.path.join(log_parent_dir, "per_env_metrics_rma_gt.yaml"), "w") as f:
        yaml.dump(rma_gt_per_env_metrics, f, sort_keys=False)
    with open(os.path.join(log_parent_dir, "aggregate_metrics_rma_gt.yaml"), "w") as f:
        yaml.dump(rma_gt_aggregate, f, sort_keys=False)
    if ppo_aggregate is not None:
        with open(os.path.join(log_parent_dir, "per_env_metrics_ppo.yaml"), "w") as f:
            yaml.dump(ppo_per_env_metrics, f, sort_keys=False)
        with open(os.path.join(log_parent_dir, "aggregate_metrics_ppo.yaml"), "w") as f:
            yaml.dump(ppo_aggregate, f, sort_keys=False)
        if comparison_report is not None:
            with open(os.path.join(log_parent_dir, "comparison_rma_vs_ppo.yaml"), "w") as f:
                yaml.dump(comparison_report, f, sort_keys=False)
        with open(os.path.join(log_parent_dir, "comparison_rma_gt_vs_ppo.yaml"), "w") as f:
            yaml.dump(comparison_report_rma_gt_vs_ppo, f, sort_keys=False)

    csv_path = os.path.join(log_parent_dir, "episode_records.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "env_id",
                "env_name",
                "episode_index",
                "seed",
                "episode_return",
                "episode_length",
                "success",
                "latent_encoder_fraction",
                "latent_adaptation_fraction",
            ],
        )
        writer_csv.writeheader()
        for row in rma_episode_rows + rma_gt_episode_rows + ppo_episode_rows:
            writer_csv.writerow(row)

    if writer is not None:
        if rma_aggregate is not None:
            writer.add_scalar("eval_rma/macro_return_mean", rma_aggregate["macro_return"]["mean"], 0)
            writer.add_scalar("eval_rma/macro_return_std", rma_aggregate["macro_return"]["std"], 0)
            writer.add_scalar("eval_rma/macro_success_mean", rma_aggregate["macro_success_rate"]["mean"], 0)
            writer.add_scalar("eval_rma/macro_adaptation_fraction_mean", rma_aggregate["macro_latent_adaptation_fraction"]["mean"], 0)
        writer.add_scalar("eval_rma_gt/macro_return_mean", rma_gt_aggregate["macro_return"]["mean"], 0)
        writer.add_scalar("eval_rma_gt/macro_return_std", rma_gt_aggregate["macro_return"]["std"], 0)
        writer.add_scalar("eval_rma_gt/macro_success_mean", rma_gt_aggregate["macro_success_rate"]["mean"], 0)
        writer.add_scalar("eval_rma_gt/macro_encoder_fraction_mean", rma_gt_aggregate["macro_latent_encoder_fraction"]["mean"], 0)
        if ppo_aggregate is not None:
            writer.add_scalar("eval_ppo/macro_return_mean", ppo_aggregate["macro_return"]["mean"], 0)
            writer.add_scalar("eval_ppo/macro_return_std", ppo_aggregate["macro_return"]["std"], 0)
            writer.add_scalar("eval_ppo/macro_success_mean", ppo_aggregate["macro_success_rate"]["mean"], 0)
            if rma_aggregate is not None:
                writer.add_scalar(
                    "eval_compare/macro_return_delta_rma_minus_ppo",
                    rma_aggregate["macro_return"]["mean"] - ppo_aggregate["macro_return"]["mean"],
                    0,
                )
                writer.add_scalar(
                    "eval_compare/macro_success_delta_rma_minus_ppo",
                    rma_aggregate["macro_success_rate"]["mean"] - ppo_aggregate["macro_success_rate"]["mean"],
                    0,
                )
            writer.add_scalar(
                "eval_compare/macro_return_delta_rma_gt_minus_ppo",
                rma_gt_aggregate["macro_return"]["mean"] - ppo_aggregate["macro_return"]["mean"],
                0,
            )
            writer.add_scalar(
                "eval_compare/macro_success_delta_rma_gt_minus_ppo",
                rma_gt_aggregate["macro_success_rate"]["mean"] - ppo_aggregate["macro_success_rate"]["mean"],
                0,
            )
        writer.close()

    save_performance_comparison_chart(
        log_parent_dir=log_parent_dir,
        rma_per_env=rma_per_env_metrics,
        rma_gt_per_env=rma_gt_per_env_metrics,
        ppo_per_env=ppo_per_env_metrics,
    )

    print(f"Saved evaluation artifacts to: {log_parent_dir}")
