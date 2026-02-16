import os
import random
import time
from dataclasses import dataclass, field
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
from scripts.smooth_policy.amp_history.amp_training.rma_adaptation import RMAAdaptationModule
from scripts.smooth_policy.encoder import EnvEncoder


class _TupleFriendlySafeLoader(yaml.SafeLoader):
    """Safe loader variant that accepts legacy !!python/tuple nodes as plain tuples."""


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


_TupleFriendlySafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def augment_policy_observation(observation, last_action, use_last_action):
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def concat_env_latent_to_policy_obs(policy_obs_base, env_latent):
    return torch.cat([policy_obs_base, env_latent], dim=-1)


def inject_latent_noise(env_latent, noise_std):
    return env_latent + torch.randn_like(env_latent) * noise_std


def get_env_spec_ranges():
    # Keep defaults aligned with stage-1 LSGAN randomization ranges.
    return {
        "paddle_density": (2500 * 0.5, 2500 * 1.5),
        "paddle_damping": (3 * 0.5, 3 * 1.5),
        "puck_density": (250 * 0.5, 250 * 1.5),
        "puck_damping": (0.5 * 0.5, 0.5 * 1.5),
        "force_scaling": (1 * 0.5, 1 * 1.5),
    }


def get_env_spec_ordered_keys():
    return [
        "paddle_density",
        "paddle_damping",
        "puck_density",
        "puck_damping",
        "force_scaling",
    ]


def validate_env_spec_pool(env_spec_pool):
    if not isinstance(env_spec_pool, list) or len(env_spec_pool) == 0:
        raise ValueError("Loaded env_spec_pool must be a non-empty list.")

    required_keys = {"env_id", *get_env_spec_ordered_keys()}
    validated_pool = []
    for idx, spec in enumerate(env_spec_pool):
        if not isinstance(spec, dict):
            raise ValueError(f"Environment spec at index {idx} must be a dict.")
        missing = required_keys - set(spec.keys())
        if missing:
            raise ValueError(f"Environment spec at index {idx} is missing keys: {sorted(missing)}")
        out = dict(spec)
        out["env_id"] = int(spec["env_id"])
        for key in get_env_spec_ordered_keys():
            out[key] = float(spec[key])
        validated_pool.append(out)
    return validated_pool


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=_TupleFriendlySafeLoader)


def _resolve_env_artifact_paths(env_spec_pool_path):
    """
    Accept either:
    - direct env pool file path (.pt/.yaml), or
    - run/checkpoint directory containing optional env_spec_pool(.pt/.yaml) and/or training_env_setup.yaml, or
    - direct path to training_env_setup.yaml.

    Returns:
      (pool_path_or_none, manifest_path_or_none)
    """
    input_path = os.path.abspath(env_spec_pool_path)
    manifest_path = None
    pool_path = None

    if os.path.isdir(input_path):
        manifest_candidate = os.path.join(input_path, "training_env_setup.yaml")
        pt_candidate = os.path.join(input_path, "env_spec_pool.pt")
        yaml_candidate = os.path.join(input_path, "env_spec_pool.yaml")
        if os.path.exists(manifest_candidate):
            manifest_path = manifest_candidate
        if os.path.exists(pt_candidate):
            pool_path = pt_candidate
        elif os.path.exists(yaml_candidate):
            pool_path = yaml_candidate
        if pool_path is None and manifest_path is None:
            raise FileNotFoundError(
                f"No recognized env artifacts found in directory '{input_path}'. "
                "Expected training_env_setup.yaml and/or env_spec_pool.pt/.yaml."
            )
        return pool_path, manifest_path

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"env_spec_pool_path '{env_spec_pool_path}' does not exist. "
            "Use a stage-1 run/checkpoint directory or env_spec_pool(.pt/.yaml) artifact."
        )

    if os.path.basename(input_path) == "training_env_setup.yaml":
        manifest_path = input_path
        manifest = _load_yaml(manifest_path) or {}
        source_path = manifest.get("env_spec_pool_source_path")
        if source_path:
            source_path = os.path.abspath(source_path)
            if os.path.exists(source_path):
                pool_path = source_path
        if pool_path is None:
            parent = os.path.dirname(manifest_path)
            pt_candidate = os.path.join(parent, "env_spec_pool.pt")
            yaml_candidate = os.path.join(parent, "env_spec_pool.yaml")
            if os.path.exists(pt_candidate):
                pool_path = pt_candidate
            elif os.path.exists(yaml_candidate):
                pool_path = yaml_candidate
        # Manifest-only setups are valid for on-demand range sampling.
        return pool_path, manifest_path

    if input_path.endswith(".pt") or input_path.endswith(".yaml") or input_path.endswith(".yml"):
        return input_path, None

    raise ValueError(
        f"Unsupported env_spec_pool_path format: '{env_spec_pool_path}'. "
        "Use a directory, training_env_setup.yaml, env_spec_pool.pt, or env_spec_pool.yaml."
    )


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


def _resolve_stage1_module_paths(module_dir):
    """
    Resolve required stage-1 artifacts from one directory.
    Expected files:
      - model:   model.pth
      - encoder: encoder.pth
    Optional:
      - training_env_setup.yaml and/or env_spec_pool.(pt/.yaml) for range metadata.
    """
    base_dir = os.path.abspath(module_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"module_dir does not exist or is not a directory: '{module_dir}'")

    model_path = os.path.join(base_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find model artifact in module_dir '{base_dir}'. Expected: '{model_path}'."
        )

    encoder_path = os.path.join(base_dir, "encoder.pth")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Could not find encoder artifact in module_dir '{base_dir}'. Expected: '{encoder_path}'."
        )

    try:
        pool_path, manifest_path = _resolve_env_artifact_paths(base_dir)
    except FileNotFoundError:
        pool_path, manifest_path = None, None
    return {
        "module_dir": base_dir,
        "model_path": model_path,
        "encoder_path": encoder_path,
        "env_spec_pool_path": pool_path,
        "training_manifest_path": manifest_path,
    }


def _coerce_env_spec_ranges(raw_ranges):
    if raw_ranges is None:
        return get_env_spec_ranges()

    ordered_keys = get_env_spec_ordered_keys()
    out = {}
    for key in ordered_keys:
        if key not in raw_ranges:
            raise ValueError(
                f"randomization_ranges in manifest missing key '{key}'. "
                f"Expected keys: {ordered_keys}"
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


def extract_env_var_vector_from_spec(spec, env_var_dim, env_spec_ranges=None):
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


def parse_env_vars_from_infos(infos, num_envs, env_var_dim, device, fallback_env_vars):
    if not (isinstance(infos, dict) and "rma_env_vars" in infos):
        return fallback_env_vars
    raw = infos["rma_env_vars"]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in raw], axis=0)
    else:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1, env_var_dim)
    if raw.shape[0] == num_envs:
        return torch.as_tensor(raw, dtype=torch.float32, device=device)
    mask = infos.get("_rma_env_vars")
    if mask is not None and raw.shape[0] == int(np.asarray(mask, dtype=bool).sum()):
        out = fallback_env_vars.clone()
        out[torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool, device=device)] = torch.as_tensor(
            raw, dtype=torch.float32, device=device
        )
        return out
    return fallback_env_vars


class ResetSampledEnvWrapper(gym.Wrapper):
    def __init__(self, env, env_spec_pool, env_var_dim, rng_seed):
        super().__init__(env)
        self.env_spec_pool = env_spec_pool
        self.env_var_dim = env_var_dim
        self.rng = np.random.default_rng(rng_seed)
        self.current_env_var_vec = np.zeros(env_var_dim, dtype=np.float32)
        self.current_env_id = -1

    def _apply_env_spec(self, env_spec):
        self.env.unwrapped.paddle_density = env_spec["paddle_density"]
        self.env.unwrapped.paddle_damping = env_spec["paddle_damping"]
        self.env.unwrapped.puck_density = env_spec["puck_density"]
        self.env.unwrapped.puck_damping = env_spec["puck_damping"]
        self.env.unwrapped.force_scaling = env_spec["force_scaling"]
        self.current_env_var_vec = extract_env_var_vector_from_spec(env_spec, self.env_var_dim)
        self.current_env_id = int(env_spec["env_id"])

    def _sample_and_apply_spec(self):
        idx = int(self.rng.integers(0, len(self.env_spec_pool)))
        self._apply_env_spec(self.env_spec_pool[idx])

    def reset(self, **kwargs):
        self._sample_and_apply_spec()
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["rma_env_vars"] = self.current_env_var_vec.copy()
        info["rma_env_id"] = self.current_env_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["rma_env_vars"] = self.current_env_var_vec.copy()
        info["rma_env_id"] = self.current_env_id
        return obs, reward, terminated, truncated, info


def sample_env_spec_from_ranges(rng, env_id, env_spec_ranges):
    return {
        "env_id": int(env_id),
        "paddle_density": float(rng.uniform(*env_spec_ranges["paddle_density"])),
        "paddle_damping": float(rng.uniform(*env_spec_ranges["paddle_damping"])),
        "puck_density": float(rng.uniform(*env_spec_ranges["puck_density"])),
        "puck_damping": float(rng.uniform(*env_spec_ranges["puck_damping"])),
        "force_scaling": float(rng.uniform(*env_spec_ranges["force_scaling"])),
    }


class ResetRangeSampledEnvWrapper(gym.Wrapper):
    def __init__(self, env, env_var_dim, rng_seed, env_spec_ranges, env_id_offset=0):
        super().__init__(env)
        self.env_var_dim = env_var_dim
        self.rng = np.random.default_rng(rng_seed)
        self.env_spec_ranges = _coerce_env_spec_ranges(env_spec_ranges)
        self.current_env_var_vec = np.zeros(env_var_dim, dtype=np.float32)
        self.current_env_id = -1
        self._sample_idx = 0
        self._env_id_offset = int(env_id_offset)

    def _apply_env_spec(self, env_spec):
        self.env.unwrapped.paddle_density = env_spec["paddle_density"]
        self.env.unwrapped.paddle_damping = env_spec["paddle_damping"]
        self.env.unwrapped.puck_density = env_spec["puck_density"]
        self.env.unwrapped.puck_damping = env_spec["puck_damping"]
        self.env.unwrapped.force_scaling = env_spec["force_scaling"]
        self.current_env_var_vec = extract_env_var_vector_from_spec(
            spec=env_spec,
            env_var_dim=self.env_var_dim,
            env_spec_ranges=self.env_spec_ranges,
        )
        self.current_env_id = int(env_spec["env_id"])

    def _sample_and_apply_spec(self):
        env_id = self._env_id_offset + self._sample_idx
        self._sample_idx += 1
        sampled_spec = sample_env_spec_from_ranges(
            rng=self.rng,
            env_id=env_id,
            env_spec_ranges=self.env_spec_ranges,
        )
        self._apply_env_spec(sampled_spec)

    def reset(self, **kwargs):
        self._sample_and_apply_spec()
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["rma_env_vars"] = self.current_env_var_vec.copy()
        info["rma_env_id"] = self.current_env_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["rma_env_vars"] = self.current_env_var_vec.copy()
        info["rma_env_id"] = self.current_env_id
        return obs, reward, terminated, truncated, info


def make_env(env_id, env_var_dim, air_hockey_config, seed, env_spec_ranges, env_id_offset):
    def _thunk():
        cfg = dict(air_hockey_config)
        cfg["seed"] = random.randint(0, int(1e8))
        env = AirHockeyEnv(cfg)
        return ResetRangeSampledEnvWrapper(
            env=env,
            env_var_dim=env_var_dim,
            rng_seed=seed + env_id * 131,
            env_spec_ranges=env_spec_ranges,
            env_id_offset=env_id_offset + env_id * 1_000_000,
        )

    return _thunk


def load_env_spec_pool(env_spec_pool_path):
    pool_path, manifest_path = _resolve_env_artifact_paths(env_spec_pool_path)
    if pool_path.endswith(".pt"):
        pool = torch.load(pool_path, map_location="cpu")
    else:
        pool = _load_yaml(pool_path)
    return validate_env_spec_pool(pool), manifest_path, pool_path


def split_env_pool(env_spec_pool, train_env_count, eval_env_count, seed):
    pool_size = len(env_spec_pool)
    if eval_env_count < 0:
        raise ValueError("eval_env_count must be >= 0.")
    if train_env_count <= 0:
        raise ValueError("train_env_count must be > 0.")
    if train_env_count + eval_env_count > pool_size:
        raise ValueError(
            f"train_env_count + eval_env_count ({train_env_count + eval_env_count}) "
            f"exceeds pool size ({pool_size})."
        )
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(pool_size)
    eval_idxs = idxs[:eval_env_count]
    train_idxs = idxs[eval_env_count : eval_env_count + train_env_count]

    train_specs = [env_spec_pool[int(i)] for i in train_idxs]
    if eval_env_count <= 0:
        return train_specs, train_specs
    eval_specs = [env_spec_pool[int(i)] for i in eval_idxs]
    return train_specs, eval_specs


def compute_context_lengths(done_flags):
    # done_flags: [T, N] where done[t, n] indicates transition t ended episode.
    t_steps, n_envs = done_flags.shape
    context = torch.zeros_like(done_flags, dtype=torch.long)
    for t in range(t_steps):
        if t == 0:
            context[t] = 1
        else:
            context[t] = torch.where(done_flags[t - 1], torch.ones(n_envs, dtype=torch.long, device=done_flags.device), context[t - 1] + 1)
    return context


def build_window(states, actions, start_idx, end_idx, env_idx, fixed_context_len):
    # Inclusive [start_idx, end_idx]
    state_seq = states[start_idx : end_idx + 1, env_idx]
    action_seq = actions[start_idx : end_idx + 1, env_idx]
    curr_len = state_seq.shape[0]

    if curr_len > fixed_context_len:
        # Keep the most recent context when sequence is longer than fixed size.
        state_seq = state_seq[-fixed_context_len:]
        action_seq = action_seq[-fixed_context_len:]
        curr_len = fixed_context_len

    if curr_len < fixed_context_len:
        pad = fixed_context_len - curr_len
        state_pad = state_seq[:1].repeat(pad, 1)
        action_pad = action_seq[:1].repeat(pad, 1)
        state_seq = torch.cat([state_pad, state_seq], dim=0)
        action_seq = torch.cat([action_pad, action_seq], dim=0)
        valid_mask = torch.cat(
            [
                torch.zeros(pad, dtype=torch.float32, device=state_seq.device),
                torch.ones(curr_len, dtype=torch.float32, device=state_seq.device),
            ],
            dim=0,
        )
    else:
        valid_mask = torch.ones(fixed_context_len, dtype=torch.float32, device=state_seq.device)
    return state_seq, action_seq, valid_mask


def collect_rollout_dataset(envs, agent, env_encoder, args, device):
    num_envs = args.num_envs
    rollout_len = args.rollout_len
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    action_dim = int(np.prod(act_shape))

    states = torch.zeros((rollout_len, num_envs) + obs_shape, device=device)
    actions = torch.zeros((rollout_len, num_envs) + act_shape, device=device)
    env_vars = torch.zeros((rollout_len, num_envs, args.env_var_dim), device=device)
    target_latents = torch.zeros((rollout_len, num_envs, args.env_latent_dim), device=device)
    done_flags = torch.zeros((rollout_len, num_envs), dtype=torch.bool, device=device)

    next_obs, infos = envs.reset(seed=args.seed + args.rollout_seed_offset)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    current_env_vars = parse_env_vars_from_infos(
        infos=infos,
        num_envs=num_envs,
        env_var_dim=args.env_var_dim,
        device=device,
        fallback_env_vars=torch.zeros((num_envs, args.env_var_dim), dtype=torch.float32, device=device),
    )
    last_action = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in range(rollout_len):
            states[t] = next_obs
            env_vars[t] = current_env_vars
            policy_obs_base = augment_policy_observation(next_obs, last_action, args.use_last_action_in_policy_state)
            latent_clean = env_encoder(current_env_vars)
            latent_noisy = inject_latent_noise(latent_clean, args.latent_noise_std)
            target_latents[t] = latent_clean.detach()
            policy_obs = concat_env_latent_to_policy_obs(policy_obs_base, latent_noisy)
            action, _, _, _ = agent.get_action_and_value(policy_obs)
            actions[t] = action

            next_obs_np, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            done_t = torch.as_tensor(done, dtype=torch.bool, device=device)
            done_flags[t] = done_t
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            current_env_vars = parse_env_vars_from_infos(
                infos=infos,
                num_envs=num_envs,
                env_var_dim=args.env_var_dim,
                device=device,
                fallback_env_vars=current_env_vars,
            )
            last_action = action.detach()
            last_action[done_t] = 0.0

    return {
        "states": states,
        "actions": actions,
        "env_vars": env_vars,
        "done_flags": done_flags,
        "target_latents": target_latents,
        "context_lengths": compute_context_lengths(done_flags),
    }


def sample_batch_from_rollout(rollout, batch_size, args, mode):
    states = rollout["states"]
    actions = rollout["actions"]
    targets = rollout["target_latents"]
    context_lengths = rollout["context_lengths"]

    t_steps, n_envs = context_lengths.shape
    min_k = args.prior_min
    max_k = args.prior_max
    fixed_context_len = 100

    if mode not in {"train", "eval_max"}:
        raise ValueError(f"Unknown mode: {mode}")

    available_flat = context_lengths.reshape(-1)
    if mode == "train":
        valid_positions = torch.nonzero(available_flat >= min_k, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            raise RuntimeError(
                "No valid training samples found in rollout. "
                "Increase rollout_len or reduce prior_min."
            )
    else:
        valid_positions = torch.arange(available_flat.numel(), device=available_flat.device)

    chosen_flat = valid_positions[
        torch.randint(
            low=0,
            high=valid_positions.numel(),
            size=(batch_size,),
            device=valid_positions.device,
        )
    ]
    sampled_t = torch.div(chosen_flat, n_envs, rounding_mode="floor")
    sampled_e = torch.remainder(chosen_flat, n_envs)
    sampled_available = context_lengths[sampled_t, sampled_e].to(torch.long)

    if mode == "train":
        local_max = torch.minimum(sampled_available, torch.full_like(sampled_available, max_k))
        span = (local_max - min_k + 1).to(torch.float32)
        sampled_k = (torch.floor(torch.rand(batch_size, device=targets.device) * span).to(torch.long) + min_k)
    else:
        sampled_k = torch.minimum(sampled_available, torch.full_like(sampled_available, max_k))

    effective_len = torch.minimum(sampled_k, torch.full_like(sampled_k, fixed_context_len))
    start_idx = sampled_t - sampled_k + 1
    start_eff = start_idx + (sampled_k - effective_len)
    pad_len = fixed_context_len - effective_len

    rel = torch.arange(fixed_context_len, device=targets.device).unsqueeze(0).expand(batch_size, -1)
    time_idx = torch.where(
        rel < pad_len.unsqueeze(1),
        start_eff.unsqueeze(1),
        start_eff.unsqueeze(1) + (rel - pad_len.unsqueeze(1)),
    )
    env_idx = sampled_e.unsqueeze(1).expand(-1, fixed_context_len)
    valid_mask = (rel >= pad_len.unsqueeze(1)).to(dtype=torch.float32)

    sampled_states = states[time_idx, env_idx]
    sampled_actions = actions[time_idx, env_idx]
    sampled_targets = targets[sampled_t, sampled_e]

    return {
        "states": sampled_states,
        "actions": sampled_actions,
        "valid_mask": valid_mask,
        "targets": sampled_targets,
        "k": sampled_k.to(dtype=torch.float32),
        "available": sampled_available.to(dtype=torch.float32),
    }


def latent_metrics(pred, target):
    pred_2d = pred.reshape(-1, pred.shape[-1])
    target_2d = target.reshape(-1, target.shape[-1])
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    rmse = torch.sqrt(mse + 1e-12)
    cos = nn.functional.cosine_similarity(pred, target, dim=-1).mean()
    pred_norm = pred.norm(dim=-1).mean()
    target_norm = target.norm(dim=-1).mean()

    # Multi-dimensional explained variance:
    # EV_d = 1 - Var(y_d - yhat_d) / Var(y_d), then average over dimensions
    # with non-negligible target variance.
    target_var = torch.var(target_2d, dim=0, unbiased=False)
    error_var = torch.var(target_2d - pred_2d, dim=0, unbiased=False)
    valid_dims = target_var > 1e-12
    if torch.any(valid_dims):
        explained_variance = (1.0 - (error_var[valid_dims] / target_var[valid_dims])).mean()
    else:
        explained_variance = torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "cosine": cos,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "explained_variance": explained_variance,
    }


def tensor_stats(tensor):
    return tensor.mean(), tensor.std(unbiased=False), tensor.min(), tensor.max()


def param_and_grad_norm(module):
    p_norm_sq = 0.0
    g_norm_sq = 0.0
    for p in module.parameters():
        p_norm_sq += float(torch.sum(p.detach() ** 2).item())
        if p.grad is not None:
            g_norm_sq += float(torch.sum(p.grad.detach() ** 2).item())
    return p_norm_sq ** 0.5, g_norm_sq ** 0.5


@dataclass
class Args:
    # Paths and configs.
    config: str = "scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_config.yaml"
    args_file: str | None = None
    module_dir: str = ""
    model_path: str = ""
    encoder_path: str = ""
    env_spec_pool_path: str = ""  # Optional stage-1 directory/manifest/pool path for randomization-range metadata.
    log_parent_dir: str | None = None
    run_name: str = "rma_adaptation_supervised"

    # Runtime.
    seed: int = 0
    device: str = "cuda:0"
    num_envs: int = 32

    # Stage-1 architecture compatibility.
    use_last_action_in_policy_state: bool = True
    action_scale: float = 1.0
    agent_hidden_size: int = 512
    env_var_dim: int = 8
    env_latent_dim: int = 8
    env_encoder_hidden_size: list[int] = field(default_factory=lambda: [128, 128])

    # Stage-2 adaptation model sizes.
    adaptation_conv_in_channels: int = 32
    adaptation_hidden_size: int = 64

    # Data collection and supervision.
    train_env_count: int = 450
    eval_env_count: int = 50
    rollout_len: int = 200
    prior_min: int = 50
    prior_max: int = 100
    min_model_context_len: int = 16
    latent_noise_std: float = 0.10
    rollout_seed_offset: int = 1000

    # Optimization.
    num_iterations: int = 2000
    train_steps_per_iter: int = 16
    minibatch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    # Logging/checkpointing.
    eval_interval: int = 10
    checkpoint_interval: int = 50
    print_interval: int = 1


if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        if isinstance(file_args_dict.get("env_encoder_hidden_size"), int):
            file_args_dict["env_encoder_hidden_size"] = [int(file_args_dict["env_encoder_hidden_size"])]
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)

    resolved_stage1 = None
    if args.module_dir:
        resolved_stage1 = _resolve_stage1_module_paths(args.module_dir)
        if not args.model_path:
            args.model_path = resolved_stage1["model_path"]
        if not args.encoder_path:
            args.encoder_path = resolved_stage1["encoder_path"]
        print(
            "Resolved stage-1 artifacts from module_dir: "
            f"{resolved_stage1['module_dir']}"
        )

    if not args.model_path:
        raise ValueError("model_path must be provided.")
    if not args.encoder_path:
        raise ValueError("encoder_path must be provided.")
    if args.prior_min > args.prior_max:
        raise ValueError("prior_min must be <= prior_max.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    resolved_pool_path = None
    manifest_path = None
    if resolved_stage1 is not None and resolved_stage1.get("training_manifest_path") is not None:
        manifest_path = resolved_stage1["training_manifest_path"]
    if args.env_spec_pool_path:
        maybe_pool_path, maybe_manifest_path = _resolve_env_artifact_paths(args.env_spec_pool_path)
        if maybe_manifest_path is not None:
            manifest_path = maybe_manifest_path
        if maybe_pool_path is not None:
            resolved_pool_path = maybe_pool_path

    manifest = (_load_yaml(manifest_path) or {}) if manifest_path is not None else {}
    if manifest:
        _validate_manifest_compatibility(manifest, args)
        print(f"Loaded stage-1 env manifest: {manifest_path}")

    env_spec_ranges = _coerce_env_spec_ranges(manifest.get("randomization_ranges"))
    print(
        "Using on-demand range sampling for adaptation envs with ranges: "
        f"{env_spec_ranges}"
    )

    train_envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_id=i,
                env_var_dim=args.env_var_dim,
                air_hockey_config=config["air_hockey"],
                seed=args.seed,
                env_spec_ranges=env_spec_ranges,
                env_id_offset=0,
            )
            for i in range(args.num_envs)
        ]
    )
    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_id=i,
                env_var_dim=args.env_var_dim,
                air_hockey_config=config["air_hockey"],
                seed=args.seed + 991,
                env_spec_ranges=env_spec_ranges,
                env_id_offset=100_000_000,
            )
            for i in range(args.num_envs)
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task", "task")
    if args.log_parent_dir is None:
        log_parent_dir = f"runs/rma_adaptation/{task_name}/{args.run_name}_{timestamp}"
    else:
        log_parent_dir = args.log_parent_dir
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    env_artifact_usage = {
        "input_module_dir": os.path.abspath(args.module_dir) if args.module_dir else None,
        "resolved_model_path": os.path.abspath(args.model_path),
        "resolved_encoder_path": os.path.abspath(args.encoder_path),
        "input_env_spec_pool_path": os.path.abspath(args.env_spec_pool_path) if args.env_spec_pool_path else None,
        "resolved_env_spec_pool_path": resolved_pool_path,
        "resolved_training_manifest_path": manifest_path,
        "sampling_mode": "on_demand_range_sampling",
        "randomization_ranges": env_spec_ranges,
        "loaded_env_pool_size": None,
        "train_env_count": None,
        "eval_env_count": None,
        "split_seed": int(args.seed),
    }
    with open(f"{log_parent_dir}/env_spec_pool_usage.yaml", "w") as f:
        yaml.dump(env_artifact_usage, f, sort_keys=False)

    base_policy_obs_dim = int(np.prod(train_envs.single_observation_space.shape))
    action_dim = int(np.prod(train_envs.single_action_space.shape))
    if args.use_last_action_in_policy_state:
        base_policy_obs_dim += action_dim
    policy_obs_dim = base_policy_obs_dim + args.env_latent_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=train_envs.single_action_space,
    )

    device = torch.device(args.device)
    agent = Agent(
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
    adaptation_module = RMAAdaptationModule(
        action_dim=action_dim,
        state_dim=int(np.prod(train_envs.single_observation_space.shape)),
        conv_in_channels=args.adaptation_conv_in_channels,
        latent_dim=args.env_latent_dim,
        hidden_size=args.adaptation_hidden_size,
    ).to(device)

    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    env_encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    agent.eval()
    env_encoder.eval()
    for p in agent.parameters():
        p.requires_grad = False
    for p in env_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        adaptation_module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-6,
    )

    best_eval_mse = float("inf")
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        adaptation_module.train()
        rollout = collect_rollout_dataset(train_envs, agent, env_encoder, args, device)
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_rmse_sum = 0.0
        train_cos_sum = 0.0
        train_ev_sum = 0.0

        for _ in range(args.train_steps_per_iter):
            batch = sample_batch_from_rollout(
                rollout=rollout,
                batch_size=args.minibatch_size,
                args=args,
                mode="train",
            )
            should_log = ((global_step + 1) % 20) == 0
            model_out = adaptation_module(
                batch["actions"],
                batch["states"],
                valid_mask=batch["valid_mask"],
                return_intermediates=should_log,
            )
            if should_log:
                preds = model_out["latent"]
            else:
                preds = model_out
            metrics = latent_metrics(preds, batch["targets"])
            loss = metrics["mse"]

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adaptation_module.parameters(), args.max_grad_norm)
            optimizer.step()
            global_step += 1

            train_loss_sum += metrics["mse"].item()
            train_mae_sum += metrics["mae"].item()
            train_rmse_sum += metrics["rmse"].item()
            train_cos_sum += metrics["cosine"].item()
            train_ev_sum += metrics["explained_variance"].item()

            if should_log:
                writer.add_scalar("train/loss_mse", metrics["mse"].item(), global_step)
                writer.add_scalar("train/loss_mae", metrics["mae"].item(), global_step)
                writer.add_scalar("train/loss_rmse", metrics["rmse"].item(), global_step)
                writer.add_scalar("train/latent_cosine", metrics["cosine"].item(), global_step)
                writer.add_scalar("train/explained_variance", metrics["explained_variance"].item(), global_step)
                writer.add_scalar("train/pred_norm", metrics["pred_norm"].item(), global_step)
                writer.add_scalar("train/target_norm", metrics["target_norm"].item(), global_step)
                writer.add_scalar("train/prior_k_mean", batch["k"].mean().item(), global_step)
                writer.add_scalar("train/prior_available_mean", batch["available"].mean().item(), global_step)
                writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("optim/grad_norm", float(grad_norm.item()), global_step)
                writer.add_scalar(
                    "debug/nan_count_pred",
                    float(torch.isnan(preds).sum().item()),
                    global_step,
                )
                writer.add_scalar(
                    "debug/inf_count_pred",
                    float(torch.isinf(preds).sum().item()),
                    global_step,
                )

                emb_stats = tensor_stats(model_out["embedded"])
                temporal_stats = tensor_stats(model_out["temporal_in"])
                pooled_stats = tensor_stats(model_out["pooled"])
                writer.add_scalar("cnn/embedded_mean", emb_stats[0].item(), global_step)
                writer.add_scalar("cnn/embedded_std", emb_stats[1].item(), global_step)
                writer.add_scalar("cnn/temporal_in_mean", temporal_stats[0].item(), global_step)
                writer.add_scalar("cnn/temporal_in_std", temporal_stats[1].item(), global_step)
                writer.add_scalar("cnn/pooled_mean", pooled_stats[0].item(), global_step)
                writer.add_scalar("cnn/pooled_std", pooled_stats[1].item(), global_step)

                for name, layer in adaptation_module.named_modules():
                    if isinstance(layer, nn.Conv1d):
                        w = layer.weight.detach()
                        writer.add_scalar(f"cnn/{name}_weight_norm", float(torch.norm(w).item()), global_step)
                        if layer.weight.grad is not None:
                            writer.add_scalar(
                                f"cnn/{name}_grad_norm",
                                float(torch.norm(layer.weight.grad.detach()).item()),
                                global_step,
                            )

                p_norm, g_norm = param_and_grad_norm(adaptation_module)
                writer.add_scalar("optim/param_norm", p_norm, global_step)
                writer.add_scalar("optim/global_grad_norm_sqroot", g_norm, global_step)

        mean_train_mse = train_loss_sum / args.train_steps_per_iter

        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            adaptation_module.eval()
            with torch.no_grad():
                eval_rollout = collect_rollout_dataset(eval_envs, agent, env_encoder, args, device)
                eval_batch = sample_batch_from_rollout(
                    rollout=eval_rollout,
                    batch_size=args.minibatch_size * 4,
                    args=args,
                    mode="eval_max",
                )
                eval_preds = adaptation_module(
                    eval_batch["actions"], eval_batch["states"], valid_mask=eval_batch["valid_mask"]
                )
                eval_metrics = latent_metrics(eval_preds, eval_batch["targets"])
                writer.add_scalar("eval/loss_mse", eval_metrics["mse"].item(), global_step)
                writer.add_scalar("eval/loss_mae", eval_metrics["mae"].item(), global_step)
                writer.add_scalar("eval/loss_rmse", eval_metrics["rmse"].item(), global_step)
                writer.add_scalar("eval/latent_cosine", eval_metrics["cosine"].item(), global_step)
                writer.add_scalar("eval/explained_variance", eval_metrics["explained_variance"].item(), global_step)
                writer.add_scalar("eval/context_k_mean", eval_batch["k"].mean().item(), global_step)
                writer.add_scalar("eval/context_available_mean", eval_batch["available"].mean().item(), global_step)

                eval_payload = {
                    "iteration": int(iteration),
                    "global_step": int(global_step),
                    "eval_mse": float(eval_metrics["mse"].item()),
                    "eval_mae": float(eval_metrics["mae"].item()),
                    "eval_rmse": float(eval_metrics["rmse"].item()),
                    "eval_cosine": float(eval_metrics["cosine"].item()),
                    "eval_explained_variance": float(eval_metrics["explained_variance"].item()),
                    "eval_context_mean": float(eval_batch["k"].mean().item()),
                    "eval_context_max": float(eval_batch["k"].max().item()),
                }
                with open(os.path.join(log_parent_dir, f"eval_iter_{iteration}.yaml"), "w") as f:
                    yaml.dump(eval_payload, f, sort_keys=False)
                print(
                    "[eval] "
                    f"iter={iteration} mse={eval_payload['eval_mse']:.6f} "
                    f"mae={eval_payload['eval_mae']:.6f} rmse={eval_payload['eval_rmse']:.6f} "
                    f"cos={eval_payload['eval_cosine']:.6f} ev={eval_payload['eval_explained_variance']:.6f} "
                    f"context_mean={eval_payload['eval_context_mean']:.2f}"
                )

                if eval_metrics["mse"].item() < best_eval_mse:
                    best_eval_mse = float(eval_metrics["mse"].item())
                    torch.save(adaptation_module.state_dict(), os.path.join(log_parent_dir, "adaptation_module_best.pth"))

        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            ckpt_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(adaptation_module.state_dict(), os.path.join(ckpt_dir, "adaptation_module.pth"))

        if args.print_interval > 0 and iteration % args.print_interval == 0:
            sps = int(global_step / max(1e-6, (time.time() - start_time)))
            print(
                f"[train] iter={iteration} mse={mean_train_mse:.6f} "
                f"mae={train_mae_sum / args.train_steps_per_iter:.6f} "
                f"rmse={train_rmse_sum / args.train_steps_per_iter:.6f} "
                f"cos={train_cos_sum / args.train_steps_per_iter:.6f} "
                f"ev={train_ev_sum / args.train_steps_per_iter:.6f} sps={sps}"
            )
            if global_step % 20 == 0:
                writer.add_scalar("charts/sps", sps, global_step)

    torch.save(adaptation_module.state_dict(), os.path.join(log_parent_dir, "adaptation_module.pth"))
    writer.close()
    train_envs.close()
    eval_envs.close()
