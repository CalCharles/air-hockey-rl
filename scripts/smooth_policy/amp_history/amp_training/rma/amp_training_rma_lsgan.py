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

from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.encoder import EnvEncoder

# AMP components
from scripts.smooth_policy.amp_history.amp_training.discriminator import Discriminator
from scripts.smooth_policy.amp_history.amp_training.replay_buffer import ReplayBuffer
from scripts.smooth_policy.amp_history.amp_training.normalizer import Normalizer
from scripts.smooth_policy.amp_history.amp_training.demo_loader_position_history import DemoLoaderPositionHistory
from scripts.smooth_policy.amp_history.amp_training.feature_processing import (
    PUCK_FEATURE_DIM,
    build_puck_discriminator_features_torch,
    normalize_action_history_batch,
    normalize_position_history_batch,
    normalize_position_sequence_batch,
    sample_bucketed_indices_torch,
)


class _TupleFriendlySafeLoader(yaml.SafeLoader):
    """Safe loader variant that accepts legacy !!python/tuple nodes as plain tuples."""


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


_TupleFriendlySafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def parse_discriminator_hidden_dims(hidden_sizes):
    """Validate and normalize discriminator hidden layer sizes."""
    hidden_dims = [int(size) for size in hidden_sizes]
    if len(hidden_dims) == 0:
        raise ValueError("disc_hidden_sizes must contain at least one layer size.")
    if any(size <= 0 for size in hidden_dims):
        raise ValueError(f"disc_hidden_sizes must be positive, got: {hidden_dims}")
    return hidden_dims

def augment_policy_observation(observation, last_action, use_last_action):
    """Append last action to policy observation when enabled."""
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


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


def concat_env_latent_to_policy_obs(policy_obs_base, env_latent):
    """Concatenate encoded environment latent to base policy observation."""
    return torch.cat([policy_obs_base, env_latent], dim=-1)


def inject_latent_noise(env_latent, noise_std, enabled):
    return env_latent + torch.randn_like(env_latent) * noise_std


def summarize_latent_stats(latents):
    """Compute scalar statistics for encoder latent vectors."""
    flat = latents.reshape(-1)
    per_sample_norm = torch.norm(latents, dim=-1)
    return {
        "mean": flat.mean(),
        "std": flat.std(unbiased=False),
        "min": flat.min(),
        "max": flat.max(),
        "norm_mean": per_sample_norm.mean(),
        "norm_std": per_sample_norm.std(unbiased=False),
    }


def _sample_rows_random(rows, num_samples):
    """Randomly sample rows from [N, D], with replacement if needed."""
    if num_samples <= 0:
        return rows[:0]
    if rows.shape[0] == 0:
        return rows
    if num_samples <= rows.shape[0]:
        indices = torch.randperm(rows.shape[0], device=rows.device)[:num_samples]
    else:
        indices = torch.randint(0, rows.shape[0], (num_samples,), device=rows.device)
    return rows[indices]


def extract_puck_feature_slice(disc_obs_batch, use_action_disc):
    """Extract puck feature sub-vector [direction_sign, downward_speed_bin, vertical_pos_bin_5]."""
    start = 16 if use_action_disc else 8
    return disc_obs_batch[:, start : start + PUCK_FEATURE_DIM]


def sample_large_demo_disc_obs(demo_loader, total_samples, chunk_size):
    """Sample a large random set of demo discriminator observations in chunks."""
    if total_samples <= 0:
        return None
    chunk_size = int(chunk_size)
    total_samples = int(total_samples)
    full_chunks, tail = divmod(total_samples, chunk_size)
    chunks = [demo_loader.sample(chunk_size) for _ in range(full_chunks)]
    if tail:
        chunks.append(demo_loader.sample(tail))
    return torch.cat(chunks, dim=0)


def sample_large_agent_disc_obs(valid_disc_obs, replay_buffer, total_samples):
    """Sample a large random set of agent discriminator observations from current + replay."""
    if total_samples <= 0:
        return None, 0, 0
    total_samples, valid_count, replay_count = int(total_samples), int(valid_disc_obs.shape[0]), int(len(replay_buffer))
    if valid_count == 0 and replay_count == 0:
        return None, 0, 0
    current_target = total_samples if replay_count == 0 else (total_samples // 2 if valid_count > 0 else 0)
    replay_target = total_samples - current_target
    parts = []
    if current_target > 0:
        parts.append(_sample_rows_random(valid_disc_obs, current_target))
    if replay_target > 0:
        parts.append(replay_buffer.sample(replay_target))
    return torch.cat(parts, dim=0), current_target, replay_target


def _summarize_1d(values):
    """Scalar summary for one feature dimension."""
    values = values.float()
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "p10": torch.quantile(values, 0.10).item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p90": torch.quantile(values, 0.90).item(),
    }


def _hist_overlap_score(x_demo, x_agent, bins=40):
    """
    Histogram overlap in [0, 1] for two 1D tensors.
    1 means identical histograms; 0 means no overlap.
    """
    low = min(float(x_demo.min().item()), float(x_agent.min().item()))
    high = max(float(x_demo.max().item()), float(x_agent.max().item()))
    if high - low <= 1e-8:
        return 1.0
    demo_hist = torch.histc(x_demo, bins=bins, min=low, max=high)
    agent_hist = torch.histc(x_agent, bins=bins, min=low, max=high)
    demo_prob = demo_hist / demo_hist.sum().clamp_min(1e-8)
    agent_prob = agent_hist / agent_hist.sum().clamp_min(1e-8)
    return torch.minimum(demo_prob, agent_prob).sum().item()


def _categorical_probs(values, categories=(-1.0, 0.0, 1.0)):
    probs = []
    denom = max(1, int(values.shape[0]))
    for cat in categories:
        probs.append((values == cat).float().sum().item() / float(denom))
    return probs


def compute_puck_feature_diagnostics(demo_puck_features, agent_puck_features):
    """Compute distribution and separability metrics for puck discriminator features."""
    dim_names = ["direction_sign", "downward_speed_bin", "vertical_pos_bin_5"]
    per_dim = {}
    easy_flags = []

    for dim_idx, dim_name in enumerate(dim_names):
        demo_dim = demo_puck_features[:, dim_idx]
        agent_dim = agent_puck_features[:, dim_idx]

        demo_stats = _summarize_1d(demo_dim)
        agent_stats = _summarize_1d(agent_dim)
        pooled_std = np.sqrt(0.5 * (demo_stats["std"] ** 2 + agent_stats["std"] ** 2) + 1e-8)
        z_gap = abs(demo_stats["mean"] - agent_stats["mean"]) / pooled_std

        item = {
            "demo_stats": demo_stats,
            "agent_stats": agent_stats,
            "z_gap": float(z_gap),
        }
        demo_probs = _categorical_probs(demo_dim, categories=(-1.0, -0.5, 0.0, 0.5, 1.0) if dim_name == "vertical_pos_bin_5" else (-1.0, 0.0, 1.0))
        agent_probs = _categorical_probs(agent_dim, categories=(-1.0, -0.5, 0.0, 0.5, 1.0) if dim_name == "vertical_pos_bin_5" else (-1.0, 0.0, 1.0))
        tv_distance = 0.5 * float(sum(abs(d - a) for d, a in zip(demo_probs, agent_probs)))
        item["demo_probs"] = demo_probs
        item["agent_probs"] = agent_probs
        item["tv_distance"] = tv_distance
        if tv_distance > 0.35:
            easy_flags.append(f"{dim_name}:tv={tv_distance:.2f}")
        per_dim[dim_name] = item

    return {
        "per_dim": per_dim,
        "easy_flags": easy_flags,
    }


def print_puck_feature_diagnostics(
    diagnostics,
    iteration,
    valid_count,
    replay_count,
    sampled_current,
    sampled_replay,
    sampled_demo,
    sampled_agent,
):
    """Pretty-print puck feature diagnostics for quick discriminator leakage checks."""
    print("\n" + "-" * 92)
    print(
        f"[puck-diag][iter={iteration}] "
        f"valid_current={valid_count}, replay_size={replay_count}, "
        f"sampled_current={sampled_current}, sampled_replay={sampled_replay}, "
        f"sampled_demo={sampled_demo}, sampled_agent={sampled_agent}"
    )
    per_dim = diagnostics["per_dim"]
    for dim_name in ["direction_sign", "downward_speed_bin", "vertical_pos_bin_5"]:
        dim_info = per_dim[dim_name]
        d = dim_info["demo_stats"]
        a = dim_info["agent_stats"]
        base = (
            f"  {dim_name:<20} "
            f"demo(mean={d['mean']:+.4f},std={d['std']:.4f},p10/p50/p90={d['p10']:+.3f}/{d['p50']:+.3f}/{d['p90']:+.3f}) | "
            f"agent(mean={a['mean']:+.4f},std={a['std']:.4f},p10/p50/p90={a['p10']:+.3f}/{a['p50']:+.3f}/{a['p90']:+.3f}) | "
            f"z_gap={dim_info['z_gap']:.3f}"
        )
        if "tv_distance" in dim_info:
            base += (
                f", tv_distance={dim_info['tv_distance']:.3f}, "
                f"demo_probs={np.array2string(np.array(dim_info['demo_probs']), precision=3)}, "
                f"agent_probs={np.array2string(np.array(dim_info['agent_probs']), precision=3)}"
            )
        print(base)

    if diagnostics["easy_flags"]:
        print("  Potential easy discriminator shortcuts in puck features: " + "; ".join(diagnostics["easy_flags"]))
    else:
        print("  No obvious easy discriminator shortcut from puck features under current diagnostics.")
    print("-" * 92 + "\n")


def get_env_spec_ranges():
    """Single source of truth for uniform randomization ranges."""
    return {
        # Baselines match scripts/smooth_policy/amp_history/configs/new_juggle/new_pid_noise.yaml
        "paddle_density": (2000.0 * 0.5, 2000.0 * 1.5),
        "paddle_damping": (2.5 * 0.5, 2.5 * 1.5),
        "puck_density": (250.0 * 0.5, 250.0 * 1.5),
        "puck_damping": (0.35 * 0.5, 0.35 * 1.5),
        "force_scaling": (1.0 * 0.5, 1.0 * 1.5),
        "pid_kp": (1250.0 * 0.5, 1250.0 * 1.5),
        "pid_kd": (100.0 * 0.5, 100.0 * 1.5),
        "wall_bounce_scale": (0.2 * 0.5, 0.2 * 1.5),
    }


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=_TupleFriendlySafeLoader)


def _serialize_env_spec_ranges(env_spec_ranges):
    return {key: [float(bounds[0]), float(bounds[1])] for key, bounds in env_spec_ranges.items()}


def _coerce_env_spec_ranges(raw_ranges, base_ranges):
    """Merge optional external ranges with defaults and validate shape/value constraints."""
    ordered_keys = get_env_spec_ordered_keys()
    out = {}
    raw_ranges = {} if raw_ranges is None else raw_ranges
    if not isinstance(raw_ranges, dict):
        raise ValueError("randomization_ranges must be a mapping from key -> [low, high].")
    unknown_keys = sorted(set(raw_ranges.keys()) - set(ordered_keys))
    if unknown_keys:
        raise ValueError(f"randomization_ranges contains unknown keys: {unknown_keys}")

    for key in ordered_keys:
        if key in raw_ranges:
            bounds = raw_ranges[key]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Invalid range for '{key}': {bounds}. Expected [low, high].")
            low = float(bounds[0])
            high = float(bounds[1])
            if not high > low:
                raise ValueError(f"Invalid range for '{key}': low={low}, high={high}.")
            out[key] = (low, high)
        else:
            out[key] = tuple(base_ranges[key])
    return out


def _coerce_randomized_keys(randomized_keys, ordered_keys):
    if randomized_keys is None:
        return list(ordered_keys)
    if not isinstance(randomized_keys, list) or len(randomized_keys) == 0:
        raise ValueError("randomized_keys must be a non-empty list when provided.")
    unknown = sorted(set(randomized_keys) - set(ordered_keys))
    if unknown:
        raise ValueError(f"randomized_keys contains unknown entries: {unknown}")
    deduped = []
    seen = set()
    for key in randomized_keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def resolve_env_randomization_config(args):
    """Resolve randomization ranges + randomized subset from defaults, file, and CLI."""
    ordered_keys = get_env_spec_ordered_keys()
    default_ranges = get_env_spec_ranges()
    setup_source_path = None
    file_ranges = None
    file_randomized_keys = None

    if args.env_randomization_setup_path is not None:
        setup_source_path = os.path.abspath(args.env_randomization_setup_path)
        if not os.path.exists(setup_source_path):
            raise FileNotFoundError(f"env_randomization_setup_path does not exist: {args.env_randomization_setup_path}")
        setup_data = _load_yaml(setup_source_path) or {}
        if not isinstance(setup_data, dict):
            raise ValueError("env_randomization_setup_path must point to a YAML mapping/object.")
        file_ranges = setup_data.get("randomization_ranges")
        file_randomized_keys = setup_data.get("randomized_keys")

    configured_ranges = _coerce_env_spec_ranges(file_ranges, base_ranges=default_ranges)
    randomized_keys = _coerce_randomized_keys(file_randomized_keys, ordered_keys)
    if args.env_randomized_keys is not None:
        randomized_keys = _coerce_randomized_keys(args.env_randomized_keys, ordered_keys)

    resolved_ranges = {}
    for key in ordered_keys:
        low, high = configured_ranges[key]
        if key in randomized_keys:
            resolved_ranges[key] = (low, high)
        else:
            fixed_value = float(0.5 * (low + high))
            resolved_ranges[key] = (fixed_value, fixed_value)

    return {
        "source_path": setup_source_path,
        "ordered_keys": ordered_keys,
        "default_ranges": default_ranges,
        "configured_ranges": configured_ranges,
        "resolved_ranges": resolved_ranges,
        "randomized_keys": randomized_keys,
    }


def save_env_randomization_config_artifact(env_randomization_config, output_dir):
    """Persist resolved env randomization config used for this run/checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    artifact_path = os.path.join(output_dir, "env_randomization_config_resolved.yaml")
    randomized = set(env_randomization_config["randomized_keys"])
    payload = {
        "env_randomization_setup_source_path": env_randomization_config["source_path"],
        "ordered_keys": env_randomization_config["ordered_keys"],
        "randomized_keys": env_randomization_config["randomized_keys"],
        "fixed_keys": [key for key in env_randomization_config["ordered_keys"] if key not in randomized],
        "configured_ranges": _serialize_env_spec_ranges(env_randomization_config["configured_ranges"]),
        "resolved_randomization_ranges": _serialize_env_spec_ranges(env_randomization_config["resolved_ranges"]),
    }
    with open(artifact_path, "w") as f:
        yaml.dump(payload, f, sort_keys=False)
    print(f"✓ Saved env randomization config: {artifact_path}")


def get_env_spec_ordered_keys():
    """Canonical environment parameter ordering for env-var vector encoding."""
    return [
        "paddle_density",
        "paddle_damping",
        "puck_density",
        "puck_damping",
        "force_scaling",
        "pid_kp",
        "pid_kd",
        "wall_bounce_scale",
    ]


def validate_env_spec_pool(env_spec_pool):
    """Validate and coerce env spec pool entries into a consistent format."""
    if not isinstance(env_spec_pool, list) or len(env_spec_pool) == 0:
        raise ValueError("Environment spec pool must be a non-empty list.")

    ordered_keys = get_env_spec_ordered_keys()
    required_keys = {"env_id", *ordered_keys}
    ranges = get_env_spec_ranges()
    defaults = {key: float(np.mean(ranges[key])) for key in ordered_keys}
    validated_pool = []
    for idx, spec in enumerate(env_spec_pool):
        if not isinstance(spec, dict):
            raise ValueError(f"Environment spec at index {idx} must be a dict.")
        spec_out = dict(spec)
        # Backward compatibility: older pools may miss newly added randomized variables.
        for key, default_value in defaults.items():
            spec_out.setdefault(key, default_value)
        missing = required_keys - set(spec_out.keys())
        if missing:
            raise ValueError(f"Environment spec at index {idx} is missing keys: {sorted(missing)}")
        spec_out["env_id"] = int(spec_out["env_id"])
        for key in ordered_keys:
            spec_out[key] = float(spec_out[key])
        validated_pool.append(spec_out)
    return validated_pool


def load_env_spec_pool(env_spec_pool_path):
    """Load env spec pool from disk (.pt preferred, YAML fallback)."""
    if not os.path.exists(env_spec_pool_path):
        raise FileNotFoundError(f"env_spec_pool_path does not exist: {env_spec_pool_path}")

    raw_pool = None
    load_errors = []
    try:
        raw_pool = torch.load(env_spec_pool_path, map_location="cpu")
    except Exception as exc:
        load_errors.append(f"torch.load failed: {exc}")

    if raw_pool is None:
        try:
            with open(env_spec_pool_path, "r") as f:
                raw_pool = yaml.safe_load(f)
        except Exception as exc:
            load_errors.append(f"yaml.safe_load failed: {exc}")

    if raw_pool is None:
        raise ValueError(
            "Could not load env spec pool from path "
            f"{env_spec_pool_path}. Errors: {' | '.join(load_errors)}"
        )
    return validate_env_spec_pool(raw_pool)


def build_env_spec_pool(num_randomized_envs_total, seed, env_spec_ranges=None):
    """Create placeholder environment specs for domain randomization."""
    rng = np.random.default_rng(seed)
    ranges = get_env_spec_ranges() if env_spec_ranges is None else env_spec_ranges
    pool = []
    for idx in range(num_randomized_envs_total):
        pool.append(
            {
                "env_id": idx,
                "paddle_density": float(rng.uniform(*ranges["paddle_density"])),
                "paddle_damping": float(rng.uniform(*ranges["paddle_damping"])),
                "puck_density": float(rng.uniform(*ranges["puck_density"])),
                "puck_damping": float(rng.uniform(*ranges["puck_damping"])),
                "force_scaling": float(rng.uniform(*ranges["force_scaling"])),
                "pid_kp": float(rng.uniform(*ranges["pid_kp"])),
                "pid_kd": float(rng.uniform(*ranges["pid_kd"])),
                "wall_bounce_scale": float(rng.uniform(*ranges["wall_bounce_scale"])),
            }
        )
    return pool


def sample_env_spec_from_ranges(rng, env_id, env_spec_ranges=None):
    """Sample one environment spec directly from configured randomization ranges."""
    ranges = get_env_spec_ranges() if env_spec_ranges is None else env_spec_ranges
    return {
        "env_id": int(env_id),
        "paddle_density": float(rng.uniform(*ranges["paddle_density"])),
        "paddle_damping": float(rng.uniform(*ranges["paddle_damping"])),
        "puck_density": float(rng.uniform(*ranges["puck_density"])),
        "puck_damping": float(rng.uniform(*ranges["puck_damping"])),
        "force_scaling": float(rng.uniform(*ranges["force_scaling"])),
        "pid_kp": float(rng.uniform(*ranges["pid_kp"])),
        "pid_kd": float(rng.uniform(*ranges["pid_kd"])),
        "wall_bounce_scale": float(rng.uniform(*ranges["wall_bounce_scale"])),
    }


def save_env_spec_pool_artifacts(env_spec_pool, output_dir):
    """Persist sampled env spec pool so stage-2 adaptation can reuse exact matching specs."""
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, "env_spec_pool.yaml")
    pt_path = os.path.join(output_dir, "env_spec_pool.pt")
    with open(yaml_path, "w") as f:
        yaml.dump(env_spec_pool, f, sort_keys=False)
    torch.save(env_spec_pool, pt_path)
    print(f"✓ Saved env spec pool artifacts: {yaml_path}, {pt_path}")


def save_training_env_setup_manifest(env_spec_pool, output_dir, args, source_mode, source_path, env_randomization_config):
    """Persist startup env setup metadata for reproducibility and resumed training."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "training_env_setup.yaml")
    pool_size = int(len(env_spec_pool)) if env_spec_pool is not None else None
    manifest = {
        "env_spec_pool_source": source_mode,
        "env_spec_pool_source_path": source_path,
        "env_randomization_setup_source_path": env_randomization_config["source_path"],
        "randomized_keys": list(env_randomization_config["randomized_keys"]),
        "randomization_ranges": _serialize_env_spec_ranges(env_randomization_config["resolved_ranges"]),
        "env_var_representation": {
            "env_var_dim": int(args.env_var_dim),
            "env_latent_dim": int(args.env_latent_dim),
            "ordered_keys": get_env_spec_ordered_keys(),
        },
        "env_spec_pool_summary": {
            "configured_num_randomized_envs_total": int(args.num_randomized_envs_total),
            "actual_pool_size": pool_size,
            "seed": int(args.seed),
        },
    }
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, sort_keys=False)
    print(f"✓ Saved training env setup manifest: {manifest_path}")


def build_edge_eval_specs(env_spec_ranges=None):
    """
    Build 5 fixed evaluation environments at range edges using the same
    ranges as training randomization.
    """
    ranges = get_env_spec_ranges() if env_spec_ranges is None else env_spec_ranges
    lows = {key: value[0] for key, value in ranges.items()}
    highs = {key: value[1] for key, value in ranges.items()}

    edge_specs = [
        {"env_id": 10000, "name": "all_low", **lows},
        {"env_id": 10001, "name": "all_high", **highs},
        {
            "env_id": 10002,
            "name": "high_paddle_low_puck_high_force",
            "paddle_density": highs["paddle_density"],
            "paddle_damping": highs["paddle_damping"],
            "puck_density": lows["puck_density"],
            "puck_damping": lows["puck_damping"],
            "force_scaling": highs["force_scaling"],
            "pid_kp": highs["pid_kp"],
            "pid_kd": highs["pid_kd"],
            "wall_bounce_scale": highs["wall_bounce_scale"],
        },
        {
            "env_id": 10003,
            "name": "low_paddle_high_puck_low_force",
            "paddle_density": lows["paddle_density"],
            "paddle_damping": lows["paddle_damping"],
            "puck_density": highs["puck_density"],
            "puck_damping": highs["puck_damping"],
            "force_scaling": lows["force_scaling"],
            "pid_kp": lows["pid_kp"],
            "pid_kd": lows["pid_kd"],
            "wall_bounce_scale": lows["wall_bounce_scale"],
        },
        {
            "env_id": 10004,
            "name": "alternating_edges",
            "paddle_density": highs["paddle_density"],
            "paddle_damping": lows["paddle_damping"],
            "puck_density": highs["puck_density"],
            "puck_damping": lows["puck_damping"],
            "force_scaling": highs["force_scaling"],
            "pid_kp": highs["pid_kp"],
            "pid_kd": lows["pid_kd"],
            "wall_bounce_scale": lows["wall_bounce_scale"],
        },
    ]
    return edge_specs


def save_edge_eval_specs_artifact(edge_specs, output_dir):
    """Persist the fixed edge evaluation specs used by evaluate_on_edge_specs."""
    os.makedirs(output_dir, exist_ok=True)
    edge_specs_path = os.path.join(output_dir, "edge_eval_specs.yaml")
    with open(edge_specs_path, "w") as f:
        yaml.dump(edge_specs, f, sort_keys=False)
    print(f"✓ Saved edge eval specs: {edge_specs_path}")


def sample_random_validation_specs(num_specs, seed, env_id_start=20_000, env_spec_ranges=None):
    """Sample validation environment specs directly from randomization ranges."""
    rng = np.random.default_rng(seed)
    specs = []
    for idx in range(num_specs):
        spec = sample_env_spec_from_ranges(rng, env_id=env_id_start + idx, env_spec_ranges=env_spec_ranges)
        spec["name"] = f"random_validation_env_{idx}"
        specs.append(spec)
    return specs


def extract_env_var_vector_from_spec(spec, env_var_dim, env_spec_ranges=None):
    """
    Pack a fixed-size env-variable vector from one sampled environment spec.
    Variables are normalized to approximately mean 0 / std 1 using the
    uniform-randomization ranges: mean=(low+high)/2, std=(high-low)/sqrt(12).
    """
    ranges = get_env_spec_ranges() if env_spec_ranges is None else env_spec_ranges
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
    base = np.array(normalized, dtype=np.float32)
    copy_len = min(env_var_dim, len(base))
    vec[:copy_len] = base[:copy_len]
    return vec


def parse_env_vars_from_infos(infos, num_envs, env_var_dim, device, fallback_env_vars):
    """Read per-worker env vars from vectorized infos; fallback if missing."""
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
    """
    Keeps one env process alive and re-samples env config on every reset.
    This avoids process respawn overhead while still randomizing from a large pool.
    """

    def __init__(self, env, env_spec_pool, env_var_dim, rng_seed, env_spec_ranges):
        super().__init__(env)
        self.env_spec_pool = env_spec_pool
        self.env_var_dim = env_var_dim
        self.env_spec_ranges = env_spec_ranges
        self.rng = np.random.default_rng(rng_seed)
        self.current_env_spec = None
        self.current_env_var_vec = np.zeros(env_var_dim, dtype=np.float32)
        self.current_env_id = -1

    def _apply_env_spec(self, env_spec):
        self.env.unwrapped.paddle_density = env_spec["paddle_density"]
        self.env.unwrapped.paddle_damping = env_spec["paddle_damping"]
        self.env.unwrapped.puck_density = env_spec["puck_density"]
        self.env.unwrapped.puck_damping = env_spec["puck_damping"]
        self.env.unwrapped.force_scaling = env_spec["force_scaling"]
        self.env.unwrapped.pid_kp = env_spec["pid_kp"]
        self.env.unwrapped.pid_kd = env_spec["pid_kd"]
        self.env.unwrapped.wall_bounce_scale = env_spec["wall_bounce_scale"]
        if hasattr(self.env.unwrapped, "pid_controller"):
            self.env.unwrapped.pid_controller.Kp = env_spec["pid_kp"]
            self.env.unwrapped.pid_controller.Kd = env_spec["pid_kd"]
        # Keep the listener in sync because collision impulses read this value directly.
        if hasattr(self.env.unwrapped, "collision_listener"):
            self.env.unwrapped.collision_listener.wall_bounce_scale = env_spec["wall_bounce_scale"]
        
        self.current_env_var_vec = extract_env_var_vector_from_spec(
            env_spec, self.env_var_dim, env_spec_ranges=self.env_spec_ranges
        )
        self.current_env_id = int(env_spec["env_id"])

    def _sample_and_apply_spec(self):
        idx = int(self.rng.integers(0, len(self.env_spec_pool)))
        self.current_env_spec = self.env_spec_pool[idx]
        self._apply_env_spec(self.current_env_spec)

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


class ResetRangeSampledEnvWrapper(gym.Wrapper):
    """
    Keeps one env process alive and samples fresh env parameters from the
    configured randomization ranges on every reset.
    """

    def __init__(self, env, env_var_dim, rng_seed, env_spec_ranges, env_id_offset=0):
        super().__init__(env)
        self.env_var_dim = env_var_dim
        self.rng = np.random.default_rng(rng_seed)
        self.env_spec_ranges = env_spec_ranges
        self.current_env_spec = None
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
        self.env.unwrapped.pid_kp = env_spec["pid_kp"]
        self.env.unwrapped.pid_kd = env_spec["pid_kd"]
        self.env.unwrapped.wall_bounce_scale = env_spec["wall_bounce_scale"]
        if hasattr(self.env.unwrapped, "pid_controller"):
            self.env.unwrapped.pid_controller.Kp = env_spec["pid_kp"]
            self.env.unwrapped.pid_controller.Kd = env_spec["pid_kd"]
        # Keep the listener in sync because collision impulses read this value directly.
        if hasattr(self.env.unwrapped, "collision_listener"):
            self.env.unwrapped.collision_listener.wall_bounce_scale = env_spec["wall_bounce_scale"]

        self.current_env_var_vec = extract_env_var_vector_from_spec(
            env_spec, self.env_var_dim, env_spec_ranges=self.env_spec_ranges
        )
        self.current_env_id = int(env_spec["env_id"])

    def _sample_and_apply_spec(self):
        env_id = self._env_id_offset + self._sample_idx
        self._sample_idx += 1
        self.current_env_spec = sample_env_spec_from_ranges(
            self.rng, env_id=env_id, env_spec_ranges=self.env_spec_ranges
        )
        self._apply_env_spec(self.current_env_spec)

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


@dataclass
class Args:
    num_envs: int = 16
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

    # AMP hyperparameters (always enabled)
    disc_replay_buffer_size: int = 100000
    disc_replay_samples: int = 1024
    disc_batch_size: int = 512
    disc_learning_rate: float = 1e-5
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 5.0
    disc_weight_decay: float = 0.0001
    task_reward_weight: float = 0.5
    disc_reward_weight: float = 0.5
    num_discriminator_updates: int = 1
    disc_hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    use_long_discriminator: bool = False
    long_disc_reward_weight: float = 0.0
    long_disc_replay_buffer_size: int = 100000
    long_disc_replay_samples: int = 1024
    long_disc_batch_size: int = 512
    long_disc_learning_rate: float = 1e-5
    long_disc_logit_reg: float = 0.01
    long_disc_grad_penalty: float = 5.0
    long_disc_weight_decay: float = 0.0001
    long_num_discriminator_updates: int = 1
    long_disc_hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    long_history_len: int = 30
    long_num_bins: int = 3
    long_samples_per_bin: int = 2
    long_puck_current_index: int = 15
    
    # Optional auxiliary rewards (default disabled)
    temporal_alignment_reward_scale: float = 0.0
    action_magnitude_reward_scale: float = 0.0
    temporal_alignment_horizon: int = 4

    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str = None
    model_path: str = None  # Path to pre-trained model state dict
    encoder_path: str = None  # Path to pre-trained environment encoder state dict
    discriminator_path: str = None  # Path to pre-trained discriminator state dict
    long_discriminator_path: str = None  # Path to second long-window discriminator state dict
    amp_components_path: str = None  # Path to AMP components (normalizer, replay buffer)
    long_amp_components_path: str = None  # Path to long AMP components (normalizer, replay buffer)
    log_parent_dir: str = None
    run_name: str = "default"
    demo_data_path: str = "scripts/smooth_policy/amp_data/amp_dataset.pt"

    # Others
    seed: int = 0
    device: str = "cuda:0"
    stats_log_interval: int = 5  # Log scalar statistics every N iterations
    use_last_action_in_policy_state: bool = False  # Append previous action to policy input state

    # action scale for the agent (mostly deprecated, should default to 1)
    action_scale: float = 1
    
    # agent hidden layer size (2 layers with this size)
    agent_hidden_size: int = 512
    agent_weight_decay: float = 0.00001

    # RMA randomization + encoder args
    num_randomized_envs_total: int = 500
    use_on_demand_env_sampling: bool = True  # If True, sample env parameters at reset directly from ranges.
    env_spec_pool_path: str = None  # Optional path to saved env pool (.pt or .yaml) for continued training
    env_randomization_setup_path: str = None  # Optional YAML with randomization_ranges and randomized_keys
    env_randomized_keys: list[str] = None  # Optional CLI override for randomized key subset
    env_var_dim: int = 8
    env_latent_dim: int = 8
    env_encoder_hidden_size: list[int] = field(default_factory=lambda: [128, 128])
    latent_noise_std: float = 0.05
    edge_eval_episodes: int = 5
    edge_eval_interval: int = 10
    model_save_interval: int = 50
    num_random_validation_gifs: int = 3  # Additional random-range validation GIFs per checkpoint
    validation_gif_episodes: int = 3  # Episodes per validation GIF (random gifs use at least 3)
    
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
    puck_diag_interval_iters: int = 5  # Print puck-feature diagnostics every N training iterations (<=0 disables)
    puck_diag_total_agent_samples: int = 8192  # Large random sample size from current valid + replay buffers
    puck_diag_total_demo_samples: int = 8192  # Large random sample size from demo buffer
    
    

def make_env(env_id, env_spec_pool, env_var_dim, env_spec_ranges, seed=0, use_on_demand_env_sampling=False):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        
        env = AirHockeyEnv(config["air_hockey"])
        if use_on_demand_env_sampling:
            env = ResetRangeSampledEnvWrapper(
                env=env,
                env_var_dim=env_var_dim,
                rng_seed=seed + env_id * 131,
                env_spec_ranges=env_spec_ranges,
                env_id_offset=env_id * 1_000_000,
            )
        else:
            env = ResetSampledEnvWrapper(
                env=env,
                env_spec_pool=env_spec_pool,
                env_var_dim=env_var_dim,
                rng_seed=seed + env_id * 131,
                env_spec_ranges=env_spec_ranges,
            )
        
        return env
    return _thunk


def apply_env_spec_to_unwrapped_env(env, env_spec):
    """Apply one environment spec directly to an environment instance."""
    env.unwrapped.paddle_density = env_spec["paddle_density"]
    env.unwrapped.paddle_damping = env_spec["paddle_damping"]
    env.unwrapped.puck_density = env_spec["puck_density"]
    env.unwrapped.puck_damping = env_spec["puck_damping"]
    env.unwrapped.force_scaling = env_spec["force_scaling"]
    env.unwrapped.pid_kp = env_spec["pid_kp"]
    env.unwrapped.pid_kd = env_spec["pid_kd"]
    env.unwrapped.wall_bounce_scale = env_spec["wall_bounce_scale"]
    if hasattr(env.unwrapped, "pid_controller"):
        env.unwrapped.pid_controller.Kp = env_spec["pid_kp"]
        env.unwrapped.pid_controller.Kd = env_spec["pid_kd"]
    if hasattr(env.unwrapped, "collision_listener"):
        env.unwrapped.collision_listener.wall_bounce_scale = env_spec["wall_bounce_scale"]


def evaluate_on_edge_specs(agent, env_encoder, air_hockey_config, args, device, env_spec_ranges):
    """
    Evaluate the trained policy on five edge-of-range environment specs.
    Runs args.edge_eval_episodes episodes per spec.
    """
    edge_specs = build_edge_eval_specs(env_spec_ranges=env_spec_ranges)
    eval_env = AirHockeyEnv(air_hockey_config)
    action_dim = int(np.prod(eval_env.action_space.shape))

    results = []
    agent.eval()
    env_encoder.eval()
    with torch.no_grad():
        for spec in edge_specs:
            apply_env_spec_to_unwrapped_env(eval_env, spec)
            env_var_np = extract_env_var_vector_from_spec(
                spec, args.env_var_dim, env_spec_ranges=env_spec_ranges
            )
            env_var_tensor = torch.tensor(env_var_np, dtype=torch.float32, device=device).unsqueeze(0)

            episode_returns = []
            episode_lengths = []
            for episode_idx in range(args.edge_eval_episodes):
                obs, _ = eval_env.reset(seed=args.seed + episode_idx)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                done = False
                ep_return = 0.0
                ep_len = 0
                last_action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
                while not done:
                    policy_obs_base = augment_policy_observation(
                        obs_t, last_action, args.use_last_action_in_policy_state
                    )
                    latent = env_encoder(env_var_tensor)
                    policy_obs = concat_env_latent_to_policy_obs(policy_obs_base, latent)
                    action, _, _, _ = agent.get_action_and_value(policy_obs)
                    action_np = action.squeeze(0).cpu().numpy()
                    obs, reward, terminated, truncated, _ = eval_env.step(action_np)
                    done = bool(terminated or truncated)
                    ep_return += float(reward)
                    ep_len += 1
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    last_action = action.detach()
                    if done:
                        last_action.zero_()

                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)

            results.append(
                {
                    "name": spec["name"],
                    "spec": {k: float(v) if isinstance(v, (int, float)) else v for k, v in spec.items()},
                    "avg_return": float(np.mean(episode_returns)),
                    "min_return": float(np.min(episode_returns)),
                    "max_return": float(np.max(episode_returns)),
                    "avg_length": float(np.mean(episode_lengths)),
                    "episodes": int(args.edge_eval_episodes),
                }
            )
            print(
                f"[edge-eval] {spec['name']}: "
                f"avg={np.mean(episode_returns):.3f}, min={np.min(episode_returns):.3f}, max={np.max(episode_returns):.3f}"
            )

    eval_env.close()
    agent.train()
    env_encoder.train()
    return results


def save_validation_gif(
    agent,
    env_encoder,
    env_spec,
    air_hockey_config,
    args,
    gif_savepath,
    env_spec_ranges,
    num_episodes=1,
    episode_seed_offset=0,
):
    """Save a validation GIF over one or more episodes on a fixed environment spec."""
    eval_env = AirHockeyEnv(air_hockey_config.copy())
    apply_env_spec_to_unwrapped_env(eval_env, env_spec)
    renderer = AirHockeyRenderer(eval_env, show_target_position=True, show_acceleration_arrow=False)
    action_dim = int(np.prod(eval_env.action_space.shape))
    env_var_np = extract_env_var_vector_from_spec(
        env_spec, args.env_var_dim, env_spec_ranges=env_spec_ranges
    )
    env_var_tensor = torch.tensor(env_var_np, dtype=torch.float32, device=args.device).unsqueeze(0)

    frames = []
    agent.eval()
    env_encoder.eval()
    with torch.no_grad():
        for episode_idx in tqdm.tqdm(range(num_episodes), desc="validation-gif", leave=False):
            obs, _ = eval_env.reset(seed=args.seed + episode_seed_offset + episode_idx)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            last_action = torch.zeros((1, action_dim), dtype=torch.float32, device=args.device)
            done = False
            rew = 0.0
            cum_rew = 0.0
            while not done:
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
                    f"Episode: {episode_idx + 1}/{num_episodes}",
                    (10, 30),
                    font,
                    font_scale,
                    font_color,
                    line_type,
                )
                cv2.putText(frame, f"Reward: {rew:.2f}", (frame.shape[1] - 150, 30), font, font_scale, font_color, line_type)
                cv2.putText(frame, f"Return: {cum_rew:.2f}", (frame.shape[1] - 150, 60), font, font_scale, font_color, line_type)
                frames.append(frame)

                policy_obs_base = augment_policy_observation(
                    obs_tensor, last_action, args.use_last_action_in_policy_state
                )
                env_latent = env_encoder(env_var_tensor)
                policy_obs = concat_env_latent_to_policy_obs(policy_obs_base, env_latent)
                action, _, _, _ = agent.get_action_and_value(policy_obs)
                action_np = action.squeeze(0).cpu().numpy()
                obs, rew, term, trunc, _ = eval_env.step(action_np)
                done = bool(term or trunc)
                cum_rew += float(rew)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                last_action = action.detach()
                if done:
                    last_action.zero_()

    imageio.mimsave(gif_savepath, frames, format="GIF", loop=0, duration=50)
    eval_env.close()
    agent.train()
    env_encoder.train()

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
    if (
        (not args.use_on_demand_env_sampling)
        and args.env_spec_pool_path is None
        and args.num_randomized_envs_total < args.num_envs
    ):
        raise ValueError(
            f"num_randomized_envs_total ({args.num_randomized_envs_total}) must be >= num_envs ({args.num_envs})."
        )
    if args.env_var_dim <= 0 or args.env_latent_dim <= 0:
        raise ValueError("env_var_dim and env_latent_dim must be positive.")
    if args.stats_log_interval <= 0:
        raise ValueError("stats_log_interval must be a positive integer.")
    if args.num_random_validation_gifs < 0:
        raise ValueError("num_random_validation_gifs must be >= 0.")
    if args.validation_gif_episodes <= 0:
        raise ValueError("validation_gif_episodes must be a positive integer.")
    args.batch_size = args.num_envs * args.num_steps

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    use_on_demand_env_sampling = args.use_on_demand_env_sampling and args.env_spec_pool_path is None
    env_randomization_config = resolve_env_randomization_config(args)
    env_spec_ranges = env_randomization_config["resolved_ranges"]
    print("Resolved env randomization configuration:")
    print(f"  source_path: {env_randomization_config['source_path']}")
    print(f"  randomized_keys: {env_randomization_config['randomized_keys']}")
    if use_on_demand_env_sampling:
        env_spec_pool = None
        env_spec_pool_source_mode = "on_demand_range_sampling"
        env_spec_pool_source_path = None
        print("Using on-demand env sampling from randomization ranges (no fixed env pool).")
    elif args.env_spec_pool_path is not None:
        env_spec_pool = load_env_spec_pool(args.env_spec_pool_path)
        env_spec_pool_source_mode = "loaded"
        env_spec_pool_source_path = os.path.abspath(args.env_spec_pool_path)
        print(f"Loaded env spec pool from: {env_spec_pool_source_path} (size={len(env_spec_pool)})")
    else:
        env_spec_pool = build_env_spec_pool(
            args.num_randomized_envs_total, args.seed, env_spec_ranges=env_spec_ranges
        )
        env_spec_pool_source_mode = "generated"
        env_spec_pool_source_path = None
        print(f"Generated env spec pool from seed={args.seed} (size={len(env_spec_pool)})")
    if env_spec_pool is not None and len(env_spec_pool) < args.num_envs:
        raise ValueError(
            f"Environment spec pool size ({len(env_spec_pool)}) must be >= num_envs ({args.num_envs})."
        )

    edge_eval_specs = build_edge_eval_specs(env_spec_ranges=env_spec_ranges)
    validation_env_spec = edge_eval_specs[0]  # fixed validation environment across checkpoints
    # Persistent vector env: each worker re-samples from the pool inside reset().
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                i,
                env_spec_pool=env_spec_pool,
                env_var_dim=args.env_var_dim,
                env_spec_ranges=env_spec_ranges,
                seed=args.seed,
                use_on_demand_env_sampling=use_on_demand_env_sampling,
            )
            for i in range(args.num_envs)
        ]
    )

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
    if env_spec_pool is not None:
        save_env_spec_pool_artifacts(env_spec_pool=env_spec_pool, output_dir=log_parent_dir)
    save_env_randomization_config_artifact(env_randomization_config=env_randomization_config, output_dir=log_parent_dir)
    save_training_env_setup_manifest(
        env_spec_pool=env_spec_pool,
        output_dir=log_parent_dir,
        args=args,
        source_mode=env_spec_pool_source_mode,
        source_path=env_spec_pool_source_path,
        env_randomization_config=env_randomization_config,
    )
    save_edge_eval_specs_artifact(edge_specs=edge_eval_specs, output_dir=log_parent_dir)
    
    if 'use_pid' in config["air_hockey"] and config["air_hockey"]["use_pid"]:
        action_scale = 1
    else:
        action_scale = args.action_scale # use whatever action scale specified

    base_policy_obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
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
        single_action_space=envs.single_action_space,
    )
    agent = Agent(policy_env_view, action_scale=action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size, activation_type='leaky_relu').to(args.device)
    env_encoder = EnvEncoder(
        env_var_dim=args.env_var_dim,
        latent_dim=args.env_latent_dim,
        hidden_size=args.env_encoder_hidden_size,
    ).to(args.device)
    # Load pre-trained model if path is provided
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully")
    if args.encoder_path is not None:
        if not os.path.exists(args.encoder_path):
            raise FileNotFoundError(f"Encoder path {args.encoder_path} does not exist.")
        print(f"Loading pre-trained encoder from {args.encoder_path}")
        env_encoder.load_state_dict(torch.load(args.encoder_path, map_location=args.device))
        print("Encoder loaded successfully")
    
    optimizer = torch.optim.Adam(
        list(agent.parameters()) + list(env_encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.agent_weight_decay,
        eps=1e-6,
    )
    use_short_discriminator_reward = args.disc_reward_weight > 0.0
    use_long_discriminator_reward = args.use_long_discriminator and args.long_disc_reward_weight > 0.0
    use_discriminator_reward = use_short_discriminator_reward or use_long_discriminator_reward

    # Initialize AMP components (always enabled)
    print("\n" + "="*80)
    print("Initializing AMP (Adversarial Motion Priors) components")
    print("="*80)

    history_len = 5  # Number of positions to track for short discriminator
    disc_obs_dim = 8
    disc_obs_dim_long = 0
    use_action_disc = False
    use_puck_disc = False
    use_puck_disc_long = False
    discriminator, disc_optimizer, disc_normalizer, replay_buffer, demo_loader = None, None, None, None, None
    discriminator_long, disc_optimizer_long, disc_normalizer_long, replay_buffer_long, demo_loader_long = None, None, None, None, None
    
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
        disc_hidden_dims = parse_discriminator_hidden_dims(args.disc_hidden_sizes)
        
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
            hidden_dims=disc_hidden_dims,
            activation='leaky_relu'
        ).to(args.device)
        
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_learning_rate, 
        weight_decay=args.disc_weight_decay, eps=1e-6, betas=(0.5, 0.95)) # lower momentum values

        print(f"✓ Short discriminator initialized (input_dim={disc_obs_dim}, hidden={disc_hidden_dims})")
        
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

    if use_long_discriminator_reward:
        demo_loader_long = DemoLoaderPositionHistory(
            args.demo_data_path,
            device=args.device,
            use_actions=False,
            use_puck=args.use_puck_discriminator,
            puck_vertical_axis=args.puck_vertical_axis,
            puck_downward_positive_direction=args.puck_downward_positive_direction,
            puck_downward_speed_max=args.puck_downward_speed_max,
            puck_speed_dt=args.puck_speed_dt,
            puck_noise_std=args.puck_noise_std,
            puck_vertical_pos_min=args.puck_vertical_pos_min,
            puck_vertical_pos_max=args.puck_vertical_pos_max,
            position_key="position_sequences_30",
            puck_key="puck_sequences_30",
            sample_bucketed_points=True,
            bucket_window_len=args.long_history_len,
            bucket_num_bins=args.long_num_bins,
            bucket_samples_per_bin=args.long_samples_per_bin,
            puck_current_index=args.long_puck_current_index,
        )
        print(f"✓ Long demo loader initialized ({len(demo_loader_long):,} demonstrations)")
        use_puck_disc_long = demo_loader_long.use_puck
        disc_obs_dim_long = demo_loader_long.get_obs_dim()
        long_hidden_dims = parse_discriminator_hidden_dims(args.long_disc_hidden_sizes)
        print(f"  Long mode: POSITION HISTORY (bucketed 8-point from {args.long_history_len}) + "
              f"{'PUCK FEATURES (' + str(PUCK_FEATURE_DIM) + 'D)' if use_puck_disc_long else 'NO PUCK'}")
        print(f"  Long discriminator input dim: {disc_obs_dim_long}")

        discriminator_long = Discriminator(
            disc_obs_dim_long,
            hidden_dims=long_hidden_dims,
            activation='leaky_relu'
        ).to(args.device)
        disc_optimizer_long = torch.optim.Adam(
            discriminator_long.parameters(),
            lr=args.long_disc_learning_rate,
            weight_decay=args.long_disc_weight_decay,
            eps=1e-6,
            betas=(0.5, 0.95),
        )
        print(f"✓ Long discriminator initialized (input_dim={disc_obs_dim_long}, hidden={long_hidden_dims})")

        disc_normalizer_long = Normalizer(shape=(disc_obs_dim_long,), clip=10.0, device=args.device)
        replay_buffer_long = ReplayBuffer(
            capacity=args.long_disc_replay_buffer_size,
            obs_shape=(disc_obs_dim_long,),
            device=args.device
        )
        print(f"✓ Long replay buffer initialized (capacity={args.long_disc_replay_buffer_size:,})")

        if args.long_discriminator_path is not None:
            if not os.path.exists(args.long_discriminator_path):
                raise FileNotFoundError(f"Long discriminator path {args.long_discriminator_path} does not exist.")
            print(f"Loading pre-trained long discriminator from {args.long_discriminator_path}")
            discriminator_long.load_state_dict(torch.load(args.long_discriminator_path, map_location=args.device))
            print("✓ Long discriminator loaded successfully")

        if args.long_amp_components_path is not None:
            if not os.path.exists(args.long_amp_components_path):
                raise FileNotFoundError(f"Long AMP components path {args.long_amp_components_path} does not exist.")
            print(f"Loading long AMP components from {args.long_amp_components_path}")
            long_amp_components = torch.load(args.long_amp_components_path, map_location=args.device)
            disc_normalizer_long.load_state_dict(long_amp_components['normalizer'])
            replay_buffer_long.load_state_dict(long_amp_components['replay_buffer'])
            print(f"✓ Long AMP components loaded successfully (replay buffer size: {len(replay_buffer_long):,})")
    else:
        print("  Long discriminator reward disabled.")
    
    print("="*80 + "\n")


    # main training loop
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    policy_obs_base = torch.zeros((args.num_steps, args.num_envs, base_policy_obs_dim), device=args.device)
    env_var_rollout = torch.zeros((args.num_steps, args.num_envs, args.env_var_dim), device=args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    
    # AMP: Storage for discriminator observations and position history
    disc_obs = torch.zeros((args.num_steps, args.num_envs, disc_obs_dim)).to(args.device) if use_short_discriminator_reward else None
    disc_obs_long = (
        torch.zeros((args.num_steps, args.num_envs, disc_obs_dim_long), device=args.device)
        if use_long_discriminator_reward else None
    )
    paddle_positions = torch.zeros((args.num_steps, args.num_envs, 2)).to(args.device)
    temporal_alignment_reward_raw = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    temporal_alignment_reward_scaled = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    action_magnitude_reward_raw = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    action_magnitude_reward_scaled = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
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
    long_position_history = (
        torch.zeros((args.num_envs, args.long_history_len, 2), device=args.device)
        if use_long_discriminator_reward else None
    )
    long_puck_history = (
        torch.zeros((args.num_envs, args.long_history_len, 2), device=args.device)
        if (use_long_discriminator_reward and use_puck_disc_long) else None
    )
    long_position_count = (
        torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        if use_long_discriminator_reward else None
    )
    long_puck_count = (
        torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        if (use_long_discriminator_reward and use_puck_disc_long) else None
    )
    valid_transition_long = (
        torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool, device=args.device)
        if use_long_discriminator_reward else None
    )

    # Tracking lists for motion metrics
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=args.device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=args.device)
    just_reset = torch.zeros(args.num_envs, dtype=torch.bool).to(args.device)
    last_action_for_policy = torch.zeros((args.num_envs, action_dim), device=args.device)
    current_env_vars = parse_env_vars_from_infos(
        infos=infos,
        num_envs=args.num_envs,
        env_var_dim=args.env_var_dim,
        device=args.device,
        fallback_env_vars=torch.zeros((args.num_envs, args.env_var_dim), device=args.device),
    )
    

    for iteration in range(1, args.num_iterations + 1):
        should_log_stats = (iteration % args.stats_log_interval) == 0
        # Reset episodic return tracking for this iteration
        episodic_returns = []
        success_rates = []

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            policy_next_obs_base = augment_policy_observation(
                next_obs, last_action_for_policy, args.use_last_action_in_policy_state
            )
            policy_obs_base[step] = policy_next_obs_base
            env_var_rollout[step] = current_env_vars

            with torch.no_grad():
                env_latent = env_encoder(current_env_vars)
                env_latent = inject_latent_noise(
                    env_latent, args.latent_noise_std, enabled=True
                )
                policy_next_obs = concat_env_latent_to_policy_obs(policy_next_obs_base, env_latent)
                action, logprob, _, value = agent.get_action_and_value(policy_next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # REWARD SCALING is done on the environment level, not here
            next_done_np = np.logical_or(terminations, truncations)
            next_done_mask = torch.as_tensor(next_done_np, dtype=torch.bool, device=args.device)
            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=args.device).view(-1)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=args.device)
            next_done = next_done_mask.to(dtype=torch.float32)
            current_env_vars = parse_env_vars_from_infos(
                infos=infos,
                num_envs=args.num_envs,
                env_var_dim=args.env_var_dim,
                device=args.device,
                fallback_env_vars=current_env_vars,
            )
            last_action_for_policy = action.detach().clone()
            last_action_for_policy[next_done_mask] = 0
            
            # AMP: Construct discriminator observations from trajectory history
            current_paddle_pos = extract_current_paddle_position(next_obs)
            current_puck_pos = extract_current_puck_position(next_obs) if (use_puck_disc or use_puck_disc_long) else None
            paddle_positions[step] = current_paddle_pos
            
            # Optional action magnitude reward (raw value is independent of scale)
            action_magnitude = actions[step].abs().sum(dim=-1)
            action_mag_raw = (torch.maximum(-action_magnitude, torch.full_like(action_magnitude, -0.25)) + 0.125) * 8.0
            action_magnitude_reward_raw[step] = action_mag_raw
            action_magnitude_reward_scaled[step] = action_mag_raw * args.action_magnitude_reward_scale
            
            if use_short_discriminator_reward:
                # Update short position history buffer (rolling buffer: shift left, add new position at end)
                position_history = torch.roll(position_history, shifts=-1, dims=1)
                position_history[:, -1, :] = current_paddle_pos

                if use_action_disc:
                    # Update action history buffer with current transition action (aligned with next_obs)
                    action_history = torch.roll(action_history, shifts=-1, dims=1)
                    action_history[:, -1, :] = action
                if use_puck_disc:
                    puck_position_history = torch.roll(puck_position_history, shifts=-1, dims=1)
                    puck_position_history[:, -1, :] = current_puck_pos
                
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

            if use_long_discriminator_reward:
                long_position_history = torch.roll(long_position_history, shifts=-1, dims=1)
                long_position_history[:, -1, :] = current_paddle_pos
                long_position_count = torch.clamp(long_position_count + 1, max=args.long_history_len)

                if use_puck_disc_long:
                    long_puck_history = torch.roll(long_puck_history, shifts=-1, dims=1)
                    long_puck_history[:, -1, :] = current_puck_pos
                    long_puck_count = torch.clamp(long_puck_count + 1, max=args.long_history_len)

                has_enough_long_history = long_position_count >= args.long_history_len
                valid_transition_long[step] = has_enough_long_history
                if use_puck_disc_long:
                    has_enough_long_puck = long_puck_count >= args.long_history_len
                    valid_transition_long[step] = valid_transition_long[step] & has_enough_long_puck
                valid_transition_long[step, just_reset] = False

                sampled_indices = sample_bucketed_indices_torch(
                    args.num_envs,
                    window_len=args.long_history_len,
                    num_bins=args.long_num_bins,
                    samples_per_bin=args.long_samples_per_bin,
                    device=args.device,
                )
                gather_idx = sampled_indices.unsqueeze(-1).expand(-1, -1, 2)
                sampled_positions = torch.gather(long_position_history, dim=1, index=gather_idx)
                long_features = [normalize_position_sequence_batch(sampled_positions)]
                if use_puck_disc_long:
                    puck_features_long = build_puck_discriminator_features_torch(
                        long_puck_history,
                        current_index=args.long_puck_current_index,
                        vertical_axis=args.puck_vertical_axis,
                        downward_positive_direction=args.puck_downward_positive_direction,
                        downward_speed_max=args.puck_downward_speed_max,
                        speed_dt=args.puck_speed_dt,
                        noise_std=args.puck_noise_std,
                        vertical_pos_min=args.puck_vertical_pos_min,
                        vertical_pos_max=args.puck_vertical_pos_max,
                    )
                    long_features.append(puck_features_long)
                disc_obs_long[step] = torch.cat(long_features, dim=-1)
            
            # Reset position history and count for environments that are done
            if use_discriminator_reward and next_done_mask.any():
                done_mask = next_done_mask
                if use_short_discriminator_reward:
                    position_history[done_mask] = 0
                    position_history[done_mask, -1, :] = current_paddle_pos[done_mask]
                    position_count[done_mask] = 1  # We have 1 position after reset
                    if use_action_disc:
                        action_history[done_mask] = 0
                        action_count[done_mask] = 0
                    if use_puck_disc:
                        puck_position_history[done_mask] = 0
                        puck_position_history[done_mask, -1, :] = current_puck_pos[done_mask]
                        puck_count[done_mask] = 1
                if use_long_discriminator_reward:
                    long_position_history[done_mask] = 0
                    long_position_history[done_mask, -1, :] = current_paddle_pos[done_mask]
                    long_position_count[done_mask] = 1
                    if use_puck_disc_long:
                        long_puck_history[done_mask] = 0
                        long_puck_history[done_mask, -1, :] = current_puck_pos[done_mask]
                        long_puck_count[done_mask] = 1
            
            # Track which environments just reset for next step
            just_reset = next_done_mask

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        episodic_returns.append(info["episode_return"])
                        success_rates.append(1.0 if info["success"] else 0.0)
                        if should_log_stats:
                            writer.add_scalar("charts/episodic_return", info["episode_return"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode_length"], global_step)
                        if "motion_data" in info:
                            velocity_magnitudes.extend(info["motion_data"]["velocity_mags"])
                            acceleration_magnitudes.extend(info["motion_data"]["acceleration_mags"])
                            jerk_magnitudes.extend(info["motion_data"]["jerk_mags"])

        # bootstrap value if not done and compute advantages
        with torch.no_grad():
            next_policy_obs_base = augment_policy_observation(
                next_obs, last_action_for_policy, args.use_last_action_in_policy_state
            )
            next_env_latent = env_encoder(current_env_vars)
            next_env_latent = inject_latent_noise(
                next_env_latent, args.latent_noise_std, enabled=True
            )
            next_policy_obs = concat_env_latent_to_policy_obs(next_policy_obs_base, next_env_latent)
            next_value = agent.get_value(next_policy_obs).reshape(1, -1)
            
            # Optional temporal alignment reward:
            # compare realized movement over horizon vs commanded action direction from horizon steps ago.
            temporal_alignment_reward_raw.zero_()
            temporal_alignment_reward_scaled.zero_()
            horizon = args.temporal_alignment_horizon
            eps = 1e-8
            if horizon < args.num_steps:
                realized_movement = paddle_positions[horizon:] - paddle_positions[: args.num_steps - horizon]
                target_direction = actions[: args.num_steps - horizon]

                movement_norm = torch.norm(realized_movement, dim=-1).clamp_min(eps)
                target_norm = torch.norm(target_direction, dim=-1)
                cosine_sim = (realized_movement * target_direction).sum(dim=-1) / (
                    movement_norm * target_norm.clamp_min(eps)
                )

                # Apply fallback reward per environment when target direction is near zero.
                small_target_mask = target_norm < 0.03  # hard-coded threshold for now
                cosine_sim = torch.where(
                    small_target_mask,
                    torch.full_like(cosine_sim, 0.75),  # hard-coded reward
                    cosine_sim,
                )

                # Invalidate if episode reset happened between command and realized movement.
                done_int = dones.bool().to(dtype=torch.int32)
                done_prefix = torch.cat(
                    [
                        torch.zeros((1, args.num_envs), dtype=torch.int32, device=args.device),
                        torch.cumsum(done_int, dim=0),
                    ],
                    dim=0,
                )
                # For each t in [horizon, num_steps-1], check done count over [t-horizon+1, t].
                window_done_count = done_prefix[horizon + 1 :] - done_prefix[1 : args.num_steps - horizon + 1]
                temporal_valid = window_done_count == 0

                temporal_alignment_reward_raw[horizon:] = cosine_sim * temporal_valid.float()
            temporal_alignment_reward_scaled = temporal_alignment_reward_raw * args.temporal_alignment_reward_scale
            
            if use_short_discriminator_reward:
                b_disc_obs = disc_obs.reshape(-1, disc_obs_dim)
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

            if use_long_discriminator_reward:
                b_disc_obs_long = disc_obs_long.reshape(-1, disc_obs_dim_long)
                norm_disc_obs_long = disc_normalizer_long.normalize(b_disc_obs_long)
                disc_scores_long = discriminator_long(norm_disc_obs_long).squeeze(-1)
                disc_r_raw_long = torch.clamp(1 - 0.25 * (disc_scores_long - 1) ** 2, min=0)
                b_valid_long = valid_transition_long.reshape(-1)
                disc_r_raw_long = disc_r_raw_long * b_valid_long.float()
                disc_r_raw_shaped_long = disc_r_raw_long.reshape(args.num_steps, args.num_envs)
                disc_r_scaled_long = args.long_disc_reward_weight * disc_r_raw_shaped_long
            else:
                b_disc_obs_long = None
                disc_r_raw_long = torch.zeros(args.num_steps * args.num_envs, device=args.device)
                disc_r_scaled_long = torch.zeros((args.num_steps, args.num_envs), device=args.device)
            task_r_raw = rewards
            task_r_scaled = args.task_reward_weight * task_r_raw
            
            # Combine scaled reward streams only when building PPO targets.
            combined_rewards = (
                task_r_scaled
                + disc_r_scaled
                + disc_r_scaled_long
                + temporal_alignment_reward_scaled
                + action_magnitude_reward_scaled
            )
            
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
            
            # Log reward stream statistics (raw and scaled)
            if should_log_stats:
                if use_short_discriminator_reward:
                    writer.add_scalar("amp/disc_reward_raw_mean", disc_r_raw.mean().item(), global_step)
                    writer.add_scalar("amp/disc_reward_raw_std", disc_r_raw.std().item(), global_step)
                    writer.add_scalar("amp/disc_reward_scaled_mean", disc_r_scaled.mean().item(), global_step)
                    writer.add_scalar("amp/disc_reward_scaled_std", disc_r_scaled.std().item(), global_step)
                if use_long_discriminator_reward:
                    writer.add_scalar("amp_long/disc_reward_raw_mean", disc_r_raw_long.mean().item(), global_step)
                    writer.add_scalar("amp_long/disc_reward_raw_std", disc_r_raw_long.std().item(), global_step)
                    writer.add_scalar("amp_long/disc_reward_scaled_mean", disc_r_scaled_long.mean().item(), global_step)
                    writer.add_scalar("amp_long/disc_reward_scaled_std", disc_r_scaled_long.std().item(), global_step)
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

                # log statistics of the advantages, values
                writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
                writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
                writer.add_scalar("charts/value_mean", values.mean().item(), global_step)
                writer.add_scalar("charts/value_std", values.std().item(), global_step)

        # flatten the batch
        b_policy_obs_base = policy_obs_base.reshape((-1, base_policy_obs_dim))
        b_env_vars = env_var_rollout.reshape((-1, args.env_var_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Log encoder latent statistics from the rollout batch for monitoring.
        with torch.no_grad():
            latent_snapshot = env_encoder(b_env_vars)
            latent_stats = summarize_latent_stats(latent_snapshot)
            if should_log_stats:
                writer.add_scalar("encoder_latent/mean", latent_stats["mean"].item(), global_step)
                writer.add_scalar("encoder_latent/std", latent_stats["std"].item(), global_step)
                writer.add_scalar("encoder_latent/min", latent_stats["min"].item(), global_step)
                writer.add_scalar("encoder_latent/max", latent_stats["max"].item(), global_step)
                writer.add_scalar("encoder_latent/norm_mean", latent_stats["norm_mean"].item(), global_step)
                writer.add_scalar("encoder_latent/norm_std", latent_stats["norm_std"].item(), global_step)
            

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_env_latent = env_encoder(b_env_vars[mb_inds])
                mb_env_latent = inject_latent_noise(
                    mb_env_latent, args.latent_noise_std, enabled=True
                )
                mb_policy_obs = concat_env_latent_to_policy_obs(b_policy_obs_base[mb_inds], mb_env_latent)

                _, newlogprob, _, newvalue = agent.get_action_and_value(mb_policy_obs, b_actions[mb_inds])
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.parameters()) + list(env_encoder.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # AMP: Train short discriminator
        if use_short_discriminator_reward:
            for i in range(args.num_discriminator_updates):
            # Sample demonstration data
                demo_disc_obs = demo_loader.sample(args.disc_batch_size)
            
                # Sample agent data (mix current batch + replay buffer)
                agent_samples = args.disc_batch_size // 2
            
                # Filter to only valid transitions
                b_valid = valid_transition.reshape(-1)
                valid_disc_obs = b_disc_obs[b_valid]
            
                # Randomly sample from valid current batch transitions
                perm_indices = torch.randperm(len(valid_disc_obs), device=args.device)
                agent_disc_obs_current = valid_disc_obs[perm_indices[:agent_samples]]
            
                # Store only valid samples in replay buffer
                num_to_store = min(len(valid_disc_obs), args.disc_replay_samples)
                replay_buffer.push(valid_disc_obs[perm_indices[:num_to_store]])
            
                # Sample from replay buffer if available
                if len(replay_buffer) > 0:
                    agent_disc_obs_replay = replay_buffer.sample(agent_samples)
                    agent_disc_obs = torch.cat([agent_disc_obs_current, agent_disc_obs_replay], dim=0)
                else:
                    agent_disc_obs = agent_disc_obs_current
            
                # Normalize observations
                norm_demo_disc_obs = disc_normalizer.normalize(demo_disc_obs)
                norm_agent_disc_obs = disc_normalizer.normalize(agent_disc_obs)
            
                # Enable gradients for gradient penalty
                norm_demo_disc_obs.requires_grad_(True)
            
                # Forward pass through discriminator (LSGAN outputs raw scores, not logits)
                disc_demo_logit = discriminator(norm_demo_disc_obs).squeeze(-1)
                disc_agent_logit = discriminator(norm_agent_disc_obs).squeeze(-1)
            
                # LSGAN: Least Squares loss
                # Demo = 1 (expert), Agent = -1 (fake)
                # LSGAN uses MSE instead of BCE: 0.5 * E[(D(x_real) - 1)^2] + 0.5 * E[(D(x_fake) - (-1))^2]
                disc_loss_demo = 0.5 * torch.mean((disc_demo_logit - 1.0) ** 2)
                disc_loss_agent = 0.5 * torch.mean((disc_agent_logit - (-1.0)) ** 2)
                disc_loss = disc_loss_demo + disc_loss_agent
            
                # Gradient penalty (Lipschitz constraint)
                disc_demo_grad = torch.autograd.grad(
                    disc_demo_logit, norm_demo_disc_obs,
                    grad_outputs=torch.ones_like(disc_demo_logit),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                disc_grad_penalty = torch.mean(torch.sum(disc_demo_grad ** 2, dim=-1))
                disc_loss = disc_loss + args.disc_grad_penalty * disc_grad_penalty
            
                # Logit regularization
                if args.disc_logit_reg > 0:
                    logit_weights = discriminator.get_logit_weights()
                    disc_logit_reg_loss = torch.sum(logit_weights ** 2)
                    disc_loss = disc_loss + args.disc_logit_reg * disc_logit_reg_loss
            
                # Update discriminator
                disc_optimizer.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                disc_optimizer.step()
            
                # Update normalizer statistics (only with valid transitions)
                b_valid = valid_transition.reshape(-1)
                disc_normalizer.record(b_disc_obs[b_valid])
                disc_normalizer.record(demo_disc_obs)
                disc_normalizer.update()
            
                # Compute discriminator accuracy (LSGAN)
                # For LSGAN, scores closer to 1 for expert, closer to -1 for agent
                with torch.no_grad():
                    disc_agent_acc = (disc_agent_logit < 0.0).float().mean().item()
                    disc_demo_acc = (disc_demo_logit > 0.0).float().mean().item()

            if use_puck_disc and args.puck_diag_interval_iters > 0 and iteration % args.puck_diag_interval_iters == 0:
                b_valid = valid_transition.reshape(-1)
                valid_disc_obs = b_disc_obs[b_valid]
                valid_count = int(valid_disc_obs.shape[0])
                replay_count = int(len(replay_buffer))

                agent_diag_disc_obs, sampled_current_count, sampled_replay_count = sample_large_agent_disc_obs(
                    valid_disc_obs=valid_disc_obs,
                    replay_buffer=replay_buffer,
                    total_samples=args.puck_diag_total_agent_samples,
                )
                demo_diag_disc_obs = sample_large_demo_disc_obs(
                    demo_loader=demo_loader,
                    total_samples=args.puck_diag_total_demo_samples,
                    chunk_size=max(args.disc_batch_size, 512),
                )

                if agent_diag_disc_obs is not None and demo_diag_disc_obs is not None:
                    demo_puck_features = extract_puck_feature_slice(
                        demo_diag_disc_obs, use_action_disc=use_action_disc
                    )
                    agent_puck_features = extract_puck_feature_slice(
                        agent_diag_disc_obs, use_action_disc=use_action_disc
                    )
                    puck_diag = compute_puck_feature_diagnostics(
                        demo_puck_features=demo_puck_features,
                        agent_puck_features=agent_puck_features,
                    )
                    print_puck_feature_diagnostics(
                        diagnostics=puck_diag,
                        iteration=iteration,
                        valid_count=valid_count,
                        replay_count=replay_count,
                        sampled_current=sampled_current_count,
                        sampled_replay=sampled_replay_count,
                        sampled_demo=int(demo_diag_disc_obs.shape[0]),
                        sampled_agent=int(agent_diag_disc_obs.shape[0]),
                    )

                    per_dim = puck_diag["per_dim"]
                    for tag, value in (
                        ("puck_diag/direction_sign_tv", per_dim["direction_sign"]["tv_distance"]),
                        ("puck_diag/downward_speed_bin_tv", per_dim["downward_speed_bin"]["tv_distance"]),
                        ("puck_diag/vertical_pos_bin_tv", per_dim["vertical_pos_bin_5"]["tv_distance"]),
                        ("puck_diag/num_easy_flags", float(len(puck_diag["easy_flags"]))),
                    ):
                        writer.add_scalar(tag, value, iteration)
            
            # Log short discriminator metrics
            if should_log_stats:
                writer.add_scalar("amp/disc_loss", disc_loss.item(), global_step)
                writer.add_scalar("amp/disc_loss_demo", disc_loss_demo.item(), global_step)
                writer.add_scalar("amp/disc_loss_agent", disc_loss_agent.item(), global_step)
                writer.add_scalar("amp/disc_grad_penalty", disc_grad_penalty.item(), global_step)
                writer.add_scalar("amp/disc_agent_acc", disc_agent_acc, global_step)
                writer.add_scalar("amp/disc_demo_acc", disc_demo_acc, global_step)
                writer.add_scalar("amp/disc_agent_logit_mean", disc_agent_logit.mean().item(), global_step)
                writer.add_scalar("amp/disc_demo_logit_mean", disc_demo_logit.mean().item(), global_step)
                writer.add_scalar("amp/replay_buffer_size", len(replay_buffer), global_step)

        # AMP: Train long discriminator
        if use_long_discriminator_reward:
            for i in range(args.long_num_discriminator_updates):
                demo_disc_obs_long = demo_loader_long.sample(args.long_disc_batch_size)
                agent_samples_long = args.long_disc_batch_size // 2

                b_valid_long = valid_transition_long.reshape(-1)
                valid_disc_obs_long = b_disc_obs_long[b_valid_long]
                perm_indices_long = torch.randperm(len(valid_disc_obs_long), device=args.device)
                agent_disc_obs_current_long = valid_disc_obs_long[perm_indices_long[:agent_samples_long]]

                num_to_store_long = min(len(valid_disc_obs_long), args.long_disc_replay_samples)
                replay_buffer_long.push(valid_disc_obs_long[perm_indices_long[:num_to_store_long]])

                if len(replay_buffer_long) > 0:
                    agent_disc_obs_replay_long = replay_buffer_long.sample(agent_samples_long)
                    agent_disc_obs_long = torch.cat([agent_disc_obs_current_long, agent_disc_obs_replay_long], dim=0)
                else:
                    agent_disc_obs_long = agent_disc_obs_current_long

                norm_demo_disc_obs_long = disc_normalizer_long.normalize(demo_disc_obs_long)
                norm_agent_disc_obs_long = disc_normalizer_long.normalize(agent_disc_obs_long)
                norm_demo_disc_obs_long.requires_grad_(True)

                disc_demo_logit_long = discriminator_long(norm_demo_disc_obs_long).squeeze(-1)
                disc_agent_logit_long = discriminator_long(norm_agent_disc_obs_long).squeeze(-1)
                disc_loss_demo_long = 0.5 * torch.mean((disc_demo_logit_long - 1.0) ** 2)
                disc_loss_agent_long = 0.5 * torch.mean((disc_agent_logit_long - (-1.0)) ** 2)
                disc_loss_long = disc_loss_demo_long + disc_loss_agent_long

                disc_demo_grad_long = torch.autograd.grad(
                    disc_demo_logit_long,
                    norm_demo_disc_obs_long,
                    grad_outputs=torch.ones_like(disc_demo_logit_long),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                disc_grad_penalty_long = torch.mean(torch.sum(disc_demo_grad_long ** 2, dim=-1))
                disc_loss_long = disc_loss_long + args.long_disc_grad_penalty * disc_grad_penalty_long

                if args.long_disc_logit_reg > 0:
                    logit_weights_long = discriminator_long.get_logit_weights()
                    disc_logit_reg_loss_long = torch.sum(logit_weights_long ** 2)
                    disc_loss_long = disc_loss_long + args.long_disc_logit_reg * disc_logit_reg_loss_long

                disc_optimizer_long.zero_grad()
                disc_loss_long.backward()
                nn.utils.clip_grad_norm_(discriminator_long.parameters(), args.max_grad_norm)
                disc_optimizer_long.step()

                disc_normalizer_long.record(b_disc_obs_long[b_valid_long])
                disc_normalizer_long.record(demo_disc_obs_long)
                disc_normalizer_long.update()

                with torch.no_grad():
                    disc_agent_acc_long = (disc_agent_logit_long < 0.0).float().mean().item()
                    disc_demo_acc_long = (disc_demo_logit_long > 0.0).float().mean().item()

            if should_log_stats:
                writer.add_scalar("amp_long/disc_loss", disc_loss_long.item(), global_step)
                writer.add_scalar("amp_long/disc_loss_demo", disc_loss_demo_long.item(), global_step)
                writer.add_scalar("amp_long/disc_loss_agent", disc_loss_agent_long.item(), global_step)
                writer.add_scalar("amp_long/disc_grad_penalty", disc_grad_penalty_long.item(), global_step)
                writer.add_scalar("amp_long/disc_agent_acc", disc_agent_acc_long, global_step)
                writer.add_scalar("amp_long/disc_demo_acc", disc_demo_acc_long, global_step)
                writer.add_scalar("amp_long/disc_agent_logit_mean", disc_agent_logit_long.mean().item(), global_step)
                writer.add_scalar("amp_long/disc_demo_logit_mean", disc_demo_logit_long.mean().item(), global_step)
                writer.add_scalar("amp_long/replay_buffer_size", len(replay_buffer_long), global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if should_log_stats:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        if should_log_stats:
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Calculate and log episodic return statistics for this iteration
        if episodic_returns:
            avg_return = np.mean(episodic_returns)
            min_return = np.min(episodic_returns)
            max_return = np.max(episodic_returns)

            print(f"Iteration {iteration}: Avg Return: {avg_return:.2f}, Min Return: {min_return:.2f}, Max Return: {max_return:.2f}")
            print(f"Iteration {iteration}: Avg Success Rate: {np.mean(success_rates):.2f}, Max Success Rate: {np.max(success_rates):.2f}")
            if should_log_stats:
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

            print(
                f"Iteration {iteration}: Avg Velocity Mag: {avg_vel_mag:.4f}, "
                f"Avg Acceleration Mag: {avg_acc_mag:.4f}, Avg Jerk Mag: {avg_jerk_mag:.4f}"
            )

            if should_log_stats:
                writer.add_scalar("motion/avg_velocity_magnitude", avg_vel_mag, iteration)
                writer.add_scalar("motion/avg_acceleration_magnitude", avg_acc_mag, iteration)
                writer.add_scalar("motion/avg_jerk_magnitude", avg_jerk_mag, iteration)

            # Clear lists for next iteration
            velocity_magnitudes.clear()
            acceleration_magnitudes.clear()
            jerk_magnitudes.clear()

        if args.edge_eval_interval > 0 and iteration % args.edge_eval_interval == 0:
            edge_eval_results = evaluate_on_edge_specs(
                agent=agent,
                env_encoder=env_encoder,
                air_hockey_config=config["air_hockey"],
                args=args,
                device=args.device,
                env_spec_ranges=env_spec_ranges,
            )
            if should_log_stats:
                writer.add_scalar(
                    "edge_eval/overall_avg_return",
                    float(np.mean([entry["avg_return"] for entry in edge_eval_results])),
                    iteration,
                )
                for entry in edge_eval_results:
                    spec_name = entry["name"]
                    writer.add_scalar(f"edge_eval/{spec_name}/avg_return", entry["avg_return"], iteration)
                    writer.add_scalar(f"edge_eval/{spec_name}/avg_length", entry["avg_length"], iteration)

        if iteration % args.model_save_interval == 0:
            # save a checkpoint of the model
            # create a subfolder for the checkpoint
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(agent.state_dict(), model_path)
            torch.save(env_encoder.state_dict(), f"{checkpoint_dir}/encoder.pth")
            if env_spec_pool is not None:
                save_env_spec_pool_artifacts(env_spec_pool=env_spec_pool, output_dir=checkpoint_dir)
            save_env_randomization_config_artifact(
                env_randomization_config=env_randomization_config, output_dir=checkpoint_dir
            )
            save_edge_eval_specs_artifact(edge_specs=edge_eval_specs, output_dir=checkpoint_dir)
            
            # Save short AMP components in checkpoint when short discriminator reward is active
            if use_short_discriminator_reward:
                torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
                torch.save({
                    'normalizer': disc_normalizer.state_dict(),
                    'replay_buffer': replay_buffer.state_dict()
                }, f"{checkpoint_dir}/amp_components.pth")
            if use_long_discriminator_reward:
                torch.save(discriminator_long.state_dict(), f"{checkpoint_dir}/discriminator_long.pth")
                torch.save({
                    'normalizer': disc_normalizer_long.state_dict(),
                    'replay_buffer': replay_buffer_long.state_dict()
                }, f"{checkpoint_dir}/amp_components_long.pth")

            validation_gif_path = os.path.join(checkpoint_dir, "validation_fixed_env.gif")
            save_validation_gif(
                agent=agent,
                env_encoder=env_encoder,
                env_spec=validation_env_spec,
                air_hockey_config=config["air_hockey"],
                args=args,
                gif_savepath=validation_gif_path,
                env_spec_ranges=env_spec_ranges,
            )
            if args.num_random_validation_gifs > 0:
                random_specs = sample_random_validation_specs(
                    num_specs=args.num_random_validation_gifs,
                    seed=args.seed + iteration * 10_000,
                    env_id_start=20_000 + iteration * 100,
                    env_spec_ranges=env_spec_ranges,
                )
                random_specs_path = os.path.join(checkpoint_dir, "validation_random_env_specs.yaml")
                with open(random_specs_path, "w") as f:
                    yaml.dump(random_specs, f, sort_keys=False)
                print(f"✓ Saved random validation env specs: {random_specs_path}")

                random_gif_episodes = max(3, args.validation_gif_episodes)
                for spec_idx, random_spec in enumerate(random_specs):
                    random_gif_path = os.path.join(checkpoint_dir, f"validation_random_env_{spec_idx:02d}.gif")
                    save_validation_gif(
                        agent=agent,
                        env_encoder=env_encoder,
                        env_spec=random_spec,
                        air_hockey_config=config["air_hockey"],
                        args=args,
                        gif_savepath=random_gif_path,
                        env_spec_ranges=env_spec_ranges,
                        num_episodes=random_gif_episodes,
                        episode_seed_offset=iteration * 1_000 + spec_idx * 100,
                    )

            print(f"Iteration {iteration} complete")

    # save model
    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")
    torch.save(env_encoder.state_dict(), f"{log_parent_dir}/encoder.pth")
    
    # Save AMP components when discriminator reward is active
    if use_short_discriminator_reward:
        torch.save(discriminator.state_dict(), f"{log_parent_dir}/discriminator.pth")
        torch.save({
            'normalizer': disc_normalizer.state_dict(),
            'replay_buffer': replay_buffer.state_dict()
        }, f"{log_parent_dir}/amp_components.pth")
        print(f"✓ Saved short discriminator and AMP components")
    if use_long_discriminator_reward:
        torch.save(discriminator_long.state_dict(), f"{log_parent_dir}/discriminator_long.pth")
        torch.save({
            'normalizer': disc_normalizer_long.state_dict(),
            'replay_buffer': replay_buffer_long.state_dict()
        }, f"{log_parent_dir}/amp_components_long.pth")
        print(f"✓ Saved long discriminator and AMP components")

    # Evaluate on 5 edge configurations derived from the training randomization ranges.
    edge_eval_results = evaluate_on_edge_specs(
        agent=agent,
        env_encoder=env_encoder,
        air_hockey_config=config["air_hockey"],
        args=args,
        device=args.device,
        env_spec_ranges=env_spec_ranges,
    )
    if (args.num_iterations % args.stats_log_interval) == 0:
        writer.add_scalar(
            "edge_eval/overall_avg_return",
            float(np.mean([entry["avg_return"] for entry in edge_eval_results])),
            args.num_iterations,
        )
        for entry in edge_eval_results:
            spec_name = entry["name"]
            writer.add_scalar(f"edge_eval/{spec_name}/avg_return", entry["avg_return"], args.num_iterations)
            writer.add_scalar(f"edge_eval/{spec_name}/avg_length", entry["avg_length"], args.num_iterations)

    writer.close()
    envs.close()
    
    # end of training