"""Shared evaluation helpers.

Used by scripts/smooth_policy/evaluate.py and scripts/smooth_policy/sim2sim_eval.py
to load a trained policy (stochastic Agent or DeterministicAgent) from a
checkpoint and construct a matching module.
"""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.smooth_policy.residual_agent import ResidualActor


def augment_policy_observation(observation, last_action, use_last_action):
    """Concatenate last_action onto observation if the policy expects it."""
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def unwrap_eval_state_dict(loaded_obj):
    """Normalize different checkpoint formats to a policy state_dict.

    Accepts:
        - a raw actor state_dict (``OrderedDict[str, Tensor]``),
        - a wrapper dict with ``state_dict`` key,
        - a training-state bundle with an ``actor`` key.
    """
    if not isinstance(loaded_obj, dict):
        raise TypeError(f"Expected checkpoint/state_dict to be a dict, got {type(loaded_obj)}")

    candidate = loaded_obj
    if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
        candidate = loaded_obj["state_dict"]
    elif "actor" in loaded_obj and isinstance(loaded_obj["actor"], dict):
        candidate = loaded_obj["actor"]

    tensor_keys = [k for k, v in candidate.items() if isinstance(k, str) and torch.is_tensor(v)]
    if not tensor_keys:
        raise ValueError("Could not find tensor parameters in provided checkpoint/state dict.")
    return candidate


def infer_policy_class_from_state_dict(state_dict):
    """Infer whether a checkpoint is an Agent, DeterministicAgent, or ResidualActor."""
    keys = set(state_dict.keys())
    if any(k.startswith("base.") for k in keys) and any(k.startswith("residual.") for k in keys):
        return "residual_actor"

    has_agent_only_keys = (
        "actor_logstd" in keys
        or "LOG_STD_MIN" in keys
        or "LOG_STD_MAX" in keys
        or "EPS" in keys
        or any(key.startswith("critic.") for key in keys)
        or "critic_head.weight" in keys
    )
    if has_agent_only_keys:
        return "agent"

    has_actor_keys = any(
        key.startswith("actor.") or key.startswith("actor_mean_head.") for key in keys
    )
    if has_actor_keys:
        return "deterministic_agent"

    preview_keys = sorted(list(keys))[:10]
    raise ValueError(
        f"Unable to infer policy type from checkpoint keys. Example keys: {preview_keys}"
    )


def build_policy(
    policy_type,
    policy_env_view,
    action_scale,
    agent_hidden_layer_size,
    agent_num_hidden_layers,
):
    """Construct the policy module matching ``policy_type``."""
    if policy_type == "agent":
        return Agent(
            policy_env_view,
            action_scale=action_scale,
            action_bias=0.0,
            hidden_layer_size=agent_hidden_layer_size,
            num_hidden_layers=agent_num_hidden_layers,
        )
    if policy_type == "deterministic_agent":
        return DeterministicAgent(
            policy_env_view,
            action_scale=action_scale,
            action_bias=0.0,
            hidden_layer_size=agent_hidden_layer_size,
            num_hidden_layers=agent_num_hidden_layers,
        )
    if policy_type == "residual_actor":
        # Buffers (action_scale, residual.action_scale, action_low, action_high)
        # are restored from state_dict, so constructor placeholders are fine.
        base = DeterministicAgent(
            policy_env_view,
            action_scale=action_scale,
            action_bias=0.0,
            hidden_layer_size=agent_hidden_layer_size,
            num_hidden_layers=agent_num_hidden_layers,
        )
        residual = DeterministicAgent(
            policy_env_view,
            action_scale=1.0,
            action_bias=0.0,
            hidden_layer_size=agent_hidden_layer_size,
            num_hidden_layers=agent_num_hidden_layers,
        )
        return ResidualActor(base, residual, action_low=-1.0, action_high=1.0)
    raise ValueError(
        f"Unsupported policy_type '{policy_type}'. Use 'agent', 'deterministic_agent', "
        "or 'residual_actor'."
    )


def load_policy_for_evaluation(
    model_path,
    policy_env_view,
    action_scale,
    agent_hidden_layer_size,
    agent_num_hidden_layers,
    policy_type=None,
    map_location="cpu",
):
    """Load a checkpoint and return the constructed, weight-loaded policy in eval mode."""
    loaded_obj = torch.load(model_path, map_location=map_location, weights_only=False)
    state_dict = unwrap_eval_state_dict(loaded_obj)
    resolved_policy_type = policy_type or infer_policy_class_from_state_dict(state_dict)
    model = build_policy(
        policy_type=resolved_policy_type,
        policy_env_view=policy_env_view,
        action_scale=action_scale,
        agent_hidden_layer_size=agent_hidden_layer_size,
        agent_num_hidden_layers=agent_num_hidden_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_policy_env_view(envs, use_last_action_in_policy_state):
    """Build the env-shaped namespace the policy constructor expects.

    The policy's observation space dim may differ from the env's when
    ``use_last_action_in_policy_state`` is True (policy_obs = env_obs + last_action).
    """
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    policy_obs_dim = obs_dim + action_dim if use_last_action_in_policy_state else obs_dim
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=envs.single_action_space,
    )
