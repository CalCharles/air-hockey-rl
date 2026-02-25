"""
Convert a stochastic Agent checkpoint into a deterministic policy checkpoint.

This is a simple weight transfer:
- no optimization
- no behavior cloning loss
- no parameter averaging
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro
import yaml
from types import SimpleNamespace

from scripts.smooth_policy.deterministic_agent import DeterministicAgent


@dataclass
class Args:
    args_file: str = None
    model_path: str = None
    output_path: str = None
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    use_last_action_in_policy_state: bool = False
    action_scale: float = 1.0
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    agent_hidden_size: int | None = None
    device: str = "cpu"


def build_policy_env_view(config_path: str, use_last_action_in_policy_state: bool):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Import locally to keep this script lightweight for pure checkpoint conversion use.
    from airhockey import AirHockeyEnv

    env = AirHockeyEnv(config["air_hockey"])
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        policy_obs_dim = obs_dim + act_dim if use_last_action_in_policy_state else obs_dim
        return SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(policy_obs_dim,),
                dtype=np.float32,
            ),
            single_action_space=env.action_space,
        )
    finally:
        env.close()


def extract_deterministic_state_dict(stochastic_state_dict):
    deterministic_state = {}
    for key, value in stochastic_state_dict.items():
        if key.startswith("actor.") or key.startswith("actor_mean_head."):
            deterministic_state[key] = value
        if key in ("action_scale", "action_bias"):
            deterministic_state[key] = value
    return deterministic_state


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
    if args.agent_num_hidden_layers < 1:
        raise ValueError("agent_num_hidden_layers must be >= 1.")

    policy_env_view = build_policy_env_view(args.config, args.use_last_action_in_policy_state)
    deterministic_agent = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(args.device)

    source_state_dict = torch.load(args.model_path, map_location=args.device)
    deterministic_state_dict = extract_deterministic_state_dict(source_state_dict)
    load_result = deterministic_agent.load_state_dict(deterministic_state_dict, strict=False)

    unexpected = list(load_result.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected keys during deterministic load: {unexpected}")

    output_parent = Path(args.output_path).parent
    if str(output_parent) not in ("", "."):
        output_parent.mkdir(parents=True, exist_ok=True)
    torch.save(deterministic_agent.state_dict(), args.output_path)

    print("Deterministic distillation complete.")
    print(f"Source model: {args.model_path}")
    print(f"Saved deterministic model: {args.output_path}")
    print(f"Missing keys (expected if source has critic/logstd params): {list(load_result.missing_keys)}")

