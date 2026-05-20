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
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.helper.q_network import TD3QNetwork
from scripts.td3.helper.td3_cql import cql_penalty, precompute_cql_terms
from scripts.td3.helper.td3_residual import build_residual_training
from scripts.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)
from scripts.td3.helper.replay_buffer import TD3ReplayBuffer
from scripts.td3.helper.prioritized_replay_buffer import (
    TD3PrioritizedReplayBuffer,
)
from scripts.td3.helper.td3_checkpointing import (
    build_training_state,
    load_resume_training_state,
)
from scripts.td3.helper.td3_episode_collection import (
    EpisodeTrajectory,
    finalize_episode_if_done,
)
from scripts.td3.helper.td3_metrics import (
    initialize_train_metrics,
    log_scalar_metrics,
)
from scripts.td3.helper.td3_replay_sampling import (
    concat_replay_samples,
    critic_success_failure_counts,
    sample_actor_source_chunk,
    sample_critic_source_chunk,
)
from scripts.td3.evaluate import evaluate_agent
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
    gamma: float = 0.975
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    q_frequency: int = 1
    q_updates: int = 1
    # Critic ensemble (REDQ-style — Chen et al. ICLR 2021).
    # num_critics=2 reproduces vanilla TD3 (default; backwards compatible).
    # num_critics>2 with target_critic_subset_size=None → Maxmin-N (min over all N targets).
    # num_critics>2 with target_critic_subset_size=M (M<N) → REDQ-N-M (min over a random M-subset).
    num_critics: int = 2
    target_critic_subset_size: int | None = None
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
    # Age-weighted PER: multiplies sample priorities by exp(-priority_age_decay
    # * age_in_slots) before alpha-scaling. age_in_slots is 0 for the most
    # recently added transition and grows linearly with eviction order.
    # Implements "stochastic recency-weighted sampling" — orthogonal to FIFO
    # eviction (which binary-evicts) and TD-error PER (age-blind). 0.0 disables.
    # Reasonable: 1e-5 (gentle, half-life ≈ 70k slots), 1e-4 (medium, ≈7k),
    # 1e-3 (aggressive, ≈700). Used in residual_rl_paddle50_log.md v9+.
    priority_age_decay: float = 0.0
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
    exploration_primitive_weight_target_position_directional: float = 0.0
    exploration_primitive_weight_anneal_stand_still: float = 0.3
    exploration_primitive_weight_anneal_same_direction: float = 0.1
    exploration_primitive_weight_anneal_y_aligned: float = 0.6
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

    # Checkpointing
    checkpoint_interval: int = 25000
    save_replay_buffer: bool = True

    # Paths
    config: str = "configs/new_juggle/sysid_best_params.yaml"
    args_file: str | None = None
    model_path: str | None = None
    # Full checkpoint load behavior when model_path points to a training-state dict.
    # - "full_resume": restore full training runtime state (legacy/default behavior)
    # - "weights_only": restore actor/Q networks only, keep runtime fresh
    # - "residual": load source actor as frozen base, build fresh residual + fresh critic
    full_checkpoint_load: Literal["full_resume", "weights_only", "residual"] = "full_resume"
    # Residual RL: max magnitude of the residual action component (used when
    # full_checkpoint_load=="residual"). Combined action is clipped to the env
    # action bounds, so residual_scale > 0 caps |residual|_inf via tanh.
    residual_scale: float = 0.25
    # L2 weight decay on the residual actor's parameters (Adam weight_decay).
    # > 0 keeps the residual head close to zero even when the critic encourages
    # large corrections — counteracts long-horizon drift at residual_scale=0.15.
    residual_weight_decay: float = 0.0
    # CQL (Kumar et al. 2020): conservative-Q penalty on critic loss.
    # If cql_alpha > 0, add `cql_alpha * (logsumexp_a Q(s,a) - Q(s, pi(s)))` to
    # each critic's loss. logsumexp is approximated by sampling `cql_n_random`
    # uniform actions in [-1,1]^act_dim per state.
    cql_alpha: float = 0.0
    cql_n_random: int = 10
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
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2

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

    # Live episode GIF recording
    watch_ring_size: int = 10
    watch_episode_interval: int = 50
    sample_gif_interval: int = 10000
    sample_gif_max_storage_mb: float = 50.0

    # Multi-env evaluation (used by td3_training_dr.py wrapper).
    # When eval_param_seed is None (default), behavior is unchanged.
    # When set, the wrapper monkey-patches `evaluate_agent` to roll N
    # episodes through each of `eval_n_envs` fixed seed-sampled environments
    # and aggregate; per-env stats are dumped to <ckpt_dir>/multi_env_eval.json.
    eval_param_seed: int | None = None
    eval_n_envs: int = 1
    eval_eps_per_env: int = 4


def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env

    return _thunk


def validate_args(args: "Args") -> None:
    """Range / mutual-exclusion checks for args; raises ValueError on misconfig."""
    def _positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")

    def _fraction(name: str, value: float, *, exclusive: bool = False) -> None:
        lo_ok = 0.0 < value if exclusive else 0.0 <= value
        hi_ok = value < 1.0 if exclusive else value <= 1.0
        if not (lo_ok and hi_ok):
            bracket = "(0, 1)" if exclusive else "[0, 1]"
            raise ValueError(f"{name} must be in {bracket}, got {value}.")

    def _sums_to_one(name1: str, name2: str, v1: float, v2: float) -> None:
        total = float(v1 + v2)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{name1} + {name2} must equal 1.0, got {total:.6f}.")

    if args.num_envs != 1:
        raise ValueError(
            "This training script currently supports only single-environment collection. "
            f"Set num_envs=1, got {args.num_envs}."
        )
    _fraction("critic_per_fraction", args.critic_per_fraction)
    _fraction("critic_uniform_fraction", args.critic_uniform_fraction)
    _sums_to_one(
        "critic_per_fraction", "critic_uniform_fraction",
        args.critic_per_fraction, args.critic_uniform_fraction,
    )
    _positive("success_buffer_size", args.success_buffer_size)
    _positive("failure_buffer_size", args.failure_buffer_size)
    _positive("recent_episode_window_size", args.recent_episode_window_size)
    _fraction("success_top_fraction", args.success_top_fraction, exclusive=True)
    _fraction("critic_success_sample_fraction", args.critic_success_sample_fraction)
    _fraction("critic_failure_sample_fraction", args.critic_failure_sample_fraction)
    _sums_to_one(
        "critic_success_sample_fraction", "critic_failure_sample_fraction",
        args.critic_success_sample_fraction, args.critic_failure_sample_fraction,
    )
    _positive("q_updates", args.q_updates)
    _positive("target_network_frequency", args.target_network_frequency)
    _positive("actor_updates_per_iteration", args.actor_updates_per_iteration)
    if args.target_network_frequency > args.q_updates:
        # The Polyak gate counts completed critic updates globally (see
        # total_critic_updates), so this still fires — just less often than
        # once per cycle. Loud warning since it's almost always a config typo.
        print(
            f"[warn] target_network_frequency ({args.target_network_frequency}) "
            f"> q_updates ({args.q_updates}); target nets will update less than "
            f"once per training cycle."
        )
    for name in ("same_direction", "y_aligned", "target_position_directional"):
        validate_optional_exploration_range(
            primitive_name=name,
            min_angle_deg=getattr(args, f"exploration_{name}_min_angle_deg"),
            max_angle_deg=getattr(args, f"exploration_{name}_max_angle_deg"),
            min_magnitude=getattr(args, f"exploration_{name}_min_magnitude"),
            max_magnitude=getattr(args, f"exploration_{name}_max_magnitude"),
        )


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


def _entrypoint():
    """Entry point exposed so wrapper scripts (e.g. td3_training_dr.py)
    can monkey-patch module-level callables (notably `evaluate_agent`) and
    then invoke the full training loop. Behavior is identical to running
    `python -m scripts.td3.td3_training`
    directly."""
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()

    args = tyro.cli(Args, default=default_args)
    validate_args(args)

    # `config` must be a MODULE-LEVEL name because the module-level
    # `make_env(env_id)._thunk` closure (line ~661) reads it as a free
    # variable. Before the _entrypoint() refactor, `config` lived at top
    # level naturally; now we have to declare it global so the assignment
    # below writes to the module namespace where _thunk can find it.
    global config
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
    if args.num_critics < 2:
        raise ValueError(f"num_critics must be >=2, got {args.num_critics}")
    if args.target_critic_subset_size is not None and not (
        1 <= args.target_critic_subset_size <= args.num_critics
    ):
        raise ValueError(
            f"target_critic_subset_size must be in [1, num_critics={args.num_critics}], "
            f"got {args.target_critic_subset_size}"
        )
    # Build N critics + N targets. RNG advances between modules → diverse inits.
    qfs = [
        TD3QNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_layer_size=args.q_hidden_layer_size,
            num_hidden_layers=args.q_num_hidden_layers,
        ).to(args.device)
        for _ in range(args.num_critics)
    ]
    qfs_target = [
        TD3QNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_layer_size=args.q_hidden_layer_size,
            num_hidden_layers=args.q_num_hidden_layers,
        ).to(args.device)
        for _ in range(args.num_critics)
    ]
    for q, qt in zip(qfs, qfs_target):
        qt.load_state_dict(q.state_dict())
    # Legacy aliases — preserved so anything outside the training loop that
    # still references qf1/qf2 keeps working. Critics 3+ live only in qfs.
    qf1, qf2 = qfs[0], qfs[1]
    qf1_target, qf2_target = qfs_target[0], qfs_target[1]
    resume_checkpoint = None
    checkpoint_load_mode = args.full_checkpoint_load
    residual_actor_optimizer: optim.Optimizer | None = None

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
            action_low_tensor = torch.as_tensor(
                envs.single_action_space.low, dtype=torch.float32, device=args.device
            )
            action_high_tensor = torch.as_tensor(
                envs.single_action_space.high, dtype=torch.float32, device=args.device
            )
            actor, actor_target, residual_actor_optimizer = build_residual_training(
                base_actor=actor,
                policy_env_view=policy_env_view,
                action_low=action_low_tensor,
                action_high=action_high_tensor,
                device=args.device,
                residual_scale=args.residual_scale,
                residual_weight_decay=args.residual_weight_decay,
                agent_hidden_layer_size=args.agent_hidden_layer_size,
                agent_num_hidden_layers=args.agent_num_hidden_layers,
                policy_lr=args.policy_lr,
            )
        elif is_full_state:
            resume_checkpoint = loaded_obj
            if args.eval_mode:
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
            # Backwards-compat: legacy ckpts have only qf1/qf2 keys. Newer
            # ckpts (num_critics>2) add qf3, qf4, ... and corresponding _target.
            # When resuming with a LARGER ensemble than the ckpt (e.g., fine-tuning
            # a 2-critic source into a 5-critic ensemble), load what's available
            # and leave the extra critics at their fresh init.
            n_in_ckpt = sum(
                1
                for k in resume_checkpoint
                if k.startswith("qf") and not k.endswith("_target") and k[2:].isdigit()
            )
            if n_in_ckpt > args.num_critics:
                raise ValueError(
                    f"Resume mismatch: checkpoint has {n_in_ckpt} critics but "
                    f"args.num_critics={args.num_critics} is smaller. "
                    "Cannot drop critics from an ensemble checkpoint."
                )
            n_to_load = min(n_in_ckpt, args.num_critics)
            for i in range(1, n_to_load + 1):
                qfs[i - 1].load_state_dict(resume_checkpoint[f"qf{i}"])
                qfs_target[i - 1].load_state_dict(resume_checkpoint[f"qf{i}_target"])
            if n_in_ckpt < args.num_critics:
                # Critical: when expanding from a smaller ensemble (e.g., 2-critic
                # source -> 5-critic ensemble), do NOT leave qf{n+1}..qf{N} at
                # fresh init. The min-over-critics target would then be dominated
                # by the untrained fresh critics (Q ~ 0), collapsing the entire
                # Q estimate and crashing the actor. Instead, clone qf1's loaded
                # weights into all extra slots — diversity will emerge through
                # subsequent independent gradient updates.
                src_state = qfs[0].state_dict()
                src_target_state = qfs_target[0].state_dict()
                for i in range(n_to_load + 1, args.num_critics + 1):
                    qfs[i - 1].load_state_dict(src_state)
                    qfs_target[i - 1].load_state_dict(src_target_state)
                print(
                    f"Partial critic load: checkpoint has {n_in_ckpt} critics, "
                    f"args.num_critics={args.num_critics}. Loaded qf1..qf{n_to_load}, "
                    f"cloned qf1 into qf{n_to_load+1}..qf{args.num_critics} "
                    f"(extra critics start with same weights; diverge via training updates)."
                )
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
        [p for q in qfs for p in q.parameters()],
        lr=args.q_lr,
        weight_decay=args.q_weight_decay,
    )
    if residual_actor_optimizer is not None:
        actor_optimizer = residual_actor_optimizer
    else:
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.per_enabled:
        success_rb = TD3PrioritizedReplayBuffer(
            buffer_size=args.success_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            device=args.device,
            n_envs=args.num_envs,
            alpha=args.per_alpha,
            priority_eps=args.per_eps,
            age_decay=args.priority_age_decay,
        )
        failure_rb = TD3PrioritizedReplayBuffer(
            buffer_size=args.failure_buffer_size,
            obs_shape=envs.single_observation_space.shape,
            action_shape=envs.single_action_space.shape,
            device=args.device,
            n_envs=args.num_envs,
            alpha=args.per_alpha,
            priority_eps=args.per_eps,
            age_decay=args.priority_age_decay,
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

    obs, _ = envs.reset(seed=args.seed)
    last_action_for_policy = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=args.device)
    interval_paddle_puck_collisions = 0.0
    interval_env_steps = 0
    interval_primitive_env_steps = 0
    interval_primitive_horizontal_env_steps = 0
    interval_target_position_directional_env_steps = 0
    rolling_step_stats_window = deque()
    rolling_episode_stats_window = deque()
    prev_protective_stop_flags = np.zeros(args.num_envs, dtype=bool)
    episode_trajectory = EpisodeTrajectory.empty()
    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0

    initial_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
    previous_puck_position_for_trigger = extract_current_puck_position(initial_obs_tensor).clone()

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
    # Counts completed critic updates across all training cycles. Used to gate
    # Polyak averaging by `target_network_frequency` so the schedule survives
    # cycles where q_updates < target_network_frequency or where a critic step
    # is skipped (empty replay batch).
    total_critic_updates = 0

    train_metrics = initialize_train_metrics()

    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=args.device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=args.device)

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
    )
    primitive_selector.set_primitive_weights(
        stand_still=args.exploration_primitive_weight_stand_still,
        same_direction=args.exploration_primitive_weight_same_direction,
        y_aligned=args.exploration_primitive_weight_y_aligned,
        target_position_directional=args.exploration_primitive_weight_target_position_directional,
    )
    if args.eval_mode:
        primitive_selector.chance = 0.0
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
                "train_metrics": train_metrics,
                "interval_paddle_puck_collisions": interval_paddle_puck_collisions,
                "interval_env_steps": interval_env_steps,
                "interval_primitive_env_steps": interval_primitive_env_steps,
                "interval_primitive_horizontal_env_steps": interval_primitive_horizontal_env_steps,
                "interval_target_position_directional_env_steps": (
                    interval_target_position_directional_env_steps
                ),
                "recent_episode_returns": recent_episode_returns,
                "episode_return_success_threshold": episode_return_success_threshold,
                "rolling_step_stats_window": rolling_step_stats_window,
                "rolling_episode_stats_window": rolling_episode_stats_window,
            },
        )
        global_step = restored_state["global_step"]
        iteration = restored_state["iteration"]
        total_critic_updates = restored_state["total_critic_updates"]
        obs = restored_state["obs"]
        previous_puck_position_for_trigger = extract_current_puck_position(
            torch.tensor(obs, dtype=torch.float32, device=args.device)
        ).clone()
        last_action_for_policy = restored_state["last_action_for_policy"]
        train_metrics = restored_state["train_metrics"]
        interval_paddle_puck_collisions = restored_state["interval_paddle_puck_collisions"]
        interval_env_steps = restored_state["interval_env_steps"]
        interval_primitive_env_steps = restored_state["interval_primitive_env_steps"]
        interval_primitive_horizontal_env_steps = restored_state["interval_primitive_horizontal_env_steps"]
        interval_target_position_directional_env_steps = restored_state[
            "interval_target_position_directional_env_steps"
        ]
        episode_trajectory = restored_state["episode_trajectory"]
        recent_episode_returns = restored_state["recent_episode_returns"]
        episode_return_success_threshold = restored_state["episode_return_success_threshold"]
        rolling_step_stats_window = restored_state["rolling_step_stats_window"]
        rolling_episode_stats_window = restored_state["rolling_episode_stats_window"]
        print(f"Resuming training from global_step={global_step}, iteration={iteration}")

    def save_full_checkpoint(out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/config.yaml", "w") as f:
            yaml.dump(config, f)
        with open(f"{out_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f)
        model_path_local = f"{out_dir}/model.pth"
        torch.save(actor.state_dict(), model_path_local)
        torch.save(actor_target.state_dict(), f"{out_dir}/actor_target.pth")
        for ci, q in enumerate(qfs, start=1):
            torch.save(q.state_dict(), f"{out_dir}/qf{ci}.pth")
            torch.save(qfs_target[ci - 1].state_dict(), f"{out_dir}/qf{ci}_target.pth")
        state = build_training_state(
            global_step=global_step,
            iteration=iteration,
            total_critic_updates=total_critic_updates,
            actor=actor,
            actor_target=actor_target,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1_target,
            qf2_target=qf2_target,
            extra_qfs=qfs[2:] if args.num_critics > 2 else None,
            extra_qfs_target=qfs_target[2:] if args.num_critics > 2 else None,
            q_optimizer=q_optimizer,
            actor_optimizer=actor_optimizer,
            success_rb=success_rb,
            failure_rb=failure_rb,
            primitive_selector=primitive_selector,
            obs=obs,
            last_action_for_policy=last_action_for_policy,
            train_metrics=train_metrics,
            interval_paddle_puck_collisions=interval_paddle_puck_collisions,
            interval_env_steps=interval_env_steps,
            interval_primitive_env_steps=interval_primitive_env_steps,
            interval_primitive_horizontal_env_steps=interval_primitive_horizontal_env_steps,
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
        torch.save(state, f"{out_dir}/training_state.pth")
        return model_path_local

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
                    target_position_directional=(
                        args.exploration_primitive_weight_anneal_target_position_directional
                    ),
                )
            else:
                primitive_selector.set_primitive_weights(
                    stand_still=args.exploration_primitive_weight_stand_still,
                    same_direction=args.exploration_primitive_weight_same_direction,
                    y_aligned=args.exploration_primitive_weight_y_aligned,
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
        else:
            primitive_step_stats = {
                "primitive_applied_count": 0,
                "primitive_horizontal_dominant_count": 0,
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

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=args.device)

        if recording_episode:
            recording_last_rew = float(rewards_tensor[0].item())
            recording_cum_rew += recording_last_rew

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
                reward=rewards_tensor[0],
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
                    source_batch_size = int(source_data["rewards"].shape[0])
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
                sampled_rewards = data["rewards"]
                sampled_dones = data["dones"]
                sampled_prev_actions = data["prev_actions"]
                sampled_weights = data.get("weights")
                if sampled_weights is None:
                    sampled_weights = torch.ones_like(sampled_rewards)
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

                    # Subset selection for the target Q (REDQ-style; min over a
                    # sampled M-subset). When target_critic_subset_size is None
                    # OR equals num_critics, behaves as Maxmin-N.
                    if (
                        args.target_critic_subset_size is None
                        or args.target_critic_subset_size >= args.num_critics
                    ):
                        target_indices = list(range(args.num_critics))
                    else:
                        target_indices = (
                            torch.randperm(args.num_critics)[
                                : args.target_critic_subset_size
                            ]
                            .tolist()
                        )
                    next_q_h_list = [
                        qfs_target[ti](sampled_next_observations, target_next_action)
                        for ti in target_indices
                    ]
                    if len(next_q_h_list) == 1:
                        min_next_q_h = next_q_h_list[0]
                    elif len(next_q_h_list) == 2:
                        min_next_q_h = torch.min(next_q_h_list[0], next_q_h_list[1])
                    else:
                        min_next_q_h = torch.stack(next_q_h_list, dim=0).min(dim=0).values

                    min_next_q = h_inverse(min_next_q_h, eps=args.h_transform_eps).view(-1)

                    bellman_target_original = sampled_rewards + (
                        1 - sampled_dones
                    ) * args.gamma * min_next_q

                    next_q_value_h = h_transform(
                        bellman_target_original, eps=args.h_transform_eps
                    )

                    if should_update_train_metrics and q_update_idx == args.q_updates - 1:
                        train_metrics.update(
                            {
                                "debug/bellman_target_original_mean": (
                                    bellman_target_original.mean().item()
                                ),
                                "debug/next_q_h_mean": next_q_value_h.mean().item(),
                            }
                        )

                # Forward pass over all N critics; train each against the shared target.
                qi_h_list = []
                qi_err_list = []
                qi_loss_list = []
                cql_terms = None
                if args.cql_alpha > 0.0:
                    cql_policy_obs = augment_policy_observation(
                        sampled_observations,
                        sampled_prev_actions,
                        args.use_last_action_in_policy_state,
                    )
                    with torch.no_grad():
                        cql_policy_action = deterministic_actor_action(
                            actor, cql_policy_obs
                        )
                    cql_terms = precompute_cql_terms(
                        sampled_observations=sampled_observations,
                        policy_action=cql_policy_action,
                        act_dim=act_dim,
                        n_random=int(args.cql_n_random),
                    )
                for q in qfs:
                    qi_h = q(sampled_observations, sampled_actions)
                    qi_err = qi_h.view(-1) - next_q_value_h
                    qi_h_list.append(qi_h)
                    qi_err_list.append(qi_err)
                    loss_i = (sampled_weights * qi_err.pow(2)).mean()
                    if cql_terms is not None:
                        loss_i = loss_i + args.cql_alpha * cql_penalty(
                            q, sampled_observations, cql_terms
                        )
                    qi_loss_list.append(loss_i)
                q1_h = qi_h_list[0]

                q_total_loss = sum(qi_loss_list)

                q_optimizer.zero_grad()
                q_total_loss.backward()
                q_optimizer.step()

                # PER priority: mean of |TD error| across critics.
                priority_td_error = sum(e.abs() for e in qi_err_list) / args.num_critics
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
                    positive_reward_mask = sampled_rewards > 0.0
                    positive_reward_count = float(positive_reward_mask.sum().item())
                    minibatch_size = max(int(sampled_rewards.numel()), 1)
                    positive_rewards = sampled_rewards[positive_reward_mask]
                    priority_td_error_mean = (
                        priority_td_error.mean().item()
                        if args.per_enabled and per_sample_count > 0
                        else 0.0
                    )
                    train_metrics.update(
                        {
                            "losses/q_loss": sum(l.item() for l in qi_loss_list) / args.num_critics,
                            "losses/q_total_loss": q_total_loss.item(),
                            "losses/q1_mean": q1_h.mean().item(),
                            "rewards/sampled_reward_mean": sampled_rewards.mean().item(),
                            "rewards/sampled_reward_min": sampled_rewards.min().item(),
                            "rewards/sampled_reward_std": sampled_rewards.std(
                                unbiased=False
                            ).item(),
                            "rewards/sampled_reward_positive_count": positive_reward_count,
                            "rewards/sampled_reward_positive_fraction": (
                                positive_reward_count / float(minibatch_size)
                            ),
                            "rewards/sampled_reward_positive_mean": (
                                positive_rewards.mean().item()
                                if positive_rewards.numel() > 0
                                else 0.0
                            ),
                            "rewards/sampled_reward_positive_std": (
                                positive_rewards.std(unbiased=False).item()
                                if positive_rewards.numel() > 0
                                else 0.0
                            ),
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
                    if args.num_critics > 2:
                        for ci, qh in enumerate(qi_h_list, start=1):
                            if ci == 1:
                                continue  # already logged as q1_mean
                            train_metrics[f"losses/q{ci}_mean"] = qh.mean().item()
                        all_h = torch.stack(qi_h_list, dim=0)
                        train_metrics["losses/q_min_mean"] = all_h.min(dim=0).values.mean().item()
                        train_metrics["losses/q_mean_mean"] = all_h.mean().item()

                total_critic_updates += 1
                if total_critic_updates % args.target_network_frequency == 0:
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for q, qt in zip(qfs, qfs_target):
                        for param, target_param in zip(q.parameters(), qt.parameters()):
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
                q1_h = qf1(sampled_observations, current_policy_actions)
                q1 = h_inverse(q1_h, eps=args.h_transform_eps).view(-1)
                norm_q = (1.0 - args.gamma) * q1
                actor_loss = -norm_q.mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                if should_update_train_metrics and actor_update_idx == args.actor_updates_per_iteration - 1:
                    train_metrics.update(
                        {
                            "losses/actor_loss": actor_loss.item(),
                            "losses/actor_norm_q_mean": norm_q.mean().item(),
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
            primitive_horizontal_fraction = (
                interval_primitive_horizontal_env_steps / max(interval_primitive_env_steps, 1)
                if interval_primitive_env_steps > 0
                else 0.0
            )
            target_position_directional_fraction = (
                interval_target_position_directional_env_steps / max(interval_env_steps, 1)
                if interval_env_steps > 0
                else 0.0
            )
            print(
                f"Step {global_step}: Primitive Actions (last interval): "
                f"{interval_primitive_env_steps}/{interval_env_steps} env-steps ({primitive_fraction:.4f}), "
                f"horizontal-dominant: {interval_primitive_horizontal_env_steps}/{interval_primitive_env_steps} "
                f"({primitive_horizontal_fraction:.4f})"
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
            interval_target_position_directional_env_steps = 0

        if global_step > 0 and global_step % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            model_path = save_full_checkpoint(checkpoint_dir)
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
            except Exception as e:
                print(f"Evaluation failed: {e}")

        iteration += 1
        global_step += args.num_envs

    envs.close()

    if not args.eval_mode:
        save_full_checkpoint(log_parent_dir)
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
        "losses/q_loss",
        "losses/q_total_loss",
        "losses/actor_loss",
        "rewards/sampled_reward_mean",
        "replay/per_beta",
        "replay/per_is_weight_mean",
        "replay/per_sampled_priority_mean",
        "replay/per_priority_td_error_mean",
    ]
    save_tensorboard_plots(log_parent_dir, config, metrics=metrics)
    writer.close()



if __name__ == "__main__":
    _entrypoint()
