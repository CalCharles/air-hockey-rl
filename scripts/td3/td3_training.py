"""
TD3 training with transformed Bellman targets and single-head critics.

Compared to SAC+AMP:
- no discriminator
- no entropy term / alpha tuning
- deterministic actor updates (TD3)
- twin critics (REDQ-style ensemble when num_critics > 2), single scalar Q head
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Literal, Tuple

import gymnasium as gym
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
from scripts.td3.helper.td3_args_validation import validate_args
from scripts.td3.helper.td3_cql import cql_penalty, precompute_cql_terms
from scripts.td3.helper.td3_gif_recorder import GIFEpisodeRecorder
from scripts.td3.helper.td3_loop_logging import (
    build_actor_metrics,
    build_critic_metrics,
    build_target_q_debug_metrics,
    write_periodic_episode_stats,
)
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

from scripts.transformer.context_encoder import ContextEncoder
from scripts.transformer.history_buffer import HistoryBuffer
from scripts.transformer.context_vector_analysis import context_vector_analysis
from scripts.transformer.compare_performance_ID_OOD import compare_performance_ID_OOD

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


def sum_info_metric(infos: dict, metric_name: str) -> float:
    metric_values = infos.get(metric_name)
    if metric_values is None:
        return 0.0
    try:
        return float(np.asarray(metric_values, dtype=np.float32).sum())
    except Exception:
        return 0.0


@dataclass
class Args:
    """TD3 training args. Per-field docs: notes/docs/training/td3-args-reference.md."""

    # --- Run mode ---
    eval_mode: bool = False
    total_timesteps: int = 1000000
    num_envs: int = 1

    # --- TD3 core ---
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
    policy_frequency: int = 2
    target_network_frequency: int = 1
    actor_updates_per_iteration: int = 1
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    h_transform_eps: float = 1e-3

    # --- Critic ensemble (REDQ-style) ---
    num_critics: int = 2
    target_critic_subset_size: int | None = None

    # --- Prioritized experience replay ---
    per_enabled: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 200000
    per_eps: float = 1e-6
    priority_age_decay: float = 0.0

    # --- Replay buffer split + sampling mix ---
    success_buffer_size: int = int(2e5)
    failure_buffer_size: int = int(8e5)
    success_top_fraction: float = 0.2
    recent_episode_window_size: int = 500
    critic_per_fraction: float = 0.7
    critic_uniform_fraction: float = 0.3
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7

    # --- Primitive exploration takeover ---
    exploration_primitive_chance: float = 0.05
    exploration_primitive_chance_start: float = 0.5
    exploration_primitive_chance_pre_learning_starts: float | None = None
    exploration_pre_learning_action_source: Literal["random", "policy"] = "random"
    exploration_primitive_chance_anneal_steps: int = 50000
    exploration_primitive_steps: int = 3
    exploration_primitive_weight_stand_still: float = 0.5
    exploration_primitive_weight_same_direction: float = 0.5
    exploration_primitive_weight_anneal_stand_still: float = 0.3
    exploration_primitive_weight_anneal_same_direction: float = 0.7
    exploration_direction_y_component_weight: float = 1.5
    exploration_action_delta_x: float = 0.26
    exploration_action_delta_y: float = 0.12
    exploration_same_direction_min_angle_deg: float | None = None
    exploration_same_direction_max_angle_deg: float | None = None
    exploration_same_direction_min_magnitude: float | None = None
    exploration_same_direction_max_magnitude: float | None = None

    # --- Checkpointing ---
    checkpoint_interval: int = 25000
    save_replay_buffer: bool = True

    # --- Paths + checkpoint loading ---
    config: str = "configs/new_juggle/sysid_best_params.yaml"
    args_file: str | None = None
    model_path: str | None = None
    full_checkpoint_load: Literal["full_resume", "weights_only", "residual"] = "full_resume"
    log_parent_dir: str | None = None
    run_name: str = "default"

    # --- Residual RL (active when full_checkpoint_load == "residual") ---
    residual_scale: float = 0.25
    residual_weight_decay: float = 0.0

    # --- CQL (conservative Q-learning) ---
    cql_alpha: float = 0.0
    cql_n_random: int = 10

    # --- Runtime ---
    device: str = "cuda:0"
    seed: int = 0

    # --- Network architecture ---
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2

    # --- Policy observation ---
    use_last_action_in_policy_state: bool = False

    # --- Episode GIF recording ---
    watch_ring_size: int = 10
    watch_episode_interval: int = 50
    sample_gif_interval: int = 10000
    sample_gif_max_storage_mb: float = 50.0

    # --- Multi-env evaluation (used by td3_training_dr.py wrapper) ---
    eval_param_seed: int | None = None
    eval_n_envs: int = 1
    eval_eps_per_env: int = 4

    # --- Context Vector Generation ---
    use_context_vector: bool = False
    context_len: int = 7
    context_vector_dim: int = 8
    transformer_lr: float = 0.00005 

    # --- Context Vector OOD Analysis ---
    analyze_context_vectors: bool = False
    context_analysis_n_eps: int = 20
    context_analysis_n_envs: int = 10
    context_analysis_ood_scale: float = 2.0
    context_analysis_out_dir: str = "results/context_tsne"

    # --- Compare performance of ID and OOD for baseline and transformer based model
    eval_id_ood: bool = False
    eval_id_ood_n_envs: int = 10
    eval_id_ood_n_eps: int = 8
    eval_id_ood_ood_scale: float = 2.0
    eval_id_ood_out_dir: str = "results/id_ood_comparison"
    # Path to a *second* model to compare against (the context-vector model when
    # running on a baseline checkpoint, or vice versa).  Optional — if omitted,
    # only the model loaded via --model-path is evaluated.
    eval_id_ood_compare_model_path: str | None = None


def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env

    return _thunk


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

    action_scale = 1

    raw_obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    policy_obs_dim = raw_obs_dim + act_dim if args.use_last_action_in_policy_state else raw_obs_dim
    policy_obs_dim = policy_obs_dim + args.context_vector_dim if args.use_context_vector else policy_obs_dim

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
        use_context=args.use_context_vector,
        context_vector_dim=args.context_vector_dim
    ).to(args.device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
        use_context=args.use_context_vector,
        context_vector_dim=args.context_vector_dim
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


    if args.use_context_vector and (args.context_vector_dim > 0):
        transformer = ContextEncoder(
            obs_dim=raw_obs_dim,
            context_dim=args.context_vector_dim,
            context_len=args.context_len
        ).to(args.device)

        history_buf = HistoryBuffer(
            obs_dim=raw_obs_dim,
            context_len=args.context_len,
            device=args.device,
        )

        transformer_optimizer = optim.Adam(transformer.parameters(), lr=args.transformer_lr)


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
                checkpoint_load_mode = "weights_only"
            actor.load_state_dict(extract_deterministic_state_dict(resume_checkpoint["actor"]), strict=False)
            actor_target.load_state_dict(
                extract_deterministic_state_dict(resume_checkpoint["actor_target"]),
                strict=False,
            )
            n_in_ckpt = sum(
                1
                for k in resume_checkpoint
                if k.startswith("qf") and not k.endswith("_target") and k[2:].isdigit()
            )
            if n_in_ckpt != args.num_critics:
                raise ValueError(
                    f"Resume mismatch: checkpoint has {n_in_ckpt} critics but "
                    f"args.num_critics={args.num_critics}. Ensemble size must match."
                )
            for i in range(1, args.num_critics + 1):
                qfs[i - 1].load_state_dict(resume_checkpoint[f"qf{i}"])
                qfs_target[i - 1].load_state_dict(resume_checkpoint[f"qf{i}_target"])
            print("Full training checkpoint loaded (network weights).")
            if checkpoint_load_mode == "weights_only":
                # Weights-only mode: keep networks, skip optimizer/replay/runtime restore.
                resume_checkpoint = None
                print("Weights-only load enabled: skipping resume of optimizer/replay/runtime state.")
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
            actor_target.load_state_dict(actor.state_dict())
            print("Actor-only model loaded successfully.")



        # Load the transformer from path if specified
        if args.model_path is not None and args.use_context_vector:
            checkpoint_dir = os.path.dirname(args.model_path)
            transformer_path = os.path.join(checkpoint_dir, "transformer.pth")
            if os.path.exists(transformer_path):

                transformer.load_state_dict(torch.load(transformer_path, map_location=args.device))

                print(f"Transformer loaded from {transformer_path}")

            else:
                print(f"Warning: use_context_vector=True but transformer.pth not found at {transformer_path}")
                return


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
            context_len=args.context_len,
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
            context_len=args.context_len,
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
    rolling_step_stats_window = deque()
    rolling_episode_stats_window = deque()
    episode_trajectory = EpisodeTrajectory.empty()
    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0

    # --- Live episode GIF recording ---
    renderer_env = AirHockeyEnv(config["air_hockey"])
    train_renderer = AirHockeyRenderer(
        renderer_env, show_target_position=True, show_acceleration_arrow=False
    )
    gif_recorder = GIFEpisodeRecorder(
        log_parent_dir,
        watch_ring_size=args.watch_ring_size,
        watch_episode_interval=args.watch_episode_interval,
        sample_gif_interval=args.sample_gif_interval,
        sample_gif_max_storage_mb=args.sample_gif_max_storage_mb,
    )

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
        action_delta_x=args.exploration_action_delta_x,
        action_delta_y=args.exploration_action_delta_y,
        same_direction_min_angle_deg=args.exploration_same_direction_min_angle_deg,
        same_direction_max_angle_deg=args.exploration_same_direction_max_angle_deg,
        same_direction_min_magnitude=args.exploration_same_direction_min_magnitude,
        same_direction_max_magnitude=args.exploration_same_direction_max_magnitude,
    )
    primitive_selector.set_primitive_weights(
        stand_still=args.exploration_primitive_weight_stand_still,
        same_direction=args.exploration_primitive_weight_same_direction,
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
        last_action_for_policy = restored_state["last_action_for_policy"]
        train_metrics = restored_state["train_metrics"]
        interval_paddle_puck_collisions = restored_state["interval_paddle_puck_collisions"]
        interval_env_steps = restored_state["interval_env_steps"]
        interval_primitive_env_steps = restored_state["interval_primitive_env_steps"]
        interval_primitive_horizontal_env_steps = restored_state["interval_primitive_horizontal_env_steps"]
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
        
        if args.use_context_vector:
            torch.save(transformer.state_dict(), f"{out_dir}/transformer.pth")

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


    # We wun this function assuming the transformer is loaded in from a checkpoint (.pth)
    # Use the above to support argument parsing and setting up environment for context vector analysis after transformer is trained
    
    if args.eval_id_ood:
        # The model already loaded via --model-path is treated as the baseline.
        # If --eval-id-ood-compare-model-path is set, load that second model as
        # the context-vector actor (requires use_context_vector=True on that run).
        compare_actor = None
        compare_transformer = None
        compare_history_buf = None
        compare_model_path_str = ""

        if args.eval_id_ood_compare_model_path is not None:
            # from scripts.td3.deterministic_agent import DeterministicAgent
            # import gymnasium as gym
            # from types import SimpleNamespace

            _cmp_context_dim = args.context_vector_dim
            _cmp_context_len = args.context_len
            _cmp_policy_obs_dim = (
                raw_obs_dim + _cmp_context_dim
                + (act_dim if args.use_last_action_in_policy_state else 0)
            )
            _cmp_policy_env_view = SimpleNamespace(
                single_observation_space=gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(_cmp_policy_obs_dim,), dtype=np.float32,
                ),
                single_action_space=envs.single_action_space,
            )
            compare_actor = DeterministicAgent(
                _cmp_policy_env_view,
                action_scale=action_scale,
                action_bias=0.0,
                hidden_layer_size=args.agent_hidden_layer_size,
                num_hidden_layers=args.agent_num_hidden_layers,
                use_context=True,
                context_vector_dim=_cmp_context_dim,
            ).to(args.device)

            _cmp_loaded = torch.load(
                args.eval_id_ood_compare_model_path,
                map_location=args.device,
                weights_only=False,
            )
            _cmp_state = _cmp_loaded["actor"] if (
                isinstance(_cmp_loaded, dict) and "actor" in _cmp_loaded
            ) else _cmp_loaded
            compare_actor.load_state_dict(
                extract_deterministic_state_dict(_cmp_state), strict=False
            )
            compare_actor.eval()

            compare_transformer = ContextEncoder(
                obs_dim=raw_obs_dim,
                context_dim=_cmp_context_dim,
                context_len=_cmp_context_len,
            ).to(args.device)
            _cmp_dir = os.path.dirname(args.eval_id_ood_compare_model_path)
            _cmp_transformer_path = os.path.join(_cmp_dir, "transformer.pth")
            if not os.path.exists(_cmp_transformer_path):
                raise FileNotFoundError(
                    f"transformer.pth not found at {_cmp_transformer_path}. "
                    "Expected as a sibling of the compare model.pth."
                )
            compare_transformer.load_state_dict(
                torch.load(_cmp_transformer_path, map_location=args.device)
            )
            compare_transformer.eval()

            compare_history_buf = HistoryBuffer(
                obs_dim=raw_obs_dim,
                context_len=_cmp_context_len,
                device=args.device,
            )
            compare_model_path_str = args.eval_id_ood_compare_model_path

        compare_performance_ID_OOD(
            baseline_actor=actor,
            air_hockey_base=config["air_hockey"],
            raw_obs_dim=raw_obs_dim,
            act_dim=act_dim,
            use_last_action=args.use_last_action_in_policy_state,
            n_envs=args.eval_id_ood_n_envs,
            n_eps=args.eval_id_ood_n_eps,
            ood_scale=args.eval_id_ood_ood_scale,
            out_dir=args.eval_id_ood_out_dir,
            device=args.device,
            seed=args.seed,
            context_actor=compare_actor,
            context_transformer=compare_transformer,
            context_history_buf=compare_history_buf,
            baseline_model_path=args.model_path or "",
            context_model_path=compare_model_path_str,
        )
        return  # exit after eval, don't train
            
    
    
    
    
    if args.analyze_context_vectors:
                
        context_vector_analysis(
            actor=actor,
            transformer=transformer,
            air_hockey_base=config["air_hockey"],
            raw_obs_dim=raw_obs_dim,
            act_dim=act_dim,
            context_len=args.context_len,
            context_vector_dim=args.context_vector_dim,
            use_last_action=args.use_last_action_in_policy_state,
            n_eps=args.context_analysis_n_eps,
            n_envs=args.context_analysis_n_envs,
            ood_scale=args.context_analysis_ood_scale,
            out_dir=args.context_analysis_out_dir,
            device=args.device,
            seed=args.seed,
        )
        return  # exit after analysis, don't train




    episode_finished = None
    should_update_train_metrics = None
    recent_episode_returns = []

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
                )
            else:
                primitive_selector.set_primitive_weights(
                    stand_still=args.exploration_primitive_weight_stand_still,
                    same_direction=args.exploration_primitive_weight_same_direction,
                )
        prev_action_for_transition = last_action_for_policy.clone()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)


        if args.use_context_vector:
            with torch.no_grad():
                state_history = history_buf.sample()
                context_vector = transformer(state_history)
                obs_with_context_tensor = torch.cat([obs_tensor, context_vector], dim=-1)

                policy_obs_tensor = augment_policy_observation(
                    obs_with_context_tensor, last_action_for_policy, args.use_last_action_in_policy_state
                )
        else:
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
            action_tensor, primitive_step_stats = primitive_selector.apply(
                action_tensor,
                action_low=action_low,
                action_high=action_high,
                return_stats=True,
            )
        else:
            primitive_step_stats = {
                "primitive_applied_count": 0,
                "primitive_horizontal_dominant_count": 0,
            }
        actions = action_tensor.cpu().numpy()

        gif_recorder.capture_frame(train_renderer, global_step)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        history_buf.add(obs[0], done=bool(dones[0]))

        if args.use_context_vector:
            # sample() returns (1, T, obs_dim).
            history_snapshot = history_buf.sample()[0]
        else:
            history_snapshot = None
        

        step_puck_hits = sum_info_metric(infos, "paddle_puck_collision_count")
        interval_paddle_puck_collisions += step_puck_hits
        interval_env_steps += args.num_envs
        interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
        interval_primitive_horizontal_env_steps += int(
            primitive_step_stats["primitive_horizontal_dominant_count"]
        )
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=args.device)
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=args.device)
        primitive_selector.reset(done_tensor)
        last_action_for_policy = action_tensor.clone()
        last_action_for_policy[done_tensor] = 0

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=args.device)

        gif_recorder.note_reward(float(rewards_tensor[0].item()))

        # For some reason this isn't passing conditions necessary to get to add_scalar. 
        # Thus when training these metrics are not tracked. See work around further below.
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if info and "episode_return" in info:
        #             writer.add_scalar("charts/episodic_return", info["episode_return"], global_step)
        #             writer.add_scalar("charts/episodic_length", info["episode_length"], global_step)
        #             rolling_episode_stats_window.append(
        #                 (
        #                     int(global_step + args.num_envs),
        #                     float(info["episode_return"]),
        #                     float(info["episode_length"]),
        #                     1.0 if info.get("success", False) else 0.0,
        #                 )
        #             )
        
        rolling_step_stats_window.append(
            (
                int(global_step + args.num_envs),
                int(args.num_envs),
                float(step_puck_hits),
            )
        )
        rolling_cutoff_step = int(global_step + args.num_envs - ROLLING_STATS_WINDOW_STEPS)
        while rolling_step_stats_window and int(rolling_step_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_step_stats_window.popleft()
        while rolling_episode_stats_window and int(rolling_episode_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_episode_stats_window.popleft()

        real_next_obs = next_obs.copy()

        if "final_observation" in infos:
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
                history=history_snapshot,
            )

            # Work around solution to above add_scalar conditions not being met
            if bool(dones[0]):
                episode_return = episode_trajectory.episode_return
                episode_length = len(episode_trajectory.observations)
                episode_success = bool(infos["success"][0])

                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                rolling_episode_stats_window.append((
                    int(global_step + args.num_envs),
                    float(episode_return),
                    float(episode_length),
                    1.0 if episode_success else 0.0,
                ))


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

        if episode_finished:
            gif_recorder.on_episode_end(global_step)

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

                if args.use_context_vector:
                    sampled_next_history = data["history"].to(args.device)
                    with torch.no_grad():
                        sampled_next_context = transformer(sampled_next_history)

                    sampled_next_obs_with_context = torch.cat([sampled_next_observations, sampled_next_context], dim=-1)
                    sampled_next_policy_observations = augment_policy_observation(
                        sampled_next_obs_with_context,
                        sampled_next_prev_actions,
                        args.use_last_action_in_policy_state,
                    )
                else:
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
                        train_metrics.update(build_target_q_debug_metrics(
                            bellman_target_original, next_q_value_h,
                        ))

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
                    train_metrics.update(build_critic_metrics(
                        qi_h_list=qi_h_list,
                        qi_loss_list=qi_loss_list,
                        q_total_loss=q_total_loss,
                        q1_h=q1_h,
                        num_critics=args.num_critics,
                        sampled_rewards=sampled_rewards,
                        sampled_weights=sampled_weights,
                        sampled_priorities=sampled_priorities,
                        priority_td_error=priority_td_error,
                        per_beta=per_beta,
                        per_enabled=args.per_enabled,
                        per_sample_count=per_sample_count,
                        uniform_sample_count=uniform_sample_count,
                        batch_size=args.batch_size,
                        success_batch_count=success_batch_count,
                        failure_batch_count=failure_batch_count,
                        len_success_rb=len(success_rb),
                        len_failure_rb=len(failure_rb),
                        episode_return_success_threshold=episode_return_success_threshold,
                        recent_episode_window_count=len(recent_episode_returns),
                    ))

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

                    if args.use_context_vector and all("history" in chunk for chunk in actor_data_chunks):
                        data["history"] = torch.cat(
                            [chunk["history"] for chunk in actor_data_chunks], dim=0
                        )

                sampled_observations = data["observations"]
                sampled_prev_actions = data["prev_actions"]

                if args.use_context_vector:
                    # Shape: (B, T, obs_dim) -> transformer -> (B, context_dim)
                    sampled_history = data["history"].to(args.device)

                    sampled_context = transformer(sampled_history)

                    sampled_obs_with_context = torch.cat([sampled_observations, sampled_context], dim=-1)

                    sampled_policy_observations = augment_policy_observation(
                        sampled_obs_with_context, sampled_prev_actions,
                        args.use_last_action_in_policy_state,
                    )

                else:
                    sampled_policy_observations = augment_policy_observation(
                        sampled_observations, sampled_prev_actions,
                        args.use_last_action_in_policy_state,
                    )

                current_policy_actions = deterministic_actor_action(actor, sampled_policy_observations)
                q1_h = qf1(sampled_observations, current_policy_actions)
                q1 = h_inverse(q1_h, eps=args.h_transform_eps).view(-1)
                norm_q = (1.0 - args.gamma) * q1
                actor_loss = -norm_q.mean()
                actor_optimizer.zero_grad()

                # New
                if args.use_context_vector:
                    transformer_optimizer.zero_grad()

                actor_loss.backward()
                actor_optimizer.step()

                if args.use_context_vector:
                    transformer_optimizer.step()

                if should_update_train_metrics and actor_update_idx == args.actor_updates_per_iteration - 1:
                    train_metrics.update(build_actor_metrics(actor_loss, norm_q))



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



        if global_step > 0 and global_step % 500 == 0:

            write_periodic_episode_stats(
                writer, global_step,
                rolling_episode_stats_window=rolling_episode_stats_window,
                rolling_step_stats_window=rolling_step_stats_window,
                interval_paddle_puck_collisions=interval_paddle_puck_collisions,
                interval_env_steps=interval_env_steps,
                interval_primitive_env_steps=interval_primitive_env_steps,
                interval_primitive_horizontal_env_steps=interval_primitive_horizontal_env_steps,
            )
            interval_paddle_puck_collisions = 0.0
            interval_env_steps = 0
            interval_primitive_env_steps = 0
            interval_primitive_horizontal_env_steps = 0

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
                    use_context_vector=args.use_context_vector,
                    context_vector_dim=args.context_vector_dim,
                    context_len=args.context_len,
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
            use_context_vector=args.use_context_vector,
            context_vector_dim=args.context_vector_dim,
            context_len=args.context_len,
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
