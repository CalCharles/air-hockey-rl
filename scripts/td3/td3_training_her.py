"""
TD3 + hindsight experience replay (HER) for goal-conditioned tasks.

Same algorithm and recipe as ``scripts/td3/td3_training.py`` (TD3, twin
critics with transformed Bellman targets, PER, single flat replay buffer,
primitive exploration, CUDA-graph updates, CPU rollout, async checkpoint
eval) with one addition: **hindsight relabelling of every finished episode**
before it enters replay (Andrychowicz et al. 2017).  The only thing the
networks see that a non-goal task does not is the desired goal appended to
the canonical 30-dim history observation.

Goal-conditioned env contract (``AirHockeyGoalEnv`` with ``return_goal_obs:
true``): the env returns ``{"observation", "achieved_goal", "desired_goal"}``;
``env.compute_reward(achieved, desired, info)`` is the task's vectorised
reward and ``env.reward.goal_met`` its success test.  Tasks:
``puck_goal_position_sparse``, ``puck_goal_position_speed_sparse``
(``airhockey/airhockey_tasks/puck_goal_sparse.py``); the paddle reach tasks
also qualify.

Per episode the trainer stores the T original transitions plus up to
``her_k * T`` relabelled copies whose goal is an achieved goal from a later
step of the same episode (``her_strategy: future``), with the reward
recomputed by the task and the copy marked terminal when its goal is met.
See ``scripts/td3/helper/td3_her.py`` and notes/docs/training/her.md.

Run::

    python -m scripts.td3.td3_training_her --args-file configs/td3/her/puck_goal_sysid.yaml

Batch (one job per GPU)::

    python -m scripts.td3.run_experiments --mode her --configs configs/td3/her/*.yaml --gpus 0 1
"""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
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
from scripts.td3.helper.exploration_selector import (
    NumpyPrimitiveExplorationSelector,
    PrimitiveExplorationSelector,
)
from scripts.td3.helper.her_eval import evaluate_checkpoint
from scripts.td3.helper.prioritized_replay_buffer import TD3PrioritizedReplayBuffer
from scripts.td3.helper.q_network import TD3QNetwork
from scripts.td3.helper.replay_buffer import TD3ReplayBuffer
from scripts.td3.helper.td3_args_validation import validate_args
from scripts.td3.helper.td3_checkpointing import (
    build_training_state,
    load_resume_training_state,
)
from scripts.td3.helper.td3_gif_recorder import GIFEpisodeRecorder
from scripts.td3.helper.td3_graphed_update import GraphedTD3Update, deterministic_actor_action
from scripts.td3.helper.td3_her import (
    GoalEnvVector,
    HEREpisodeTrajectory,
    HERRelabeler,
    finalize_her_episode_if_done,
    make_goal_functions,
)
from scripts.td3.helper.td3_loop_logging import write_periodic_episode_stats
from scripts.td3.helper.td3_metrics import initialize_train_metrics, log_scalar_metrics
from scripts.td3.helper.td3_replay_sampling import critic_success_failure_counts
from scripts.td3.td3_training import (
    augment_policy_observation,
    extract_deterministic_state_dict,
    linear_anneal,
    primitive_exploration_chance_for_step,
    sum_info_metric,
)
from scripts.utils import save_tensorboard_plots

ROLLING_STATS_WINDOW_STEPS = 2000


@dataclass
class Args:
    """TD3+HER training args. Everything but the ``her_*`` / ``eval_*`` fields
    mirrors ``td3_training.Args`` (notes/docs/training/td3-args-reference.md)."""

    # --- Run mode ---
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
    q_weight_decay: float = 0.0
    q_frequency: int = 1
    q_updates: int = 1
    policy_frequency: int = 2
    target_network_frequency: int = 1
    actor_updates_per_iteration: int = 1
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    h_transform_eps: float = 1e-3

    # --- Critic ensemble ---
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
    success_buffer_size: int = int(1e6)
    failure_buffer_size: int = int(7e4)
    success_top_fraction: float = 0.5
    recent_episode_window_size: int = 500
    critic_per_fraction: float = 0.7
    critic_uniform_fraction: float = 0.3
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7
    single_replay_buffer: bool = True

    # --- Hindsight experience replay ---
    # Relabelled copies per transition (0 disables HER: plain goal-conditioned TD3).
    her_k: int = 4
    # Which achieved goals may replace the episode's goal: a later step of the
    # same episode ("future", the standard choice), the last one ("final"),
    # or any step ("episode").
    her_strategy: Literal["future", "final", "episode"] = "future"
    # Mark a relabelled copy terminal when its goal is met (the env ends the
    # episode on goal arrival, so the copy must not bootstrap past it).
    her_done_on_success: bool = True
    # Only propose achieved states that lie in the task's goal-sampling region
    # (env.goal_in_distribution) as hindsight goals. Off = classic HER (any
    # achieved state that passes the task's success test against itself).
    her_goal_filter: bool = True

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
    save_replay_buffer_intermediate: bool = False
    checkpoint_eval_async: bool = True

    # --- Evaluation (goal-conditioned, scripts/td3/helper/her_eval.py) ---
    eval_n_eps: int = 20            # per checkpoint
    eval_n_eps_final: int = 100     # final in-process eval
    eval_n_gifs: int = 1
    # DR runs: fixed multi-env eval (same semantics as td3_training_dr —
    # eval_n_envs dynamics overlays sampled once with eval_param_seed,
    # eval_eps_per_env episodes each, multi_env_eval.json per checkpoint).
    eval_param_seed: int | None = None
    eval_n_envs: int = 1
    eval_eps_per_env: int = 4

    # --- Paths + checkpoint loading ---
    config: str = "configs/new_juggle/tasks/sim_sysid_puck_goal.yaml"
    args_file: str | None = None
    model_path: str | None = None
    full_checkpoint_load: Literal["full_resume", "weights_only"] = "full_resume"
    log_parent_dir: str | None = None
    run_name: str = "her"

    # --- CQL (off by default; same knobs as the canonical trainer) ---
    cql_alpha: float = 0.0
    cql_n_random: int = 10

    # --- Runtime ---
    device: str = "cuda:0"
    seed: int = 0
    rollout_device: str = "cpu"
    use_cuda_graphs: bool = True
    compile_update: bool = True
    torch_num_threads: int = 1
    compile_rollout_actor: bool = True

    # --- Logging cadence ---
    train_metrics_log_interval: int = 20
    stats_log_interval: int = 5000

    # --- Network architecture ---
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2

    # --- Policy observation ---
    use_last_action_in_policy_state: bool = True

    # --- Episode GIF recording ---
    watch_ring_size: int = 10
    watch_episode_interval: int = 50
    sample_gif_interval: int = 10000
    sample_gif_max_storage_mb: float = 50.0


class AsyncGoalCheckpointEvaluator:
    """Runs ``scripts.td3.helper.her_eval`` on the CPU in a subprocess per
    checkpoint; at most one at a time (a new one waits for the previous)."""

    def __init__(self, checkpoint_interval: int, n_eps: int, n_gifs: int) -> None:
        self.checkpoint_interval = int(checkpoint_interval)
        self.n_eps = int(n_eps)
        self.n_gifs = int(n_gifs)
        self._running: List[Tuple[subprocess.Popen, int, str, object]] = []

    def launch(self, checkpoint_dir: str, global_step: int) -> None:
        self.reap(block=True)
        log_file = open(os.path.join(checkpoint_dir, "eval.log"), "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "1")
        eval_call_index = max(1, global_step // max(self.checkpoint_interval, 1))
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "scripts.td3.helper.her_eval",
                "--checkpoint-dir", checkpoint_dir,
                "--n-eps", str(self.n_eps),
                "--n-gifs", str(self.n_gifs),
                "--eval-call-index", str(eval_call_index),
            ],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        self._running.append((proc, int(global_step), checkpoint_dir, log_file))

    def reap(self, block: bool) -> None:
        still_running = []
        for proc, step, ckpt_dir, log_file in self._running:
            if block:
                proc.wait()
            if proc.poll() is None:
                still_running.append((proc, step, ckpt_dir, log_file))
                continue
            log_file.close()
            summary = ""
            try:
                with open(os.path.join(ckpt_dir, "eval.log"), "r") as f:
                    lines = [ln.strip() for ln in f if "[her_eval]" in ln or "Traceback" in ln]
                if lines:
                    summary = lines[-1]
            except OSError:
                pass
            status = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
            print(f"[eval step {step}] {status} {summary}", flush=True)
        self._running = still_running

    def wait_all(self) -> None:
        self.reap(block=True)


def make_env(air_hockey_config: dict):
    def _thunk():
        cfg = copy.deepcopy(air_hockey_config)
        cfg["seed"] = random.randint(0, int(1e8))
        cfg["return_goal_obs"] = True
        return AirHockeyEnv(cfg)

    return _thunk


def _entrypoint() -> None:
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)
    validate_args(args)
    torch.set_num_threads(max(1, int(args.torch_num_threads)))

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not config["air_hockey"].get("return_goal_obs", False):
        print("[td3_training_her] forcing return_goal_obs: true (GoalEnv dict observations)")
        config["air_hockey"]["return_goal_obs"] = True

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    log_parent_dir = args.log_parent_dir or f"runs/her/{task_name}/{args.run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        base = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base}r{i}"
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = GoalEnvVector(make_env(config["air_hockey"]))
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    train_env = envs.env
    compute_reward_fn, goal_met_fn, goal_in_distribution_fn, achieved_to_desired_fn = make_goal_functions(train_env)
    if args.her_goal_filter and goal_in_distribution_fn is None:
        print("[td3_training_her] her_goal_filter requested but the task has no goal_in_distribution(); ignoring.")
    relabeler = HERRelabeler(
        observation_dim=envs.observation_dim,
        goal_dim=envs.goal_dim,
        compute_reward=compute_reward_fn,
        goal_met=goal_met_fn,
        k=args.her_k,
        strategy=args.her_strategy,
        done_on_success=args.her_done_on_success,
        seed=args.seed,
        goal_filter=goal_in_distribution_fn if args.her_goal_filter else None,
        achieved_to_goal=achieved_to_desired_fn,
    )
    print(
        f"HER: task={task_name} observation_dim={envs.observation_dim} goal_dim={envs.goal_dim} "
        f"achieved_dim={envs.achieved_dim} k={args.her_k} strategy={args.her_strategy} "
        f"done_on_success={args.her_done_on_success} goal_filter={relabeler.goal_filter is not None}"
    )

    action_scale = 1
    device = torch.device(args.device)
    rollout_device = torch.device(args.rollout_device)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    policy_obs_dim = obs_dim + act_dim if args.use_last_action_in_policy_state else obs_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32),
        single_action_space=envs.single_action_space,
    )

    def _actor():
        return DeterministicAgent(
            policy_env_view, action_scale=action_scale, action_bias=0.0,
            hidden_layer_size=args.agent_hidden_layer_size, num_hidden_layers=args.agent_num_hidden_layers,
        ).to(device)

    def _critic():
        return TD3QNetwork(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden_layer_size=args.q_hidden_layer_size, num_hidden_layers=args.q_num_hidden_layers,
        ).to(device)

    actor = _actor()
    actor_target = _actor()
    actor_target.load_state_dict(actor.state_dict())
    if args.num_critics < 2:
        raise ValueError(f"num_critics must be >=2, got {args.num_critics}")
    qfs = [_critic() for _ in range(args.num_critics)]
    qfs_target = [_critic() for _ in range(args.num_critics)]
    for q, qt in zip(qfs, qfs_target):
        qt.load_state_dict(q.state_dict())
    qf1, qf2 = qfs[0], qfs[1]
    qf1_target, qf2_target = qfs_target[0], qfs_target[1]

    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device)
    action_low_rollout = action_low.to(rollout_device)
    action_high_rollout = action_high.to(rollout_device)
    use_cuda_graphs = bool(args.use_cuda_graphs) and device.type == "cuda"
    if use_cuda_graphs and args.target_critic_subset_size is not None and args.target_critic_subset_size < args.num_critics:
        print("target_critic_subset_size < num_critics: CUDA-graph capture disabled (eager updates).")
        use_cuda_graphs = False

    resume_checkpoint = None
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading model/checkpoint from {args.model_path}")
        loaded = torch.load(args.model_path, map_location=args.device, weights_only=False)
        is_full_state = isinstance(loaded, dict) and "actor" in loaded and "qf1" in loaded
        if is_full_state:
            actor.load_state_dict(extract_deterministic_state_dict(loaded["actor"]), strict=False)
            actor_target.load_state_dict(extract_deterministic_state_dict(loaded["actor_target"]), strict=False)
            for i in range(1, args.num_critics + 1):
                qfs[i - 1].load_state_dict(loaded[f"qf{i}"])
                qfs_target[i - 1].load_state_dict(loaded[f"qf{i}_target"])
            if args.full_checkpoint_load == "full_resume":
                resume_checkpoint = loaded
            print("Full training checkpoint loaded (network weights).")
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded), strict=False)
            actor_target.load_state_dict(actor.state_dict())
            print("Actor-only model loaded successfully.")

    adam_kwargs = dict(capturable=use_cuda_graphs, fused=(device.type == "cuda"))
    q_optimizer = optim.Adam([p for q in qfs for p in q.parameters()], lr=args.q_lr, weight_decay=args.q_weight_decay, **adam_kwargs)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, **adam_kwargs)

    buffer_kwargs = dict(obs_shape=envs.single_observation_space.shape, action_shape=envs.single_action_space.shape,
                         device=args.device, n_envs=args.num_envs)
    if args.per_enabled:
        per_kwargs = dict(alpha=args.per_alpha, priority_eps=args.per_eps, age_decay=args.priority_age_decay)
        success_rb = TD3PrioritizedReplayBuffer(buffer_size=args.success_buffer_size, **buffer_kwargs, **per_kwargs)
        failure_rb = TD3PrioritizedReplayBuffer(buffer_size=args.failure_buffer_size, **buffer_kwargs, **per_kwargs)
    else:
        success_rb = TD3ReplayBuffer(buffer_size=args.success_buffer_size, **buffer_kwargs)
        failure_rb = TD3ReplayBuffer(buffer_size=args.failure_buffer_size, **buffer_kwargs)
    print(
        f"✓ replay buffers initialized (success_capacity={args.success_buffer_size:,}, "
        f"failure_capacity={args.failure_buffer_size:,}, per={args.per_enabled}, "
        f"single_replay_buffer={args.single_replay_buffer})"
    )

    obs, _ = envs.reset(seed=args.seed)
    last_action_for_policy = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=rollout_device)
    interval_paddle_puck_collisions = 0.0
    interval_env_steps = 0
    interval_primitive_env_steps = 0
    interval_primitive_horizontal_env_steps = 0
    rolling_step_stats_window: deque = deque()
    rolling_episode_stats_window: deque = deque()
    episode_trajectory = HEREpisodeTrajectory.empty()
    recent_episode_returns: deque = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0
    her_interval_original = 0
    her_interval_relabeled = 0
    her_interval_valid = 0
    her_interval_episodes = 0

    train_renderer = AirHockeyRenderer(train_env, show_target_position=True, show_acceleration_arrow=False)
    gif_recorder = GIFEpisodeRecorder(
        log_parent_dir, watch_ring_size=args.watch_ring_size, watch_episode_interval=args.watch_episode_interval,
        sample_gif_interval=args.sample_gif_interval, sample_gif_max_storage_mb=args.sample_gif_max_storage_mb,
    )

    global_step = 0
    iteration = 0
    total_critic_updates = 0
    train_metrics = initialize_train_metrics()
    train_metrics.update({"her/relabeled_per_episode": 0.0, "her/valid_fraction": 0.0, "her/relabel_ratio": 0.0})

    same_direction_range_set = any(
        v is not None for v in (
            args.exploration_same_direction_min_angle_deg, args.exploration_same_direction_max_angle_deg,
            args.exploration_same_direction_min_magnitude, args.exploration_same_direction_max_magnitude,
        )
    )
    use_numpy_selector = rollout_device.type == "cpu" and not same_direction_range_set
    if use_numpy_selector:
        primitive_selector = NumpyPrimitiveExplorationSelector(
            num_envs=args.num_envs, chance=primitive_exploration_chance_for_step(args, global_step),
            takeover_steps=args.exploration_primitive_steps,
            direction_y_component_weight=args.exploration_direction_y_component_weight, seed=args.seed,
        )
    else:
        primitive_selector = PrimitiveExplorationSelector(
            num_envs=args.num_envs, chance=primitive_exploration_chance_for_step(args, global_step),
            takeover_steps=args.exploration_primitive_steps, device=rollout_device, dtype=torch.float32,
            direction_y_component_weight=args.exploration_direction_y_component_weight,
            action_delta_x=args.exploration_action_delta_x, action_delta_y=args.exploration_action_delta_y,
            same_direction_min_angle_deg=args.exploration_same_direction_min_angle_deg,
            same_direction_max_angle_deg=args.exploration_same_direction_max_angle_deg,
            same_direction_min_magnitude=args.exploration_same_direction_min_magnitude,
            same_direction_max_magnitude=args.exploration_same_direction_max_magnitude,
        )
    action_low_np = envs.single_action_space.low.astype(np.float32)
    action_high_np = envs.single_action_space.high.astype(np.float32)
    primitive_selector.set_primitive_weights(
        stand_still=args.exploration_primitive_weight_stand_still,
        same_direction=args.exploration_primitive_weight_same_direction,
    )

    if resume_checkpoint is not None:
        restored = load_resume_training_state(
            resume_checkpoint, device=str(rollout_device), recent_episode_window_size=args.recent_episode_window_size,
            success_rb=success_rb, failure_rb=failure_rb, primitive_selector=primitive_selector,
            q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
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
        global_step = restored["global_step"]
        iteration = restored["iteration"]
        total_critic_updates = restored["total_critic_updates"]
        obs = restored["obs"]
        last_action_for_policy = restored["last_action_for_policy"].to(rollout_device)
        train_metrics.update({k: v for k, v in restored["train_metrics"].items() if k in train_metrics})
        interval_paddle_puck_collisions = restored["interval_paddle_puck_collisions"]
        interval_env_steps = restored["interval_env_steps"]
        interval_primitive_env_steps = restored["interval_primitive_env_steps"]
        interval_primitive_horizontal_env_steps = restored["interval_primitive_horizontal_env_steps"]
        episode_trajectory = HEREpisodeTrajectory.from_state_dict(
            resume_checkpoint.get("episode_trajectory", {}), device=str(rollout_device)
        )
        recent_episode_returns = restored["recent_episode_returns"]
        episode_return_success_threshold = restored["episode_return_success_threshold"]
        rolling_step_stats_window = restored["rolling_step_stats_window"]
        rolling_episode_stats_window = restored["rolling_episode_stats_window"]
        for opt in (q_optimizer, actor_optimizer):
            for group in opt.param_groups:
                group["capturable"] = bool(adam_kwargs["capturable"])
                group["fused"] = bool(adam_kwargs["fused"])
                group["foreach"] = None
            for state in opt.state.values():
                step_t = state.get("step")
                if torch.is_tensor(step_t) and step_t.device != device and adam_kwargs["capturable"]:
                    state["step"] = step_t.to(device)
        print(f"Resuming training from global_step={global_step}, iteration={iteration}")
    start_step = global_step

    updater = GraphedTD3Update(
        actor=actor, actor_target=actor_target, qfs=qfs, qfs_target=qfs_target,
        q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
        success_rb=success_rb, failure_rb=failure_rb, batch_size=args.batch_size,
        obs_dim=obs_dim, act_dim=act_dim, device=device, gamma=args.gamma, tau=args.tau,
        policy_noise=args.policy_noise, noise_clip=args.noise_clip,
        action_low=action_low, action_high=action_high, h_transform_eps=args.h_transform_eps,
        use_last_action_in_policy_state=args.use_last_action_in_policy_state,
        per_enabled=args.per_enabled, per_eps=args.per_eps, critic_per_fraction=args.critic_per_fraction,
        cql_alpha=args.cql_alpha, cql_n_random=args.cql_n_random,
        target_critic_subset_size=args.target_critic_subset_size,
        use_graph=use_cuda_graphs, compile_update=args.compile_update,
    )
    print(
        f"Update engine: {'CUDA graphs' if updater.use_graph else 'eager'}"
        f"{' + torch.compile' if updater.compile_update else ''} on {device}; rollout on {rollout_device}"
    )

    rollout_actor = copy.deepcopy(actor).to(rollout_device).eval()
    rollout_actor_params = list(rollout_actor.parameters())
    train_actor_params = list(actor.parameters())
    rollout_policy = lambda policy_obs: deterministic_actor_action(rollout_actor, policy_obs)  # noqa: E731
    if args.compile_rollout_actor and rollout_device.type == "cpu":
        try:
            _t0 = time.time()
            _compiled = torch.compile(rollout_actor.get_action, dynamic=False)
            _probe = torch.zeros((args.num_envs, policy_obs_dim), dtype=torch.float32, device=rollout_device)
            with torch.no_grad():
                _err = (_compiled(_probe) - rollout_actor.get_action(_probe)).abs().max().item()
            if _err > 1e-4:
                raise RuntimeError(f"compiled actor mismatch {_err:.2e}")
            rollout_policy = _compiled
            print(f"Rollout actor compiled in {time.time() - _t0:.1f}s")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"torch.compile of rollout actor unavailable ({exc}); using eager.")

    def refresh_rollout_actor() -> None:
        with torch.no_grad():
            flat = torch.nn.utils.parameters_to_vector(train_actor_params).to(rollout_device)
            torch.nn.utils.vector_to_parameters(flat, rollout_actor_params)

    checkpoint_evaluator = (
        AsyncGoalCheckpointEvaluator(args.checkpoint_interval, args.eval_n_eps, args.eval_n_gifs)
        if args.checkpoint_eval_async else None
    )

    def save_full_checkpoint(out_dir: str, is_final: bool = True) -> str:
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
            global_step=global_step, iteration=iteration, total_critic_updates=total_critic_updates,
            actor=actor, actor_target=actor_target, qf1=qf1, qf2=qf2, qf1_target=qf1_target, qf2_target=qf2_target,
            extra_qfs=qfs[2:] if args.num_critics > 2 else None,
            extra_qfs_target=qfs_target[2:] if args.num_critics > 2 else None,
            q_optimizer=q_optimizer, actor_optimizer=actor_optimizer,
            success_rb=success_rb, failure_rb=failure_rb, primitive_selector=primitive_selector,
            obs=obs, last_action_for_policy=last_action_for_policy, train_metrics=train_metrics,
            interval_paddle_puck_collisions=interval_paddle_puck_collisions, interval_env_steps=interval_env_steps,
            interval_primitive_env_steps=interval_primitive_env_steps,
            interval_primitive_horizontal_env_steps=interval_primitive_horizontal_env_steps,
            episode_trajectory=episode_trajectory, recent_episode_returns=recent_episode_returns,
            episode_return_success_threshold=episode_return_success_threshold,
            rolling_step_stats_window=rolling_step_stats_window,
            rolling_episode_stats_window=rolling_episode_stats_window, args_dict=vars(args),
            include_replay_buffer=args.save_replay_buffer and (is_final or args.save_replay_buffer_intermediate),
        )
        torch.save(state, f"{out_dir}/training_state.pth")
        return model_path_local

    def run_checkpoint_eval(model_path: str, checkpoint_dir: str) -> None:
        if checkpoint_evaluator is not None:
            checkpoint_evaluator.launch(checkpoint_dir, global_step)
            return
        try:
            evaluate_checkpoint(
                checkpoint_dir, n_eps=args.eval_n_eps, n_gifs=args.eval_n_gifs,
                eval_call_index=max(1, global_step // max(args.checkpoint_interval, 1)),
                log_parent_dir=log_parent_dir,
            )
        except Exception as e:
            print(f"Evaluation failed: {e}")

    training_cycles = 0
    next_stats_log_step = ((global_step // args.stats_log_interval) + 1) * args.stats_log_interval
    start_time = time.time()
    last_critic_out: Dict[str, torch.Tensor] | None = None
    last_actor_out: Dict[str, torch.Tensor] | None = None
    per_beta = 0.0

    while global_step < args.total_timesteps:
        if iteration % 100 == 0:
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

        # ------------------------------------------------------------ rollout
        prev_action_for_transition = last_action_for_policy.clone()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=rollout_device)
        policy_obs_tensor = augment_policy_observation(obs_tensor, last_action_for_policy, args.use_last_action_in_policy_state)
        if global_step < args.learning_starts and args.exploration_pre_learning_action_source == "random":
            action_tensor = torch.as_tensor(
                np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]),
                dtype=torch.float32, device=rollout_device,
            )
        else:
            with torch.no_grad():
                action_tensor = rollout_policy(policy_obs_tensor)
                action_tensor = action_tensor + torch.randn_like(action_tensor) * args.exploration_noise
                action_tensor = torch.clamp(action_tensor, action_low_rollout, action_high_rollout)
        if use_numpy_selector:
            actions_np, primitive_step_stats = primitive_selector.apply(
                action_tensor.numpy(), action_low_np, action_high_np, return_stats=True
            )
            action_tensor = torch.from_numpy(actions_np)
        else:
            action_tensor, primitive_step_stats = primitive_selector.apply(
                action_tensor, action_low=action_low_rollout, action_high=action_high_rollout, return_stats=True
            )
        actions = action_tensor.numpy()
        gif_recorder.capture_frame(train_renderer, global_step)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        step_puck_hits = sum_info_metric(infos, "paddle_puck_collision_count")
        interval_paddle_puck_collisions += step_puck_hits
        interval_env_steps += args.num_envs
        interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
        interval_primitive_horizontal_env_steps += int(primitive_step_stats["primitive_horizontal_dominant_count"])
        done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=rollout_device)
        primitive_selector.reset(done_tensor)
        last_action_for_policy = action_tensor.clone()
        last_action_for_policy[done_tensor] = 0
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=rollout_device)
        gif_recorder.note_reward(float(rewards[0]))

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode_return" in info:
                    writer.add_scalar("charts/episodic_return", info["episode_return"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode_length"], global_step)
                    rolling_episode_stats_window.append(
                        (int(global_step + args.num_envs), float(info["episode_return"]),
                         float(info["episode_length"]), 1.0 if info.get("success", False) else 0.0)
                    )
        rolling_step_stats_window.append((int(global_step + args.num_envs), int(args.num_envs), float(step_puck_hits)))
        rolling_cutoff_step = int(global_step + args.num_envs - ROLLING_STATS_WINDOW_STEPS)
        while rolling_step_stats_window and int(rolling_step_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_step_stats_window.popleft()
        while rolling_episode_stats_window and int(rolling_episode_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_episode_stats_window.popleft()

        # The stored next_obs is the *final* observation on every episode end
        # (not only truncations, as in the canonical trainer): a transition
        # that terminated by reaching its goal becomes non-terminal under a
        # relabelled goal and then bootstraps from its next_obs.
        episode_finished = bool(dones[0])
        if episode_finished:
            real_next_obs = np.asarray(infos["final_observation"][0], dtype=np.float32)
            achieved_next = np.asarray(infos["final_achieved_goal"][0], dtype=np.float32)
        else:
            real_next_obs = np.asarray(next_obs[0], dtype=np.float32)
            achieved_next = np.asarray(infos["achieved_goal"][0], dtype=np.float32)
        episode_trajectory.append_step(
            obs=obs_tensor[0],
            next_obs=torch.as_tensor(real_next_obs, dtype=torch.float32, device=rollout_device),
            action=action_tensor[0],
            reward=rewards_tensor[0],
            done=torch.as_tensor(float(terminations[0]), dtype=torch.float32, device=rollout_device),
            prev_action=prev_action_for_transition[0],
            achieved_goal=achieved_next,
            hard_terminal=bool(infos["hard_terminal"][0]),
        )
        episode_return_success_threshold, her_stats = finalize_her_episode_if_done(
            episode_done=episode_finished, episode_trajectory=episode_trajectory,
            recent_episode_returns=recent_episode_returns, success_top_fraction=args.success_top_fraction,
            episode_return_success_threshold=episode_return_success_threshold,
            success_rb=success_rb, failure_rb=failure_rb, relabeler=relabeler,
            single_buffer=args.single_replay_buffer,
        )
        if episode_finished:
            her_interval_original += her_stats["original"]
            her_interval_relabeled += her_stats["relabeled"]
            her_interval_valid += her_stats["valid"]
            her_interval_episodes += 1
            gif_recorder.on_episode_end(global_step)
        obs = next_obs

        # ----------------------------------------------------------- training
        if global_step > args.learning_starts and episode_finished:
            per_beta = (
                linear_anneal(args.per_beta_start, args.per_beta_end, global_step, args.per_beta_anneal_steps)
                if args.per_enabled else 0.0
            )
            for _ in range(args.q_updates):
                s_count, f_count = critic_success_failure_counts(
                    batch_size=args.batch_size, success_fraction=args.critic_success_sample_fraction,
                    success_available=len(success_rb) > 0, failure_available=len(failure_rb) > 0,
                )
                if s_count + f_count == 0:
                    continue
                last_critic_out = updater.critic_update(s_count, f_count, per_beta)
                total_critic_updates += 1
                if total_critic_updates % args.target_network_frequency == 0:
                    updater.polyak()
            for _ in range(args.actor_updates_per_iteration):
                s_count, f_count = critic_success_failure_counts(
                    batch_size=args.batch_size, success_fraction=args.critic_success_sample_fraction,
                    success_available=len(success_rb) > 0, failure_available=len(failure_rb) > 0,
                )
                if s_count + f_count == 0:
                    continue
                last_actor_out = updater.actor_update(s_count, f_count)
            refresh_rollout_actor()
            training_cycles += 1

            if training_cycles % args.train_metrics_log_interval == 0:
                if last_critic_out is not None:
                    train_metrics.update({
                        "losses/q_loss": float(last_critic_out["q_loss"].item()),
                        "losses/q_total_loss": float(last_critic_out["q_total_loss"].item()),
                        "losses/q1_mean": float(last_critic_out["q1_mean"].item()),
                        "debug/bellman_target_original_mean": float(last_critic_out["bellman_target_mean"].item()),
                        "debug/next_q_h_mean": float(last_critic_out["next_q_h_mean"].item()),
                        "rewards/sampled_reward_mean": float(last_critic_out["sampled_reward_mean"].item()),
                        "replay/per_priority_td_error_mean": float(last_critic_out["priority_td_error_mean"].item()),
                    })
                if last_actor_out is not None:
                    train_metrics.update({
                        "losses/actor_loss": float(last_actor_out["actor_loss"].item()),
                        "losses/actor_norm_q_mean": float(last_actor_out["actor_norm_q_mean"].item()),
                    })
                train_metrics.update({
                    "replay/per_beta": float(per_beta),
                    "replay/success_buffer_size": float(len(success_rb)),
                    "replay/failure_buffer_size": float(len(failure_rb)),
                    "replay/episode_return_success_threshold": float(episode_return_success_threshold),
                    "her/relabeled_per_episode": her_interval_relabeled / max(her_interval_episodes, 1),
                    "her/valid_fraction": her_interval_valid / max(her_interval_original, 1),
                    "her/relabel_ratio": her_interval_relabeled / max(her_interval_original, 1),
                })
                log_scalar_metrics(writer, train_metrics, global_step)
                writer.add_scalar("charts/exploration_primitive_chance", primitive_selector.chance, global_step)
                elapsed = max(time.time() - start_time, 1e-6)
                writer.add_scalar("charts/SPS", int((global_step - start_step) / elapsed), global_step)
                her_interval_original = her_interval_relabeled = her_interval_valid = her_interval_episodes = 0

        if global_step + args.num_envs >= next_stats_log_step:
            write_periodic_episode_stats(
                writer, global_step,
                rolling_episode_stats_window=rolling_episode_stats_window,
                rolling_step_stats_window=rolling_step_stats_window,
                interval_paddle_puck_collisions=interval_paddle_puck_collisions,
                interval_env_steps=interval_env_steps,
                interval_primitive_env_steps=interval_primitive_env_steps,
                interval_primitive_horizontal_env_steps=interval_primitive_horizontal_env_steps,
            )
            elapsed = max(time.time() - start_time, 1e-6)
            print(f"Step {global_step}: SPS {(global_step - start_step) / elapsed:.0f}", flush=True)
            interval_paddle_puck_collisions = 0.0
            interval_env_steps = 0
            interval_primitive_env_steps = 0
            interval_primitive_horizontal_env_steps = 0
            next_stats_log_step += args.stats_log_interval

        if global_step > 0 and global_step % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            model_path = save_full_checkpoint(checkpoint_dir, is_final=False)
            print(f"\nCheckpoint saved at step {global_step}", flush=True)
            run_checkpoint_eval(model_path, checkpoint_dir)

        iteration += 1
        global_step += args.num_envs

    envs.close()
    gif_recorder.close()
    if checkpoint_evaluator is not None:
        checkpoint_evaluator.wait_all()
    save_full_checkpoint(log_parent_dir)
    try:
        summary = evaluate_checkpoint(
            log_parent_dir, n_eps=args.eval_n_eps_final, n_gifs=args.eval_n_gifs,
            eval_call_index=1, log_parent_dir=log_parent_dir, final=True,
        )
        success = summary.get("success_rate", summary.get("mean_success_across_envs"))
        mean_return = summary.get("mean_return", summary.get("mean_return_across_envs"))
        writer.add_scalar("eval/final_success_rate", float(success), global_step)
        writer.add_scalar("eval/final_mean_return", float(mean_return), global_step)
    except Exception as e:
        print(f"Final evaluation failed: {e}")
    writer.close()
    save_tensorboard_plots(
        log_parent_dir, config,
        metrics=["charts/episodic_return", "charts/avg_episodic_return", "charts/avg_success_rate",
                 "losses/q_loss", "losses/actor_loss", "losses/q1_mean", "her/relabeled_per_episode"],
    )


if __name__ == "__main__":
    _entrypoint()
