"""
TD3 training with transformed Bellman targets and single-head critics.

Compared to SAC+AMP:
- no discriminator
- no entropy term / alpha tuning
- deterministic actor updates (TD3)
- twin critics (REDQ-style ensemble when num_critics > 2), single scalar Q head

Throughput layout (2026-09 optimisation — see
notes/docs/training/training-throughput.md):
- Rollout runs entirely on the CPU: a CPU replica of the actor drives the
  env, the exploration selector and the per-episode trajectory staging all
  live on the CPU, and the finished episode is moved to the GPU replay buffer
  in one transfer. No per-step host<->device traffic or syncs.
- The critic / actor updates are CUDA-graph captured (`GraphedTD3Update`):
  per update we copy the sampled minibatch into static tensors and replay.
- Per-checkpoint evaluation runs in a background subprocess
  (`scripts/td3/checkpoint_eval.py`) so rollouts + GIF encoding never block
  the training loop.
- Logging is reduced to the metrics that are actually consulted, written on
  fixed intervals.
"""

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
from scripts.td3.helper.q_network import TD3QNetwork
from scripts.td3.helper.td3_args_validation import validate_args
from scripts.td3.helper.td3_gif_recorder import GIFEpisodeRecorder
from scripts.td3.helper.td3_graphed_update import (
    GraphedTD3Update,
    deterministic_actor_action,
    h_inverse,
    h_transform,
)
from scripts.td3.helper.td3_loop_logging import write_periodic_episode_stats
from scripts.td3.helper.td3_residual import build_residual_training
from scripts.td3.helper.exploration_selector import (
    NumpyPrimitiveExplorationSelector,
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
from scripts.td3.helper.td3_replay_sampling import critic_success_failure_counts
from scripts.td3.evaluate import evaluate_agent
from scripts.utils import save_tensorboard_plots

ROLLING_STATS_WINDOW_STEPS = 2000

__all__ = [
    "Args",
    "h_transform",
    "h_inverse",
    "deterministic_actor_action",
    "augment_policy_observation",
    "make_env",
]


def augment_policy_observation(observation, last_action, use_last_action):
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


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
    # Run per-checkpoint evaluation in a background subprocess (CPU only) so
    # it does not block training. The final evaluation stays in-process.
    checkpoint_eval_async: bool = True

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
    # Device that drives the environment (actor inference at batch=num_envs,
    # exploration selector, trajectory staging). CPU is ~3x faster than the
    # GPU for batch-1 inference of the 64-wide actor and avoids per-step syncs.
    rollout_device: str = "cpu"
    # Capture the critic/actor updates in CUDA graphs (GPU device only).
    use_cuda_graphs: bool = True
    # torch.compile the loss forward/backward inside the graphs (~30% fewer
    # GPU kernels per update). Falls back automatically if compilation fails.
    compile_update: bool = True
    # Intra-op CPU threads. The rollout tensors are tiny; a big OpenMP pool
    # only adds spin overhead.
    torch_num_threads: int = 1
    # torch.compile the CPU rollout actor (falls back to eager on failure).
    compile_rollout_actor: bool = True

    # --- Logging cadence ---
    # Training-loss scalars are logged every N training cycles (= episodes
    # after learning_starts). Episode return/length are always logged.
    train_metrics_log_interval: int = 20
    # Rolling-window episode stats + console line every N env steps.
    stats_log_interval: int = 5000

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


def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env

    return _thunk


class SingleEnvVector:
    """Minimal stand-in for gym.vector.SyncVectorEnv with one env.

    Same surface as the trainer uses (`step`, `reset`, `close`,
    `single_*_space`, `num_envs`, `envs`) and the same autoreset semantics
    (on done: reset, return the reset obs, stash `final_observation` /
    `final_info`). Skips gymnasium's per-key info-array bookkeeping (~60
    keys per step for this env) which cost ~0.15 ms/step.
    """

    def __init__(self, env_fn) -> None:
        self.env = env_fn()
        self.envs = [self.env]
        self.num_envs = 1
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self._obs_dtype = np.float64

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options) if options is not None else self.env.reset(seed=seed)
        return np.asarray(obs)[None].copy(), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        # Goal tasks return the reward as a 1-element array; normalise to float.
        reward = float(np.asarray(reward, dtype=np.float64).reshape(-1)[0])
        infos = {"paddle_puck_collision_count": np.array([info.get("paddle_puck_collision_count", 0)])}
        if terminated or truncated:
            final_obs = np.asarray(obs).copy()
            obs, _ = self.env.reset()
            infos["final_observation"] = [final_obs]
            infos["final_info"] = [info]
            infos["_final_observation"] = np.array([True])
        return (
            np.asarray(obs)[None].copy(),
            np.array([reward], dtype=np.float64),
            np.array([bool(terminated)]),
            np.array([bool(truncated)]),
            infos,
        )

    def close(self):
        self.env.close()


class AsyncCheckpointEvaluator:
    """Launches `scripts.td3.checkpoint_eval` as a CPU-only subprocess per
    checkpoint and reaps finished ones. At most one eval runs at a time; if a
    new checkpoint arrives while the previous eval is still running we wait
    for it (bounded backlog, and evals are ~20 s vs minutes per checkpoint)."""

    def __init__(self, checkpoint_interval: int) -> None:
        self.checkpoint_interval = int(checkpoint_interval)
        self._running: List[Tuple[subprocess.Popen, int, str, object]] = []

    def launch(self, checkpoint_dir: str, global_step: int) -> None:
        self.reap(block=True)
        log_path = os.path.join(checkpoint_dir, "eval.log")
        log_file = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "1")
        eval_call_index = max(1, global_step // max(self.checkpoint_interval, 1))
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.td3.checkpoint_eval",
                "--checkpoint-dir",
                checkpoint_dir,
                "--eval-call-index",
                str(eval_call_index),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
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
                    lines = [ln.strip() for ln in f if "Multi-env eval" in ln or "Traceback" in ln]
                if lines:
                    summary = lines[-1]
            except OSError:
                pass
            status = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
            print(f"[eval step {step}] {status} {summary}", flush=True)
        self._running = still_running

    def wait_all(self) -> None:
        self.reap(block=True)


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

    torch.set_num_threads(max(1, int(args.torch_num_threads)))

    # `config` must be a MODULE-LEVEL name because the module-level
    # `make_env(env_id)._thunk` closure reads it as a free variable (and
    # td3_training_gat.py swaps make_env out wholesale).
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

    # A single Box2D env is far cheaper to step in-process than through the
    # AsyncVectorEnv pipe (1440 vs 680 steps/s measured). Only fan out to
    # subprocesses when there is more than one env to run.
    if args.num_envs == 1:
        envs = SingleEnvVector(make_env(0))
    else:
        envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    action_scale = 1
    device = torch.device(args.device)
    rollout_device = torch.device(args.rollout_device)

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
    ).to(device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(device)
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
        ).to(device)
        for _ in range(args.num_critics)
    ]
    qfs_target = [
        TD3QNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_layer_size=args.q_hidden_layer_size,
            num_hidden_layers=args.q_num_hidden_layers,
        ).to(device)
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

    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device)
    action_low_rollout = action_low.to(rollout_device)
    action_high_rollout = action_high.to(rollout_device)

    use_cuda_graphs = bool(args.use_cuda_graphs) and device.type == "cuda"
    if use_cuda_graphs and args.target_critic_subset_size is not None and (
        args.target_critic_subset_size < args.num_critics
    ):
        print("target_critic_subset_size < num_critics: CUDA-graph capture disabled (eager updates).")
        use_cuda_graphs = False

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
            actor, actor_target, residual_actor_optimizer = build_residual_training(
                base_actor=actor,
                policy_env_view=policy_env_view,
                action_low=action_low,
                action_high=action_high,
                device=args.device,
                residual_scale=args.residual_scale,
                residual_weight_decay=args.residual_weight_decay,
                agent_hidden_layer_size=args.agent_hidden_layer_size,
                agent_num_hidden_layers=args.agent_num_hidden_layers,
                policy_lr=args.policy_lr,
            )
            if use_cuda_graphs:
                # Rebuild the residual optimizer capturable (same hyper-params).
                residual_actor_optimizer = optim.Adam(
                    actor.residual.parameters(),
                    lr=args.policy_lr,
                    weight_decay=args.residual_weight_decay,
                    capturable=True,
                    fused=True,
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

    # fused=True: one kernel per optimizer step instead of one per tensor.
    adam_kwargs = dict(capturable=use_cuda_graphs, fused=(device.type == "cuda"))
    q_optimizer = optim.Adam(
        [p for q in qfs for p in q.parameters()],
        lr=args.q_lr,
        weight_decay=args.q_weight_decay,
        **adam_kwargs,
    )
    if residual_actor_optimizer is not None:
        actor_optimizer = residual_actor_optimizer
    else:
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, **adam_kwargs)

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
    last_action_for_policy = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=rollout_device)
    interval_paddle_puck_collisions = 0.0
    interval_env_steps = 0
    interval_primitive_env_steps = 0
    interval_primitive_horizontal_env_steps = 0
    rolling_step_stats_window = deque()
    rolling_episode_stats_window = deque()
    episode_trajectory = EpisodeTrajectory.empty()
    recent_episode_returns = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0

    # --- Live episode GIF recording (renders the actual training env) ---
    gif_recorder = None
    train_renderer = None
    if isinstance(envs, SingleEnvVector):
        train_renderer = AirHockeyRenderer(
            envs.envs[0], show_target_position=True, show_acceleration_arrow=False
        )
        gif_recorder = GIFEpisodeRecorder(
            log_parent_dir,
            watch_ring_size=args.watch_ring_size,
            watch_episode_interval=args.watch_episode_interval,
            sample_gif_interval=args.sample_gif_interval,
            sample_gif_max_storage_mb=args.sample_gif_max_storage_mb,
        )
    else:
        print("num_envs > 1: training-episode GIF recording disabled (envs live in subprocesses).")

    global_step = 0
    iteration = 0
    # Counts completed critic updates across all training cycles. Used to gate
    # Polyak averaging by `target_network_frequency` so the schedule survives
    # cycles where q_updates < target_network_frequency or where a critic step
    # is skipped (empty replay batch).
    total_critic_updates = 0

    train_metrics = initialize_train_metrics()

    same_direction_range_set = any(
        v is not None
        for v in (
            args.exploration_same_direction_min_angle_deg,
            args.exploration_same_direction_max_angle_deg,
            args.exploration_same_direction_min_magnitude,
            args.exploration_same_direction_max_magnitude,
        )
    )
    # numpy backend on the CPU rollout path (53 vs 125 us per step); the torch
    # class remains for the simulator-space range mode and non-CPU rollouts.
    use_numpy_selector = rollout_device.type == "cpu" and not same_direction_range_set
    if use_numpy_selector:
        primitive_selector = NumpyPrimitiveExplorationSelector(
            num_envs=args.num_envs,
            chance=primitive_exploration_chance_for_step(args, global_step),
            takeover_steps=args.exploration_primitive_steps,
            direction_y_component_weight=args.exploration_direction_y_component_weight,
            seed=args.seed,
        )
    else:
        primitive_selector = PrimitiveExplorationSelector(
            num_envs=args.num_envs,
            chance=primitive_exploration_chance_for_step(args, global_step),
            takeover_steps=args.exploration_primitive_steps,
            device=rollout_device,
            dtype=torch.float32,
            direction_y_component_weight=args.exploration_direction_y_component_weight,
            action_delta_x=args.exploration_action_delta_x,
            action_delta_y=args.exploration_action_delta_y,
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
    if args.eval_mode:
        primitive_selector.chance = 0.0
    if resume_checkpoint is not None:
        restored_state = load_resume_training_state(
            resume_checkpoint,
            device=str(rollout_device),
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
        last_action_for_policy = restored_state["last_action_for_policy"].to(rollout_device)
        train_metrics = initialize_train_metrics()
        train_metrics.update(
            {k: v for k, v in restored_state["train_metrics"].items() if k in train_metrics}
        )
        interval_paddle_puck_collisions = restored_state["interval_paddle_puck_collisions"]
        interval_env_steps = restored_state["interval_env_steps"]
        interval_primitive_env_steps = restored_state["interval_primitive_env_steps"]
        interval_primitive_horizontal_env_steps = restored_state["interval_primitive_horizontal_env_steps"]
        episode_trajectory = restored_state["episode_trajectory"]
        recent_episode_returns = restored_state["recent_episode_returns"]
        episode_return_success_threshold = restored_state["episode_return_success_threshold"]
        rolling_step_stats_window = restored_state["rolling_step_stats_window"]
        rolling_episode_stats_window = restored_state["rolling_episode_stats_window"]
        # Optimizer.load_state_dict restores the checkpoint's param_groups, so a
        # checkpoint written by the pre-graph trainer (capturable=False,
        # fused=False, step on CPU) would break CUDA-graph capture. Re-assert
        # the runtime flags and move Adam's step counters to the device.
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

    # --- Update engine (CUDA-graph captured on GPU, eager otherwise) ---
    updater = GraphedTD3Update(
        actor=actor,
        actor_target=actor_target,
        qfs=qfs,
        qfs_target=qfs_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        success_rb=success_rb,
        failure_rb=failure_rb,
        batch_size=args.batch_size,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        action_low=action_low,
        action_high=action_high,
        h_transform_eps=args.h_transform_eps,
        use_last_action_in_policy_state=args.use_last_action_in_policy_state,
        per_enabled=args.per_enabled,
        per_eps=args.per_eps,
        critic_per_fraction=args.critic_per_fraction,
        cql_alpha=args.cql_alpha,
        cql_n_random=args.cql_n_random,
        target_critic_subset_size=args.target_critic_subset_size,
        use_graph=use_cuda_graphs,
        compile_update=args.compile_update,
    )
    print(
        f"Update engine: {'CUDA graphs' if updater.use_graph else 'eager'}"
        f"{' + torch.compile' if updater.compile_update else ''} on {device}; rollout on {rollout_device}"
    )

    # --- CPU replica of the actor that drives the environment ---
    rollout_actor = copy.deepcopy(actor).to(rollout_device).eval()
    rollout_actor_params = list(rollout_actor.parameters())
    train_actor_params = list(actor.parameters())
    # torch.compile fuses the ~40-op residual-MLP forward into one C++ call
    # (273 -> 140 us per env step on this box). Params are read by reference
    # so in-place refreshes are picked up without recompiling.
    rollout_policy = lambda policy_obs: deterministic_actor_action(rollout_actor, policy_obs)  # noqa: E731
    if args.compile_rollout_actor and rollout_device.type == "cpu" and not hasattr(rollout_actor, "get_action_mean_and_logstd"):
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
        # One flatten kernel + one device->host copy instead of one sync per tensor.
        with torch.no_grad():
            flat = torch.nn.utils.parameters_to_vector(train_actor_params).to(rollout_device)
            torch.nn.utils.vector_to_parameters(flat, rollout_actor_params)

    checkpoint_evaluator = (
        AsyncCheckpointEvaluator(args.checkpoint_interval) if args.checkpoint_eval_async else None
    )

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

    def run_checkpoint_eval(model_path: str, checkpoint_dir: str) -> None:
        if checkpoint_evaluator is not None:
            checkpoint_evaluator.launch(checkpoint_dir, global_step)
            return
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

    training_cycles = 0
    next_stats_log_step = ((global_step // args.stats_log_interval) + 1) * args.stats_log_interval
    # Opt-in per-section wall-clock accounting (TD3_PROFILE_SECTIONS=1):
    # printed with every stats line. Costs a few perf_counter calls per step.
    profile_sections = os.environ.get("TD3_PROFILE_SECTIONS", "0") == "1"
    section_time: Dict[str, float] = {"policy": 0.0, "env": 0.0, "bookkeeping": 0.0, "train_update": 0.0, "train_other": 0.0}
    _pc = time.perf_counter
    start_time = time.time()
    last_critic_out: Dict[str, torch.Tensor] | None = None
    last_actor_out: Dict[str, torch.Tensor] | None = None

    while global_step < args.total_timesteps:
        if not args.eval_mode and iteration % 100 == 0:
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
        if profile_sections:
            _t_sec = _pc()
        prev_action_for_transition = last_action_for_policy.clone()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=rollout_device)
        policy_obs_tensor = augment_policy_observation(
            obs_tensor, last_action_for_policy, args.use_last_action_in_policy_state
        )

        if (
            global_step < args.learning_starts
            and not args.eval_mode
            and args.exploration_pre_learning_action_source == "random"
        ):
            action_tensor = torch.as_tensor(
                np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]),
                dtype=torch.float32,
                device=rollout_device,
            )
        else:
            with torch.no_grad():
                action_tensor = rollout_policy(policy_obs_tensor)
                if not args.eval_mode:
                    action_tensor = action_tensor + torch.randn_like(action_tensor) * args.exploration_noise
                action_tensor = torch.clamp(action_tensor, action_low_rollout, action_high_rollout)

        if not args.eval_mode and use_numpy_selector:
            actions_np, primitive_step_stats = primitive_selector.apply(
                action_tensor.numpy(), action_low_np, action_high_np, return_stats=True
            )
            action_tensor = torch.from_numpy(actions_np)
        elif not args.eval_mode:
            action_tensor, primitive_step_stats = primitive_selector.apply(
                action_tensor,
                action_low=action_low_rollout,
                action_high=action_high_rollout,
                return_stats=True,
            )
        else:
            primitive_step_stats = {
                "primitive_applied_count": 0,
                "primitive_horizontal_dominant_count": 0,
            }
        actions = action_tensor.numpy()

        if gif_recorder is not None:
            gif_recorder.capture_frame(train_renderer, global_step)

        if profile_sections:
            _t_now = _pc(); section_time["policy"] += _t_now - _t_sec; _t_sec = _t_now
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if profile_sections:
            _t_now = _pc(); section_time["env"] += _t_now - _t_sec; _t_sec = _t_now
        dones = np.logical_or(terminations, truncations)
        step_puck_hits = sum_info_metric(infos, "paddle_puck_collision_count")
        interval_paddle_puck_collisions += step_puck_hits
        interval_env_steps += args.num_envs
        interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
        interval_primitive_horizontal_env_steps += int(
            primitive_step_stats["primitive_horizontal_dominant_count"]
        )
        done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=rollout_device)
        primitive_selector.reset(done_tensor)
        last_action_for_policy = action_tensor.clone()
        last_action_for_policy[done_tensor] = 0

        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=rollout_device)

        if gif_recorder is not None:
            gif_recorder.note_reward(float(rewards[0]))

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
            )
        )
        rolling_cutoff_step = int(global_step + args.num_envs - ROLLING_STATS_WINDOW_STEPS)
        while rolling_step_stats_window and int(rolling_step_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_step_stats_window.popleft()
        while rolling_episode_stats_window and int(rolling_episode_stats_window[0][0]) <= rolling_cutoff_step:
            rolling_episode_stats_window.popleft()

        real_next_obs = next_obs
        if truncations.any():
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
        if not args.eval_mode:
            real_next_obs_tensor = torch.as_tensor(real_next_obs, dtype=torch.float32, device=rollout_device)
            terminations_tensor = torch.as_tensor(terminations, dtype=torch.float32, device=rollout_device)
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

        if episode_finished and gif_recorder is not None:
            gif_recorder.on_episode_end(global_step)

        obs = next_obs
        if profile_sections:
            _t_now = _pc(); section_time["bookkeeping"] += _t_now - _t_sec; _t_sec = _t_now

        # ----------------------------------------------------------- training
        if global_step > args.learning_starts and episode_finished and not args.eval_mode:
            if args.per_enabled:
                per_beta = linear_anneal(
                    args.per_beta_start,
                    args.per_beta_end,
                    global_step,
                    args.per_beta_anneal_steps,
                )
            else:
                per_beta = 0.0

            for q_update_idx in range(args.q_updates):
                success_batch_count, failure_batch_count = critic_success_failure_counts(
                    batch_size=args.batch_size,
                    success_fraction=args.critic_success_sample_fraction,
                    success_available=len(success_rb) > 0,
                    failure_available=len(failure_rb) > 0,
                )
                if success_batch_count + failure_batch_count == 0:
                    continue

                last_critic_out = updater.critic_update(success_batch_count, failure_batch_count, per_beta)
                if profile_sections:
                    _t_now = _pc(); section_time["train_update"] += _t_now - _t_sec; _t_sec = _t_now

                total_critic_updates += 1
                if total_critic_updates % args.target_network_frequency == 0:
                    updater.polyak()

            for actor_update_idx in range(args.actor_updates_per_iteration):
                actor_success_count, actor_failure_count = critic_success_failure_counts(
                    batch_size=args.batch_size,
                    success_fraction=args.critic_success_sample_fraction,
                    success_available=len(success_rb) > 0,
                    failure_available=len(failure_rb) > 0,
                )
                if actor_success_count + actor_failure_count == 0:
                    continue
                last_actor_out = updater.actor_update(actor_success_count, actor_failure_count)
                if profile_sections:
                    _t_now = _pc(); section_time["train_update"] += _t_now - _t_sec; _t_sec = _t_now

            # Push the updated actor weights to the CPU replica that drives the env.
            refresh_rollout_actor()
            training_cycles += 1
            if profile_sections:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                _t_now = _pc(); section_time["train_other"] += _t_now - _t_sec; _t_sec = _t_now

            if training_cycles % args.train_metrics_log_interval == 0:
                if last_critic_out is not None:
                    train_metrics.update(
                        {
                            "losses/q_loss": float(last_critic_out["q_loss"].item()),
                            "losses/q_total_loss": float(last_critic_out["q_total_loss"].item()),
                            "losses/q1_mean": float(last_critic_out["q1_mean"].item()),
                            "debug/bellman_target_original_mean": float(
                                last_critic_out["bellman_target_mean"].item()
                            ),
                            "debug/next_q_h_mean": float(last_critic_out["next_q_h_mean"].item()),
                            "rewards/sampled_reward_mean": float(
                                last_critic_out["sampled_reward_mean"].item()
                            ),
                            "replay/per_priority_td_error_mean": float(
                                last_critic_out["priority_td_error_mean"].item()
                            ),
                        }
                    )
                if last_actor_out is not None:
                    train_metrics.update(
                        {
                            "losses/actor_loss": float(last_actor_out["actor_loss"].item()),
                            "losses/actor_norm_q_mean": float(last_actor_out["actor_norm_q_mean"].item()),
                        }
                    )
                train_metrics.update(
                    {
                        "replay/per_beta": float(per_beta),
                        "replay/success_buffer_size": float(len(success_rb)),
                        "replay/failure_buffer_size": float(len(failure_rb)),
                        "replay/episode_return_success_threshold": float(episode_return_success_threshold),
                    }
                )
                log_scalar_metrics(writer, train_metrics, global_step)
                writer.add_scalar("charts/exploration_primitive_chance", primitive_selector.chance, global_step)
                elapsed = max(time.time() - start_time, 1e-6)
                writer.add_scalar("charts/SPS", int((global_step - start_step) / elapsed), global_step)

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
            print(
                f"Step {global_step}: SPS {(global_step - start_step) / elapsed:.0f}",
                flush=True,
            )
            if profile_sections:
                total_sec = max(sum(section_time.values()), 1e-9)
                print(
                    "  sections: "
                    + ", ".join(f"{k} {v:.1f}s ({100 * v / total_sec:.0f}%)" for k, v in section_time.items()),
                    flush=True,
                )
            interval_paddle_puck_collisions = 0.0
            interval_env_steps = 0
            interval_primitive_env_steps = 0
            interval_primitive_horizontal_env_steps = 0
            next_stats_log_step += args.stats_log_interval

        if global_step > 0 and global_step % args.checkpoint_interval == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            model_path = save_full_checkpoint(checkpoint_dir)
            print(f"\nCheckpoint saved at step {global_step}", flush=True)
            run_checkpoint_eval(model_path, checkpoint_dir)

        iteration += 1
        global_step += args.num_envs

    envs.close()
    if gif_recorder is not None:
        gif_recorder.close()
    if checkpoint_evaluator is not None:
        checkpoint_evaluator.wait_all()

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
        "charts/avg_episodic_return",
        "losses/q_loss",
        "losses/actor_loss",
        "losses/q1_mean",
        "replay/per_priority_td_error_mean",
    ]
    save_tensorboard_plots(log_parent_dir, config, metrics=metrics)
    writer.close()


if __name__ == "__main__":
    _entrypoint()
