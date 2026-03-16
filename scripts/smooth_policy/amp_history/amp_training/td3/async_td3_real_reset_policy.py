"""Async-style on-robot TD3 for reset-policy adaptation.

This script keeps the actor from an existing checkpoint (if provided), reinitializes
all critic state, and trains only on reset-window transitions:
- reset window starts when the first upward reset motion completes
- reset window ends on success/failure/estop terminal conditions

Rewards are single-stream:
- +1 on success terminal
- -1 on e-stop/protective stop terminal
- 0 on non-estop failure terminal
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM, compute_failure
from scripts.smooth_policy.agent import ResidualMLPTrunk, layer_init
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    load_split_optional_data,
    load_split_trajectory_data,
)


_TRAIN_VALS_CUR_TIME = 0
_TRAIN_VALS_STEP_INDEX = 2
_TRAIN_VALS_POSE = slice(5, 11)
_TRAIN_VALS_DESIRED_POSE = slice(26, 32)
_TRAIN_VALS_PUCK = slice(32, 35)


@dataclass
class Args:
    args_file: str | None = None
    config: str = "scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml"
    model_path: str | None = None
    device: str = "cuda:0"
    seed: int = 0

    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    policy_lr: float = 3e-4
    q_updates: int = 1
    actor_updates_per_iteration: int = 1
    target_network_frequency: int = 1
    min_replay_size_before_learning: int = 1000
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    h_transform_eps: float = 1e-3

    buffer_size: int = int(3e5)
    warm_start_hdf5_dirs: Tuple[str, ...] = ()
    warm_start_hdf5_recursive: bool = True
    min_episode_timesteps: int = 20

    action_scale: float = 0.02
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 2
    q_hidden_layer_size: int = 128
    q_num_hidden_layers: int = 2
    use_last_action_in_policy_state: bool = False

    max_reset_window_steps: int = 120
    reset_failure_bottom_margin: float = 0.25
    reset_failure_bottom_fail_count: int = 2
    reset_failure_occluded_fail_count: int = 6

    max_env_steps: int = 150000
    collector_log_interval_sec: float = 30.0
    log_parent_dir: str = "runs/async_td3_reset_policy"
    run_name: str = "reset_policy_online"


def _build_args_file_defaults(args_file_path: str) -> dict:
    with open(args_file_path, "r") as f:
        loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    if loaded_yaml is None:
        return {}
    if not isinstance(loaded_yaml, dict):
        raise ValueError(f"Expected args_file YAML to be a mapping, got {type(loaded_yaml)}")
    valid = {field.name for field in fields(Args)}
    out: dict = {}
    for key, value in loaded_yaml.items():
        if key in valid and value is not None:
            out[key] = value
    return out


def _parse_args() -> Args:
    pre = tyro.cli(Args, args=[])
    cli_args = list(__import__("sys").argv[1:])
    args_file = pre.args_file
    if "--args-file" in cli_args:
        idx = cli_args.index("--args-file")
        if idx + 1 < len(cli_args):
            args_file = cli_args[idx + 1]
    elif "--args_file" in cli_args:
        idx = cli_args.index("--args_file")
        if idx + 1 < len(cli_args):
            args_file = cli_args[idx + 1]

    if args_file is not None:
        defaults = _build_args_file_defaults(args_file)
        return tyro.cli(Args, default=Args(**defaults))
    return tyro.cli(Args)


def _prepare_air_hockey_config(config: dict, seed: int = 0, return_goal_obs: bool = False) -> dict:
    ah = dict(config["air_hockey"])
    if "seed" not in ah:
        seed_cfg = config.get("seed", seed)
        if isinstance(seed_cfg, (list, tuple)):
            seed_cfg = seed_cfg[0] if len(seed_cfg) > 0 else 0
        ah["seed"] = int(seed_cfg)
    if "n_training_steps" not in ah:
        ah["n_training_steps"] = config.get("n_training_steps", 1_000_000)
    if "return_goal_obs" not in ah:
        ah["return_goal_obs"] = return_goal_obs
    return ah


def h_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def h_inverse(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    abs_x = torch.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    sqrt_inner = torch.sqrt(inner)
    quotient = (sqrt_inner - 1) / (2 * eps)
    return torch.sign(x) * (quotient**2 - 1)


def build_policy_env_view(obs_dim: int, act_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
    )


def augment_policy_observation(observation: torch.Tensor, last_action: torch.Tensor, use_last_action: bool) -> torch.Tensor:
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def deterministic_actor_action(actor: DeterministicAgent, policy_obs: torch.Tensor) -> torch.Tensor:
    return actor.get_action(policy_obs)


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


class TD3SingleHeadQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_layer_size: int = 128, num_hidden_layers: int = 2):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
        input_dim = int(obs_dim + act_dim)
        self.trunk = ResidualMLPTrunk(
            input_dim=input_dim,
            hidden_layer_size=hidden_layer_size,
            num_residual_blocks=int(num_hidden_layers),
            units_per_block=4,
        )
        self.value_head = layer_init(nn.Linear(hidden_layer_size, 1), std=0.01)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = self.trunk(x)
        return self.value_head(x)


class ReplayBufferSingle:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.prev_actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.bootstrap_terminals = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.position = 0
        self.size = 0

    def add_batch(self, batch: dict[str, torch.Tensor]) -> int:
        n = int(batch["observations"].shape[0])
        if n <= 0:
            return 0
        if n > self.capacity:
            start = n - self.capacity
            batch = {k: v[start:] for k, v in batch.items()}
            n = self.capacity
        first = min(n, self.capacity - self.position)
        s1 = slice(self.position, self.position + first)
        self.observations[s1] = batch["observations"][:first]
        self.next_observations[s1] = batch["next_observations"][:first]
        self.actions[s1] = batch["actions"][:first]
        self.prev_actions[s1] = batch["prev_actions"][:first]
        self.rewards[s1] = batch["rewards"][:first]
        self.dones[s1] = batch["dones"][:first]
        self.bootstrap_terminals[s1] = batch["bootstrap_terminals"][:first]
        second = n - first
        if second > 0:
            s2 = slice(0, second)
            self.observations[s2] = batch["observations"][first:]
            self.next_observations[s2] = batch["next_observations"][first:]
            self.actions[s2] = batch["actions"][first:]
            self.prev_actions[s2] = batch["prev_actions"][first:]
            self.rewards[s2] = batch["rewards"][first:]
            self.dones[s2] = batch["dones"][first:]
            self.bootstrap_terminals[s2] = batch["bootstrap_terminals"][first:]
        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        return n

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations": self.observations[idx],
            "next_observations": self.next_observations[idx],
            "actions": self.actions[idx],
            "prev_actions": self.prev_actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "bootstrap_terminals": self.bootstrap_terminals[idx],
        }

    def __len__(self) -> int:
        return int(self.size)


def _classify_stop_event(env: AirHockeyEnv, step_info: dict | None = None) -> bool:
    if isinstance(step_info, dict):
        if "protective_stop" in step_info and bool(step_info.get("protective_stop", False)):
            return True
        if "controller_connected" in step_info and not bool(step_info.get("controller_connected", True)):
            return True
        if "estop" in step_info:
            arr = np.asarray(step_info.get("estop"), dtype=np.float64).reshape(-1)
            if arr.size > 0 and bool(arr[0] > 0.5):
                return True
    simulator = getattr(env, "simulator", None)
    readiness_fn = getattr(simulator, "robot_command_readiness", None)
    if callable(readiness_fn):
        try:
            readiness = readiness_fn()
            if isinstance(readiness, dict):
                if bool(readiness.get("protective_stop", False)):
                    return True
                if "controller_connected" in readiness and not bool(readiness.get("controller_connected", True)):
                    return True
        except Exception:
            return True
    return False


def _list_hdf5_files(input_roots: Sequence[str], recursive: bool, rng: np.random.Generator) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root_str in input_roots:
        root = Path(str(root_str)).expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() == ".hdf5" and root not in seen:
                seen.add(root)
                out.append(root)
            continue
        itr = root.rglob("*.hdf5") if recursive else root.glob("*.hdf5")
        for path in itr:
            p = path.resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
    rng.shuffle(out)
    return out


def _episode_batch_from_hdf5(path: Path, env: AirHockeyEnv, device: torch.device) -> dict[str, torch.Tensor] | None:
    vals = np.asarray(load_split_trajectory_data(str(path)), dtype=np.float64)
    if vals.ndim != 2 or vals.shape[0] < 2:
        return None
    opt = load_split_optional_data(str(path))
    pose_xy = np.asarray(vals[:, _TRAIN_VALS_POSE][:, :2], dtype=np.float64)
    desired_pose_xy = np.asarray(vals[:, _TRAIN_VALS_DESIRED_POSE][:, :2], dtype=np.float64)
    puck_xy = np.asarray(vals[:, _TRAIN_VALS_PUCK][:, :2], dtype=np.float64)
    puck_occ = np.asarray(vals[:, _TRAIN_VALS_PUCK][:, 2], dtype=np.float64)
    move_lims = np.asarray(getattr(env.simulator, "move_lims", (1.0, 1.0)), dtype=np.float64).reshape(-1)[:2]
    move_lims[np.abs(move_lims) < 1e-6] = 1.0
    actions_xy = np.clip((desired_pose_xy - pose_xy) / move_lims[None, :], -1.0, 1.0)

    # Conservative terminal labeling for reset-HDF5 episodes:
    # only final transition is terminal; reward based on estop/success/failure.
    stop_flags = opt.get("stop_flags")
    estop_any = bool(np.any(vals[:, 3] > 0.5))
    if isinstance(stop_flags, np.ndarray) and stop_flags.ndim >= 2 and stop_flags.shape[1] > 0:
        estop_any = estop_any or bool(np.any(stop_flags[:, 0] > 0.5))
    success_terminal = bool(puck_xy[-1, 0] < pose_xy[-1, 0]) and (not estop_any)
    terminal_reward = -1.0 if estop_any else (1.0 if success_terminal else 0.0)

    # We use current env observation helper on lightweight synthetic states.
    def _state_at(i: int) -> dict:
        return {
            "paddles": {"paddle_ego": {"position": pose_xy[i], "velocity": np.zeros(2), "acceleration": np.zeros(2), "jerk": np.zeros(2)}},
            "pucks": [{"position": puck_xy[i], "velocity": np.zeros(2), "occluded": np.array([puck_occ[i]], dtype=np.float64), "history": []}],
        }

    obs_list = []
    next_obs_list = []
    act_list = []
    prev_act_list = []
    rew_list = []
    done_list = []
    bt_list = []
    for i in range(1, vals.shape[0]):
        s_prev = _state_at(i - 1)
        s_next = _state_at(i)
        obs = env.get_observation(s_prev, obs_type=env.obs_type, puck_history=[], paddle_history=[])
        next_obs = env.get_observation(s_next, obs_type=env.obs_type, puck_history=[], paddle_history=[])
        done = 1.0 if i == (vals.shape[0] - 1) else 0.0
        reward = terminal_reward if done > 0.5 else 0.0
        obs_list.append(torch.as_tensor(obs, dtype=torch.float32, device=device))
        next_obs_list.append(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
        act_list.append(torch.as_tensor(actions_xy[i], dtype=torch.float32, device=device))
        prev_act_list.append(torch.as_tensor(actions_xy[i - 1], dtype=torch.float32, device=device))
        rew_list.append(torch.tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(done, dtype=torch.float32, device=device))
        bt_list.append(torch.tensor(done, dtype=torch.float32, device=device))
    if len(obs_list) == 0:
        return None
    return {
        "observations": torch.stack(obs_list, dim=0),
        "next_observations": torch.stack(next_obs_list, dim=0),
        "actions": torch.stack(act_list, dim=0),
        "prev_actions": torch.stack(prev_act_list, dim=0),
        "rewards": torch.stack(rew_list, dim=0).view(-1),
        "dones": torch.stack(done_list, dim=0).view(-1),
        "bootstrap_terminals": torch.stack(bt_list, dim=0).view(-1),
    }


def _warm_start_from_hdf5(args: Args, replay: ReplayBufferSingle, env: AirHockeyEnv, device: torch.device) -> tuple[int, int]:
    rng = np.random.default_rng(args.seed)
    paths = _list_hdf5_files(args.warm_start_hdf5_dirs, bool(args.warm_start_hdf5_recursive), rng)
    loaded_eps = 0
    dropped_short = 0
    for path in paths:
        batch = _episode_batch_from_hdf5(path, env, device)
        if batch is None:
            continue
        if int(batch["observations"].shape[0]) < int(args.min_episode_timesteps):
            dropped_short += 1
            continue
        replay.add_batch(batch)
        loaded_eps += 1
    return loaded_eps, dropped_short


def main(args: Args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_cfg = _prepare_air_hockey_config(config, seed=args.seed)
    sim_params = env_cfg.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
    env = AirHockeyEnv(env_cfg)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = torch.as_tensor(np.asarray(env.action_space.low, dtype=np.float32), device=device).unsqueeze(0)
    action_high = torch.as_tensor(np.asarray(env.action_space.high, dtype=np.float32), device=device).unsqueeze(0)

    policy_obs_dim = obs_dim + act_dim if args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # Actor-only resume: load actor weights, always keep freshly initialized critics.
    if args.model_path is not None and os.path.exists(args.model_path):
        loaded_obj = torch.load(args.model_path, map_location=device, weights_only=False)
        if isinstance(loaded_obj, dict) and "actor" in loaded_obj:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj["actor"]), strict=False)
        else:
            actor.load_state_dict(extract_deterministic_state_dict(loaded_obj), strict=False)
        actor_target.load_state_dict(actor.state_dict())
        print(f"[reset_td3] loaded actor-only from {args.model_path}; critics reinitialized")

    qf1 = TD3SingleHeadQNetwork(obs_dim, act_dim, args.q_hidden_layer_size, args.q_num_hidden_layers).to(device)
    qf2 = TD3SingleHeadQNetwork(obs_dim, act_dim, args.q_hidden_layer_size, args.q_num_hidden_layers).to(device)
    qf1_target = TD3SingleHeadQNetwork(obs_dim, act_dim, args.q_hidden_layer_size, args.q_num_hidden_layers).to(device)
    qf2_target = TD3SingleHeadQNetwork(obs_dim, act_dim, args.q_hidden_layer_size, args.q_num_hidden_layers).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.q_weight_decay)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    replay = ReplayBufferSingle(args.buffer_size, (obs_dim,), (act_dim,), device)
    loaded_eps, dropped_short_hdf5 = _warm_start_from_hdf5(args, replay, env, device)
    if loaded_eps > 0 or dropped_short_hdf5 > 0:
        print(f"[warm_start_reset] loaded_episodes={loaded_eps} dropped_short={dropped_short_hdf5}")

    run_dir = Path(args.log_parent_dir) / f"{args.run_name}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    total_steps = 0
    total_updates = 0
    total_actor_updates = 0
    dropped_short_online = 0
    accepted_online_episodes = 0
    last_log = time.time()
    reset_rng = np.random.default_rng(args.seed)

    while total_steps < int(args.max_env_steps):
        # Phase-1: run reset FSM until first upward phase is completed.
        fsm = ResetPolicyFSM(env, reset_rng)
        start_state = env.simulator.get_current_state()
        while (not fsm.done) and fsm.phase != "wait_for_puck":
            action = fsm.step(start_state)
            _, _, _, _, step_info = env.step(action)
            start_state = env.simulator.get_current_state()
            if _classify_stop_event(env, step_info):
                break
        fsm.close()

        if _classify_stop_event(env):
            # Estop transition with no actor data; reset and continue.
            env.reset(seed=None, write_traj=False) if "write_traj" in str(env.reset) else env.reset(seed=None)
            continue

        # If first upward already achieved success/terminal, skip this cycle.
        if getattr(fsm, "done", False):
            continue

        # Reset-policy actor episode starts here.
        state = env.simulator.get_current_state()
        obs = env.get_observation(
            state,
            obs_type=env.obs_type,
            puck_history=getattr(env.simulator, "puck_history", []),
            paddle_history=getattr(env.simulator, "paddle_history", []),
        )
        last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
        ep_obs, ep_next_obs, ep_actions, ep_prev_actions = [], [], [], []
        ep_rewards, ep_dones, ep_bootstrap = [], [], []
        fail_counters = {"bottom": 0, "occ": 0}

        for ep_step in range(int(args.max_reset_window_steps)):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            pol_obs = augment_policy_observation(obs_t, last_action, args.use_last_action_in_policy_state)
            action_t = deterministic_actor_action(actor, pol_obs)
            if args.exploration_noise > 0:
                action_t = action_t + torch.randn_like(action_t) * float(args.exploration_noise)
            action_t = torch.clamp(action_t, action_low, action_high)
            env_action = action_t.squeeze(0).detach().cpu().numpy()

            prev_action = last_action.clone()
            next_obs, _, terminated, truncated, step_info = env.step(env_action)
            state_now = getattr(env, "current_state", env.simulator.get_current_state())
            estop = _classify_stop_event(env, step_info)
            puck_x = float(state_now["pucks"][0]["position"][0])
            paddle_x = float(state_now["paddles"]["paddle_ego"]["position"][0])
            success = bool(puck_x < paddle_x)
            fail = compute_failure(
                state_now,
                table_x_bot=float(getattr(env, "table_x_bot", 0.0)),
                bottom_margin=float(args.reset_failure_bottom_margin),
                bottom_fail_count=int(args.reset_failure_bottom_fail_count),
                occluded_fail_count=int(args.reset_failure_occluded_fail_count),
                counters=fail_counters,
            )

            done = bool(estop or success or fail or terminated or truncated or ep_step == int(args.max_reset_window_steps) - 1)
            if done:
                reward = -1.0 if estop else (1.0 if success else 0.0)
                bootstrap_terminal = 1.0
            else:
                reward = 0.0
                bootstrap_terminal = 0.0

            ep_obs.append(obs_t[0])
            ep_next_obs.append(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
            ep_actions.append(action_t[0])
            ep_prev_actions.append(prev_action[0])
            ep_rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            ep_dones.append(torch.tensor(1.0 if done else 0.0, dtype=torch.float32, device=device))
            ep_bootstrap.append(torch.tensor(bootstrap_terminal, dtype=torch.float32, device=device))
            total_steps += 1
            last_action = action_t.detach().clone()
            obs = next_obs

            if done:
                break

        ep_len = len(ep_obs)
        if ep_len >= int(args.min_episode_timesteps):
            batch = {
                "observations": torch.stack(ep_obs, dim=0),
                "next_observations": torch.stack(ep_next_obs, dim=0),
                "actions": torch.stack(ep_actions, dim=0),
                "prev_actions": torch.stack(ep_prev_actions, dim=0),
                "rewards": torch.stack(ep_rewards, dim=0).view(-1),
                "dones": torch.stack(ep_dones, dim=0).view(-1),
                "bootstrap_terminals": torch.stack(ep_bootstrap, dim=0).view(-1),
            }
            replay.add_batch(batch)
            accepted_online_episodes += 1
            writer.add_scalar("collector/accepted_episode_length", float(ep_len), total_steps)
        else:
            dropped_short_online += 1
            writer.add_scalar("collector/dropped_short_episode_length", float(ep_len), total_steps)

        if len(replay) >= int(args.min_replay_size_before_learning):
            for q_idx in range(int(args.q_updates)):
                batch = replay.sample(int(args.batch_size))
                with torch.no_grad():
                    next_pol_obs = augment_policy_observation(
                        batch["next_observations"],
                        batch["actions"] * (1.0 - batch["bootstrap_terminals"].unsqueeze(-1)),
                        args.use_last_action_in_policy_state,
                    )
                    next_action = deterministic_actor_action(actor_target, next_pol_obs)
                    noise = torch.randn_like(next_action) * float(args.policy_noise)
                    noise = torch.clamp(noise, -float(args.noise_clip), float(args.noise_clip))
                    next_action = torch.clamp(next_action + noise, action_low, action_high)
                    q1_next = h_inverse(qf1_target(batch["next_observations"], next_action), eps=float(args.h_transform_eps)).view(-1)
                    q2_next = h_inverse(qf2_target(batch["next_observations"], next_action), eps=float(args.h_transform_eps)).view(-1)
                    min_next = torch.min(q1_next, q2_next)
                    bellman = batch["rewards"] + (1.0 - batch["bootstrap_terminals"]) * float(args.gamma) * min_next
                    target_h = h_transform(bellman, eps=float(args.h_transform_eps))
                q1 = qf1(batch["observations"], batch["actions"]).view(-1)
                q2 = qf2(batch["observations"], batch["actions"]).view(-1)
                q1_loss = torch.nn.functional.mse_loss(q1, target_h)
                q2_loss = torch.nn.functional.mse_loss(q2, target_h)
                q_loss = q1_loss + q2_loss
                q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                q_optimizer.step()
                total_updates += 1

                if (q_idx + 1) % int(args.target_network_frequency) == 0:
                    with torch.no_grad():
                        for source, target in ((qf1, qf1_target), (qf2, qf2_target), (actor, actor_target)):
                            for p, tp in zip(source.parameters(), target.parameters()):
                                tp.data.copy_(float(args.tau) * p.data + (1.0 - float(args.tau)) * tp.data)

            for _ in range(int(args.actor_updates_per_iteration)):
                b = replay.sample(int(args.batch_size))
                actor_obs = augment_policy_observation(b["observations"], b["prev_actions"], args.use_last_action_in_policy_state)
                pi_action = deterministic_actor_action(actor, actor_obs)
                q1_val = h_inverse(qf1(b["observations"], pi_action), eps=float(args.h_transform_eps)).view(-1)
                actor_loss = -q1_val.mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()
                total_actor_updates += 1

            writer.add_scalar("losses/q1_loss", float(q1_loss.item()), total_steps)
            writer.add_scalar("losses/q2_loss", float(q2_loss.item()), total_steps)
            writer.add_scalar("losses/actor_loss", float(actor_loss.item()), total_steps)

        now = time.time()
        if now - last_log >= float(args.collector_log_interval_sec):
            writer.add_scalar("replay/size", float(len(replay)), total_steps)
            writer.add_scalar("collector/dropped_short_online_episodes", float(dropped_short_online), total_steps)
            writer.add_scalar("collector/dropped_short_warmstart_episodes", float(dropped_short_hdf5), total_steps)
            writer.add_scalar("collector/accepted_online_episodes", float(accepted_online_episodes), total_steps)
            writer.add_scalar("learner/q_updates", float(total_updates), total_steps)
            writer.add_scalar("learner/actor_updates", float(total_actor_updates), total_steps)
            print(
                "[reset_td3] "
                f"steps={total_steps} replay={len(replay)} q_updates={total_updates} actor_updates={total_actor_updates} "
                f"accepted_eps={accepted_online_episodes} dropped_short_online={dropped_short_online}"
            )
            last_log = now

    ckpt_path = run_dir / "reset_policy_checkpoint.pth"
    torch.save(
        {
            "actor": actor.state_dict(),
            "actor_target": actor_target.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "args": vars(args),
            "steps": total_steps,
            "q_updates": total_updates,
            "actor_updates": total_actor_updates,
        },
        ckpt_path,
    )
    print(f"[reset_td3] saved checkpoint to {ckpt_path}")
    writer.close()


if __name__ == "__main__":
    main(_parse_args())
