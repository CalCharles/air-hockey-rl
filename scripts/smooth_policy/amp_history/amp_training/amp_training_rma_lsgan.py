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


def normalize_position_history_batch(position_history):
    """
    Normalize batched position history to relative coordinates (translation only).
    
    Process:
    1. Translate all positions so the first position is at (0, 0)
    2. Remove the first position (now [0, 0], contains no information)
    3. Return the remaining 4 relative positions (8 dimensions)
    
    Args:
        position_history: Tensor of shape [batch, 5, 2] containing 5 consecutive (x, y) positions
    
    Returns:
        torch.Tensor: Normalized positions of shape [batch, 8]
                     [pos2_x, pos2_y, pos3_x, pos3_y, pos4_x, pos4_y, pos5_x, pos5_y]
                     (all relative to pos1 which is at origin)
    """
    # Extract the first position
    pos1 = position_history[:, 0, :]  # [batch, 2]
    # Translate all positions so first is at origin
    translated = position_history - pos1.unsqueeze(1)  # [batch, 5, 2]
    # Remove first position (now [0, 0]) and flatten the remaining 4 positions
    normalized_state = translated[:, 1:, :].reshape(-1, 8)  # [batch, 8]
    return normalized_state


def normalize_action_history_batch(action_history):
    """
    Normalize batched transition action history to unit norm and flatten.
    Args:
        action_history: Tensor of shape [batch, 4, 2]
    Returns:
        torch.Tensor: Normalized flattened actions of shape [batch, 8]
    """
    action_norms = torch.norm(action_history, dim=-1, keepdim=True)
    normalized_actions = action_history / (action_norms + 1e-8)
    return normalized_actions.reshape(action_history.shape[0], 8)

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


def concat_env_latent_to_policy_obs(policy_obs_base, env_latent):
    """Concatenate encoded environment latent to base policy observation."""
    return torch.cat([policy_obs_base, env_latent], dim=-1)


def inject_latent_noise(env_latent, noise_std, enabled):
    return env_latent + torch.randn_like(env_latent) * noise_std


def get_env_spec_ranges():
    """Single source of truth for uniform randomization ranges."""
    return {
        "paddle_density": (2500 * 0.8, 2500 * 1.2),
        "paddle_damping": (3 * 0.8, 3 * 1.2),
        "puck_density": (250 * 0.8, 250 * 1.2),
        "puck_damping": (0.5 * 0.8, 0.5 * 1.2),
        "force_scaling": (1 * 0.8, 1 * 1.2),
    }


def build_env_spec_pool(num_randomized_envs_total, seed):
    """Create placeholder environment specs for domain randomization."""
    rng = np.random.default_rng(seed)
    ranges = get_env_spec_ranges()
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
            }
        )
    return pool


def save_env_spec_pool_artifacts(env_spec_pool, output_dir):
    """Persist sampled env spec pool so stage-2 adaptation can reuse exact matching specs."""
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, "env_spec_pool.yaml")
    pt_path = os.path.join(output_dir, "env_spec_pool.pt")
    with open(yaml_path, "w") as f:
        yaml.dump(env_spec_pool, f, sort_keys=False)
    torch.save(env_spec_pool, pt_path)
    print(f"✓ Saved env spec pool artifacts: {yaml_path}, {pt_path}")


def build_edge_eval_specs():
    """
    Build 5 fixed evaluation environments at range edges using the same
    ranges as training randomization.
    """
    ranges = get_env_spec_ranges()
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
        },
        {
            "env_id": 10003,
            "name": "low_paddle_high_puck_low_force",
            "paddle_density": lows["paddle_density"],
            "paddle_damping": lows["paddle_damping"],
            "puck_density": highs["puck_density"],
            "puck_damping": highs["puck_damping"],
            "force_scaling": lows["force_scaling"],
        },
        {
            "env_id": 10004,
            "name": "alternating_edges",
            "paddle_density": highs["paddle_density"],
            "paddle_damping": lows["paddle_damping"],
            "puck_density": highs["puck_density"],
            "puck_damping": lows["puck_damping"],
            "force_scaling": highs["force_scaling"],
        },
    ]
    return edge_specs


def extract_env_var_vector_from_spec(spec, env_var_dim):
    """
    Pack a fixed-size env-variable vector from one sampled environment spec.
    Variables are normalized to approximately mean 0 / std 1 using the
    uniform-randomization ranges: mean=(low+high)/2, std=(high-low)/sqrt(12).
    """
    ranges = get_env_spec_ranges()
    ordered_keys = [
        "paddle_density",
        "paddle_damping",
        "puck_density",
        "puck_damping",
        "force_scaling",
    ]

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

    def __init__(self, env, env_spec_pool, env_var_dim, rng_seed):
        super().__init__(env)
        self.env_spec_pool = env_spec_pool
        self.env_var_dim = env_var_dim
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
        
        self.current_env_var_vec = extract_env_var_vector_from_spec(env_spec, self.env_var_dim)
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
    amp_components_path: str = None  # Path to AMP components (normalizer, replay buffer)
    log_parent_dir: str = None
    run_name: str = "default"
    demo_data_path: str = "scripts/smooth_policy/amp_data/amp_dataset.pt"

    # Others
    seed: int = 0
    device: str = "cuda:0"
    use_last_action_in_policy_state: bool = False  # Append previous action to policy input state

    # action scale for the agent (mostly deprecated, should default to 1)
    action_scale: float = 1
    
    # agent hidden layer size (2 layers with this size)
    agent_hidden_size: int = 512

    # RMA randomization + encoder args
    num_randomized_envs_total: int = 500
    env_var_dim: int = 8
    env_latent_dim: int = 8
    env_encoder_hidden_size: int = 64
    latent_noise_std: float = 0.05
    edge_eval_episodes: int = 5
    edge_eval_interval: int = 10
    model_save_interval: int = 50
    
    # Action-conditioned discriminator
    use_action_discriminator: bool = False  # If True, discriminator uses position + 4 transition actions (16D)
    disc_debug_interval: int = 5000  # Print discriminator feature samples every N env steps (<=0 disables)
    
    

def make_env(env_id, env_spec_pool, env_var_dim, seed=0):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        
        env = AirHockeyEnv(config["air_hockey"])
        env = ResetSampledEnvWrapper(
            env=env,
            env_spec_pool=env_spec_pool,
            env_var_dim=env_var_dim,
            rng_seed=seed + env_id * 131,
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


def evaluate_on_edge_specs(agent, env_encoder, air_hockey_config, args, device):
    """
    Evaluate the trained policy on five edge-of-range environment specs.
    Runs args.edge_eval_episodes episodes per spec.
    """
    edge_specs = build_edge_eval_specs()
    eval_env = AirHockeyEnv(air_hockey_config)
    action_dim = int(np.prod(eval_env.action_space.shape))

    results = []
    agent.eval()
    env_encoder.eval()
    with torch.no_grad():
        for spec in edge_specs:
            apply_env_spec_to_unwrapped_env(eval_env, spec)
            env_var_np = extract_env_var_vector_from_spec(spec, args.env_var_dim)
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


def save_validation_gif(agent, env_encoder, env_spec, air_hockey_config, args, gif_savepath):
    """Save a validation GIF on a fixed environment spec."""
    eval_env = AirHockeyEnv(air_hockey_config.copy())
    apply_env_spec_to_unwrapped_env(eval_env, env_spec)
    renderer = AirHockeyRenderer(eval_env, show_target_position=True, show_acceleration_arrow=False)
    action_dim = int(np.prod(eval_env.action_space.shape))
    env_var_np = extract_env_var_vector_from_spec(env_spec, args.env_var_dim)
    env_var_tensor = torch.tensor(env_var_np, dtype=torch.float32, device=args.device).unsqueeze(0)

    frames = []
    agent.eval()
    env_encoder.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(1), desc="validation-gif", leave=False):
            obs, _ = eval_env.reset(seed=args.seed)
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
    if args.num_randomized_envs_total < args.num_envs:
        raise ValueError(
            f"num_randomized_envs_total ({args.num_randomized_envs_total}) must be >= num_envs ({args.num_envs})."
        )
    if args.env_var_dim <= 0 or args.env_latent_dim <= 0:
        raise ValueError("env_var_dim and env_latent_dim must be positive.")
    args.batch_size = args.num_envs * args.num_steps

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_spec_pool = build_env_spec_pool(args.num_randomized_envs_total, args.seed)
    validation_env_spec = build_edge_eval_specs()[0]  # fixed validation environment across checkpoints
    # Persistent vector env: each worker re-samples from the pool inside reset().
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                i,
                env_spec_pool=env_spec_pool,
                env_var_dim=args.env_var_dim,
                seed=args.seed,
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
    save_env_spec_pool_artifacts(env_spec_pool=env_spec_pool, output_dir=log_parent_dir)
    
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
    agent = Agent(policy_env_view, action_scale=action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size).to(args.device)
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
        eps=1e-6,
    )
    use_discriminator_reward = args.disc_reward_weight > 0.0

    # Initialize AMP components (always enabled)
    print("\n" + "="*80)
    print("Initializing AMP (Adversarial Motion Priors) components")
    print("="*80)

    history_len = 5  # Number of positions to track
    disc_obs_dim = 8
    use_action_disc = False
    discriminator, disc_optimizer, disc_normalizer, replay_buffer, demo_loader = None, None, None, None, None
    
    if use_discriminator_reward:
        # Load demonstration data first to determine observation dimension
        demo_loader = DemoLoaderPositionHistory(args.demo_data_path, device=args.device)
        print(f"✓ Demo loader initialized ({len(demo_loader):,} demonstrations)")
        
        # Check if demo data has actions and if we should use them
        use_action_disc = args.use_action_discriminator
        if use_action_disc and not demo_loader.has_actions:
            print("  ⚠ WARNING: --use_action_discriminator is True but demo data has no actions!")
            print("    Falling back to position-only discriminator.")
            use_action_disc = False
        
        # Get discriminator observation dimension (position-only is always 8D)
        disc_obs_dim = demo_loader.get_obs_dim() if use_action_disc else 8
        disc_hidden_dims = parse_discriminator_hidden_dims(args.disc_hidden_sizes)
        
        if use_action_disc:
            print("  Mode: POSITION + ACTION HISTORY (5 positions + 4 transition actions)")
            print(f"  Discriminator input dim: {disc_obs_dim} (8 position + 8 action)")
        else:
            print(f"  Mode: POSITION HISTORY (5 positions → 4 relative positions)")
            print(f"  Discriminator input dim: {disc_obs_dim}")
        
        # Initialize discriminator
        discriminator = Discriminator(
            disc_obs_dim,
            hidden_dims=disc_hidden_dims,
            activation='leaky_relu'
        ).to(args.device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_learning_rate, eps=1e-6)
        print(f"✓ Discriminator initialized (input_dim={disc_obs_dim}, hidden={disc_hidden_dims})")
        
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
        
        # Load pre-trained discriminator if path is provided
        if args.discriminator_path is not None:
            if not os.path.exists(args.discriminator_path):
                raise FileNotFoundError(f"Discriminator path {args.discriminator_path} does not exist.")
            print(f"Loading pre-trained discriminator from {args.discriminator_path}")
            discriminator.load_state_dict(torch.load(args.discriminator_path, map_location=args.device))
            print("✓ Discriminator loaded successfully")
        
        # Load AMP components (normalizer, replay buffer) if path is provided
        if args.amp_components_path is not None:
            if not os.path.exists(args.amp_components_path):
                raise FileNotFoundError(f"AMP components path {args.amp_components_path} does not exist.")
            print(f"Loading AMP components from {args.amp_components_path}")
            amp_components = torch.load(args.amp_components_path, map_location=args.device)
            disc_normalizer.load_state_dict(amp_components['normalizer'])
            replay_buffer.load_state_dict(amp_components['replay_buffer'])
            print(f"✓ AMP components loaded successfully (replay buffer size: {len(replay_buffer):,})")
    else:
        print("  Discriminator reward disabled (disc_reward_weight <= 0). Skipping discriminator setup.")
    
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
    disc_obs = torch.zeros((args.num_steps, args.num_envs, disc_obs_dim)).to(args.device) if use_discriminator_reward else None
    paddle_positions = torch.zeros((args.num_steps, args.num_envs, 2)).to(args.device)
    temporal_alignment_reward_raw = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    temporal_alignment_reward_scaled = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    action_magnitude_reward_raw = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    action_magnitude_reward_scaled = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    # Position history buffer: [num_envs, history_len, 2] for (x, y) positions
    position_history = torch.zeros((args.num_envs, history_len, 2)).to(args.device) if use_discriminator_reward else None
    # Action history buffer: [num_envs, 4, 2] for transition actions between 5 states
    action_history_len = history_len - 1
    action_history = torch.zeros((args.num_envs, action_history_len, 2)).to(args.device) if use_discriminator_reward else None
    # Track how many valid positions we have per environment (need history_len before valid)
    position_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device) if use_discriminator_reward else None
    action_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device) if use_discriminator_reward else None
    valid_transition = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool).to(args.device) if use_discriminator_reward else None

    # Tracking lists for motion metrics
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
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

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # REWARD SCALING is done on the environment level, not here
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)
            current_env_vars = parse_env_vars_from_infos(
                infos=infos,
                num_envs=args.num_envs,
                env_var_dim=args.env_var_dim,
                device=args.device,
                fallback_env_vars=current_env_vars,
            )
            last_action_for_policy = action.detach().clone()
            last_action_for_policy[next_done.bool()] = 0
            
            # AMP: Construct discriminator observations from position history
            current_paddle_pos = next_obs[:, 12:14]  # [batch, 2] - just x, y position
            paddle_positions[step] = current_paddle_pos
            
            # Optional action magnitude reward (raw value is independent of scale)
            action_magnitude = actions[step].abs().sum(dim=-1)
            action_mag_raw = (torch.maximum(-action_magnitude, torch.full_like(action_magnitude, -0.25)) + 0.125) * 8.0
            action_magnitude_reward_raw[step] = action_mag_raw
            action_magnitude_reward_scaled[step] = action_mag_raw * args.action_magnitude_reward_scale
            
            if use_discriminator_reward:
                # Update position history buffer (rolling buffer: shift left, add new position at end)
                position_history = torch.roll(position_history, shifts=-1, dims=1)
                position_history[:, -1, :] = current_paddle_pos

                # Update action history buffer with current transition action (aligned with next_obs)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                action_history[:, -1, :] = action
                
                # Increment position count (capped at history_len)
                position_count = torch.clamp(position_count + 1, max=history_len)
                action_count = torch.clamp(action_count + 1, max=action_history_len)
                
                # Valid transition only if we have history_len positions AND not just reset
                has_enough_history = position_count >= history_len
                has_enough_actions = action_count >= action_history_len
                valid_transition[step] = has_enough_history
                if use_action_disc:
                    valid_transition[step] = valid_transition[step] & has_enough_actions
                
                # Invalidate for environments that just reset
                valid_transition[step, just_reset] = False
                
                # Normalize the position history to get relative positions
                # Result: [batch, 8] = 4 relative positions × 2 coords
                normalized_positions = normalize_position_history_batch(position_history)
                
                # Store the discriminator observations
                if use_action_disc:
                    # Concatenate positions with normalized transition action history: [batch, 8] + [batch, 8] -> [batch, 16]
                    normalized_actions = normalize_action_history_batch(action_history)
                    disc_obs[step] = torch.cat([normalized_positions, normalized_actions], dim=-1)
                else:
                    # Position-only mode: [batch, 8]
                    disc_obs[step] = normalized_positions

                # Occasional debug logging for discriminator feature formatting.
                if args.disc_debug_interval > 0 and global_step % args.disc_debug_interval == 0:
                    sample_idx = 0
                    sample_pos = normalized_positions[sample_idx].detach().cpu().numpy()
                    if use_action_disc:
                        sample_action = normalize_action_history_batch(action_history)[sample_idx].detach().cpu().numpy()
                        sample_disc = disc_obs[step, sample_idx].detach().cpu().numpy()
                        print(
                            f"[disc-debug][step={global_step}] pos_shape={normalized_positions.shape}, "
                            f"action_shape={action_history.shape}, disc_shape={disc_obs[step].shape}, "
                            f"valid={bool(valid_transition[step, sample_idx].item())}"
                        )
                        print(
                            "  sample pos[0:4]="
                            f"{np.array2string(sample_pos[:4], precision=4, suppress_small=True)} "
                            "action[0:4]="
                            f"{np.array2string(sample_action[:4], precision=4, suppress_small=True)} "
                            "disc[0:8]="
                            f"{np.array2string(sample_disc[:8], precision=4, suppress_small=True)}"
                        )
                    else:
                        print(
                            f"[disc-debug][step={global_step}] pos_shape={normalized_positions.shape}, "
                            f"disc_shape={disc_obs[step].shape}, valid={bool(valid_transition[step, sample_idx].item())}"
                        )
                        print(
                            "  sample pos[0:4]="
                            f"{np.array2string(sample_pos[:4], precision=4, suppress_small=True)}"
                        )
            
            # Reset position history and count for environments that are done
            if use_discriminator_reward and next_done.any():
                done_mask = next_done.bool()
                position_history[done_mask] = 0
                position_history[done_mask, -1, :] = current_paddle_pos[done_mask]
                position_count[done_mask] = 1  # We have 1 position after reset
                action_history[done_mask] = 0
                action_count[done_mask] = 0
            
            # Track which environments just reset for next step
            just_reset = next_done.bool()

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        episodic_returns.append(info["episode_return"])
                        success_rates.append(1.0 if info["success"] else 0.0)
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
            for t in range(horizon, args.num_steps):
                realized_movement = paddle_positions[t] - paddle_positions[t - horizon]
                target_direction = actions[t - horizon]

                movement_norm = torch.norm(realized_movement, dim=-1).clamp_min(eps)
                target_norm = torch.norm(target_direction, dim=-1).clamp_min(eps)
                cosine_sim = (realized_movement * target_direction).sum(dim=-1) / (movement_norm * target_norm)

                # Apply fallback reward per environment when target direction is near zero.
                small_target_mask = torch.norm(target_direction, dim=-1) < 0.03  # hard-coded threshold for now
                cosine_sim = torch.where(
                    small_target_mask,
                    torch.full_like(cosine_sim, 0.75),  # hard-coded reward
                    cosine_sim,
                )
                
                # Invalidate if episode reset happened between command and realized movement.
                temporal_valid = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
                for k in range(t - horizon + 1, t + 1):
                    temporal_valid = temporal_valid & (~dones[k].bool())
                
                temporal_alignment_reward_raw[t] = cosine_sim * temporal_valid.float()
            temporal_alignment_reward_scaled = temporal_alignment_reward_raw * args.temporal_alignment_reward_scale
            
            if use_discriminator_reward:
                # AMP: Compute discriminator rewards and combine with task rewards
                # Flatten discriminator observations
                b_disc_obs = disc_obs.reshape(-1, disc_obs_dim)
                
                # Normalize discriminator observations
                norm_disc_obs = disc_normalizer.normalize(b_disc_obs)
                
                # Compute discriminator rewards (LSGAN)
                disc_scores = discriminator(norm_disc_obs).squeeze(-1)
                # For LSGAN, discriminator outputs raw scores (not logits)
                # Reward formula: quadratic function peaked at disc_scores=1, clamped to [0, 1]
                disc_r_raw = torch.clamp(1 - 0.25 * (disc_scores - 1) ** 2, min=0)
                
                # Mask invalid transitions (set reward to 0)
                b_valid = valid_transition.reshape(-1)
                disc_r_raw = disc_r_raw * b_valid.float()  # Zero out invalid transitions
                
                # Reshape discriminator rewards to match original shape
                disc_r_raw_shaped = disc_r_raw.reshape(args.num_steps, args.num_envs)
                disc_r_scaled = args.disc_reward_weight * disc_r_raw_shaped
            else:
                b_disc_obs = None
                disc_r_raw = torch.zeros(args.num_steps * args.num_envs, device=args.device)
                disc_r_scaled = torch.zeros((args.num_steps, args.num_envs), device=args.device)
            task_r_raw = rewards
            task_r_scaled = args.task_reward_weight * task_r_raw
            
            # Combine scaled reward streams only when building PPO targets.
            combined_rewards = (
                task_r_scaled
                + disc_r_scaled
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
            if use_discriminator_reward:
                writer.add_scalar("amp/disc_reward_raw_mean", disc_r_raw.mean().item(), global_step)
                writer.add_scalar("amp/disc_reward_raw_std", disc_r_raw.std().item(), global_step)
                writer.add_scalar("amp/disc_reward_scaled_mean", disc_r_scaled.mean().item(), global_step)
                writer.add_scalar("amp/disc_reward_scaled_std", disc_r_scaled.std().item(), global_step)
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
            if use_discriminator_reward:
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
        
        # AMP: Train discriminator
        if use_discriminator_reward:
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
            
                # Weight decay
                if args.disc_weight_decay > 0:
                    disc_weights = discriminator.get_all_weights()
                    disc_weight_decay_loss = torch.sum(disc_weights ** 2)
                    disc_loss = disc_loss + args.disc_weight_decay * disc_weight_decay_loss
            
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
            
            # Log discriminator metrics
            writer.add_scalar("amp/disc_loss", disc_loss.item(), global_step)
            writer.add_scalar("amp/disc_loss_demo", disc_loss_demo.item(), global_step)
            writer.add_scalar("amp/disc_loss_agent", disc_loss_agent.item(), global_step)
            writer.add_scalar("amp/disc_grad_penalty", disc_grad_penalty.item(), global_step)
            writer.add_scalar("amp/disc_agent_acc", disc_agent_acc, global_step)
            writer.add_scalar("amp/disc_demo_acc", disc_demo_acc, global_step)
            writer.add_scalar("amp/disc_agent_logit_mean", disc_agent_logit.mean().item(), global_step)
            writer.add_scalar("amp/disc_demo_logit_mean", disc_demo_logit.mean().item(), global_step)
            writer.add_scalar("amp/replay_buffer_size", len(replay_buffer), global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Calculate and log episodic return statistics for this iteration
        if episodic_returns:
            avg_return = np.mean(episodic_returns)
            min_return = np.min(episodic_returns)
            max_return = np.max(episodic_returns)

            print(f"Iteration {iteration}: Avg Return: {avg_return:.2f}, Min Return: {min_return:.2f}, Max Return: {max_return:.2f}")
            print(f"Iteration {iteration}: Avg Success Rate: {np.mean(success_rates):.2f}, Max Success Rate: {np.max(success_rates):.2f}")
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
            )
            writer.add_scalar(
                "edge_eval/overall_avg_return",
                float(np.mean([entry["avg_return"] for entry in edge_eval_results])),
                iteration,
            )
            for entry in edge_eval_results:
                spec_name = entry["name"]
                writer.add_scalar(f"edge_eval/{spec_name}/avg_return", entry["avg_return"], iteration)
                writer.add_scalar(f"edge_eval/{spec_name}/min_return", entry["min_return"], iteration)
                writer.add_scalar(f"edge_eval/{spec_name}/max_return", entry["max_return"], iteration)
                writer.add_scalar(f"edge_eval/{spec_name}/avg_length", entry["avg_length"], iteration)
            edge_eval_path_iter = os.path.join(log_parent_dir, f"edge_eval_results_iter_{iteration}.yaml")
            with open(edge_eval_path_iter, "w") as f:
                yaml.dump(edge_eval_results, f, sort_keys=False)

        if iteration % args.model_save_interval == 0:
            # save a checkpoint of the model
            # create a subfolder for the checkpoint
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(agent.state_dict(), model_path)
            torch.save(env_encoder.state_dict(), f"{checkpoint_dir}/encoder.pth")
            save_env_spec_pool_artifacts(env_spec_pool=env_spec_pool, output_dir=checkpoint_dir)
            
            # Save AMP components in checkpoint when discriminator reward is active
            if use_discriminator_reward:
                torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
                torch.save({
                    'normalizer': disc_normalizer.state_dict(),
                    'replay_buffer': replay_buffer.state_dict()
                }, f"{checkpoint_dir}/amp_components.pth")

            validation_gif_path = os.path.join(checkpoint_dir, "validation_fixed_env.gif")
            save_validation_gif(
                agent=agent,
                env_encoder=env_encoder,
                env_spec=validation_env_spec,
                air_hockey_config=config["air_hockey"],
                args=args,
                gif_savepath=validation_gif_path,
            )

            print(f"Iteration {iteration} complete")

    # save model
    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")
    torch.save(env_encoder.state_dict(), f"{log_parent_dir}/encoder.pth")
    
    # Save AMP components when discriminator reward is active
    if use_discriminator_reward:
        torch.save(discriminator.state_dict(), f"{log_parent_dir}/discriminator.pth")
        torch.save({
            'normalizer': disc_normalizer.state_dict(),
            'replay_buffer': replay_buffer.state_dict()
        }, f"{log_parent_dir}/amp_components.pth")
        print(f"✓ Saved discriminator and AMP components")

    # Evaluate on 5 edge configurations derived from the training randomization ranges.
    edge_eval_results = evaluate_on_edge_specs(
        agent=agent,
        env_encoder=env_encoder,
        air_hockey_config=config["air_hockey"],
        args=args,
        device=args.device,
    )
    edge_eval_path = os.path.join(log_parent_dir, "edge_eval_results.yaml")
    with open(edge_eval_path, "w") as f:
        yaml.dump(edge_eval_results, f, sort_keys=False)
    print(f"✓ Saved edge evaluation results to {edge_eval_path}")

    writer.close()
    envs.close()
    
    # end of training