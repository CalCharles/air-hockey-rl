"""
SAC training with AMP (Adversarial Motion Priors), least-squares discriminator.

This script implements SAC with AMP for learning smooth motion patterns from
expert demonstrations while maintaining task performance.

Uses the Transformed Bellman Operator for Q-function targets, which applies a
non-linear rescaling function h(x) to handle varying reward scales and improve
learning stability. The transformation is:
    h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
    h_inv(x) = sign(x) * ((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1

The Bellman target becomes: Q_target = h(r + γ * h_inv(Q_target_value))
"""

import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

from airhockey import AirHockeyEnv
from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.smooth_policy.agent import Agent
from scripts.utils import save_tensorboard_plots

# SAC replay buffer
from scripts.smooth_policy.amp_history.amp_training.sac.sac_replay_buffer import SACReplayBuffer

# AMP components
from scripts.smooth_policy.amp_history.amp_training.discriminator import Discriminator
from scripts.smooth_policy.amp_history.amp_training.replay_buffer import ReplayBuffer as AMPReplayBuffer
from scripts.smooth_policy.amp_history.amp_training.normalizer import Normalizer
from scripts.smooth_policy.amp_history.amp_training.demo_loader_position_history import DemoLoaderPositionHistory


# ================================================
# Transformed Bellman Operator Functions
# ================================================
def h_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Apply the h transformation to rescale values.
    h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
    
    This squashes large values while preserving signs, making learning
    more stable across varying reward scales.
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def h_inverse(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Apply the inverse h transformation to recover original scale.
    h_inv(x) = sign(x) * ((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1
    
    This is the exact inverse of h_transform.
    """
    # For numerical stability, handle the inverse carefully
    abs_x = torch.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    sqrt_inner = torch.sqrt(inner)
    quotient = (sqrt_inner - 1) / (2 * eps)
    return torch.sign(x) * (quotient ** 2 - 1)


class ReferenceStateWrapper(gym.Wrapper):
    """Wrapper that initializes paddle from demonstration states."""
    
    def __init__(self, env, reference_states):
        super().__init__(env)
        self.reference_states = reference_states  # numpy array [N, 4]
    
    def reset(self, **kwargs):
        # Sample random reference state [x, y, vx, vy]
        idx = np.random.randint(0, len(self.reference_states))
        ref_state = self.reference_states[idx]

        # Set reference state on underlying environment
        self.env.unwrapped._ref_paddle_state = (
            (float(ref_state[0]), float(ref_state[1])),  # pos
            (float(ref_state[2]), float(ref_state[3]))   # vel
        )
        
        return self.env.reset(**kwargs)


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


@dataclass
class Args:
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    
    # SAC specific arguments
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    q_weight_decay: float = 1e-4
    """L2 regularization (weight decay) for Q networks"""
    q_hidden_size: int = 128
    """hidden layer size for Q networks (2-layer MLP)"""
    q_frequency: int = 1
    """Q-network update frequency (update every q_frequency iterations)"""
    q_updates: int = 1
    """number of Q-network gradient updates when updating"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
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
    discriminator_path: str = None  # Path to pre-trained discriminator state dict
    amp_components_path: str = None  # Path to AMP components (normalizer, replay buffer)
    log_parent_dir: str = None
    run_name: str = "default"
    demo_data_path: str = "scripts/smooth_policy/amp_data/amp_dataset.pt"
    
    # Others
    device: str = "cuda:0"
    
    # action scale for the agent
    action_scale: float = 0.02
    seed: int = 0
    
    # agent hidden layer size (2 layers with this size)
    agent_hidden_size: int = 64
    
    # Reference state initialization
    use_reference_state_init: bool = False  # Enable/disable feature
    reference_data_path: str = None  # Path to raw demo data (defaults to demo_data_path)
    ref_max_episode_steps: int = 20  # Episode length when using reference init
    
    # Transformed Bellman operator
    h_transform_eps: float = 1e-3  # Epsilon for h transformation (ensures invertibility)
    
    # Action-conditioned discriminator
    use_action_discriminator: bool = False  # If True, discriminator uses position + 4 transition actions (16D)
    disc_debug_interval: int = 5000  # Print discriminator feature samples every N env steps (<=0 disables)
    use_last_action_in_policy_state: bool = False  # Append previous action to policy input state


def make_env(env_id, reference_states=None, max_episode_steps=20):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        
        # Override max_timesteps if using reference state initialization
        if reference_states is not None:
            config["air_hockey"]["max_timesteps"] = max_episode_steps
        
        env = AirHockeyEnv(config["air_hockey"])
        
        # Wrap with reference state initialization if provided
        if reference_states is not None:
            env = ReferenceStateWrapper(env, reference_states)
        
        return env
    return _thunk


if __name__ == "__main__":

    temp_args = tyro.cli(Args)  # checks for a passed in args file
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()  # Use class defaults

    # command line args override file args
    args = tyro.cli(Args, default=default_args)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load reference states if enabled
    reference_states = None
    if args.use_reference_state_init:
        ref_data_path = args.reference_data_path or args.demo_data_path
        print(f"\n{'='*80}")
        print(f"Loading reference states from: {ref_data_path}")
        raw_data = torch.load(ref_data_path, map_location='cpu')
        # Extract first position of each sequence: [N, 2] = [x, y]
        # Note: We only have position data, not velocity, so we'll set velocity to 0
        position_sequences = raw_data['position_sequences']  # Shape: [N, 5, 2]
        first_positions = position_sequences[:, 0, :].numpy()  # [N, 2]
        # get velocities with finite difference
        velocities = np.diff(position_sequences, axis=1)[:, :, 0] / 0.05 # 20hz
        # Create reference states with velocities: [N, 4] = [x, y, vx, vy]
        reference_states = np.concatenate([
            first_positions,
            velocities  # Zero velocity
        ], axis=1)
        print(f"✓ Loaded {len(reference_states):,} reference states (position-based)")
        print(f"  Note: Velocities initialized to zero")
        print(f"  Episode length will be {args.ref_max_episode_steps} timesteps")
        print(f"{'='*80}\n")

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

    # env setup
    envs = gym.vector.AsyncVectorEnv([
        make_env(i, reference_states, args.ref_max_episode_steps) 
        for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    print("envs.single_observation_space.shape:", envs.single_observation_space.shape)

    if 'use_pid' in config["air_hockey"] and config["air_hockey"]["use_pid"]:
        action_scale = 1
    else:
        action_scale = args.action_scale # use whatever action scale specified

    # SAC networks: Use Agent for actor (PPO-style), add Q-networks for SAC
    raw_obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    policy_obs_dim = raw_obs_dim + act_dim if args.use_last_action_in_policy_state else raw_obs_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=envs.single_action_space,
    )
    actor = Agent(policy_env_view, action_scale=action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size).to(args.device)
    
    # Q-networks for SAC (simple MLPs)
    class SoftQNetwork(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = self.fc3(x)
            return x
    
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    
    qf1 = SoftQNetwork(obs_dim, act_dim, args.q_hidden_size).to(args.device)
    qf2 = SoftQNetwork(obs_dim, act_dim, args.q_hidden_size).to(args.device)
    qf1_target = SoftQNetwork(obs_dim, act_dim, args.q_hidden_size).to(args.device)
    qf2_target = SoftQNetwork(obs_dim, act_dim, args.q_hidden_size).to(args.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Load pre-trained model if path is provided
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        actor.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully")
    
    # q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, weight_decay=args.q_weight_decay)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        # Calculate target entropy based on desired policy standard deviation
        # 
        # For a multivariate Gaussian with diagonal covariance and equal std σ per dimension:
        #   H = (d/2) * ln(2πe * σ²)
        # 
        # We want the policy std to be 1/4 of the action scale (precise, small actions):
        #   σ = action_scale / 4
        # 
        # For d=2 action dimensions:
        #   H = (2/2) * ln(2πe * σ²) = ln(2πe * σ²)
        #   H = ln(2πe * (action_scale/4)²)
        #   H = ln(2πe * action_scale² / 16)
        #
        action_dim = 2  # Assume 2D actions (x, y movement)
        desired_std = action_scale / 4.0  # Target std is 1/4 of action scale
        
        # Entropy of multivariate Gaussian: H = (d/2) * ln(2πe * σ²)
        target_entropy = (action_dim / 2.0) * np.log(2 * np.pi * np.e * (desired_std ** 2))
        
        print(f"\n{'='*60}")
        print(f"Entropy Autotune Configuration:")
        print(f"  Action dimension: {action_dim}")
        print(f"  Action scale: {action_scale}")
        print(f"  Desired policy std: {desired_std:.6f} (= action_scale / 4)")
        print(f"  Target entropy: {target_entropy:.4f}")
        print(f"  (Original heuristic would be: {-action_dim})")
        print(f"{'='*60}\n")
        
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # Initialize AMP components (only if disc_reward_weight > 0)
    use_amp = args.disc_reward_weight > 0
    disc_hidden_dims = parse_discriminator_hidden_dims(args.disc_hidden_sizes)
    disc_obs_dim = 8  # Position-only discriminator input dim
    history_len = 5  # Number of positions to track
    use_action_disc = False
    
    if use_amp:
        print("\n" + "="*80)
        print("Initializing AMP (Adversarial Motion Priors) components")
        print("="*80)
        
        # Load demonstration data first so action-conditioned mode can infer observation dim.
        demo_loader = DemoLoaderPositionHistory(args.demo_data_path, device=args.device)
        print(f"✓ Demo loader initialized ({len(demo_loader):,} demonstrations)")

        use_action_disc = args.use_action_discriminator
        if use_action_disc and not demo_loader.has_actions:
            print("  ⚠ WARNING: --use_action_discriminator is True but demo data has no actions!")
            print("    Falling back to position-only discriminator.")
            use_action_disc = False

        disc_obs_dim = demo_loader.get_obs_dim() if use_action_disc else 8
        if use_action_disc:
            print("  Mode: POSITION + ACTION HISTORY (5 positions + 4 transition actions)")
            print(f"  Discriminator input dim: {disc_obs_dim} (8 position + 8 action)")
        else:
            print("  Mode: POSITION HISTORY (5 positions → 4 relative positions)")
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
        amp_replay_buffer = AMPReplayBuffer(
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
            amp_replay_buffer.load_state_dict(amp_components['replay_buffer'])
            print(f"✓ AMP components loaded successfully (replay buffer size: {len(amp_replay_buffer):,})")
        
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("AMP disabled (disc_reward_weight = 0)")
        print("="*80 + "\n")

    # SAC Replay Buffer
    envs.single_observation_space.dtype = np.float32
    rb = SACReplayBuffer(
        buffer_size=args.buffer_size,
        obs_shape=envs.single_observation_space.shape,
        action_shape=envs.single_action_space.shape,
        device=args.device,
        n_envs=args.num_envs,
        disc_obs_dim=disc_obs_dim if use_amp else None  # Only store disc obs if AMP is enabled
    )
    print(f"✓ SAC replay buffer initialized (capacity={args.buffer_size:,})\n")
    
    # AMP: Storage for position history tracking (only if AMP enabled)
    if use_amp:
        # Position history buffer: [num_envs, history_len, 2] for (x, y) positions
        position_history = torch.zeros((args.num_envs, history_len, 2)).to(args.device)
        action_history_len = history_len - 1
        action_history = torch.zeros((args.num_envs, action_history_len, 2)).to(args.device)
        # Track how many valid positions we have per environment (need history_len before valid)
        position_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device)
        action_count = torch.zeros(args.num_envs, dtype=torch.long).to(args.device)
    
    # Tracking lists for motion metrics
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []
    
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    last_action_for_policy = torch.zeros((args.num_envs, act_dim), dtype=torch.float32, device=args.device)
    
    # Track episodic returns for logging
    episodic_returns = []
    success_rates = []
    just_reset = torch.zeros(args.num_envs, dtype=torch.bool).to(args.device)
    
    # Auxiliary reward histories (temporal alignment / action magnitude)
    use_auxiliary_rewards = (
        args.temporal_alignment_reward_scale > 0.0
        or args.action_magnitude_reward_scale > 0.0
    )
    if use_auxiliary_rewards and args.temporal_alignment_horizon > 0:
        temporal_horizon = args.temporal_alignment_horizon
        temporal_position_history = torch.zeros((args.num_envs, temporal_horizon + 1, 2), device=args.device)
        temporal_action_history = torch.zeros((args.num_envs, temporal_horizon, 2), device=args.device)
        temporal_done_history = torch.zeros((args.num_envs, temporal_horizon), dtype=torch.bool, device=args.device)
        temporal_position_count = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        temporal_action_count = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        
        # Seed history with the initial post-reset paddle position.
        initial_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
        temporal_position_history[:, -1, :] = initial_obs_tensor[:, 12:14]
        temporal_position_count[:] = 1
    
    # Track last loss values for logging (to avoid undefined variable errors)
    actor_loss_val = 0.0
    alpha_loss_val = 0.0
    bellman_target_original_mean = 0.0
    next_q_value_mean = 0.0
    next_state_log_pi_mean = 0.0
    min_qf_next_target_mean = 0.0
    entropy_term_mean = 0.0
    qf1_loss_val = 0.0
    qf2_loss_val = 0.0
    qf_loss_val = 0.0
    qf1_a_values_mean = 0.0
    qf2_a_values_mean = 0.0
    sampled_task_rewards_mean = 0.0
    sampled_combined_rewards_mean = 0.0
    temporal_alignment_reward_raw_mean = 0.0
    temporal_alignment_reward_scaled_mean = 0.0
    action_magnitude_reward_raw_mean = 0.0
    action_magnitude_reward_scaled_mean = 0.0
    
    # AMP logging variables (only used if AMP enabled)
    disc_loss_val = None
    disc_agent_acc = None
    disc_demo_acc = None
    sampled_disc_r_mean = 0.0
    
    # Use while loop so global_step increments by num_envs (total samples collected)
    # iteration counts loop iterations (increments by 1 each time) for frequency checks
    global_step = 0
    iteration = 0
    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        prev_action_for_transition = last_action_for_policy.clone()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=args.device)
        policy_obs_tensor = augment_policy_observation(
            obs_tensor, last_action_for_policy, args.use_last_action_in_policy_state
        )
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _, _ = actor.get_action_and_value(policy_obs_tensor)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Track dones for all cases
        dones = np.logical_or(terminations, truncations)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=args.device)
        current_paddle_pos = next_obs_tensor[:, 12:14]  # [batch, 2]
        action_tensor = torch.tensor(actions, dtype=torch.float32, device=args.device)
        done_tensor = torch.tensor(dones, dtype=torch.bool, device=args.device)
        last_action_for_policy = action_tensor.clone()
        last_action_for_policy[done_tensor] = 0
        
        # Optional action magnitude reward (raw value is independent of scale)
        action_magnitude = action_tensor.abs().sum(dim=-1)
        action_magnitude_reward_raw = (
            torch.maximum(-action_magnitude, torch.full_like(action_magnitude, -0.25)) + 0.125
        ) * 8.0
        action_magnitude_reward_scaled = action_magnitude_reward_raw * args.action_magnitude_reward_scale
        
        # Optional temporal alignment reward:
        # compare realized movement over horizon vs commanded action direction from horizon steps ago.
        temporal_alignment_reward_raw = torch.zeros(args.num_envs, device=args.device)
        temporal_alignment_reward_scaled = torch.zeros(args.num_envs, device=args.device)
        if use_auxiliary_rewards and args.temporal_alignment_horizon > 0:
            temporal_position_history = torch.roll(temporal_position_history, shifts=-1, dims=1)
            temporal_position_history[:, -1, :] = current_paddle_pos
            temporal_action_history = torch.roll(temporal_action_history, shifts=-1, dims=1)
            temporal_action_history[:, -1, :] = action_tensor
            temporal_done_history = torch.roll(temporal_done_history, shifts=-1, dims=1)
            temporal_done_history[:, -1] = done_tensor
            
            temporal_position_count = torch.clamp(temporal_position_count + 1, max=args.temporal_alignment_horizon + 1)
            temporal_action_count = torch.clamp(temporal_action_count + 1, max=args.temporal_alignment_horizon)
            
            realized_movement = temporal_position_history[:, -1, :] - temporal_position_history[:, 0, :]
            target_direction = temporal_action_history[:, 0, :]
            eps = 1e-8
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
            
            temporal_valid = (
                (temporal_position_count >= args.temporal_alignment_horizon + 1)
                & (temporal_action_count >= args.temporal_alignment_horizon)
                & (~temporal_done_history.any(dim=1))
            )
            temporal_alignment_reward_raw = cosine_sim * temporal_valid.float()
            temporal_alignment_reward_scaled = temporal_alignment_reward_raw * args.temporal_alignment_reward_scale
        
        # Merge auxiliary rewards into the environment task reward during collection.
        rewards = (
            rewards
            + temporal_alignment_reward_scaled.detach().cpu().numpy()
            + action_magnitude_reward_scaled.detach().cpu().numpy()
        )
        
        # Cache auxiliary reward means for scalar logging.
        temporal_alignment_reward_raw_mean = temporal_alignment_reward_raw.mean().item()
        temporal_alignment_reward_scaled_mean = temporal_alignment_reward_scaled.mean().item()
        action_magnitude_reward_raw_mean = action_magnitude_reward_raw.mean().item()
        action_magnitude_reward_scaled_mean = action_magnitude_reward_scaled.mean().item()
    
        # ================================================
        # AMP: Track position history (only if AMP enabled)
        # ================================================
        if use_amp:
            # Update position history buffer (rolling buffer: shift left, add new position at end)
            position_history = torch.roll(position_history, shifts=-1, dims=1)
            position_history[:, -1, :] = current_paddle_pos
            action_history = torch.roll(action_history, shifts=-1, dims=1)
            action_history[:, -1, :] = action_tensor
            
            # Increment position count (capped at history_len)
            position_count = torch.clamp(position_count + 1, max=history_len)
            action_count = torch.clamp(action_count + 1, max=action_history_len)
            
            # Valid transition only if we have history_len positions AND not just reset
            has_enough_history = position_count >= history_len
            has_enough_actions = action_count >= action_history_len
            valid_transition_mask = has_enough_history & (~just_reset)
            if use_action_disc:
                valid_transition_mask = valid_transition_mask & has_enough_actions
            
            # Normalize the position history to get relative positions
            # Result: [batch, 8] = 4 relative positions × 2 coords
            normalized_positions = normalize_position_history_batch(position_history)
            if use_action_disc:
                normalized_actions = normalize_action_history_batch(action_history)
                current_disc_obs = torch.cat([normalized_positions, normalized_actions], dim=-1)
            else:
                current_disc_obs = normalized_positions
            
            # Store all valid discriminator observations in AMP replay buffer
            if valid_transition_mask.any():
                valid_disc_obs = current_disc_obs[valid_transition_mask]
                amp_replay_buffer.push(valid_disc_obs)

            # Occasional debug logging for discriminator feature formatting.
            if args.disc_debug_interval > 0 and global_step % args.disc_debug_interval == 0:
                sample_idx = 0
                sample_pos = normalized_positions[sample_idx].detach().cpu().numpy()
                if use_action_disc:
                    sample_action = normalize_action_history_batch(action_history)[sample_idx].detach().cpu().numpy()
                    sample_disc = current_disc_obs[sample_idx].detach().cpu().numpy()
                    print(
                        f"[disc-debug][step={global_step}] pos_shape={normalized_positions.shape}, "
                        f"action_shape={action_history.shape}, disc_shape={current_disc_obs.shape}, "
                        f"valid={bool(valid_transition_mask[sample_idx].item())}"
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
                        f"disc_shape={current_disc_obs.shape}, valid={bool(valid_transition_mask[sample_idx].item())}"
                    )
                    print(
                        "  sample pos[0:4]="
                        f"{np.array2string(sample_pos[:4], precision=4, suppress_small=True)}"
                    )
            
            # Reset position history and count for environments that are done
            if dones.any():
                done_mask = done_tensor
                position_history[done_mask] = 0
                position_history[done_mask, -1, :] = current_paddle_pos[done_mask]
                position_count[done_mask] = 1  # We have 1 position after reset
                action_history[done_mask] = 0
                action_count[done_mask] = 0
            
            # Track which environments just reset for next step
            just_reset = done_tensor
        
        # Reset auxiliary histories for environments that are done.
        if use_auxiliary_rewards and args.temporal_alignment_horizon > 0 and dones.any():
            temporal_position_history[done_tensor] = 0
            temporal_position_history[done_tensor, -1, :] = current_paddle_pos[done_tensor]
            temporal_action_history[done_tensor] = 0
            temporal_done_history[done_tensor] = False
            temporal_position_count[done_tensor] = 1
            temporal_action_count[done_tensor] = 0

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode_return" in info:
                    episodic_returns.append(info['episode_return'])
                    success_rates.append(1.0 if info.get('success', False) else 0.0)
                    # print(f"global_step={global_step}, episodic_return={info['episode_return']}")
                    writer.add_scalar("charts/episodic_return", info['episode_return'], global_step)
                    writer.add_scalar("charts/episodic_length", info['episode_length'], global_step)
                    
                    # Extract motion data if available
                    if 'motion_data' in info:
                        velocity_magnitudes.extend(info['motion_data']['velocity_mags'])
                        acceleration_magnitudes.extend(info['motion_data']['acceleration_mags'])
                        jerk_magnitudes.extend(info['motion_data']['jerk_mags'])

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Store data in RL buffer
        # If AMP enabled, also store discriminator observations and validity mask for later reward computation
        if use_amp:
            rb.add(
                obs, 
                real_next_obs, 
                actions, 
                rewards,  # Environment reward + auxiliary rewards (disc reward recomputed on sampling)
                terminations,
                prev_action=prev_action_for_transition.cpu().numpy(),
                disc_obs=current_disc_obs.cpu().numpy(),
                disc_valid=valid_transition_mask.cpu().numpy()
            )
        else:
            rb.add(
                obs, 
                real_next_obs, 
                actions, 
                rewards,
                terminations,
                prev_action=prev_action_for_transition.cpu().numpy(),
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # ================================================
            # Q-network updates (controlled by q_frequency)
            # ================================================
            if iteration % args.q_frequency == 0:
                # Perform q_updates gradient updates
                for _ in range(args.q_updates):
                    data = rb.sample(args.batch_size)
                    sampled_observations = data['observations']
                    sampled_next_observations = data['next_observations']
                    sampled_actions = data['actions']
                    sampled_task_rewards = data['rewards']  # Environment reward + auxiliary rewards
                    sampled_dones = data['dones']
                    sampled_prev_actions = data['prev_actions']
                    sampled_next_prev_actions = sampled_actions * (1.0 - sampled_dones.unsqueeze(-1))
                    sampled_policy_observations = augment_policy_observation(
                        sampled_observations, sampled_prev_actions, args.use_last_action_in_policy_state
                    )
                    sampled_next_policy_observations = augment_policy_observation(
                        sampled_next_observations, sampled_next_prev_actions, args.use_last_action_in_policy_state
                    )
                    
                    # Compute combined rewards (with or without AMP)
                    if use_amp:
                        sampled_disc_obs = data['disc_obs']
                        sampled_disc_valid = data['disc_valid']
                        
                        # Recompute discriminator rewards with current discriminator
                        with torch.no_grad():
                            # Normalize discriminator observations
                            norm_sampled_disc_obs = disc_normalizer.normalize(sampled_disc_obs)
                            
                            # Compute discriminator rewards (least-squares scores)
                            disc_scores = discriminator(norm_sampled_disc_obs).squeeze(-1)
                            sampled_disc_r = torch.clamp(1 - 0.25*(disc_scores-1)**2, min=0)
                            
                            # Mask invalid transitions (set reward to 0)
                            sampled_disc_r = sampled_disc_r * sampled_disc_valid.float()
                            
                            # Combine task and discriminator rewards
                            sampled_combined_rewards = args.task_reward_weight * sampled_task_rewards + args.disc_reward_weight * sampled_disc_r
                    else:
                        # No AMP: use task rewards only (with task_reward_weight for consistency)
                        sampled_combined_rewards = args.task_reward_weight * sampled_task_rewards
                    
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _, _ = actor.get_action_and_value(sampled_next_policy_observations)
                        qf1_next_target = qf1_target(sampled_next_observations, next_state_actions)
                        qf2_next_target = qf2_target(sampled_next_observations, next_state_actions)
                        
                        # Transformed Bellman Operator:
                        # 1. Q-networks output values in h-transformed space
                        # 2. Apply h_inverse to convert back to original scale
                        # 3. Compute standard Bellman target: r + γ * (Q_next - α * log_pi)
                        # 4. Apply h to transform back to the space Q-networks are trained in
                        min_qf_next_target_h = torch.min(qf1_next_target, qf2_next_target)
                        min_qf_next_target = h_inverse(min_qf_next_target_h, eps=args.h_transform_eps)
                        
                        # Standard soft Q-value target (in original scale)
                        soft_q_next = min_qf_next_target - alpha * next_state_log_pi.unsqueeze(-1)
                        bellman_target_original = sampled_combined_rewards.flatten() + (1 - sampled_dones.flatten()) * args.gamma * soft_q_next.view(-1)
                        
                        # Transform to h-space for training
                        next_q_value = h_transform(bellman_target_original, eps=args.h_transform_eps)
                        
                        # Store values for logging (debug negative Q-targets)
                        next_state_log_pi_mean = next_state_log_pi.mean().item()
                        next_q_value_mean = next_q_value.mean().item()
                        min_qf_next_target_mean = min_qf_next_target.mean().item()
                        entropy_term_mean = (alpha * next_state_log_pi).mean().item()
                        # Additional logging for transformed values
                        bellman_target_original_mean = bellman_target_original.mean().item()

                    # Q-networks predict in h-transformed space
                    qf1_a_values = qf1(sampled_observations, sampled_actions).view(-1)
                    qf2_a_values = qf2(sampled_observations, sampled_actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    # optimize the model
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()
                    
                    # Store values for logging
                    qf1_loss_val = qf1_loss.item()
                    qf2_loss_val = qf2_loss.item()
                    qf_loss_val = qf_loss.item()
                    qf1_a_values_mean = qf1_a_values.mean().item()
                    qf2_a_values_mean = qf2_a_values.mean().item()
                    sampled_task_rewards_mean = sampled_task_rewards.mean().item()
                    sampled_combined_rewards_mean = sampled_combined_rewards.mean().item()
                    if use_amp:
                        sampled_disc_r_mean = sampled_disc_r.mean().item()

            # ================================================
            # Policy updates (controlled by policy_frequency)
            # ================================================
            if iteration % args.policy_frequency == 0:  # TD 3 Delayed update support
                # Sample fresh data for policy update
                data = rb.sample(args.batch_size)
                sampled_observations = data['observations']
                sampled_prev_actions = data['prev_actions']
                sampled_policy_observations = augment_policy_observation(
                    sampled_observations, sampled_prev_actions, args.use_last_action_in_policy_state
                )
                
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _, _ = actor.get_action_and_value(sampled_policy_observations)
                    qf1_pi = qf1(sampled_observations, pi)
                    qf2_pi = qf2(sampled_observations, pi)
                    min_qf_pi_h = torch.min(qf1_pi, qf2_pi)
                    # Transform Q-values back to original scale for actor loss
                    min_qf_pi = h_inverse(min_qf_pi_h, eps=args.h_transform_eps)
                    actor_loss = ((alpha * log_pi.unsqueeze(-1)) - min_qf_pi_h).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Store for logging
                    actor_loss_val = actor_loss.item()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _, _ = actor.get_action_and_value(sampled_policy_observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        
                        # Store for logging
                        alpha_loss_val = alpha_loss.item()

            # ================================================
            # Target network updates
            # ================================================
            if iteration % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            

            # ================================================
            # AMP: Train discriminator (only if AMP enabled)
            # ================================================
            if use_amp:
                # Sample demonstration data
                demo_disc_obs = demo_loader.sample(args.disc_batch_size)
                
                # Sample agent data from replay buffer
                agent_samples = args.disc_batch_size // 2
                
                # Sample from current batch (already stored valid transitions in amp_replay_buffer above)
                if len(amp_replay_buffer) > 0:
                    agent_disc_obs = amp_replay_buffer.sample(args.disc_batch_size)
                else:
                    # If replay buffer is empty, sample from current valid transitions
                    if valid_transition_mask.any():
                        valid_disc_obs = current_disc_obs[valid_transition_mask]
                        if len(valid_disc_obs) >= args.disc_batch_size:
                            perm_indices = torch.randperm(len(valid_disc_obs), device=args.device)[:args.disc_batch_size]
                            agent_disc_obs = valid_disc_obs[perm_indices]
                        else:
                            agent_disc_obs = valid_disc_obs
                    else:
                        agent_disc_obs = None  # Skip discriminator training if no valid transitions
                
                # Only train discriminator if we have agent data
                if agent_disc_obs is not None:
                    # Normalize observations
                    norm_demo_disc_obs = disc_normalizer.normalize(demo_disc_obs)
                    norm_agent_disc_obs = disc_normalizer.normalize(agent_disc_obs)
                    
                    # Enable gradients for gradient penalty
                    norm_demo_disc_obs.requires_grad_(True)
                    
                    # Forward pass through discriminator (raw scores, not logits)
                    disc_demo_logit = discriminator(norm_demo_disc_obs).squeeze(-1)
                    disc_agent_logit = discriminator(norm_agent_disc_obs).squeeze(-1)
                    
                    # Least squares (MSE) discriminator loss
                    # Demo = 1 (expert), Agent = -1 (fake)
                    # MSE instead of BCE: 0.5 * E[(D(x_real) - 1)^2] + 0.5 * E[(D(x_fake) - (-1))^2]
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
                    nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
                    disc_optimizer.step()
                    
                    # Update normalizer statistics (only with valid transitions)
                    if valid_transition_mask.any():
                        valid_disc_obs = current_disc_obs[valid_transition_mask]
                        disc_normalizer.record(valid_disc_obs)
                    disc_normalizer.record(demo_disc_obs)
                    disc_normalizer.update()
                    
                    # Discriminator accuracy (scores ~1 expert, ~-1 agent)
                    # Threshold at 0: expert positive, agent negative
                    with torch.no_grad():
                        disc_agent_acc = (disc_agent_logit < 0.0).float().mean().item()
                        disc_demo_acc = (disc_demo_logit > 0.0).float().mean().item()
                    
                    # Store for logging
                    disc_loss_val = disc_loss.item()


            # Logging
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values_mean, global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values_mean, global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss_val, global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_val, global_step)
                writer.add_scalar("losses/qf_loss", qf_loss_val / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_val, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                
                # Q-value target debug metrics
                writer.add_scalar("debug/next_q_value_target_mean", next_q_value_mean, global_step)
                writer.add_scalar("debug/next_state_log_pi_mean", next_state_log_pi_mean, global_step)
                writer.add_scalar("debug/min_qf_next_target_mean", min_qf_next_target_mean, global_step)
                writer.add_scalar("debug/entropy_term_mean", entropy_term_mean, global_step)
                # Transformed Bellman metrics (original scale vs h-transformed)
                writer.add_scalar("debug/bellman_target_original_mean", bellman_target_original_mean, global_step)
                
                # AMP metrics (only if AMP enabled and discriminator was trained)
                if use_amp and disc_loss_val is not None:
                    writer.add_scalar("amp/disc_loss", disc_loss_val, global_step)
                    writer.add_scalar("amp/disc_agent_acc", disc_agent_acc, global_step)
                    writer.add_scalar("amp/disc_demo_acc", disc_demo_acc, global_step)
                    writer.add_scalar("amp/replay_buffer_size", len(amp_replay_buffer), global_step)
                    # Log current step discriminator reward (from most recent environment step)
                    writer.add_scalar("amp/disc_reward_mean", sampled_disc_r_mean, global_step)
                
                # Log sampled batch task reward for reference
                writer.add_scalar("rewards/sampled_task_reward_mean", sampled_task_rewards_mean, global_step)
                writer.add_scalar("rewards/sampled_combined_reward_mean", sampled_combined_rewards_mean, global_step)
                writer.add_scalar("amp/temporal_alignment_reward_raw_mean", temporal_alignment_reward_raw_mean, global_step)
                writer.add_scalar("amp/temporal_alignment_reward_scaled_mean", temporal_alignment_reward_scaled_mean, global_step)
                writer.add_scalar("amp/action_magnitude_reward_raw_mean", action_magnitude_reward_raw_mean, global_step)
                writer.add_scalar("amp/action_magnitude_reward_scaled_mean", action_magnitude_reward_scaled_mean, global_step)
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss_val, global_step)
        
        # Log average returns every 500 steps
        if global_step > 0 and global_step % 500 == 0:
            if episodic_returns:
                avg_return = np.mean(episodic_returns)
                min_return = np.min(episodic_returns)
                max_return = np.max(episodic_returns)
                avg_success = np.mean(success_rates) if success_rates else 0.0
                
                print(f"Step {global_step}: Avg Return: {avg_return:.2f}, Min: {min_return:.2f}, Max: {max_return:.2f}, Success Rate: {avg_success:.2f}, Episodes: {len(episodic_returns)}")
                
                writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
                writer.add_scalar("charts/min_episodic_return", min_return, global_step)
                writer.add_scalar("charts/max_episodic_return", max_return, global_step)
                writer.add_scalar("charts/avg_success_rate", avg_success, global_step)
                
                # Clear lists for next logging period
                episodic_returns.clear()
                success_rates.clear()
            else:
                print(f"Step {global_step}: No episodes completed in last 500 steps")
            
            # Log motion statistics if available
            if velocity_magnitudes:
                avg_vel_mag = np.mean(velocity_magnitudes)
                avg_acc_mag = np.mean(acceleration_magnitudes)
                avg_jerk_mag = np.mean(jerk_magnitudes)
                
                print(f"Step {global_step}: Avg Velocity: {avg_vel_mag:.4f}, Avg Acceleration: {avg_acc_mag:.4f}, Avg Jerk: {avg_jerk_mag:.4f}")
                
                writer.add_scalar("motion/avg_velocity_magnitude", avg_vel_mag, global_step)
                writer.add_scalar("motion/avg_acceleration_magnitude", avg_acc_mag, global_step)
                writer.add_scalar("motion/avg_jerk_magnitude", avg_jerk_mag, global_step)
                
                velocity_magnitudes.clear()
                acceleration_magnitudes.clear()
                jerk_magnitudes.clear()
        
        # Checkpointing and evaluation every 10000 steps
        if global_step > 0 and global_step % 10000 == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(actor.state_dict(), model_path)
            
            # Save Q-network targets
            torch.save(qf1_target.state_dict(), f"{checkpoint_dir}/qf1_target.pth")
            torch.save(qf2_target.state_dict(), f"{checkpoint_dir}/qf2_target.pth")
            
            # Save AMP components in checkpoint (only if AMP enabled)
            if use_amp:
                torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
                torch.save({
                    'normalizer': disc_normalizer.state_dict(),
                    'replay_buffer': amp_replay_buffer.state_dict()
                }, f"{checkpoint_dir}/amp_components.pth")
            
            print(f"\nCheckpoint saved at step {global_step}")
            
            # Evaluate the model
            try:
                evaluate_agent(
                    model_path, 
                    checkpoint_dir, 
                    config["air_hockey"], 
                    n_eps=4, 
                    n_gifs=1,
                    reference_states=reference_states,
                    ref_max_episode_steps=args.ref_max_episode_steps if args.use_reference_state_init else None,
                    action_scale=action_scale,
                    agent_hidden_size=args.agent_hidden_size,
                    use_last_action_in_policy_state=args.use_last_action_in_policy_state,
                )
            except Exception as e:
                print(f"Evaluation failed: {e}")
            
            print(f"Step {global_step} complete\n")
        
        # Increment counters
        iteration += 1  # Iteration count for frequency checks
        global_step += args.num_envs  # Total samples collected (for timestep-based checks)

    envs.close()
    
    # Save final model
    torch.save(actor.state_dict(), f"{log_parent_dir}/model.pth")
    
    # Save Q-network targets
    torch.save(qf1_target.state_dict(), f"{log_parent_dir}/qf1_target.pth")
    torch.save(qf2_target.state_dict(), f"{log_parent_dir}/qf2_target.pth")
    
    # Save AMP components (only if AMP enabled)
    if use_amp:
        torch.save(discriminator.state_dict(), f"{log_parent_dir}/discriminator.pth")
        torch.save({
            'normalizer': disc_normalizer.state_dict(),
            'replay_buffer': amp_replay_buffer.state_dict()
        }, f"{log_parent_dir}/amp_components.pth")
        print(f"✓ Saved actor, Q-networks, discriminator and AMP components")
    else:
        print(f"✓ Saved actor and Q-networks")

    # Evaluate the final model
    try:
        evaluate_agent(
            f"{log_parent_dir}/model.pth", 
            log_parent_dir, 
            config["air_hockey"],
            reference_states=reference_states,
            ref_max_episode_steps=args.ref_max_episode_steps if args.use_reference_state_init else None,
            action_scale=action_scale,
            agent_hidden_size=args.agent_hidden_size,
            use_last_action_in_policy_state=args.use_last_action_in_policy_state,
        )
    except Exception as e:
        print(f"Final evaluation failed: {e}")
    
    # Define metrics to plot
    base_metrics = [
        'charts/episodic_return', 
        'losses/qf1_loss', 
        'losses/qf2_loss', 
        'losses/actor_loss',
        'losses/alpha'
    ]
    
    # Add AMP metrics (only if AMP enabled)
    if use_amp:
        amp_metrics = [
            'amp/disc_loss',
            'amp/disc_agent_acc',
            'amp/disc_demo_acc',
            'amp/disc_reward_mean',
        ]
        base_metrics.extend(amp_metrics)
    
    save_tensorboard_plots(log_parent_dir, config, metrics=base_metrics)
    
    writer.close()