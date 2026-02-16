"""
AMP Training with LSGAN Objective

This script implements Adversarial Motion Priors (AMP) using the 
Least Squares GAN (LSGAN) objective instead of the typical binary cross-entropy loss.

LSGAN uses MSE loss: 0.5 * E[(D(x_real) - 1)^2] + 0.5 * E[(D(x_fake) - (-1))^2]
This variant uses [-1, 1] targets instead of [0, 1] for better gradient flow.
This often provides more stable gradients compared to standard GAN training.

Supports two discriminator modes:
1. Position-only (default): Discriminator input is 8D (4 relative positions x 2 coords)
2. Position + Action: Discriminator input is 10D (8 positions + 2 action dims)
   Enable with --use_action_discriminator flag and provide a dataset with action_sequences.
"""

import random
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml

from airhockey import AirHockeyEnv
import gymnasium as gym

from dataclasses import dataclass, field
import tyro

import os
from datetime import datetime
from types import SimpleNamespace

from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.smooth_policy.agent import Agent

# AMP components
from scripts.smooth_policy.amp_history.amp_training.discriminator import Discriminator
from scripts.smooth_policy.amp_history.amp_training.replay_buffer import ReplayBuffer
from scripts.smooth_policy.amp_history.amp_training.normalizer import Normalizer
from scripts.smooth_policy.amp_history.amp_training.demo_loader_position_history import DemoLoaderPositionHistory

# used for possible reference state initialization from demonstrations
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
    num_envs: int = 8
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
    agent_hidden_size: int = 128
    
    # Reference state initialization
    use_reference_state_init: bool = False  # Enable/disable feature
    reference_data_path: str = None  # Path to raw demo data (defaults to demo_data_path)
    ref_max_episode_steps: int = 200  # Episode length when using reference init
    
    # Action-conditioned discriminator
    use_action_discriminator: bool = False  # If True, discriminator uses position + 4 transition actions (16D)
    disc_debug_interval: int = 5000  # Print discriminator feature samples every N env steps (<=0 disables)
    
    

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
    args.batch_size = args.num_envs * args.num_steps

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
        # Create reference states with zero velocity: [N, 4] = [x, y, 0, 0]
        reference_states = np.concatenate([
            first_positions,
            np.zeros_like(first_positions)  # Zero velocity
        ], axis=1)
        print(f"✓ Loaded {len(reference_states):,} reference states (position-based)")
        print(f"  Note: Velocities initialized to zero")
        print(f"  Episode length will be {args.ref_max_episode_steps} timesteps")
        print(f"{'='*80}\n")

    # should just create parallel envs for future use (can just use sync, async as placeholders)
    envs = gym.vector.AsyncVectorEnv([
        make_env(i, reference_states, args.ref_max_episode_steps) 
        for i in range(args.num_envs)
    ])

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
    
    if 'use_pid' in config["air_hockey"] and config["air_hockey"]["use_pid"]:
        action_scale = 1
    else:
        action_scale = args.action_scale # use whatever action scale specified

    policy_obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    if args.use_last_action_in_policy_state:
        policy_obs_dim += action_dim
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
    # Load pre-trained model if path is provided
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-6)
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
    policy_obs = torch.zeros((args.num_steps, args.num_envs, policy_obs_dim), device=args.device)
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    just_reset = torch.zeros(args.num_envs, dtype=torch.bool).to(args.device)
    last_action_for_policy = torch.zeros((args.num_envs, action_dim), device=args.device)
    

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
            policy_next_obs = augment_policy_observation(
                next_obs, last_action_for_policy, args.use_last_action_in_policy_state
            )
            policy_obs[step] = policy_next_obs

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(policy_next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # REWARD SCALING is done on the environment level, not here
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)
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
                        episodic_returns.append(info['episode_return'])
                        success_rates.append(1.0 if info['success'] else 0.0)
                        # print(f"global_step={global_step}, episodic_return={info['episode_return']}")
                        writer.add_scalar("charts/episodic_return", info['episode_return'], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode_length'], global_step)
                        
                        # Extract motion data if available
                        if 'motion_data' in info:
                            velocity_magnitudes.extend(info['motion_data']['velocity_mags'])
                            acceleration_magnitudes.extend(info['motion_data']['acceleration_mags'])
                            jerk_magnitudes.extend(info['motion_data']['jerk_mags'])

        # bootstrap value if not done and compute advantages
        with torch.no_grad():
            next_policy_obs = augment_policy_observation(
                next_obs, last_action_for_policy, args.use_last_action_in_policy_state
            )
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_policy_obs = policy_obs.reshape((-1, policy_obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1).bool()
            

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            noise_std = 0.01
            nearby_policy_obs = b_policy_obs + torch.randn_like(b_policy_obs) * noise_std # new sample of noise
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, _, newvalue = agent.get_action_and_value(b_policy_obs[mb_inds], b_actions[mb_inds])
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
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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
            
            print(f"Iteration {iteration}: Avg Velocity Mag: {avg_vel_mag:.4f}, Avg Acceleration Mag: {avg_acc_mag:.4f}, Avg Jerk Mag: {avg_jerk_mag:.4f}")
            
            writer.add_scalar("motion/avg_velocity_magnitude", avg_vel_mag, iteration)
            writer.add_scalar("motion/avg_acceleration_magnitude", avg_acc_mag, iteration)
            writer.add_scalar("motion/avg_jerk_magnitude", avg_jerk_mag, iteration)
            
            # Clear lists for next iteration
            velocity_magnitudes.clear()
            acceleration_magnitudes.clear()
            jerk_magnitudes.clear()

        if iteration % 10 == 0 or min_return >= 5000: # start cherry-picking good policies
            # save a checkpoint of the model
            # create a subfolder for the checkpoint
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(agent.state_dict(), model_path)
            
            # Save AMP components in checkpoint when discriminator reward is active
            if use_discriminator_reward:
                torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
                torch.save({
                    'normalizer': disc_normalizer.state_dict(),
                    'replay_buffer': replay_buffer.state_dict()
                }, f"{checkpoint_dir}/amp_components.pth")

            # evaluate the model
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
            
            print(f"Iteration {iteration} complete")

    # save model
    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")
    
    # Save AMP components when discriminator reward is active
    if use_discriminator_reward:
        torch.save(discriminator.state_dict(), f"{log_parent_dir}/discriminator.pth")
        torch.save({
            'normalizer': disc_normalizer.state_dict(),
            'replay_buffer': replay_buffer.state_dict()
        }, f"{log_parent_dir}/amp_components.pth")
        print(f"✓ Saved discriminator and AMP components")

    # evaluate the model and save results
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
    
    # end of training