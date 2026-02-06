"""
AMP Training with Triplet States (LSGAN Objective)

This script implements Adversarial Motion Priors (AMP) using state triplets
(s_t, s_{t+1}, s_{t+2}) instead of pairs, with translation-only normalization (no rotation).

This captures richer motion dynamics including acceleration patterns while using
the Least Squares GAN (LSGAN) objective for stable training.
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

from dataclasses import dataclass
import tyro

import os
from datetime import datetime

from scripts.smooth_policy.evaluate import evaluate_agent
from scripts.smooth_policy.agent import Agent
from scripts.utils import save_tensorboard_plots

# AMP components
from scripts.smooth_policy.amp_triplet.discriminator import Discriminator
from scripts.smooth_policy.amp_triplet.replay_buffer import ReplayBuffer
from scripts.smooth_policy.amp_triplet.normalizer import Normalizer
from scripts.smooth_policy.amp_triplet.demo_loader import DemoLoader


def normalize_state_triplet_batch(state_triplets):
    """
    Normalize batched state triplets to relative coordinates (translation only, no rotation).
    
    Process:
    1. Translate all states so first position is at (0, 0)
    2. Keep first velocity and all relative positions/velocities
    
    Args:
        state_triplets: Tensor of shape [batch, 3, 4] containing [state1, state2, state3]
                       where each state is [x_pos, y_pos, x_vel, y_vel]
    
    Returns:
        torch.Tensor: Normalized states of shape [batch, 10]
                     [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y,
                      rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]
    """
    # Extract states
    state1 = state_triplets[:, 0, :]  # [batch, 4]
    state2 = state_triplets[:, 1, :]  # [batch, 4]
    state3 = state_triplets[:, 2, :]  # [batch, 4]
    
    # Extract positions and velocities
    pos1 = state1[:, :2]  # [batch, 2]
    vel1 = state1[:, 2:]  # [batch, 2]
    pos2 = state2[:, :2]  # [batch, 2]
    vel2 = state2[:, 2:]  # [batch, 2]
    pos3 = state3[:, :2]  # [batch, 2]
    vel3 = state3[:, 2:]  # [batch, 2]
    
    # Step 1: Translate so first position is at origin
    pos2_translated = pos2 - pos1  # [batch, 2]
    pos3_translated = pos3 - pos1  # [batch, 2]
    
    # Step 2: Concatenate first velocity and all relative states
    # No rotation is applied - keep velocities as-is
    normalized_state = torch.cat([
        vel1,              # First velocity (2D)
        pos2_translated,   # Relative position 2 (2D)
        vel2,              # Second velocity (2D)
        pos3_translated,   # Relative position 3 (2D)
        vel3               # Third velocity (2D)
    ], dim=-1)  # [batch, 10]
    
    return normalized_state


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
    minibatch_size: int = 64
    batch_size: int = 0 # computed at runtime
    norm_adv: bool = True

    # CAPS hyperparameters
    caps_coef_nearby: float = 0.0
    caps_coef_consecutive: float = 0.0

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
    
    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str = None
    model_path: str = None  # Path to pre-trained model state dict
    discriminator_path: str = None  # Path to pre-trained discriminator state dict
    amp_components_path: str = None  # Path to AMP components (normalizer, replay buffer)
    log_parent_dir: str = None
    run_name: str = "default"
    demo_data_path: str = "scripts/smooth_policy/amp_triplet_data/amp_triplet_dataset.pt"

    # Others
    seed: int = 0
    device: str = "cuda:0"

    # action scale for the agent
    action_scale: float = 0.02
    

def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
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

    # should just create parallel envs for future use (can just use sync, async as placeholders)
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(args.num_envs)])

    # hard-coded base reward scaling for now
    envs.call('set_base_reward_scaling', 0.1)

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
    
    agent = Agent(envs, action_scale=args.action_scale, action_bias=0.0).to(args.device)
    # Load pre-trained model if path is provided
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-6)

    # Initialize AMP components (always enabled)
    print("\n" + "="*80)
    print("Initializing AMP (Adversarial Motion Priors) with Triplet States")
    print("="*80)
    
    # Discriminator observation dimension: normalized state triplets = 10D
    # [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y,
    #  rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]
    disc_obs_dim = 10
    
    # Initialize discriminator
    discriminator = Discriminator(disc_obs_dim, hidden_dims=[128, 128], activation='relu').to(args.device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_learning_rate, eps=1e-6)
    print(f"✓ Discriminator initialized (input_dim={disc_obs_dim}, hidden=[128, 128])")
    
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
    
    # Load demonstration data
    demo_loader = DemoLoader(args.demo_data_path, device=args.device)
    print(f"✓ Demo loader initialized ({len(demo_loader):,} demonstrations)")
    
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
    
    print("="*80 + "\n")

    # Component-wise statistics tracking
    component_names = ['vel1_x', 'vel1_y', 'rel_pos2_x', 'rel_pos2_y', 
                       'rel_vel2_x', 'rel_vel2_y', 'rel_pos3_x', 'rel_pos3_y',
                       'rel_vel3_x', 'rel_vel3_y']
    agent_stats = {name: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')} 
                   for name in component_names}
    demo_stats = {name: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')} 
                  for name in component_names}

    # main training loop
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    
    # AMP: Storage for discriminator observations (10D for triplets)
    disc_obs = torch.zeros((args.num_steps, args.num_envs, 10)).to(args.device)
    prev_paddle_state_1 = torch.zeros((args.num_envs, 4)).to(args.device)
    prev_paddle_state_2 = torch.zeros((args.num_envs, 4)).to(args.device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    
    # Tracking lists for motion metrics
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []
    
    for iteration in range(1, args.num_iterations + 1):
        # Reset episodic return tracking for this iteration
        episodic_returns = []
        success_rates = []
        
        # Reset component statistics for this iteration
        for name in component_names:
            agent_stats[name] = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
            demo_stats[name] = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # REWARD SCALING is done on the environment level, not here

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)
            
            # AMP: Construct discriminator observations (consecutive paddle state triplets)
            # Extract current paddle state (position + velocity) from observation
            # Assuming obs_type="vel": [paddle_x, paddle_y, paddle_vx, paddle_vy, puck_x, puck_y, puck_vx, puck_vy]
            current_paddle_state = next_obs[:, :4]  # [batch, 4]
            
            # Create state triplets [prev_prev_state, prev_state, current_state]
            if step == 0:
                # First step: initialize both previous states with current state
                prev_paddle_state_1 = current_paddle_state.clone()
                prev_paddle_state_2 = current_paddle_state.clone()
            
            # Create raw state triplets [batch, 3, 4]
            raw_state_triplets = torch.stack([prev_paddle_state_2, prev_paddle_state_1, current_paddle_state], dim=1)
            
            # Apply normalization to get [batch, 10]
            normalized_state_triplets = normalize_state_triplet_batch(raw_state_triplets)
            
            # Store the normalized state triplets
            disc_obs[step] = normalized_state_triplets
            
            # Update previous states for next step (but reset on done)
            prev_paddle_state_2 = prev_paddle_state_1.clone()
            prev_paddle_state_1 = current_paddle_state.clone()
            
            # Reset previous states for environments that are done
            if next_done.any():
                prev_paddle_state_1[next_done.bool()] = current_paddle_state[next_done.bool()]
                prev_paddle_state_2[next_done.bool()] = current_paddle_state[next_done.bool()]

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
            next_value = agent.get_value(next_obs).reshape(1, -1)
            
            # AMP: Compute discriminator rewards and combine with task rewards
            # Flatten discriminator observations
            b_disc_obs = disc_obs.reshape(-1, 10)
            
            # Normalize discriminator observations
            norm_disc_obs = disc_normalizer.normalize(b_disc_obs)
            
            # Compute discriminator rewards (LSGAN)
            disc_scores = discriminator(norm_disc_obs).squeeze(-1)
            # For LSGAN, discriminator outputs raw scores (not logits)
            # Reward formula: quadratic function peaked at disc_scores=1, clamped to [0, 1]
            disc_r = torch.clamp(1 - 0.25*(disc_scores-1)**2, min=0)
            
            # Reshape discriminator rewards to match original shape
            disc_r_shaped = disc_r.reshape(args.num_steps, args.num_envs)
            
            # Combine task and discriminator rewards
            combined_rewards = args.task_reward_weight * rewards + args.disc_reward_weight * disc_r_shaped
            
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
            
            # Log discriminator reward statistics
            writer.add_scalar("amp/disc_reward_mean", disc_r.mean().item(), global_step)
            writer.add_scalar("amp/disc_reward_std", disc_r.std().item(), global_step)
            writer.add_scalar("amp/task_reward_mean", rewards.mean().item(), global_step)
            writer.add_scalar("amp/combined_reward_mean", combined_rewards.mean().item(), global_step)

            # log statistics of the advantages, values
            writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
            writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
            writer.add_scalar("charts/value_mean", values.mean().item(), global_step)
            writer.add_scalar("charts/value_std", values.std().item(), global_step)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1).bool()
        
        # EVALUATE the loss before optimization
        with torch.no_grad():
            noise_std = 0.01
            nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std # new sample of noise
            nearby_actions, _, _, _ = agent.get_action_and_value(nearby_obs)
            next_actions = b_actions[np.clip(np.arange(args.batch_size) + 1, 0, args.batch_size - 1)] # ignore bias from the last action
            non_done_mask = ~b_dones

            # L2 losses
            nearby_action_loss_l2 = ((nearby_actions - b_actions) ** 2.0).mean()
            consecutive_action_loss_l2 = ((next_actions - b_actions) ** 2.0)[non_done_mask].sum() / non_done_mask.sum() # average over non-done steps
            action_loss_l2 = (b_actions ** 2.0).mean()
            caps_loss = nearby_action_loss_l2 * args.caps_coef_nearby + consecutive_action_loss_l2 * args.caps_coef_consecutive

            # L1 losses
            nearby_action_loss_l1 = ((nearby_actions - b_actions).abs()).mean()
            consecutive_action_loss_l1 = ((next_actions - b_actions).abs())[non_done_mask].sum() / non_done_mask.sum() # average over non-done steps
            action_loss_l1 = (b_actions.abs()).mean()

            writer.add_scalar("losses/consecutive_action_loss_l2", consecutive_action_loss_l2.item(), global_step)
            writer.add_scalar("losses/nearby_action_loss_l2", nearby_action_loss_l2.item(), global_step)
            writer.add_scalar("losses/caps_loss", caps_loss.item(), global_step)
            writer.add_scalar("losses/action_loss_l2", action_loss_l2.item(), global_step) # plot out, but not used in training
            writer.add_scalar("losses/consecutive_action_loss_l1", consecutive_action_loss_l1.item(), global_step)
            writer.add_scalar("losses/nearby_action_loss_l1", nearby_action_loss_l1.item(), global_step)
            writer.add_scalar("losses/action_loss_l1", action_loss_l1.item(), global_step) # plot out, but not used in training
            

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            noise_std = 0.01
            nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std # new sample of noise
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, _, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                # caps loss
                if args.caps_coef_nearby > 0 or args.caps_coef_consecutive > 0:
                    next_inds = np.clip(mb_inds + 1, 0, len(b_obs) - 1)
                    curr_actions, _, _, _ = agent.get_action_and_value(b_obs[mb_inds])
                    nearby_actions, _, _, _ = agent.get_action_and_value(nearby_obs[mb_inds]) # for now just use sample from gaussian
                    next_actions, _, _, _ = agent.get_action_and_value(b_obs[next_inds])

                    nearby_action_loss = ((nearby_actions - curr_actions) ** 2.0).mean()
                    non_done_mask = ~b_dones[mb_inds]
                    consecutive_action_loss = ((next_actions - curr_actions) ** 2.0)[non_done_mask].sum() / non_done_mask.sum() # average over non-done steps

                    caps_loss = nearby_action_loss * args.caps_coef_nearby + consecutive_action_loss * args.caps_coef_consecutive
                else:
                    caps_loss = 0.0

                entropy_loss = (-newlogprob).mean() # unbiased estimate of entropy
                # entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + caps_loss
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # AMP: Train discriminator
        # Sample demonstration data
        demo_disc_obs = demo_loader.sample(args.disc_batch_size)
        
        # Record demo statistics (before normalization)
        with torch.no_grad():
            for i, name in enumerate(component_names):
                component_data = demo_disc_obs[:, i]
                demo_stats[name]['sum'] += component_data.sum().item()
                demo_stats[name]['sum_sq'] += (component_data ** 2).sum().item()
                demo_stats[name]['count'] += len(component_data)
                demo_stats[name]['min'] = min(demo_stats[name]['min'], component_data.min().item())
                demo_stats[name]['max'] = max(demo_stats[name]['max'], component_data.max().item())
        
        # Sample agent data (mix current batch + replay buffer)
        agent_samples = args.disc_batch_size // 2
        
        # Randomly sample from current batch
        perm_indices = torch.randperm(len(b_disc_obs), device=args.device)
        agent_disc_obs_current = b_disc_obs[perm_indices[:agent_samples]]
        
        # Store samples in replay buffer
        num_to_store = min(len(b_disc_obs), args.disc_replay_samples)
        replay_buffer.push(b_disc_obs[perm_indices[:num_to_store]])
        
        # Sample from replay buffer if available
        if len(replay_buffer) > 0:
            agent_disc_obs_replay = replay_buffer.sample(agent_samples)
            agent_disc_obs = torch.cat([agent_disc_obs_current, agent_disc_obs_replay], dim=0)
        else:
            agent_disc_obs = agent_disc_obs_current
        
        # Record agent statistics (before normalization)
        with torch.no_grad():
            for i, name in enumerate(component_names):
                component_data = agent_disc_obs[:, i]
                agent_stats[name]['sum'] += component_data.sum().item()
                agent_stats[name]['sum_sq'] += (component_data ** 2).sum().item()
                agent_stats[name]['count'] += len(component_data)
                agent_stats[name]['min'] = min(agent_stats[name]['min'], component_data.min().item())
                agent_stats[name]['max'] = max(agent_stats[name]['max'], component_data.max().item())
        
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
        
        # Update normalizer statistics
        disc_normalizer.record(b_disc_obs)
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
        
        # Log component-wise statistics for agent vs demo
        for name in component_names:
            # Agent statistics
            if agent_stats[name]['count'] > 0:
                agent_mean = agent_stats[name]['sum'] / agent_stats[name]['count']
                agent_var = (agent_stats[name]['sum_sq'] / agent_stats[name]['count']) - (agent_mean ** 2)
                agent_std = np.sqrt(max(agent_var, 0))
                
                writer.add_scalar(f"amp_components/agent_{name}_mean", agent_mean, global_step)
                writer.add_scalar(f"amp_components/agent_{name}_std", agent_std, global_step)
                writer.add_scalar(f"amp_components/agent_{name}_min", agent_stats[name]['min'], global_step)
                writer.add_scalar(f"amp_components/agent_{name}_max", agent_stats[name]['max'], global_step)
            
            # Demo statistics
            if demo_stats[name]['count'] > 0:
                demo_mean = demo_stats[name]['sum'] / demo_stats[name]['count']
                demo_var = (demo_stats[name]['sum_sq'] / demo_stats[name]['count']) - (demo_mean ** 2)
                demo_std = np.sqrt(max(demo_var, 0))
                
                writer.add_scalar(f"amp_components/demo_{name}_mean", demo_mean, global_step)
                writer.add_scalar(f"amp_components/demo_{name}_std", demo_std, global_step)
                writer.add_scalar(f"amp_components/demo_{name}_min", demo_stats[name]['min'], global_step)
                writer.add_scalar(f"amp_components/demo_{name}_max", demo_stats[name]['max'], global_step)
            
            # Log the difference between agent and demo
            if agent_stats[name]['count'] > 0 and demo_stats[name]['count'] > 0:
                mean_diff = abs(agent_mean - demo_mean)
                std_ratio = agent_std / (demo_std + 1e-8)
                writer.add_scalar(f"amp_components/diff_{name}_mean_abs", mean_diff, global_step)
                writer.add_scalar(f"amp_components/diff_{name}_std_ratio", std_ratio, global_step)
        
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
            
            # Save AMP components in checkpoint
            torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator.pth")
            torch.save({
                'normalizer': disc_normalizer.state_dict(),
                'replay_buffer': replay_buffer.state_dict()
            }, f"{checkpoint_dir}/amp_components.pth")

            # evaluate the model
            evaluate_agent(model_path, checkpoint_dir, config["air_hockey"], n_eps=4, n_gifs=1)
            
            print(f"Iteration {iteration} complete")

    # save model
    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")
    
    # Save AMP components
    torch.save(discriminator.state_dict(), f"{log_parent_dir}/discriminator.pth")
    torch.save({
        'normalizer': disc_normalizer.state_dict(),
        'replay_buffer': replay_buffer.state_dict()
    }, f"{log_parent_dir}/amp_components.pth")
    print(f"✓ Saved discriminator and AMP components")

    # evaluate the model and save results
    evaluate_agent(f"{log_parent_dir}/model.pth", log_parent_dir, config["air_hockey"])
    
    # Define metrics to plot
    base_metrics = [
        'charts/avg_episodic_return', 
        'charts/max_episodic_return', 
        'charts/min_episodic_return', 
        'charts/episodic_return', 
        'losses/approx_kl', 
        'losses/value_loss', 
        'losses/policy_loss', 
        'charts/avg_success_rate',
        'losses/action_loss',
        'losses/caps_loss',
        'motion/avg_velocity_magnitude',
        'motion/avg_acceleration_magnitude',
        'motion/avg_jerk_magnitude'
    ]
    
    # Add AMP metrics
    amp_metrics = [
        'amp/disc_loss',
        'amp/disc_agent_acc',
        'amp/disc_demo_acc',
        'amp/disc_reward_mean',
        'amp/task_reward_mean',
        'amp/combined_reward_mean'
    ]
    base_metrics.extend(amp_metrics)
    
    save_tensorboard_plots(log_parent_dir, config, metrics=base_metrics)
