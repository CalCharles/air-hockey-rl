"""
Discriminator Verification Script

This script verifies that the discriminator can successfully distinguish between
demonstration data and agent-generated observations. It:

1. Loads demonstration data from amp_no_rotation_data
2. Collects agent observations by running a trained model in the environment
3. Trains a discriminator using LSGAN objective (same as amp_training_lsgan.py)
4. Monitors comprehensive discriminator statistics via TensorBoard

This is a standalone verification tool - no policy training involved.
"""

import random
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path

from airhockey import AirHockeyEnv
import gymnasium as gym

from dataclasses import dataclass
import tyro

import os
from datetime import datetime

from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.amp.discriminator import Discriminator
from scripts.smooth_policy.amp.normalizer import Normalizer
from scripts.smooth_policy.amp.demo_loader import DemoLoader


def normalize_state_pair_batch(state_pairs):
    """
    Normalize batched state pairs to relative coordinates (translation only).
    
    Process:
    1. Translate both states so first position is at (0, 0)
    2. Return first velocity and second state (preserving original velocity directions)
    
    Args:
        state_pairs: Tensor of shape [batch, 2, 4] containing [state1, state2]
                    where each state is [x_pos, y_pos, x_vel, y_vel]
    
    Returns:
        torch.Tensor: Normalized states of shape [batch, 6]
                     [vel1_x, vel1_y, pos2_x, pos2_y, vel2_x, vel2_y]
    """
    # Extract states
    state1 = state_pairs[:, 0, :]  # [batch, 4]
    state2 = state_pairs[:, 1, :]  # [batch, 4]
    
    # Extract positions and velocities
    pos1 = state1[:, :2]  # [batch, 2]
    vel1 = state1[:, 2:]  # [batch, 2]
    pos2 = state2[:, :2]  # [batch, 2]
    vel2 = state2[:, 2:]  # [batch, 2]
    
    # Step 1: Translate so first position is at origin
    pos2_translated = pos2 - pos1  # [batch, 2]
    
    # Step 2: Concatenate first velocity and second state
    # First position [0, 0] contains no information so not included
    # No rotation applied - velocities maintain original direction
    normalized_state = torch.cat([vel1, pos2_translated, vel2], dim=-1)  # [batch, 6]
    
    return normalized_state


def split_data(data, val_split, seed=0):
    """
    Split data into train and validation sets.
    
    Args:
        data: Tensor to split
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_data, val_data)
    """
    torch.manual_seed(seed)
    n_samples = len(data)
    indices = torch.randperm(n_samples)
    
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]
    
    return train_data, val_data


def collect_agent_observations(agent, env, num_steps, device):
    """
    Collect agent observations by running the model in the environment.
    
    Args:
        agent: Trained agent model
        env: Environment to run in
        num_steps: Number of steps to collect
        device: Device to store tensors on
    
    Returns:
        torch.Tensor: Normalized agent observations [N, 6]
    """
    print("\n" + "="*80)
    print("COLLECTING AGENT OBSERVATIONS")
    print("="*80)
    
    agent.eval()
    
    all_state_pairs = []
    prev_paddle_state = None
    
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to(device)
    
    steps_collected = 0
    episodes = 0
    
    with torch.no_grad():
        while steps_collected < num_steps:
            # Get action from agent
            action = agent.forward(obs)
            
            # Step environment (flatten action to 1D array)
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            
            # Extract current paddle state [x, y, vx, vy]
            current_paddle_state = torch.Tensor(next_obs[:4]).to(device)
            
            # Create state pair if we have a previous state
            if prev_paddle_state is not None:
                state_pair = torch.stack([prev_paddle_state, current_paddle_state], dim=0)
                all_state_pairs.append(state_pair)
                steps_collected += 1
            
            # Update previous state
            prev_paddle_state = current_paddle_state.clone()
            
            # Update observation
            obs = torch.Tensor(next_obs).to(device)
            
            # Reset if done
            if done:
                obs, _ = env.reset()
                obs = torch.Tensor(obs).to(device)
                prev_paddle_state = None
                episodes += 1
                
                if steps_collected % 1000 == 0 and steps_collected > 0:
                    print(f"  Collected {steps_collected}/{num_steps} steps ({episodes} episodes)")
    
    print(f"\n✓ Collected {len(all_state_pairs):,} agent state pairs from {episodes} episodes")
    
    # Stack all pairs and normalize
    state_pairs_tensor = torch.stack(all_state_pairs, dim=0)  # [N, 2, 4]
    normalized_obs = normalize_state_pair_batch(state_pairs_tensor)  # [N, 6]
    
    print(f"✓ Normalized agent observations shape: {normalized_obs.shape}")
    print("="*80 + "\n")
    
    return normalized_obs


@dataclass
class Args:
    # Model and data paths
    model_path: str = "pid/no_rotation/runr1/checkpoint_380/model.pth"
    demo_data_path: str = "scripts/smooth_policy/amp_no_rotation_data/amp_full_dataset_raw.pt"
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    
    # Data collection
    num_collection_steps: int = 10000
    
    # Train/validation split
    val_split: float = 0.2  # Fraction of data to use for validation
    
    # Discriminator training
    num_training_iterations: int = 1000
    batch_size: int = 512
    learning_rate: float = 1e-5
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 5.0
    disc_weight_decay: float = 0.0001
    max_grad_norm: float = 0.5
    
    # Logging
    log_dir: str = None
    log_interval: int = 10
    
    # Device
    device: str = "cuda:0"
    seed: int = 0


def make_env(config):
    """Create a single environment."""
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env
    return _thunk()


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    print("\n" + "="*80)
    print("DISCRIMINATOR VERIFICATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Demo data path: {args.demo_data_path}")
    print(f"  Config: {args.config}")
    print(f"  Collection steps: {args.num_collection_steps:,}")
    print(f"  Training iterations: {args.num_training_iterations:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print("="*80 + "\n")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_dir is None:
        args.log_dir = f"runs/disc_verification/run_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)
    
    writer = SummaryWriter(args.log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Save config and args
    with open(f"{args.log_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{args.log_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    
    # Load demonstration data
    print("\n" + "="*80)
    print("LOADING DEMONSTRATION DATA")
    print("="*80)
    demo_loader = DemoLoader(args.demo_data_path, device=args.device)
    demo_obs_all = demo_loader.get_all()
    print(f"✓ Loaded {len(demo_obs_all):,} demonstration observations")
    
    # Split demo data
    demo_train, demo_val = split_data(demo_obs_all, args.val_split, seed=args.seed)
    print(f"✓ Split into train: {len(demo_train):,}, validation: {len(demo_val):,}")
    print("="*80 + "\n")
    
    # Create environment and load agent
    print("\n" + "="*80)
    print("LOADING AGENT MODEL")
    print("="*80)
    env = make_env(config)
    envs = gym.vector.SyncVectorEnv([lambda: env])
    agent = Agent(envs).to(args.device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at: {args.model_path}")
    
    agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print(f"✓ Loaded agent model from: {args.model_path}")
    print("="*80 + "\n")
    
    # Collect agent observations
    agent_obs_all = collect_agent_observations(agent, env, args.num_collection_steps, args.device)
    
    # Split agent data
    print("\n" + "="*80)
    print("SPLITTING AGENT DATA")
    print("="*80)
    agent_train, agent_val = split_data(agent_obs_all, args.val_split, seed=args.seed)
    print(f"✓ Split into train: {len(agent_train):,}, validation: {len(agent_val):,}")
    print("="*80 + "\n")
    
    # Initialize discriminator
    print("\n" + "="*80)
    print("INITIALIZING DISCRIMINATOR")
    print("="*80)
    disc_obs_dim = 6
    discriminator = Discriminator(disc_obs_dim, hidden_dims=[32, 32], activation='relu').to(args.device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, eps=1e-6)
    disc_normalizer = Normalizer(shape=(disc_obs_dim,), clip=10.0, device=args.device)
    print(f"✓ Discriminator initialized (input_dim={disc_obs_dim}, hidden=[128, 128])")
    print(f"✓ Normalizer initialized (clip=10.0)")
    print("="*80 + "\n")
    
    # Component-wise statistics tracking
    component_names = ['vel1_x', 'vel1_y', 'rel_pos_x', 'rel_pos_y', 'rel_vel_x', 'rel_vel_y']
    
    # Pre-compute statistics for demo and agent data
    print("\n" + "="*80)
    print("DATA STATISTICS (Before Training)")
    print("="*80)
    print("\nDemonstration data (TRAIN):")
    for i, name in enumerate(component_names):
        values = demo_train[:, i]
        print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}, "
              f"min={values.min():.4f}, max={values.max():.4f}")
    
    print("\nAgent data (TRAIN):")
    for i, name in enumerate(component_names):
        values = agent_train[:, i]
        print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}, "
              f"min={values.min():.4f}, max={values.max():.4f}")
    print("="*80 + "\n")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING DISCRIMINATOR")
    print("="*80)
    
    start_time = time.time()
    
    for iteration in range(1, args.num_training_iterations + 1):
        # Sample batches from TRAIN set
        demo_indices = torch.randperm(len(demo_train), device=args.device)[:args.batch_size]
        agent_indices = torch.randperm(len(agent_train), device=args.device)[:args.batch_size]
        
        demo_batch = demo_train[demo_indices]
        agent_batch = agent_train[agent_indices]
        
        # Normalize observations
        norm_demo_batch = disc_normalizer.normalize(demo_batch)
        norm_agent_batch = disc_normalizer.normalize(agent_batch)
        
        # Enable gradients for gradient penalty
        norm_demo_batch.requires_grad_(True)
        
        # Forward pass through discriminator
        disc_demo_logit = discriminator(norm_demo_batch).squeeze(-1)
        disc_agent_logit = discriminator(norm_agent_batch).squeeze(-1)
        
        # LSGAN loss: Demo = 1 (expert), Agent = -1 (fake)
        disc_loss_demo = 0.5 * torch.mean((disc_demo_logit - 1.0) ** 2)
        disc_loss_agent = 0.5 * torch.mean((disc_agent_logit - (-1.0)) ** 2)
        disc_loss = disc_loss_demo + disc_loss_agent
        
        # Gradient penalty (Lipschitz constraint)
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit, norm_demo_batch,
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
        disc_normalizer.record(demo_batch)
        disc_normalizer.record(agent_batch)
        disc_normalizer.update()
        
        # Compute discriminator accuracy on TRAIN set
        with torch.no_grad():
            train_agent_acc = (disc_agent_logit < 0.0).float().mean().item()
            train_demo_acc = (disc_demo_logit > 0.0).float().mean().item()
        
        # Logging
        if iteration % args.log_interval == 0 or iteration == 1:
            # Evaluate on VALIDATION set
            with torch.no_grad():
                # Sample validation batches
                val_demo_indices = torch.randperm(len(demo_val), device=args.device)[:args.batch_size]
                val_agent_indices = torch.randperm(len(agent_val), device=args.device)[:args.batch_size]
                
                val_demo_batch = demo_val[val_demo_indices]
                val_agent_batch = agent_val[val_agent_indices]
                
                # Normalize validation batches
                norm_val_demo_batch = disc_normalizer.normalize(val_demo_batch)
                norm_val_agent_batch = disc_normalizer.normalize(val_agent_batch)
                
                # Forward pass
                val_disc_demo_logit = discriminator(norm_val_demo_batch).squeeze(-1)
                val_disc_agent_logit = discriminator(norm_val_agent_batch).squeeze(-1)
                
                # Compute validation losses
                val_disc_loss_demo = 0.5 * torch.mean((val_disc_demo_logit - 1.0) ** 2)
                val_disc_loss_agent = 0.5 * torch.mean((val_disc_agent_logit - (-1.0)) ** 2)
                val_disc_loss = val_disc_loss_demo + val_disc_loss_agent
                
                # Compute validation accuracies
                val_agent_acc = (val_disc_agent_logit < 0.0).float().mean().item()
                val_demo_acc = (val_disc_demo_logit > 0.0).float().mean().item()
            
            elapsed = time.time() - start_time
            iter_per_sec = iteration / elapsed
            
            print(f"Iteration {iteration}/{args.num_training_iterations} "
                  f"({iter_per_sec:.2f} iter/s)")
            print(f"  TRAIN Loss: {disc_loss.item():.4f} "
                  f"(Demo: {disc_loss_demo.item():.4f}, Agent: {disc_loss_agent.item():.4f})")
            print(f"  TRAIN Accuracy: Demo={train_demo_acc:.3f}, Agent={train_agent_acc:.3f}")
            print(f"  VAL Loss: {val_disc_loss.item():.4f} "
                  f"(Demo: {val_disc_loss_demo.item():.4f}, Agent: {val_disc_loss_agent.item():.4f})")
            print(f"  VAL Accuracy: Demo={val_demo_acc:.3f}, Agent={val_agent_acc:.3f}")
            print(f"  Grad Penalty: {disc_grad_penalty.item():.4f}")
            
            # TensorBoard logging - TRAIN
            writer.add_scalar("train/loss", disc_loss.item(), iteration)
            writer.add_scalar("train/loss_demo", disc_loss_demo.item(), iteration)
            writer.add_scalar("train/loss_agent", disc_loss_agent.item(), iteration)
            writer.add_scalar("train/grad_penalty", disc_grad_penalty.item(), iteration)
            writer.add_scalar("train/agent_acc", train_agent_acc, iteration)
            writer.add_scalar("train/demo_acc", train_demo_acc, iteration)
            writer.add_scalar("train/agent_logit_mean", disc_agent_logit.mean().item(), iteration)
            writer.add_scalar("train/demo_logit_mean", disc_demo_logit.mean().item(), iteration)
            
            # TensorBoard logging - VALIDATION
            writer.add_scalar("val/loss", val_disc_loss.item(), iteration)
            writer.add_scalar("val/loss_demo", val_disc_loss_demo.item(), iteration)
            writer.add_scalar("val/loss_agent", val_disc_loss_agent.item(), iteration)
            writer.add_scalar("val/agent_acc", val_agent_acc, iteration)
            writer.add_scalar("val/demo_acc", val_demo_acc, iteration)
            writer.add_scalar("val/agent_logit_mean", val_disc_agent_logit.mean().item(), iteration)
            writer.add_scalar("val/demo_logit_mean", val_disc_demo_logit.mean().item(), iteration)
            
            # Component-wise statistics
            for i, name in enumerate(component_names):
                # Demo statistics
                demo_component = demo_batch[:, i]
                writer.add_scalar(f"components/demo_{name}_mean", demo_component.mean().item(), iteration)
                writer.add_scalar(f"components/demo_{name}_std", demo_component.std().item(), iteration)
                
                # Agent statistics
                agent_component = agent_batch[:, i]
                writer.add_scalar(f"components/agent_{name}_mean", agent_component.mean().item(), iteration)
                writer.add_scalar(f"components/agent_{name}_std", agent_component.std().item(), iteration)
                
                # Difference
                mean_diff = abs(demo_component.mean().item() - agent_component.mean().item())
                writer.add_scalar(f"components/diff_{name}_mean_abs", mean_diff, iteration)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Save discriminator and normalizer
    torch.save(discriminator.state_dict(), f"{args.log_dir}/discriminator.pth")
    torch.save({
        'normalizer': disc_normalizer.state_dict()
    }, f"{args.log_dir}/disc_components.pth")
    
    print(f"\n✓ Saved discriminator to: {args.log_dir}/discriminator.pth")
    print(f"✓ Saved normalizer to: {args.log_dir}/disc_components.pth")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    with torch.no_grad():
        # Evaluate on TRAIN set
        norm_demo_train = disc_normalizer.normalize(demo_train)
        norm_agent_train = disc_normalizer.normalize(agent_train)
        
        disc_demo_train_scores = discriminator(norm_demo_train).squeeze(-1)
        disc_agent_train_scores = discriminator(norm_agent_train).squeeze(-1)
        
        train_demo_acc = (disc_demo_train_scores > 0.0).float().mean().item()
        train_agent_acc = (disc_agent_train_scores < 0.0).float().mean().item()
        
        # Evaluate on VALIDATION set
        norm_demo_val = disc_normalizer.normalize(demo_val)
        norm_agent_val = disc_normalizer.normalize(agent_val)
        
        disc_demo_val_scores = discriminator(norm_demo_val).squeeze(-1)
        disc_agent_val_scores = discriminator(norm_agent_val).squeeze(-1)
        
        val_demo_acc = (disc_demo_val_scores > 0.0).float().mean().item()
        val_agent_acc = (disc_agent_val_scores < 0.0).float().mean().item()
        
        print(f"\nFinal Accuracy on TRAIN Set ({len(demo_train):,} samples):")
        print(f"  Demo accuracy: {train_demo_acc:.3f}")
        print(f"  Agent accuracy: {train_agent_acc:.3f}")
        print(f"  Overall accuracy: {(train_demo_acc + train_agent_acc) / 2:.3f}")
        
        print(f"\nFinal Accuracy on VALIDATION Set ({len(demo_val):,} samples):")
        print(f"  Demo accuracy: {val_demo_acc:.3f}")
        print(f"  Agent accuracy: {val_agent_acc:.3f}")
        print(f"  Overall accuracy: {(val_demo_acc + val_agent_acc) / 2:.3f}")
        
        print(f"\nFinal Logit Statistics (TRAIN):")
        print(f"  Demo logits: mean={disc_demo_train_scores.mean():.3f}, std={disc_demo_train_scores.std():.3f}")
        print(f"  Agent logits: mean={disc_agent_train_scores.mean():.3f}, std={disc_agent_train_scores.std():.3f}")
        print(f"  Separation: {disc_demo_train_scores.mean() - disc_agent_train_scores.mean():.3f}")
        
        print(f"\nFinal Logit Statistics (VALIDATION):")
        print(f"  Demo logits: mean={disc_demo_val_scores.mean():.3f}, std={disc_demo_val_scores.std():.3f}")
        print(f"  Agent logits: mean={disc_agent_val_scores.mean():.3f}, std={disc_agent_val_scores.std():.3f}")
        print(f"  Separation: {disc_demo_val_scores.mean() - disc_agent_val_scores.mean():.3f}")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print(f"\nLogs saved to: {args.log_dir}")
    print("View results with: tensorboard --logdir " + args.log_dir)
    print("="*80 + "\n")
    
    writer.close()
