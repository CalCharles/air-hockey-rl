#!/usr/bin/env python3
"""
Simple script to load a trained agent and run it for one episode.
"""


"""
EXAMPLE USAGE:

python scripts/smooth_policy/run_single_episode.py 
--model-path runs/smoothing/caps_loss_regular_001/checkpoint_70/iterative_smoothing_model.pth
--config-path runs/smoothing/caps_loss_regular_001/config.yaml 
--plot-dir plots/actions

optionally add '--no-plot' to skip action plotting
"""

import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from airhockey import AirHockeyEnv
from scripts.smooth_policy.agent import Agent
import gymnasium as gym


def run_single_episode(model_path, config_path, render=False, device="cpu", save_plot=True, plot_dir=None, max_steps=200):
    """
    Load an agent from model_path and run it for one episode.
    
    Args:
        model_path (str): Path to the saved model state dict
        config_path (str): Path to the config YAML file
        render (bool): Whether to render the episode
        device (str): Device to run the model on
        save_plot (bool): Whether to save action plot
        plot_dir (str): Directory to save the plot (if None, saves in current directory)
    
    Returns:
        dict: Episode results including return, length, success, and recorded data
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    air_hockey_params = config['air_hockey']
    
    # Create environment
    env = AirHockeyEnv(air_hockey_params)
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    # Create and load agent
    agent = Agent(envs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    episode_length = 0
    while not episode_length >= max_steps:

        print("Running episode...")
        # Initialize data recording lists
        states = []
        actions = []
        rewards = []
        next_states = []
        
        # Run one episode
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        
        episode_return = 0.0
        episode_length = 0
        done = False
        while not done and episode_length < max_steps:
            # Record current state
            states.append(obs.cpu().numpy().copy())
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
            
            # Record action
            actions.append(action.copy())
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            
            # Record reward and next state
            rewards.append(reward)
            next_states.append(next_obs.cpu().numpy().copy())
            
            obs = next_obs
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
    
    # Extract final episode info
    success = info.get('success', False)
    
    # Convert lists to numpy arrays for easier handling
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    
    # Create action plot if requested
    if save_plot and len(actions) > 0:
        create_action_plot(actions, model_path, plot_dir)
    
    results = {
        'episode_return': episode_return,
        'episode_length': episode_length,
        'success': success,
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states
    }
    
    print(f"Episode completed!")
    print(f"Return: {episode_return:.2f}")
    print(f"Length: {episode_length}")
    print(f"Success: {success}")
    print(f"Recorded {len(actions)} action steps")
    
    return results


def create_action_plot(actions, model_path, plot_dir=None):
    """
    Create a plot showing the 2D actions over time.
    
    Args:
        actions (np.ndarray): Array of actions with shape (timesteps, action_dim)
        model_path (str): Path to model (used for plot title and filename)
        plot_dir (str): Directory to save the plot (if None, saves in current directory)
    """
    if actions.shape[1] < 2:
        print("Warning: Actions are not 2-dimensional, cannot create side-by-side plot")
        return
    
    timesteps = np.arange(len(actions))
    
    # Create figure with two rows and two columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Top row: original axis limits
    # First action dimension
    axs[0, 0].plot(timesteps, actions[:, 0], 'b-', linewidth=1.5, alpha=0.8)
    axs[0, 0].set_xlabel('Timestep')
    axs[0, 0].set_ylabel('Action Value')
    axs[0, 0].set_title('Action Dimension X')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_ylim(-10, 10)

    # Second action dimension
    axs[0, 1].plot(timesteps, actions[:, 1], 'r-', linewidth=1.5, alpha=0.8)
    axs[0, 1].set_xlabel('Timestep')
    axs[0, 1].set_ylabel('Action Value')
    axs[0, 1].set_title('Action Dimension Y')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_ylim(-10, 10)

    # Bottom row: -3 to 3 y-axis range
    # First action dimension, limited to [-3, 3]
    axs[1, 0].plot(timesteps, actions[:, 0], 'b-', linewidth=1.5, alpha=0.8)
    axs[1, 0].set_xlabel('Timestep')
    axs[1, 0].set_ylabel('Action Value')
    axs[1, 0].set_title('Action Dimension X (x ∈ [-3, 3])')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_ylim(-3, 3)

    # Second action dimension, limited to [-3, 3]
    axs[1, 1].plot(timesteps, actions[:, 1], 'r-', linewidth=1.5, alpha=0.8)
    axs[1, 1].set_xlabel('Timestep')
    axs[1, 1].set_ylabel('Action Value')
    axs[1, 1].set_title('Action Dimension Y (y ∈ [-3, 3])')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_ylim(-3, 3)


    # Add statistics for each action dimension
    # 1) mean of the absolute action values
    # 2) mean of the squared action values
    # 4) mean of the absolute difference between consecutive action values
    # 3) mean of the squared difference between consecutive action values

    print(f"Mean of absolute action values: {np.mean(np.abs(actions[:, 0]))}")
    print(f"Mean of squared action values: {np.mean(actions[:, 0] ** 2)}")
    print(f"Mean of absolute difference between consecutive action values: {np.mean(np.abs(actions[1:] - actions[:-1]))}")
    print(f"Mean of squared difference between consecutive action values: {np.mean((actions[1:] - actions[:-1]) ** 2)}")

    # put these statistics below the plots
    
    # Calculate statistics for action dimension 1 and round to 2 decimals
    mean_abs = np.round(np.mean(np.abs(actions[:, 0])), 2)
    mean_sq = np.round(np.mean(actions[:, 0] ** 2), 2)
    mean_abs_diff = np.round(np.mean(np.abs(actions[1:, 0] - actions[:-1, 0])), 2)
    mean_sq_diff = np.round(np.mean((actions[1:, 0] - actions[:-1, 0]) ** 2), 2)
    
    # Prepare the statistics string
    stats_text = (
        f"Mean of absolute action values: {mean_abs}\n"
        f"Mean of squared action values: {mean_sq}\n"
        f"Mean abs diff between consecutive actions: {mean_abs_diff}\n"
        f"Mean sq diff between consecutive actions: {mean_sq_diff}"
    )

    # Place the statistics below the plots in an empty space using the main figure (outside subplots)
    fig.text(0.5, 0.03, stats_text, ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add overall title
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    fig.suptitle(f'Actions Over Time - {model_name}', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # slightly shrink for suptitle

    # Determine save path
    plot_filename = f"actions_plot_{model_name.split('/')[-1].replace('.pth', '')}.png"
    plot_subpath = f"{model_path.replace('.pth', '')}"
    
    if plot_dir is not None:
        plot_dir = os.path.join(plot_dir, plot_subpath)
    else:
        plot_dir = plot_subpath

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, plot_filename)

    # Save plot
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Action plot saved as: {plot_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a single episode with a trained agent')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved model state dict')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to the config YAML file')
    parser.add_argument('--render', action='store_true',
                       help='Render the episode')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run the model on (cpu/cuda)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip creating action plot')
    parser.add_argument('--plot-dir', type=str, default=None,
                       help='Directory to save the action plot (default: current directory)')
    
    args = parser.parse_args()
    
    results = run_single_episode(
        model_path=args.model_path,
        config_path=args.config_path,
        render=args.render,
        device=args.device,
        save_plot=not args.no_plot,
        plot_dir=args.plot_dir
    )
