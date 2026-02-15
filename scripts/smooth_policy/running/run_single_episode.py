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
from scripts.smooth_policy.deprecated.agent import Agent
from scripts.domain_adaptation.encode_env_params import assign_values
import gymnasium as gym

#Need to add the reset_from_state vector here


def run_single_episode(model_path, param_vector, param_names, action_sequence, 
                       base_config, render=False, device="cpu", ground_truth = True, 
                       state_vector = [], traj_length = 0, save_plot=True, plot_dir=None):
    """
    Load an agent from model_path and run it for one episode.
    
    Args:
        model_path (str): Path to the saved model state dict
        param_vector (dict) : keys are the names of the parameters, values are their corresponding values
        config_path (str): Path to the config YAML file
        render (bool): Whether to render the episode
        device (str): Device to run the model on
        save_plot (bool): Whether to save action plot
        plot_dir (str): Directory to save the plot (if None, saves in current directory)
    
    Returns:
        dict: Episode results including return, length, success, and recorded data
    """

    #In the observation vector, the paddle position is in indices 0 and 1 and the puck position is in indices 4 and 5
    if not ground_truth:
        new_config = assign_values(param_vector, param_names, base_config)
        env = AirHockeyEnv(new_config)
        envs = gym.vector.SyncVectorEnv([lambda: env])
        if state_vector is not None:
            #Starts in the middle of the trajectory to evaluate
            obs, _ = env.reset_from_state(state_vector)
    else:
        new_config = base_config
        env = AirHockeyEnv(base_config)
        envs = gym.vector.SyncVectorEnv([lambda: env])
        obs, _ = env.reset()
    
    agent = Agent(envs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    states = []
    actions = []
    rewards = []
    next_states = []
    hits = []
    
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    episode_return = 0.0
    episode_length = 0
    done = False
    # previous while loop condition: (ground_truth and not done) or (not ground_truth and (episode_length < traj_length))
    while (ground_truth and not done) or (not ground_truth and not done and episode_length < traj_length): # or episode_length < len(action_sequence)
        states.append(obs.cpu().numpy().copy())
        if ground_truth:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
            
            actions.append(action.copy())
        else:
            action = action_sequence[episode_length]
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        rewards.append(reward)
        next_states.append(next_obs.cpu().numpy().copy())

        
        obs = next_obs

        paddle_pos = obs[0:2]
        puck_pos = obs[4:6]
        puck_paddle_distance = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        in_close_distance = env.puck_radius + env.paddle_radius + 0.02
        #taking care of the wall collisions
        wall_distance_boundaries = ((env.width/2) - env.puck_radius)
        wall_hit_threshold = 0.01
        if puck_paddle_distance <= in_close_distance or (abs(puck_pos[1])) + wall_hit_threshold >= wall_distance_boundaries:
            hits.append(1)
        else:
            hits.append(0)



        episode_return += reward
        episode_length += 1

        #for testing only
        # done = False

        #for simulation
        done = terminated or truncated
        if done:
            break

        if render:
            env.render()
    success = info.get('success', False)
    
    states = np.array(states)
    actions = np.array(actions) if ground_truth else np.array(action_sequence)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    
    # Create action plot if requested
    # if save_plot and len(actions) > 0:
    #     create_action_plot(actions, model_path, plot_dir)
    
    results = {
        'episode_return': episode_return,
        'episode_length': episode_length,
        'success': success,
        'observations': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'new_config': new_config,
        'hits_array': hits
    }
    
    # print(f"Episode completed!")
    # print(f"Return: {episode_return:.2f}")
    # print(f"Length: {episode_length}")
    # print(f"Success: {success}")
    # print(f"Recorded {len(actions)} action steps")
    
    return results

def run_single_episode_test(model_path, param_vector, param_names, action_sequence, 
                       base_config, render=False, device="cpu", ground_truth = True, 
                       state_vector = None, traj_length = 0, save_plot=True, plot_dir=None):
    """
    Load an agent from model_path and run it for one episode.
    
    Args:
        model_path (str): Path to the saved model state dict
        param_vector (dict) : keys are the names of the parameters, values are their corresponding values
        config_path (str): Path to the config YAML file
        render (bool): Whether to render the episode
        device (str): Device to run the model on
        save_plot (bool): Whether to save action plot
        plot_dir (str): Directory to save the plot (if None, saves in current directory)
    
    Returns:
        dict: Episode results including return, length, success, and recorded data
    """

    #In the observation vector, the paddle position is in indices 0 and 1 and the puck position is in indices 4 and 5
    if not ground_truth:
        new_config = assign_values(param_vector, param_names, base_config)
        env = AirHockeyEnv(new_config)
        envs = gym.vector.SyncVectorEnv([lambda: env])
        obs, _ = env.reset()
        if state_vector is not None:
            #Starts in the middle of the trajectory to evaluate
            obs, _ = env.reset_from_state(state_vector)
    else:
        new_config = base_config
        env = AirHockeyEnv(base_config)
        envs = gym.vector.SyncVectorEnv([lambda: env])
        obs, _ = env.reset()
    
    agent = Agent(envs).to(device)
    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    states = []
    actions = []
    rewards = []
    next_states = []
    hits = []
    
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    episode_return = 0.0
    episode_length = 0
    done = False
    while (ground_truth and not done) or (not ground_truth and not done and episode_length < len(action_sequence)): # or episode_length < len(action_sequence)
        states.append(obs.cpu().numpy().copy())
        if ground_truth:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
            
            actions.append(action.copy())
        else:
            action = action_sequence[episode_length]
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        rewards.append(reward)
        next_states.append(next_obs.cpu().numpy().copy())

        
        obs = next_obs

        paddle_pos = obs[0:2]
        puck_pos = obs[4:6]
        puck_paddle_distance = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        in_close_distance = env.puck_radius + env.paddle_radius + 0.02
        #taking care of the wall collisions
        wall_distance_boundaries = ((env.width/2) - env.puck_radius)
        wall_hit_threshold = 0.01
        if puck_paddle_distance <= in_close_distance or (abs(puck_pos[1])) + wall_hit_threshold >= wall_distance_boundaries:
            hits.append(1)
        else:
            hits.append(0)



        episode_return += reward
        episode_length += 1

        #for testing only
        # done = False

        #for simulation
        done = terminated or truncated
        if done:
            break

        if render:
            env.render()
    success = info.get('success', False)
    
    states = np.array(states)
    actions = np.array(actions) if ground_truth else np.array(action_sequence)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    
    # Create action plot if requested
    # if save_plot and len(actions) > 0:
    #     create_action_plot(actions, model_path, plot_dir)
    
    results = {
        'episode_return': episode_return,
        'episode_length': episode_length,
        'success': success,
        'observations': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'new_config': new_config,
        'hits_array': hits
    }
    
    # print(f"Episode completed!")
    # print(f"Return: {episode_return:.2f}")
    # print(f"Length: {episode_length}")
    # print(f"Success: {success}")
    # print(f"Recorded {len(actions)} action steps")
    
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
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first action dimension
    ax1.plot(timesteps, actions[:, 0], 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Action Value')
    ax1.set_title('Action Dimension 1')
    ax1.grid(True, alpha=0.3)
    
    # Plot second action dimension
    ax2.plot(timesteps, actions[:, 1], 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Action Value')
    ax2.set_title('Action Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    fig.suptitle(f'Actions Over Time - {model_name}', fontsize=14)
    
    plt.tight_layout()
    
    # Determine save path
    plot_filename = f"actions_plot_{model_name.replace('.pth', '')}.png"
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, plot_filename)
    else:
        plot_path = plot_filename
    
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
