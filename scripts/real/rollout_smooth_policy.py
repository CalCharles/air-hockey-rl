#!/usr/bin/env python3
"""
Rollout script for smooth_policy Agent on real air hockey environment.
Loads a trained smooth policy model and runs it on the real air hockey table.
"""

import torch
import argparse
import yaml
import os
import numpy as np
import gymnasium as gym
from airhockey import AirHockeyEnv
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from scripts.smooth_policy.agent import Agent


def load_model(model_path, air_hockey_params, action_scale=0.02, hidden_size=64, device='cuda:0'):
    """
    Load the smooth policy Agent model from a checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        air_hockey_params (dict): Air hockey environment parameters
        action_scale (float): Action scaling parameter for the agent
        hidden_size (int): Hidden layer size of the agent network
        device (str): Device to run the model on
        
    Returns:
        Agent: Loaded agent model in eval mode
    """
    # Create a dummy environment to get observation/action space dimensions
    env = AirHockeyEnv(air_hockey_params)
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    # Create agent with specified parameters
    agent = Agent(envs, action_scale=action_scale, action_bias=0.0, hidden_size=hidden_size)
    
    # Load the model state dict
    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    agent = agent.to(device)
    agent.eval()
    
    print(f"✓ Loaded model from: {model_path}")
    print(f"  - Action scale: {action_scale}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Device: {device}")
    
    return agent


def run_policy(air_hockey_cfg, agent, obs_type='pos', device='cuda:0'):
    """
    Run the smooth policy agent on the real air hockey environment.
    
    Args:
        air_hockey_cfg (dict): Full air hockey configuration
        agent (Agent): Loaded agent model
        obs_type (str): Observation type ('pos', 'vel', 'history', etc.)
        device (str): Device the model is running on
    """
    air_hockey_params = air_hockey_cfg['air_hockey'].copy()
    air_hockey_params['seed'] = 42
    air_hockey_params['max_timesteps'] = 200
    
    # Create the real environment
    eval_env = AirHockeyEnv(air_hockey_params)
    
    # Get initial observation
    state_dict = eval_env.simulator.get_current_state()
    obs = get_observation_by_type(
        state_dict, 
        obs_type=obs_type, 
        puck_history=state_dict["pucks"][0]["history"]
    )
    
    print("\n" + "="*60)
    print("Starting rollout on real air hockey table")
    print("="*60)
    print("Controls:")
    print("  'y' - Reset and SAVE trajectory")
    print("  'q' - Reset without saving")
    print("="*60 + "\n")
    
    with NonBlockingConsole() as nbc:
        delay_counter = 0
        step_count = 0
        
        while True:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device).float()
            
            # Get action from agent (forward method returns action tensor)
            with torch.no_grad():
                action = agent(obs_tensor)
            
            # Apply delay at the start (zeroed actions for first 10 steps)
            if delay_counter < 10:
                action = action * 0.0
            else:
                action = action.clip(-1, 1)
            
            delay_counter += 1
            step_count += 1
            
            # Print action and observation periodically
            if step_count % 10 == 0:
                print(f"Step {step_count}: action = {action.squeeze().cpu().numpy()}")
            
            # Step the environment
            action_np = action.squeeze().detach().cpu().numpy()
            obs, reward, is_finished, truncated, info = eval_env.step(action_np)
            
            # Get next observation
            state_dict = eval_env.simulator.get_current_state()
            obs = get_observation_by_type(
                state_dict, 
                obs_type=obs_type, 
                puck_history=state_dict["pucks"][0]["history"]
            )
            
            # Check for keyboard input
            key = nbc.get_data()
            if key == 'y':
                print(f"\n[RESET] Saving trajectory and resetting environment")
                eval_env.reset(seed=None, write_traj=True)
                delay_counter = 0
                step_count = 0
            elif key == 'q':
                print(f"\n[RESET] Resetting environment without saving")
                eval_env.reset(seed=None, write_traj=False)
                delay_counter = 0
                step_count = 0


def main():
    parser = argparse.ArgumentParser(
        description='Run smooth_policy Agent on real air hockey environment.'
    )
    parser.add_argument(
        '--cfg', 
        type=str, 
        default='configs/real_configs/rollout_config.yaml',
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--obs-type', 
        type=str, 
        default='pos',
        help='Observation type (pos, vel, history, etc.)'
    )
    parser.add_argument(
        '--action-scale', 
        type=float, 
        default=0.02,
        help='Action scale parameter for the agent'
    )
    parser.add_argument(
        '--hidden-size', 
        type=int, 
        default=64,
        help='Hidden layer size of the agent network'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0',
        help='Device to run the model on (cuda:0, cpu, etc.)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.cfg):
        print(f"Error: Configuration file not found: {args.cfg}")
        return
    
    with open(args.cfg, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    
    print(f"✓ Loaded config from: {args.cfg}")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Load the agent model
    agent = load_model(
        args.model,
        air_hockey_cfg['air_hockey'],
        action_scale=args.action_scale,
        hidden_size=args.hidden_size,
        device=args.device
    )
    
    # Run the policy
    run_policy(air_hockey_cfg, agent, obs_type=args.obs_type, device=args.device)


if __name__ == "__main__":
    main()
