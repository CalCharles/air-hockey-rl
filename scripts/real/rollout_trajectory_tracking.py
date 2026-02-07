#!/usr/bin/env python3
"""
Rollout script for trajectory tracking policy on real air hockey environment.
Loads a demonstration trajectory from HDF5 file and tracks it by matching
current position to closest trajectory position and executing the corresponding action.
"""

import torch
import argparse
import yaml
import os
import numpy as np
import h5py
import gymnasium as gym
from airhockey import AirHockeyEnv
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole


class TrajectoryTrackingPolicy:
    """
    Policy that tracks a demonstration trajectory by matching current position
    to the closest position in the trajectory and executing the corresponding action.
    """
    
    def __init__(self, trajectory_path, action_x_ratio=0.26, action_y_ratio=0.12, 
                 reach_threshold=0.02, final_reach_threshold=0.05):
        """
        Initialize the trajectory tracking policy.
        
        Args:
            trajectory_path (str): Path to HDF5 trajectory file
            action_x_ratio (float): Action scaling ratio for x dimension
            action_y_ratio (float): Action scaling ratio for y dimension
            reach_threshold (float): Distance threshold to consider initial position reached
            final_reach_threshold (float): Distance threshold to consider final position reached
        """
        self.action_x_ratio = action_x_ratio
        self.action_y_ratio = action_y_ratio
        self.reach_threshold = reach_threshold
        self.final_reach_threshold = final_reach_threshold
        
        # Load trajectory data
        self._load_trajectory(trajectory_path)
        
        self.reached_initial = False
        self.reached_final = False
        
        print(f"✓ Loaded trajectory from: {trajectory_path}")
        print(f"  - Number of timesteps: {len(self.positions)}")
        print(f"  - Initial position: {self.positions[0]}")
        print(f"  - Final position: {self.positions[-1]}")
        print(f"  - Reach threshold: {reach_threshold}")
        print(f"  - Final reach threshold: {final_reach_threshold}")
    
    def _load_trajectory(self, trajectory_path):
        """
        Load trajectory data from HDF5 file and extract positions and actions.
        
        The trajectory data contains 32 fields per timestep:
        - Fields 5-6: pose_x, pose_y (paddle position)
        - Fields 26-27: desired_x, desired_y (commanded position)
        
        Actions are computed as: (desired_pose - pose) / [action_x_ratio, action_y_ratio]
        """
        with h5py.File(trajectory_path, 'r') as f:
            train_vals = f['train_vals'][:]
        
        # Extract paddle positions (fields 5-6)
        self.positions = train_vals[:, 5:7].copy()
        
        # Extract desired positions (fields 26-27)
        desired_positions = train_vals[:, 26:28].copy()
        
        # Compute normalized actions
        # action = (desired_pose - pose) / [action_x_ratio, action_y_ratio]
        position_deltas = desired_positions - self.positions
        self.actions = position_deltas / np.array([self.action_x_ratio, self.action_y_ratio])
        
        # Clip actions to [-1, 1] range
        self.actions = np.clip(self.actions, -1.0, 1.0)
        
        print(f"  - Action range: x=[{self.actions[:, 0].min():.3f}, {self.actions[:, 0].max():.3f}], "
              f"y=[{self.actions[:, 1].min():.3f}, {self.actions[:, 1].max():.3f}]")
    
    def find_closest_trajectory_point(self, current_pos):
        """
        Find the index of the closest position in the trajectory to the current position.
        
        Args:
            current_pos (np.ndarray): Current paddle position [x, y]
            
        Returns:
            int: Index of closest trajectory point
        """
        # Compute Euclidean distances to all trajectory points
        distances = np.linalg.norm(self.positions - current_pos, axis=1)
        
        # Return index of minimum distance
        return np.argmin(distances)
    
    def get_action(self, current_pos):
        """
        Get the action to execute based on current position.
        
        Phase 1: If initial position not reached, output action toward initial position
        Phase 2: Once initial position reached, match to closest trajectory point
        
        Args:
            current_pos (np.ndarray): Current paddle position [x, y]
            
        Returns:
            np.ndarray: Action to execute [delta_x, delta_y] in [-1, 1] range
            str: Phase description for logging
        """
        # Phase 1: Reach initial position
        if not self.reached_initial:
            distance_to_initial = np.linalg.norm(current_pos - self.positions[0])
            
            if distance_to_initial < self.reach_threshold:
                self.reached_initial = True
                print(f"\n✓ Reached initial position (distance: {distance_to_initial:.4f}m)")
                print(f"  Starting trajectory tracking...\n")
                # Return the first action from trajectory
                return self.actions[0], "tracking"
            else:
                # Compute action toward initial position
                position_delta = self.positions[0] - current_pos
                action = position_delta / np.array([self.action_x_ratio, self.action_y_ratio])
                action = np.clip(action, -1.0, 1.0)
                return action, f"reaching_initial (dist: {distance_to_initial:.4f}m)"
        
        # Phase 2: Track trajectory by matching to closest point
        closest_idx = self.find_closest_trajectory_point(current_pos)
        
        # Check if we've reached the final position
        distance_to_final = np.linalg.norm(current_pos - self.positions[-1])
        if distance_to_final < self.final_reach_threshold and not self.reached_final:
            self.reached_final = True
            print(f"\n✓ Reached final position (distance: {distance_to_final:.4f}m)")
            print(f"  Trajectory tracking complete!\n")
        
        return self.actions[closest_idx], f"tracking (idx: {closest_idx}/{len(self.positions)-1})"


def run_policy(air_hockey_cfg, policy, obs_type='pos'):
    """
    Run the trajectory tracking policy on the real air hockey environment.
    
    Args:
        air_hockey_cfg (dict): Full air hockey configuration
        policy (TrajectoryTrackingPolicy): Loaded trajectory tracking policy
        obs_type (str): Observation type ('pos', 'vel', 'history', etc.)
    """
    air_hockey_params = air_hockey_cfg['air_hockey'].copy()
    air_hockey_params['seed'] = 42
    air_hockey_params['max_timesteps'] = 500  # Increase for trajectory tracking
    
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
    print("Starting trajectory tracking on real air hockey table")
    print("="*60)
    print("Controls:")
    print("  'y' - Reset and SAVE trajectory")
    print("  'q' - Reset without saving")
    print("="*60 + "\n")
    
    with NonBlockingConsole() as nbc:
        delay_counter = 0
        step_count = 0
        
        while True:
            # Get current paddle position from state
            state_dict = eval_env.simulator.get_current_state()
            current_pos = np.array(state_dict['paddles']['paddle_ego']['position'])
            
            # Get action from policy
            action, phase = policy.get_action(current_pos)
            
            # Apply delay at the start (zeroed actions for first 10 steps)
            if delay_counter < 10:
                action = action * 0.0
                phase = "delay"
            else:
                action = np.clip(action, -1.0, 1.0)
            
            delay_counter += 1
            step_count += 1
            
            # Print action and observation periodically
            if step_count % 10 == 0:
                print(f"Step {step_count}: {phase}")
                print(f"  Current pos: [{current_pos[0]:.4f}, {current_pos[1]:.4f}]")
                print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}]")
            
            # Step the environment
            action_np = np.array(action, dtype=np.float32)
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
                policy.reached_initial = False
                policy.reached_final = False
            elif key == 'q':
                print(f"\n[RESET] Resetting environment without saving")
                eval_env.reset(seed=None, write_traj=False)
                delay_counter = 0
                step_count = 0
                policy.reached_initial = False
                policy.reached_final = False


def main():
    parser = argparse.ArgumentParser(
        description='Run trajectory tracking policy on real air hockey environment.'
    )
    parser.add_argument(
        '--cfg', 
        type=str, 
        default='configs/real_configs/rollout_config.yaml',
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--trajectory', 
        type=str, 
        required=True,
        help='Path to the demonstration trajectory HDF5 file'
    )
    parser.add_argument(
        '--obs-type', 
        type=str, 
        default='pos',
        help='Observation type (pos, vel, history, etc.)'
    )
    parser.add_argument(
        '--action-x-ratio', 
        type=float, 
        default=0.26,
        help='Action scaling ratio for x dimension'
    )
    parser.add_argument(
        '--action-y-ratio', 
        type=float, 
        default=0.12,
        help='Action scaling ratio for y dimension'
    )
    parser.add_argument(
        '--reach-threshold', 
        type=float, 
        default=0.02,
        help='Distance threshold (m) to consider initial position reached'
    )
    parser.add_argument(
        '--final-threshold', 
        type=float, 
        default=0.05,
        help='Distance threshold (m) to consider final position reached'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.cfg):
        print(f"Error: Configuration file not found: {args.cfg}")
        return
    
    with open(args.cfg, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    
    print(f"✓ Loaded config from: {args.cfg}")
    
    # Check if trajectory file exists
    if not os.path.exists(args.trajectory):
        print(f"Error: Trajectory file not found: {args.trajectory}")
        return
    
    # Load the trajectory tracking policy
    policy = TrajectoryTrackingPolicy(
        args.trajectory,
        action_x_ratio=args.action_x_ratio,
        action_y_ratio=args.action_y_ratio,
        reach_threshold=args.reach_threshold,
        final_reach_threshold=args.final_threshold
    )
    
    # Run the policy
    run_policy(air_hockey_cfg, policy, obs_type=args.obs_type)


if __name__ == "__main__":
    main()
