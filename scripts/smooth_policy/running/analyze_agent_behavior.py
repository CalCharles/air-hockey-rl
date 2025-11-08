import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
import gymnasium as gym

from airhockey import AirHockeyEnv
from scripts.smooth_policy.agent import Agent


@dataclass
class AnalysisArgs:
    model_path: str = "runs/finetune/checkpoint_70/iterative_smoothing_model.pth"
    config_path: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    n_episodes: int = 10
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_dir: str = "analysis_results"
    seed: int = 42


def load_agent_and_config(model_path: str, config_path: str, device: str) -> Tuple[Agent, dict]:
    """Load the trained agent and environment configuration."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create a dummy environment to get observation/action spaces
    env = AirHockeyEnv(config["air_hockey"])
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    # Load agent
    agent = Agent(envs).to(device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    else:
        print(f"Warning: Model path {model_path} does not exist. Using randomly initialized agent.")
    
    agent.eval()  # Set to evaluation mode
    
    return agent, config


def collect_trajectories(agent: Agent, config: dict, n_episodes: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run the agent on the environment and collect state-action trajectories."""
    env = AirHockeyEnv(config["air_hockey"])
    
    all_states = []
    all_actions = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_states = []
        episode_actions = []
        
        done = False
        step_count = 0
        max_steps = config["air_hockey"].get("max_timesteps", 300)
        
        while not done and step_count < max_steps:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action from agent
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
                action = action.cpu().numpy().flatten()
            
            # Store state and action
            episode_states.append(obs.copy())
            episode_actions.append(action.copy())
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        
        print(f"Episode {episode + 1}/{n_episodes} completed with {len(episode_states)} steps")
    
    return np.array(all_states), np.array(all_actions)


def plot_state_distributions(states: np.ndarray, save_dir: str):
    """Plot distributions of state variables."""
    n_state_dims = states.shape[1]
    
    # Create subplots for state distributions
    n_cols = min(4, n_state_dims)
    n_rows = (n_state_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_state_dims):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot histogram
        ax.hist(states[:, i], bins=50, alpha=0.7, density=True, edgecolor='black')
        ax.set_title(f'State Dimension {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_state_dims, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'state_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_action_distributions(actions: np.ndarray, save_dir: str):
    """Plot distributions of action variables."""
    n_action_dims = actions.shape[1]
    
    # Create subplots for action distributions
    n_cols = min(3, n_action_dims)
    n_rows = (n_action_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_action_dims):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot histogram
        ax.hist(actions[:, i], bins=50, alpha=0.7, density=True, edgecolor='black')
        ax.set_title(f'Action Dimension {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(actions[:, i])
        std_val = np.std(actions[:, i])
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, label=f'±1σ: {std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8)
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_action_dims, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_action_correlation_matrix(actions: np.ndarray, save_dir: str):
    """Plot correlation matrix between action dimensions."""
    if actions.shape[1] > 1:
        correlation_matrix = np.corrcoef(actions.T)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Action Correlation Matrix')
        plt.xlabel('Action Dimension')
        plt.ylabel('Action Dimension')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'action_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_action_trajectory_samples(actions: np.ndarray, save_dir: str, n_samples: int = 5):
    """Plot sample action trajectories over time."""
    n_action_dims = actions.shape[1]
    trajectory_length = min(500, len(actions))  # Show first 500 steps or all if fewer
    
    fig, axes = plt.subplots(n_action_dims, 1, figsize=(12, 2 * n_action_dims))
    if n_action_dims == 1:
        axes = [axes]
    
    for i in range(n_action_dims):
        axes[i].plot(actions[:trajectory_length, i], alpha=0.8, linewidth=1)
        axes[i].set_title(f'Action Dimension {i} Over Time')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Action Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_statistics(states: np.ndarray, actions: np.ndarray, save_dir: str):
    """Generate and save summary statistics."""
    stats = {
        'states': {
            'shape': states.shape,
            'mean': np.mean(states, axis=0).tolist(),
            'std': np.std(states, axis=0).tolist(),
            'min': np.min(states, axis=0).tolist(),
            'max': np.max(states, axis=0).tolist(),
        },
        'actions': {
            'shape': actions.shape,
            'mean': np.mean(actions, axis=0).tolist(),
            'std': np.std(actions, axis=0).tolist(),
            'min': np.min(actions, axis=0).tolist(),
            'max': np.max(actions, axis=0).tolist(),
        }
    }
    
    # Save statistics to file
    with open(os.path.join(save_dir, 'summary_statistics.yaml'), 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total timesteps collected: {len(states)}")
    print(f"State dimensions: {states.shape[1]}")
    print(f"Action dimensions: {actions.shape[1]}")
    print(f"\nAction statistics:")
    for i in range(actions.shape[1]):
        print(f"  Dimension {i}: mean={np.mean(actions[:, i]):.3f}, "
              f"std={np.std(actions[:, i]):.3f}, "
              f"range=[{np.min(actions[:, i]):.3f}, {np.max(actions[:, i]):.3f}]")


def main():
    parser = argparse.ArgumentParser(description='Analyze agent behavior by collecting and plotting state-action distributions')
    parser.add_argument('--model-path', type=str, default="runs/finetune/checkpoint_70/iterative_smoothing_model.pth",
                       help='Path to the trained model')
    parser.add_argument('--config-path', type=str, default="scripts/smooth_policy/configs/puck_juggle/velocity.yaml",
                       help='Path to the environment config')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                       help='Device to run on')
    parser.add_argument('--save-dir', type=str, default="analysis_results",
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading agent and configuration...")
    agent, config = load_agent_and_config(args.model_path, args.config_path, args.device)
    
    print(f"Collecting trajectories from {args.n_episodes} episodes...")
    states, actions = collect_trajectories(agent, config, args.n_episodes, args.device)
    
    print("Generating plots and statistics...")
    plot_state_distributions(states, args.save_dir)
    plot_action_distributions(actions, args.save_dir)
    plot_action_correlation_matrix(actions, args.save_dir)
    plot_action_trajectory_samples(actions, args.save_dir)
    generate_summary_statistics(states, actions, args.save_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.save_dir}")
    print("Generated files:")
    print("  - state_distributions.png")
    print("  - action_distributions.png")
    print("  - action_correlation_matrix.png")
    print("  - action_trajectories.png")
    print("  - summary_statistics.yaml")


if __name__ == "__main__":
    main()
