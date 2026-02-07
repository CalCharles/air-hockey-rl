from scripts.utils import save_task_gif
import torch
import argparse
import yaml
import os
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.pid_controller_policy.agent import Agent
import gymnasium as gym
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from scripts.utils import save_tensorboard_plots

def evaluate_agent(model_path, save_dir, air_hockey_params, air_hockey_config_path=None, n_eps=5, n_gifs=3, base_reward_scaling=1.0):
    # Create environment to get dimensions
    envs = gym.vector.SyncVectorEnv([lambda : AirHockeyEnv(air_hockey_params)])
    
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    
    # Agent expects augmented observation [obs, target_position]
    augmented_obs_dim = obs_dim + act_dim
    
    model = Agent(obs_dim=augmented_obs_dim, act_dim=act_dim)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    env = envs.envs[0]
    renderer = AirHockeyRenderer(env)

    env.set_base_reward_scaling(base_reward_scaling)

    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Modified save_task_gif to handle PID controller with target positions
    def save_pid_task_gif(n_eps, n_gifs, env, model, renderer, save_dir):
        """
        Modified version that maintains target positions across episodes.
        """
        episode_rewards = []
        episode_successes = []
        
        for ep in range(n_eps):
            obs, _ = env.reset()
            # Initialize target position to current paddle position
            target_pos = obs[:act_dim].copy()
            
            done = False
            episode_reward = 0
            frames = []
            
            while not done:
                # Create augmented observation
                augmented_obs = np.concatenate([obs, target_pos])
                
                # Get delta target from agent
                with torch.no_grad():
                    delta_target = model(torch.FloatTensor(augmented_obs))
                    if isinstance(delta_target, torch.Tensor):
                        delta_target = delta_target.cpu().numpy()
                    if delta_target.ndim > 1:
                        delta_target = delta_target.squeeze(0)
                
                # Update target position
                target_pos = target_pos + delta_target
                
                # Step with target position as action
                obs, reward, terminated, truncated, info = env.step(target_pos)
                done = terminated or truncated
                episode_reward += reward
                
                # Render frame if we're saving this episode
                if ep < n_gifs:
                    frame = renderer.render()
                    frames.append(frame)
            
            episode_rewards.append(episode_reward)
            episode_successes.append(1.0 if info.get('success', False) else 0.0)
            
            # Save gif for first n_gifs episodes
            if ep < n_gifs:
                import imageio
                gif_path = os.path.join(save_dir, f"episode_{ep}.gif")
                imageio.mimsave(gif_path, frames, fps=20)
                print(f"Saved episode {ep} to {gif_path}")
        
        # Print statistics
        print(f"\nEvaluation Results ({n_eps} episodes):")
        print(f"  Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"  Success Rate: {np.mean(episode_successes):.2%}")
        print(f"  Min Reward: {np.min(episode_rewards):.2f}")
        print(f"  Max Reward: {np.max(episode_rewards):.2f}")
        
        # Save statistics to file
        stats_path = os.path.join(save_dir, "eval_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Evaluation Results ({n_eps} episodes):\n")
            f.write(f"  Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}\n")
            f.write(f"  Success Rate: {np.mean(episode_successes):.2%}\n")
            f.write(f"  Min Reward: {np.min(episode_rewards):.2f}\n")
            f.write(f"  Max Reward: {np.max(episode_rewards):.2f}\n")
    
    save_pid_task_gif(n_eps, n_gifs, env, model, renderer, save_dir)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a PID controller policy model.')

    parser.add_argument('--use-parent-log-dir', type=bool, default=False, help='Use the parent log directory to save the evaluation gifs and tensorboard plots.')
    parser.add_argument('--parent-log-dir', type=str, default="runs/pid_training", help='Path to the parent log directory.')

    # optional arguments if use-parent-log-dir is False
    parser.add_argument('--model', type=str, default="runs/pid_training/model.pth", help='Path to the model to evaluate.')
    parser.add_argument('--save-dir', type=str, default="runs/pid_training/eval", help='Path to save the evaluation gifs to.')
    parser.add_argument('--config-path', type=str, default="runs/pid_training/config.yaml", help='Path to the config file.')
    parser.add_argument('--log-dir', type=str, default="runs/pid_training", help='Path to the tensorboard log directory.')

    # arguments for how many episodes and gifs to save
    parser.add_argument('--n-eps', type=int, default=4, help='Number of episodes to evaluate.')
    parser.add_argument('--n-gifs', type=int, default=3, help='Number of gifs to save.')

    # how much base reward scaling to use
    parser.add_argument('--base-reward-scaling', type=float, default=1.0, help='Base reward scaling to use.')
    # whether to loop through base reward scaling values
    parser.add_argument('--loop-base-reward-scaling', type=bool, default=False, help='Whether to loop through base reward scaling values.')

    args = parser.parse_args()
    
    if args.use_parent_log_dir:
        args.model = os.path.join(args.parent_log_dir, "model.pth")
        args.save_dir = os.path.join(args.parent_log_dir, "eval")
        args.config_path = os.path.join(args.parent_log_dir, "config.yaml")
        args.log_dir = args.parent_log_dir
    
    air_hockey_config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    air_hockey_params = air_hockey_config['air_hockey']
    log_dir = args.log_dir

    if args.loop_base_reward_scaling:
        # save config file into the save directory
        # make sure save directory exists
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "config.yaml"), "w") as f:
            yaml.dump(air_hockey_config, f)
        import numpy as np
        base_reward_scaling_values = np.linspace(0.1, 1.0, 10)
        for base_reward_scaling in base_reward_scaling_values:
            save_dir = os.path.join(args.save_dir, f"base_reward_scaling_{base_reward_scaling: .2f}")
            os.makedirs(save_dir, exist_ok=True)
            evaluate_agent(args.model, save_dir, air_hockey_params, air_hockey_config_path=args.config_path, n_eps=args.n_eps, n_gifs=args.n_gifs, base_reward_scaling=base_reward_scaling)
    else:
        evaluate_agent(args.model, args.save_dir, air_hockey_params, air_hockey_config_path=args.config_path, n_eps=args.n_eps, n_gifs=args.n_gifs, base_reward_scaling=args.base_reward_scaling)

