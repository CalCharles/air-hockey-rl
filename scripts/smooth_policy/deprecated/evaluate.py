from scripts.utils import save_task_gif
import torch
import argparse
import yaml
import os
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.agent import Agent
import gymnasium as gym
from tensorboard.backend.event_processing import event_accumulator
from scripts.utils import save_tensorboard_plots
from scripts.smooth_policy.running.run_single_episode import run_single_episode

def evaluate_iterative_smoothing(model_path, save_dir, air_hockey_params, air_hockey_config_path=None, n_eps=5, n_gifs=3, base_reward_scaling=1.0):
    # save an action plot
    if air_hockey_config_path is not None:
        run_single_episode(model_path, air_hockey_config_path, plot_dir=save_dir, max_steps=100)

    envs = gym.vector.SyncVectorEnv([lambda : AirHockeyEnv(air_hockey_params)])
    model = Agent(envs)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    env = envs.envs[0]
    renderer = AirHockeyRenderer(env)

    env.set_base_reward_scaling(base_reward_scaling) # forgot to do this

    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    save_task_gif(n_eps, n_gifs, env, model, renderer, save_dir)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate an iterative smoothing model.')

    parser.add_argument('--use-parent-log-dir', type=bool, default=False, help='Use the parent log directory to save the evaluation gifs and tensorboard plots.')
    parser.add_argument('--parent-log-dir', type=str, default="runs/iterative_smoothing", help='Path to the parent log directory.')

    # optional arguments if use-parent-log-dir is False
    parser.add_argument('--model', type=str, default="runs/iterative_smoothing/iterative_smoothing_model_0.pth", help='Path to the model to evaluate.')
    parser.add_argument('--save-dir', type=str, default="runs/iterative_smoothing/eval", help='Path to save the evaluation gifs to.')
    parser.add_argument('--config-path', type=str, default="runs/iterative_smoothing/default_config.yaml", help='Path to the config file.')
    parser.add_argument('--log-dir', type=str, default="runs/iterative_smoothing", help='Path to the tensorboard log directory.')

    # arguments for how many episodes and gifs to save
    parser.add_argument('--n-eps', type=int, default=4, help='Number of episodes to evaluate.')
    parser.add_argument('--n-gifs', type=int, default=3, help='Number of gifs to save.')

    # how much base reward scaling to use
    parser.add_argument('--base-reward-scaling', type=float, default=1.0, help='Base reward scaling to use.')
    # whether to loop through base reward scaling values
    parser.add_argument('--loop-base-reward-scaling', type=bool, default=False, help='Whether to loop through base reward scaling values.')

    args = parser.parse_args()
    
    if args.use_parent_log_dir:
        args.model = os.path.join(args.parent_log_dir, "iterative_smoothing_model_0.pth")
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
            evaluate_iterative_smoothing(args.model, save_dir, air_hockey_params, air_hockey_config_path=args.config_path, n_eps=args.n_eps, n_gifs=args.n_gifs, base_reward_scaling=base_reward_scaling)
    else:
        evaluate_iterative_smoothing(args.model, args.save_dir, air_hockey_params, air_hockey_config_path=args.config_path, n_eps=args.n_eps, n_gifs=args.n_gifs, base_reward_scaling=args.base_reward_scaling)


    # save_tensorboard_plots(log_dir, air_hockey_config, 
    # metrics=['charts/avg_episodic_return', 
    # 'charts/max_episodic_return', 
    # 'charts/min_episodic_return', 
    # 'charts/episodic_return', 
    # 'losses/approx_kl', 
    # 'losses/value_loss', 
    # 'losses/policy_loss', 
    # 'charts/avg_success_rate',
    # 'losses/action_loss'])