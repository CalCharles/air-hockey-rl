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

def evaluate_iterative_smoothing(model_path, save_dir, air_hockey_params, n_eps=5, n_gifs=3):
    envs = gym.vector.SyncVectorEnv([lambda : AirHockeyEnv(air_hockey_params)])
    model = Agent(envs)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    env = envs.envs[0]
    renderer = AirHockeyRenderer(env)

    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    save_task_gif(n_eps, n_gifs, env, model, renderer, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate an iterative smoothing model.')

    parser.add_argument('--use-parent-log-dir', type=bool, default=True, help='Use the parent log directory to save the evaluation gifs and tensorboard plots.')
    parser.add_argument('--parent-log-dir', type=str, default="runs/iterative_smoothing", help='Path to the parent log directory.')

    # optional arguments if use-parent-log-dir is False
    parser.add_argument('--model', type=str, default="runs/iterative_smoothing/iterative_smoothing_model_0.pth", help='Path to the model to evaluate.')
    parser.add_argument('--save-dir', type=str, default="runs/iterative_smoothing/eval", help='Path to save the evaluation gifs to.')
    parser.add_argument('--air-hockey-params', type=str, default="runs/iterative_smoothing/default_config.yaml", help='Path to the config file.')
    parser.add_argument('--log-dir', type=str, default="runs/iterative_smoothing", help='Path to the tensorboard log directory.')

    args = parser.parse_args()
    
    if args.use_parent_log_dir:
        args.model = os.path.join(args.parent_log_dir, "iterative_smoothing_model_0.pth")
        args.save_dir = os.path.join(args.parent_log_dir, "eval")
        args.air_hockey_params = os.path.join(args.parent_log_dir, "config.yaml")
        args.log_dir = args.parent_log_dir
    
    air_hockey_config = yaml.load(open(args.air_hockey_params, 'r'), Loader=yaml.FullLoader)
    air_hockey_params = air_hockey_config['air_hockey']
    log_dir = args.log_dir
    evaluate_iterative_smoothing(args.model, args.save_dir, air_hockey_params)

    save_tensorboard_plots(log_dir, air_hockey_config, 
    metrics=['charts/avg_episodic_return', 
    'charts/max_episodic_return', 
    'charts/min_episodic_return', 
    'charts/episodic_return', 
    'losses/approx_kl', 
    'losses/value_loss', 
    'losses/policy_loss', 
    'charts/avg_success_rate',
    'losses/action_loss'])