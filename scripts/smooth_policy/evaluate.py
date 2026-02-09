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
import numpy as np


class ReferenceStateWrapper(gym.Wrapper):
    """Wrapper that initializes paddle from demonstration states."""
    
    def __init__(self, env, reference_states):
        super().__init__(env)
        self.reference_states = reference_states  # numpy array [N, 4]
    
    def reset(self, **kwargs):
        # Sample random reference state [x, y, vx, vy]
        idx = np.random.randint(0, len(self.reference_states))
        ref_state = self.reference_states[idx]
        
        # Set reference state on underlying environment
        self.env.unwrapped._ref_paddle_state = (
            (float(ref_state[0]), float(ref_state[1])),  # pos
            (float(ref_state[2]), float(ref_state[3]))   # vel
        )
        
        return self.env.reset(**kwargs)

def evaluate_agent(model_path, save_dir, air_hockey_params, air_hockey_config_path=None, n_eps=5, n_gifs=3, base_reward_scaling=1.0, reference_states=None, ref_max_episode_steps=None, action_scale=0.02, agent_hidden_size=64):
    # save an action plot
    if air_hockey_config_path is not None:
        run_single_episode(model_path, air_hockey_config_path, plot_dir=save_dir, max_steps=100)

    # Override max_timesteps if reference state initialization is enabled
    eval_air_hockey_params = air_hockey_params.copy()
    if ref_max_episode_steps is not None:
        eval_air_hockey_params['max_timesteps'] = ref_max_episode_steps
    
    # Create environment factory function
    def make_eval_env():
        env = AirHockeyEnv(eval_air_hockey_params)
        # Wrap with reference state initialization if enabled
        if reference_states is not None:
            env = ReferenceStateWrapper(env, reference_states)
        return env
    
    envs = gym.vector.SyncVectorEnv([make_eval_env])
    model = Agent(envs, action_scale=action_scale, action_bias=0.0, hidden_size=agent_hidden_size)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    env = envs.envs[0]
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)

    env.set_base_reward_scaling(base_reward_scaling)

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
            evaluate_agent(args.model, save_dir, air_hockey_params, air_hockey_config_path=args.config_path, n_eps=args.n_eps, n_gifs=args.n_gifs, base_reward_scaling=base_reward_scaling)
    else:
        evaluate_agent(args.model, args.save_dir, air_hockey_params, air_hockey_config_path=args.config_path, n_eps=args.n_eps, n_gifs=args.n_gifs, base_reward_scaling=args.base_reward_scaling)


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