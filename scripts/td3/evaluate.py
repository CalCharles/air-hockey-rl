"""Checkpoint evaluation: load a trained policy, run rollouts, and save GIFs.

See notes/docs/training/architecture.md for training entrypoints.
"""

from scripts.utils import save_task_gif
import torch
import argparse
import yaml
import os
import cv2
import imageio
import tqdm
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.td3.eval_utils import (
    augment_policy_observation,
    build_policy_env_view,
    load_policy_for_evaluation,
)
import gymnasium as gym
from tensorboard.backend.event_processing import event_accumulator
from scripts.utils import save_tensorboard_plots
import numpy as np
from types import SimpleNamespace
from scripts.transformer.context_encoder import ContextEncoder
from scripts.transformer.history_buffer import HistoryBuffer


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


def _save_task_gif_with_last_action(
    n_eps_viz, n_gifs, env_test, policy, renderer, log_dir, action_dim, use_last_action_in_policy_state,
    transformer=None, history_buf=None, use_history=False,
):
    env_test.max_timesteps = 200
    for gif_idx in range(n_gifs):
        frames = []
        for _ in tqdm.tqdm(range(n_eps_viz)):
            obs, _ = env_test.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            last_action = torch.zeros((1, action_dim), dtype=torch.float32)
            done = False
            rew = 0
            cum_rew = 0
            while not done:
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (0, 0, 0)
                line_type = 2
                text_position = (frame.shape[1] - 150, 30)
                cv2.putText(frame, f"Reward: {rew:.2f}", text_position, font, font_scale, font_color, line_type)
                text_position = (frame.shape[1] - 150, 60)
                cv2.putText(frame, f"Return: {cum_rew:.2f}", text_position, font, font_scale, font_color, line_type)
                frames.append(frame)


                history_buf.add(obs)

                if use_history:
                    
                    state_history = history_buf.sample()

                    if transformer is not None:
                        with torch.no_grad():
                            context = transformer(state_history)  # (1, context_dim)
                    
                    else:
                        context = state_history.view(1, -1)
                    
                    
                    obs_with_context = torch.cat([obs_tensor.unsqueeze(0), context], dim=-1)
                    policy_obs = augment_policy_observation(
                        obs_with_context, last_action, use_last_action_in_policy_state
                    )
                else:
                    policy_obs = augment_policy_observation(
                        obs_tensor.unsqueeze(0), last_action, use_last_action_in_policy_state
                    )


                action = policy(policy_obs).numpy().squeeze()
                obs, rew, term, trunc, _ = env_test.step(action)
                done = term or trunc
                cum_rew += rew
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
                if done:
                    last_action.zero_()

        gif_savepath = os.path.join(log_dir, f'eval_{gif_idx}.gif')
        fps = 20
        imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=int(1000 * 1 / fps))


def evaluate_agent(
    model_path,
    save_dir,
    air_hockey_params,
    air_hockey_config_path=None,
    n_eps=5,
    n_gifs=3,
    base_reward_scaling=1.0,
    reference_states=None,
    ref_max_episode_steps=None,
    action_scale=0.02,
    agent_hidden_layer_size=64,
    agent_num_hidden_layers=2,
    agent_hidden_size=None,
    use_last_action_in_policy_state=False,
    policy_type=None,

    HISTORY_ENTRY_DIM=0,
    use_transformer=False,
    use_history=False,
    context_vector_dim=8,
    context_len=7,
):
    if agent_hidden_size is not None:
        agent_hidden_layer_size = int(agent_hidden_size)
    if agent_num_hidden_layers < 1:
        raise ValueError("agent_num_hidden_layers must be >= 1.")

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
    action_dim = int(np.prod(envs.single_action_space.shape))

    if use_history:

        raw_obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        augmented_obs_dim = raw_obs_dim + act_dim if use_last_action_in_policy_state else raw_obs_dim
        
        if use_transformer:
            augmented_obs_dim += context_vector_dim
        else:
            augmented_obs_dim += (context_len * HISTORY_ENTRY_DIM)

        policy_env_view = SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(augmented_obs_dim,), dtype=np.float32
            ),
            single_action_space=envs.single_action_space,
        )

    else:
        policy_env_view = build_policy_env_view(envs, use_last_action_in_policy_state)


    model = load_policy_for_evaluation(
        model_path=model_path,
        policy_env_view=policy_env_view,
        action_scale=action_scale,
        agent_hidden_layer_size=agent_hidden_layer_size,
        agent_num_hidden_layers=agent_num_hidden_layers,
        policy_type=policy_type,
    )

    transformer = None

    history_buf = HistoryBuffer(
        context_len=context_len,
    )
    
    if use_transformer:
        
        # raw_obs_dim = int(np.prod(envs.single_observation_space.shape))

        # TODO: need to update obs_dim input
        transformer = ContextEncoder(
            obs_dim=HISTORY_ENTRY_DIM,
            context_dim=context_vector_dim,
            context_len=context_len,
        )
        transformer_path = os.path.join(os.path.dirname(model_path), "transformer.pth")
        if os.path.exists(transformer_path):
            transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
            transformer.eval()
        else:
            print(f"Warning: transformer.pth not found at {transformer_path}, using random weights")
        

    env = envs.envs[0]
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)

    env.set_base_reward_scaling(base_reward_scaling)

    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    if use_last_action_in_policy_state:
        _save_task_gif_with_last_action(
            n_eps,
            n_gifs,
            env,
            model,
            renderer,
            save_dir,
            action_dim=action_dim,
            use_last_action_in_policy_state=use_last_action_in_policy_state,
            transformer=transformer,
            history_buf=history_buf,
            use_history=use_history,
        )
    else:
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