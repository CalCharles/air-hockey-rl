from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from airhockey import AirHockeyEnv
from render import AirHockeyRenderer
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import argparse
import yaml
import os
import re
import time
import imageio
import cv2
import tqdm


def train_air_hockey_model(air_hockey_cfg):
    """
    Train an air hockey paddle model using stable baselines.

    This script loads the configuration file, creates an AirHockey2D environment,
    wraps the environment with necessary components, trains the model,
    and saves the trained model and environment statistics.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    env = AirHockeyEnv.from_dict(air_hockey_params)

    # check_env(env)
    def wrap_env(env):
        wrapped_env = Monitor(env) # needed for extracting eprewmean and eplenmean
        wrapped_env = DummyVecEnv([lambda: wrapped_env]) # Needed for all environments (e.g. used for multi-processing)
        wrapped_env = VecNormalize(wrapped_env) # probably something to try when tuning
        return wrapped_env

    check_env(env)
    env = wrap_env(env)
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['reward_type']:
        # SAC hyperparams:
        # Create 4 artificial transitions per real transitionair_hockey_simulator
        n_sampled_goal = 4
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
            ),
            learning_starts=10000,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=512,
            tensorboard_log=air_hockey_cfg['tb_log_dir']
            # device='cuda',
            # device="cuda"
            # policy_kwargs=dict(net_arch=[64, 64]),
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, 
                tensorboard_log=air_hockey_cfg['tb_log_dir'], 
                device="cpu", # cpu is actually faster!
                gamma=air_hockey_cfg['gamma']) 
    
    model.learn(total_timesteps=air_hockey_cfg['n_training_steps'],
                tb_log_name=air_hockey_cfg['tb_log_name'], 
                progress_bar=True)
    
    log_dir = air_hockey_cfg['tb_log_dir']
    os.makedirs(log_dir, exist_ok=True)
    # get log dir ending with highest number
    subdirs = [x for x in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, x))]
    subdirs.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
    log_dir = os.path.join(log_dir, subdirs[-1])
    
    # let's save model and vec normalize here too
    model_filepath = os.path.join(log_dir, air_hockey_cfg['model_save_filepath'])
    env_filepath = os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath'])
    # copy cfg to same folder
    cfg_filepath = os.path.join(log_dir, 'model_cfg.yaml')
    with open(cfg_filepath, 'w') as f:
        yaml.dump(air_hockey_cfg, f)

    model.save(model_filepath)
    env.save(env_filepath)
    
    # let's also evaluate the policy and save the results!
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    env_test = AirHockeyEnv.from_dict(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)
    
    env_test = DummyVecEnv([lambda : env_test])
    env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['reward_type']:
        model = SAC.load(model_filepath, env=env_test)
    else:
        model = PPO.load(model_filepath)

    # env_test.training = False
    # env_test.norm_reward = False
    
    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance={  # see below regarding this argument
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                                                event_accumulator.IMAGES: 0,
                                                event_accumulator.AUDIO: 0,
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.HISTOGRAMS: 0,
                                            })

    # Load all events from the directory
    ea.Reload()

    # uncomment below to see the tags in the tensorboard log file, then you can add them to metrics
    # print("Available tags: ", ea.Tags()['scalars'])
    
    metrics = [ 
        'rollout/ep_rew_mean', 
        'train/approx_kl', 
        'train/entropy_loss', 
        'train/learning_rate', 
        'train/loss', 
        'train/value_loss']

    def save_plot(metrics):
        # Create a 2x3 subplot
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Air Hockey Training Summary')

        # Flatten the axs array for easy iteration
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            if metric in ea.Tags()['scalars']:
                # Extract time steps and values for the metric
                times, step_nums, values = zip(*ea.Scalars(metric))

                # Plot on the i-th subplot
                axs[i].plot(step_nums, values, label=metric)
                axs[i].set_title(metric)
                axs[i].set_xlabel("Steps")
                axs[i].set_ylabel("Value")
                axs[i].legend()
            else:
                print(f"Metric {metric} not found in logs.")
                axs[i].set_title(f"{metric} (not found)")
                axs[i].set_xlabel("Steps")
                axs[i].set_ylabel("Value")

        # Adjust layout for better readability
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # let's save in same folder
        plot_fp = os.path.join(log_dir, 'training_summary.png')
        plt.savefig(plot_fp)
        plt.close()

    save_plot(metrics)

    obs = env_test.reset()
    start = time.time()
    done = False
    
    # first let's create some videos offline into gifs
    print("Saving gifs...(this will tqdm for EACH gif to save)")
    n_eps_viz = 5
    n_gifs = 5
    for gif_idx in range(n_gifs):
        frames = []
        for i in tqdm.tqdm(range(n_eps_viz)):
            obs = env_test.reset()
            done = False
            while not done:
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # decrease width to 160 but keep aspect ratio
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                frames.append(frame)
                action = model.predict(obs, deterministic=True)[0]
                obs, rew, done, info = env_test.step(action)
        gif_savepath = os.path.join(log_dir, f'eval_{gif_idx}.gif')
        def fps_to_duration(fps):
            return int(1000 * 1/fps)
        imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(30))
    
    # print('Running policy live...Ctrl+C twice to stop.')
    # for i in range(1000000):
    #     if i % 1000 == 0:
    #         print("fps", 1000 / (time.time() - start))
    #         start = time.time()
    #     # Draw the world
    #     renderer.render()
    #     action = model.predict(obs, deterministic=True)[0]
    #     obs, rew, done, info = env_test.step(action)
    #     if done:
    #         obs = env_test.reset()

    env_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    args = parser.parse_args()
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, 'configs', 'train_ppo.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    train_air_hockey_model(air_hockey_cfg)
