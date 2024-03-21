from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from airhockey2d import AirHockey2D
from render import AirHockeyRenderer
from matplotlib import pyplot as plt
import threading
import time
import argparse
import yaml
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import imageio
import cv2
import tqdm

def evaluate_air_hockey_model(air_hockey_cfg, log_dir):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.
    Note: This evalutes the latest training directory in the tensorboard log directory. 
    TODO: May need to change this later!

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    model_fp = os.path.join(log_dir, air_hockey_cfg['model_save_filepath'])
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
    
    env_test = AirHockey2D.from_dict(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)
    
    env_test = DummyVecEnv([lambda : env_test])
    env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['reward_type']:
        model = SAC.load(model_fp, env=env_test)
    else:
        model = PPO.load(model_fp)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the tensorboard log directory.')
    args = parser.parse_args()
    log_dir = args.log_dir
    air_hockey_cfg_fp = os.path.join(log_dir, 'model_cfg.yaml')
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    
    evaluate_air_hockey_model(air_hockey_cfg, log_dir)
