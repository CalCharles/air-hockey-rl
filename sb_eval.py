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

def evaluate_air_hockey_model(air_hockey_cfg):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.
    Note: This evalutes the latest training directory in the tensorboard log directory. 
    TODO: May need to change this later!

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    model_fp = air_hockey_cfg['model_save_filepath']
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
    
    
    env_test = AirHockey2D.from_dict(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)
    
    env_test = DummyVecEnv([lambda : env_test])
    env_test = VecNormalize.load(air_hockey_cfg['vec_normalize_save_filepath'], env_test)
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['reward_type']:
        model = SAC.load(model_fp, env=env_test)
    else:
        model = PPO.load(model_fp)

    # env_test.training = False
    # env_test.norm_reward = False
    
    log_dir = air_hockey_cfg['tb_log_dir']
    # get log dir ending with highest number
    subdirs = [x for x in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, x))]
    log_dir = os.path.join(log_dir, sorted(subdirs)[-1])
    
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

    # List all tags in the log file
    print("Available tags: ", ea.Tags()['scalars'])
    
    metrics = [ 
        'rollout/ep_rew_mean', 
        'train/approx_kl', 
        'train/entropy_loss', 
        'train/learning_rate', 
        'train/loss', 
        'train/value_loss']

    def create_plot(metrics):
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
        plt.show()
    thread = threading.Thread(target=create_plot, args=(metrics,))
    thread.start()
    time.sleep(3) # load plot before game

    obs = env_test.reset()
    start = time.time()
    done = False
    
    for i in range(1000000):
        if i % 1000 == 0:
            print("fps", 1000 / (time.time() - start))
            start = time.time()
        # Draw the world
        renderer.render()
        action = model.predict(obs, deterministic=True)[0]
        obs, rew, done, info = env_test.step(action)
        if done:
            obs = env_test.reset()

    env_test.close()

if __name__ == '__main__':
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
    
    evaluate_air_hockey_model(air_hockey_cfg)
