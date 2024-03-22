from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from air_hockey_simulator.airhockey_box2d import AirHockey2D
from render import AirHockeyRenderer
from matplotlib import pyplot as plt
import threading
import time
import argparse
import yaml
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

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
    parser.add_argument('--data_cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the tensorboard log directory.')
    args = parser.parse_args()
    log_dir = args.log_dir
    air_hockey_cfg_fp = os.path.join(log_dir, 'model_cfg.yaml')
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)

    evaluate_air_hockey_model(air_hockey_cfg, log_dir)
