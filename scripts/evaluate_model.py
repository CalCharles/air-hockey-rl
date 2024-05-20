import os
import cv2
import imageio
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
import argparse
from utils import save_evaluation_gifs

# Takes in a folder with a model zip and the config for the model, and uses it to generate evaluation gifs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save an evaluation gif of a trained model.')
    parser.add_argument('--model', type=str, default=None, help='Folder that contains model and model_cfg.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--save-dir', type=str, default=None, help='Path to save the evaluation gifs to.')
    parser.add_argument('--seed', type=int, default=0, help='The random seed for the environment')
    args = parser.parse_args()

    with open(os.path.join(args.model, "model_cfg.yaml"), 'r') as f:
        model_cfg = yaml.safe_load(f)

    air_hockey_params = model_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = model_cfg['n_training_steps']


    # Set the return_goal_obs parameter
    model_cfg['air_hockey']['return_goal_obs'] = 'goal' in model_cfg['air_hockey']['task'] and 'sac' == model_cfg['algorithm']
    
    model_cfg['air_hockey']['max_timesteps'] = 200
            
    air_hockey_params = model_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = model_cfg['n_training_steps']
    air_hockey_params['seed'] = args.seed

    env_test = AirHockeyEnv(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)

    env_test = DummyVecEnv([lambda: env_test])

    # Load the correct model based on the algorithm specified in the config
    if model_cfg['algorithm'] == 'sac':
        model = SAC.load(os.path.join(args.model, "model.zip"), env=env_test)
    else:
        model = PPO.load(os.path.join(args.model, "model.zip"), env=env_test)

    print("Saving gifs...(this will tqdm for EACH gif to save)")
    save_evaluation_gifs(5, 3, env_test, model, renderer, args.save_dir, False)
