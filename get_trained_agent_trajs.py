from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from airhockey import AirHockeyEnv
from render import AirHockeyRenderer
from matplotlib import pyplot as plt
import threading
import time
import argparse
import yaml
import os
import tqdm
import numpy as np
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
    
    # randomly generate seeds, should be different from training..
    air_hockey_cfg['seed'] = np.random.randint(0, 1000)
    air_hockey_params = air_hockey_cfg['air_hockey']
    model_fp = os.path.join(log_dir, air_hockey_cfg['model_save_filepath'])
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
    
    env_test = AirHockeyEnv.from_dict(air_hockey_params)
    # renderer = AirHockeyRenderer(env_test)
    
    env_test = DummyVecEnv([lambda : env_test])
    env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['task']:
        model = SAC.load(model_fp, env=env_test)
    else:
        model = PPO.load(model_fp)

    obs = env_test.reset()
    start = time.time()
    done = False
    # let's save
    # s,a,r,s', timestep
    trajs = []
    timestep = 0
    
    for i in tqdm.tqdm(range(1000000)):
        # Draw the world
        # renderer.render()
        action = model.predict(obs, deterministic=True)[0]
        next_obs, rew, done, info = env_test.step(action)
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            # then it's an ordered dict
            s = obs['observation']
            g = obs['desired_goal']
            s = np.concatenate([s.flatten(), g.flatten()])
            # acheived goal already part of s
        else:
            s = obs.flatten()
        a = action.flatten()
        r = np.array(rew)
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            s_prime = next_obs['observation']
            g_prime = next_obs['desired_goal']
            s_prime = np.concatenate([s_prime.flatten(), g_prime.flatten()]) # g_prime should be the same
        else:
            s_prime = next_obs.flatten()
        t = np.array([timestep])
        
        trajs.append(np.concatenate([s, a, r, s_prime, t]))
        obs = next_obs
        timestep += 1
        if done:
            obs = env_test.reset()
            timestep = 0
    env_test.close()
    
    trajs = np.array(trajs)
    np.save(os.path.join(log_dir, 'trajs.npy'), trajs)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the tensorboard log directory.')
    args = parser.parse_args()
    log_dir = args.log_dir
    air_hockey_cfg_fp = os.path.join(log_dir, 'model_cfg.yaml')
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)

    evaluate_air_hockey_model(air_hockey_cfg, log_dir)
