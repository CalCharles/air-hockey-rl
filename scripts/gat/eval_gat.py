from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
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
import h5py
from grounded_action_transformation import GATWrapper, ForwardKinematicsCNN, InverseKinematicsCNN
from utils import EvalCallback, save_tensorboard_plots

import torch
import cv2
import imageio
import wandb


def save_rew(rewards, log_dir, idx):
    print("rewards", rewards.shape)
    plt.plot(rewards, label='Rewards', color='b', linestyle='-', marker='o', markersize=1)
    plt.xlabel('Epoch')
    plt.ylabel('Rewards')
    plt.title('Eval Rewards')
    plt.legend()
    rewards_fp = log_dir + '/rewards' + str(idx) + '.png'
    print('Saving rew to', rewards_fp, '. Mean reward is', rewards.mean())
    plt.savefig(rewards_fp)

def save_evaluation_gifs(n_eps_viz, n_gifs, env_test, model, renderer, log_dir, use_wandb, wandb_run=None):
    env_test.max_timesteps = 200
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
                if i == 0:
                    rewards = rew
                else:
                    rewards = np.hstack((rewards, rew))
        save_rew(rewards, log_dir, idx=gif_idx)
        gif_savepath = os.path.join(log_dir, f'eval_{gif_idx}.gif')
        def fps_to_duration(fps):
            return int(1000 * 1/fps)
        fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
        imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))
    # upload last gif to wandb
    if use_wandb:
        wandb_run.log({"Evaluation Video": wandb.Video(gif_savepath, fps=20)})

def evaluate_air_hockey_gat_model(model_fp, air_hockey_cfg, log_dir, wrapped, forward_fp, inverse_fp):
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
    # model_fp = os.path.join(log_dir, air_hockey_cfg['model_save_filepath'])
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
    
    env_test = AirHockeyEnv(air_hockey_params) #.from_dict(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)
    if wrapped:
        obs = env_test.get_observation(env_test.simulator.get_current_state())
        n_obs = obs.shape[0]
        n_act = 2
        inverse_model = InverseKinematicsCNN(n_obs, n_act)
        forward_model = ForwardKinematicsCNN(n_obs, n_act)
        inverse_model.load_state_dict(torch.load(inverse_fp))
        forward_model.load_state_dict(torch.load(forward_fp))
        
        env_test = GATWrapper(env_test, forward_model, inverse_model)
    # renderer = AirHockeyRenderer(env_test)
    env_test = DummyVecEnv([lambda : env_test])
    # env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    print(model_fp)

    if 'goal' in air_hockey_cfg['air_hockey']['task']:
        model = SAC.load(model_fp, env=env_test)
    else:
        model = PPO.load(model_fp)

    obs = env_test.reset()


    print("Saving gifs... to", log_dir)
    save_evaluation_gifs(10, 6, env_test, model, renderer, log_dir, use_wandb=False)
    
    env_test.close()

    
def write_trajectory(pth, tidx, d, keys): # (obs, act, rew, term, trunc, info) , trunc is always false, info is empty dictionary
    file_path = os.path.join(pth, 'trajectory_data/trajectory_data' + str(tidx) + '.hdf5')
    print("h5py file saved to", file_path)
    with h5py.File(file_path, 'w') as hf:
        for key in keys:
            vals = d[key]
            # print(vals)
            if isinstance(vals, np.ndarray):
                shape = vals.shape
                hf.create_dataset(key,
                            shape=shape,
                            compression="gzip",
                            compression_opts=9,
                            data = vals)
            else:
                shape = (len(vals),)
                hf.create_dataset(key,
                                compression="gzip",
                                compression_opts=9,
                                data = vals)
            print(tidx, hf)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--model_fp', type=str, default=None, help='Path to the model directory.')
    parser.add_argument('--air_hockey_cfg', type=str, default=None, help='Path to the model directory.')
    parser.add_argument('--wrapped', type=str, default=False, help='Path to the model directory.')
    args = parser.parse_args()
    model_fp = args.model_fp
    air_hockey_cfg = args.air_hockey_cfg
    # log_dir = args.log_dir
    with open(air_hockey_cfg, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    forward_fp ='gat_log/puck_height_real/forward_model_pt.pth'#  'gat_log/forward_model_pt.pth'
    inverse_fp ='gat_log/puck_height_real/inverse_model_pt.pth'# 'gat_log/inverse_model_pt.pth'
    if args.wrapped:
        log_dir = 'gat_log/eval/sim_wrapped'
    else:
        log_dir = 'gat_log/eval/real_unwrapped'
    os.makedirs(log_dir, exist_ok=True)
    evaluate_air_hockey_gat_model(model_fp, air_hockey_cfg, log_dir, args.wrapped, forward_fp, inverse_fp)

    # python scripts/eval_gat.py --model_fp gat_log/optimized_policy9 --air_hockey_cfg configs/gat/puck_height_real.yaml --wrapped False
    # python scripts/eval_gat.py --model_fp gat_log/optimized_policy9 --air_hockey_cfg configs/gat/puck_height3.yaml --wrapped True
