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
from utils import EvalCallback, save_evaluation_gifs, save_tensorboard_plots


from envs.reacher_trace import ReacherTraceEnv
from envs.locomotion.ant import AntEnv

def train_env_model():
    eval_env = AntEnv()
    env = AntEnv()
    seeds = [0, 1, 2, 3, 4]
    gamma = 0.99
    tb_log_dir = 'baseline_models'
    tb_log_name = 'AntEnv'
    n_training_steps = 1000000
    subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
    subdir_nums = [int(x.split(tb_log_name + '_')[1]) for x in subdirs]
    next_num = max(subdir_nums) + 1 if subdir_nums else 1
    log_dir = os.path.join(log_parent_dir, tb_log_name + f'_{next_num}')
    print('log_dir', log_dir)
    callback = EvalCallback(eval_env, 
                            log_dir=log_dir, 
                            n_eval_eps=30, 
                            eval_freq=5000)

    log_parent_dir = os.path.join(tb_log_dir, tb_log_name)
    os.makedirs(log_parent_dir, exist_ok=True)
    print('log_parent_dir', log_parent_dir)

    # for seed in seeds:
    #     model = PPO("MlpPolicy", env, verbose=1, 
    #                 tensorboard_log=log_parent_dir, 
    #                 device=device, 
    #                 seed=seed,
    #                 # batch_size=512,
    #                 #n_epochs=5,
    #                 gamma=gamma)
    #     model.learn(total_timesteps=n_training_steps,
    #                 tb_log_name=tb_log_name, 
    #                 callback=callback,
    #                 progress_bar=progress_bar)
    
def collect_env_data(air_hockey_cfg, log_dir):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.
    Note: This evalutes the latest training directory in the tensorboard log directory. 
    TODO: May need to change this later!

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """

    env_test = ReacherTraceEnv() #.from_dict(air_hockey_params)
    # renderer = AirHockeyRenderer(env_test)
    
    env_test = DummyVecEnv([lambda : env_test])
    # env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['task']:
        model = SAC.load(model_fp, env=env_test)
    else:
        model = PPO.load(model_fp)

    obs = env_test.reset()
    print(obs, type(obs)) # ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel
    start = time.time()
    done = False
    # let's save
    # s,a,r,s', timestep
    # trajs0 = []
    trajs = []
    timestep = 0
    # saved_obs = np.array([])
    # saved_act = np.array([])
    # saved_rew = np.array([])

    for i in tqdm.tqdm(range(1000000)):
        print("i:", i, "time:", timestep)
        # Draw the world
        # renderer.render()
        action = model.predict(obs, deterministic=True)[0]
        # print("env_test", env_test)
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
        # saving trajectory  # 'state', 'next_state', 'action', 'rew', 'term', 'trunc', 'info'
        if timestep == 0:
            saved_obs = np.array(s)
            saved_s_prime = np.array(s_prime)
            saved_act = np.array(a)
            saved_rew = np.array(r)
            saved_term = np.zeros(1)
            saved_trunc = np.array([1])
            saved_info = np.array([info])
        else:
            saved_obs = np.vstack([saved_obs, s])
            saved_s_prime = np.vstack([saved_s_prime, s_prime])
            saved_act = np.vstack([saved_act, a])
            saved_rew = np.vstack([saved_rew, r])
            if done:
                saved_term = np.vstack([saved_term, np.ones(1)])
            else:
                saved_term = np.vstack([saved_term, np.zeros(1)])
            saved_trunc = np.vstack([saved_trunc, np.array([1])])
            saved_info = np.vstack([saved_info, np.array([info])])
        
        obs = next_obs
        timestep += 1
        if done: # term
            obs = env_test.reset()
            timestep = 0
            d = {}
            keys = ["obs", 'next_obs', "act", "rew", "term", "trunc"] #, "info"]
            d["obs"] = saved_obs.astype(np.float64)
            d["next_obs"] = saved_s_prime.astype(np.float64)
            d["act"] = saved_act.astype(np.float64)
            d["rew"] = saved_rew.astype(np.float64)
            d["term"] = saved_term.astype(np.float64)
            d["trunc"] = saved_trunc.astype(np.float64)
            # d["info"] = str(saved_info)
            write_trajectory(log_dir, i, d, keys)

    env_test.close()
    
    # trajs = np.array(trajs)
    # np.save(os.path.join(log_dir, 'trajs.npy'), trajs)

    

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
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the tensorboard log directory.')
    args = parser.parse_args()
    log_dir = args.log_dir

    # use_wandb = args.wandb
    # device = args.device
    # clear_prior_task_results = args.clear
    train_env_model()


    # collect_env_data(log_dir)

    # python scripts/get_trained_agent_trajs.py --log_dir baseline_models/reacher_trace/?
