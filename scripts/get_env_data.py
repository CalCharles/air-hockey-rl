from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
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
import matplotlib.pyplot as plt


from envs.reacher_trace import ReacherTraceEnv
from envs.locomotion.ant import AntEnv
import gymnasium as gym

def save_rew(rew, label, plot_fp):
    # plt.plot(rew, label=label, color='b', linestyle='-', marker='o', markersize=1)
    plt.scatter(range(len(rew)), rew, label=label, color='b', marker='o', s=1)

    # plt.plot(loss_values_f, label='Forward Model', color='r', linestyle='--', marker='x', markersize=1)
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Timestep')
    plt.legend()
    plt.savefig(plot_fp)
    plt.close()

def train_env_model(tb_log_name, device='cpu', progress_bar=True):
    # eval_env = AntEnv()
    # env = AntEnv()
    # env = gym.make('Ant-v2')'

    example_map = [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]]

    env = gym.make('PointMaze_UMaze-v3', maze_map=example_map)

    def wrap_env(env):
        wrapped_env = Monitor(env) # needed for extracting eprewmean and eplenmean
        wrapped_env = DummyVecEnv([lambda: wrapped_env]) # Needed for all environments (e.g. used for multi-processing)
        # wrapped_env = VecNormalize(wrapped_env) # probably something to try when tuning
        return wrapped_env
    env = wrap_env(env)
    seeds = [0, 1, 2, 3, 4]
    gamma = 0.99
    tb_log_dir = 'baseline_models'
    n_training_steps = 1000000
    log_parent_dir = os.path.join(tb_log_dir, tb_log_name)
    os.makedirs(log_parent_dir, exist_ok=True)
    print('log_parent_dir', log_parent_dir)

    model_save_filepath = 'model'
    vec_normalize_save_filepath = 'vec_normalize.pkl'
    # make configs
    env_cfg = {}
    env_cfg['gamma'] = gamma
    env_cfg['tb_log_dir'] = tb_log_dir
    env_cfg['tb_log_name'] = tb_log_name
    env_cfg['n_training_steps'] = n_training_steps
    env_cfg['model_save_filepath'] = model_save_filepath
    env_cfg['vec_normalize_save_filepath'] = vec_normalize_save_filepath

    subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
    subdir_nums = [int(x.split(tb_log_name + '_')[1]) for x in subdirs]
    next_num = max(subdir_nums) + 1 if subdir_nums else 1
    log_dir = os.path.join(log_parent_dir, tb_log_name + f'_{next_num}')
    print('log_dir', subdirs, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    for seed in seeds:
        env_cfg['seed'] = seed
        model = PPO("MultiInputPolicy", env, verbose=1, 
                    tensorboard_log=log_parent_dir, 
                    device=device, 
                    seed=seed,
                    # batch_size=512,
                    #n_epochs=5,
                    gamma=gamma)
        model.learn(total_timesteps=n_training_steps,
                    tb_log_name=tb_log_name, 
                    progress_bar=progress_bar)
        
        model_filepath = os.path.join(log_dir, model_save_filepath)
        print('model_filepath', model_filepath)
        env_filepath = os.path.join(log_dir, vec_normalize_save_filepath)
        # copy cfg to same folder
        cfg_filepath = os.path.join(log_dir, 'model_cfg.yaml')
        with open(cfg_filepath, 'w') as f:
            yaml.dump(env_cfg, f)

        model.save(model_filepath)
        # env.save(env_filepath)

def collect_env_data_trained_network(log_dir, model_fp):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.
    Note: This evalutes the latest training directory in the tensorboard log directory. 
    TODO: May need to change this later!

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """
    # env_test = gym.make('Ant-v2')
    example_map = [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]]

    env_test = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
    
    env_test = DummyVecEnv([lambda : env_test])
    # env_test = VecNormalize.load(os.path.join(log_dir, env_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    # model = SAC.load(model_fp, env=env_test)
    model = PPO.load(model_fp)

    device='cpu'
    gamma = 0.99
    tb_log_dir = 'baseline_models'
    tb_log_name = 'PointMazeRandom'
    log_parent_dir = os.path.join(tb_log_dir, tb_log_name)
    os.makedirs(log_parent_dir, exist_ok=True)
    # model = PPO("MultiInputPolicy", env_test, verbose=1, 
    #         tensorboard_log=log_parent_dir, 
    #         device=device, 
    #         seed=0,
    #         # batch_size=512,
    #         #n_epochs=5,
    #         gamma=gamma)

    state_dict = env_test.reset()
    # print('obs', obs, type(obs)) # ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel
    obs = state_dict #['observation'][0]
    # print('obs', obs, type(obs))
    start = time.time()
    done = False
    trajs = []
    timestep = 0
    
    scatter_x = []
    scatter_y = []
    rew_plot = []
    rew_sum = 0
    num_trajectories = 0
    for i in tqdm.tqdm(range(10000000)): #10000000
        action = model.predict(obs, deterministic=True)[0]
        # , action.shape) # [[ 0.20949568,  0.13751644,  0.00859012,  0.15008973, -0.7435889 , 0.26740882,  0.1946652 ,  0.1315605 ]], dtype=float32)
        # print("env_test", env_test)
        next_obs, rew, done, info = env_test.step(action)
        rew_sum += rew
        if done: # term
            rew_plot.append(rew_sum)
            # print("timestep", timestep)
            obs = env_test.reset()
            timestep = 0
            rew_sum = 0
            d = {}
            keys = ["obs", 'next_obs', "act", "rew", "term", "trunc"] #, "info"]
            d["obs"] = saved_obs.astype(np.float64)
            d["next_obs"] = saved_s_prime.astype(np.float64)
            d["act"] = saved_act.astype(np.float64)
            d["rew"] = saved_rew.astype(np.float64)
            d["term"] = saved_term.astype(np.float64)
            d["trunc"] = saved_trunc.astype(np.float64)
            write_trajectory(log_dir, num_trajectories, d, keys)
            num_trajectories+=1
        else:
            if i % 1000 == 0: # 10000, 20000, etc
                scatter_x.append(action[0][0])
                scatter_y.append(action[0][1])
            goal = True
            if goal:
                # then it's an ordered dict
                s = obs['observation']
                g = obs['desired_goal']
                s = np.concatenate([s.flatten(), g.flatten()])
                # acheived goal already part of s
            else:
                s = obs.flatten()
            a = action.flatten()
            r = np.array(rew)
                
            if goal:
                s_prime = next_obs['observation']
                g_prime = next_obs['desired_goal']
                s_prime = np.concatenate([s_prime.flatten(), g_prime.flatten()]) # g_prime should be the same
            else:
                s_prime = next_obs.flatten()
            t = np.array([timestep])

            trajs.append(np.concatenate([s, a, r, s_prime, t]))
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

    env_test.close()
    plt.scatter(scatter_x, scatter_y, color='blue', marker='o')
    # Add titles and labels (optional)
    plt.title('2D Scatter Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # Define the range for x and y axes
    plt.xlim(-1, 1)  # Set x-axis limits from 0 to 6
    plt.ylim(-1, 1) # Set y-axis limits from 0 to 12
    plt.savefig(log_dir + "/action_stats.png")
    plt.close()
    plot_fp = log_dir + "/rewards.png"
    save_rew(rew_plot, "Reward", plot_fp)

def collect_env_data_random_action(log_dir):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.
    Note: This evalutes the latest training directory in the tensorboard log directory. 
    TODO: May need to change this later!

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """
    # env_test = gym.make('Ant-v2')
    example_map = [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]]

    env_test = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
    
    env_test = DummyVecEnv([lambda : env_test])
    # env_test = VecNormalize.load(os.path.join(log_dir, env_cfg['vec_normalize_save_filepath']), env_test)

    tb_log_dir = 'baseline_models'
    tb_log_name = 'PointMazeRandom'
    log_parent_dir = os.path.join(tb_log_dir, tb_log_name)
    os.makedirs(log_parent_dir, exist_ok=True)

    state_dict = env_test.reset()
    # print('obs', obs, type(obs)) # ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel
    obs = state_dict #['observation'][0]
    # print('obs', obs, type(obs))
    start = time.time()
    done = False
    # let's save
    # s,a,r,s', timestep
    # trajs0 = []
    trajs = []
    timestep = 0
    
    scatter_x = []
    scatter_y = []
    rew_plot = []
    rew_sum = 0
    num_trajectories = 0
    for i in tqdm.tqdm(range(10000000)): #10M 10000000
        # Draw the world
        # renderer.render()
        action = env_test.action_space.sample()
        action = np.expand_dims(action, axis=0)
        # print("action", action)
        
        # action2 = model.predict(obs, deterministic=True)[0]
        # , action.shape) # [[ 0.20949568,  0.13751644,  0.00859012,  0.15008973, -0.7435889 , 0.26740882,  0.1946652 ,  0.1315605 ]], dtype=float32)
        next_obs, rew, done, info = env_test.step(action)
        rew_sum += rew
        if done: # term
            # print("num_trajectories", num_trajectories, timestep)
            rew_plot.append(rew_sum)
            obs = env_test.reset()
            timestep = 0
            rew_sum = 0
            d = {}
            keys = ["obs", 'next_obs', "act", "rew", "term", "trunc"] #, "info"]
            d["obs"] = saved_obs.astype(np.float64)
            d["next_obs"] = saved_s_prime.astype(np.float64)
            d["act"] = saved_act.astype(np.float64)
            d["rew"] = saved_rew.astype(np.float64)
            d["term"] = saved_term.astype(np.float64)
            d["trunc"] = saved_trunc.astype(np.float64)
            write_trajectory(log_dir, num_trajectories, d, keys)
            num_trajectories+=1
        else:
            if i % 1000 == 0:
                print("i:", i, "time:", timestep)
                scatter_x.append(action[0][0])
                scatter_y.append(action[0][1])
            goal = True
            if goal:
                # then it's an ordered dict
                s = obs['observation']
                g = obs['desired_goal']
                s = np.concatenate([s.flatten(), g.flatten()])
                # acheived goal already part of s
            else:
                s = obs.flatten()
            a = action.flatten()
            r = np.array(rew)
                
            if goal:
                s_prime = next_obs['observation']
                g_prime = next_obs['desired_goal']
                s_prime = np.concatenate([s_prime.flatten(), g_prime.flatten()]) # g_prime should be the same
            else:
                s_prime = next_obs.flatten()
            t = np.array([timestep])

            trajs.append(np.concatenate([s, a, r, s_prime, t]))
            # print('trajs', trajs)
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

    env_test.close()
    plt.scatter(scatter_x, scatter_y, color='blue', marker='o', s=5)
    # Add titles and labels (optional)
    plt.title('2D Scatter Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # Define the range for x and y axes
    plt.xlim(-1, 1)  # Set x-axis limits from 0 to 6
    plt.ylim(-1, 1) # Set y-axis limits from 0 to 12
    plt.savefig(log_dir + "/action_stats.png")
    plt.close()
    plot_fp = log_dir + "/rewards.png"
    save_rew(rew_plot, "Reward", plot_fp)


def collect_env_data_random_network(log_dir):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.
    Note: This evalutes the latest training directory in the tensorboard log directory. 
    TODO: May need to change this later!

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """
    # env_test = gym.make('Ant-v2')
    example_map = [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]]

    env_test = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
    
    env_test = DummyVecEnv([lambda : env_test])
    # env_test = VecNormalize.load(os.path.join(log_dir, env_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    # model = SAC.load(model_fp, env=env_test)
    # model = PPO.load(model_fp)

    device='cpu'
    gamma = 0.99
    tb_log_dir = 'baseline_models'
    tb_log_name = 'PointMazeRandom'
    log_parent_dir = os.path.join(tb_log_dir, tb_log_name)
    os.makedirs(log_parent_dir, exist_ok=True)
    # using off-the-shelf model
    model = PPO("MultiInputPolicy", env_test, verbose=1, 
            tensorboard_log=log_parent_dir, 
            device=device, 
            seed=0,
            # batch_size=512,
            #n_epochs=5,
            gamma=gamma)

    state_dict = env_test.reset()
    # print('obs', obs, type(obs)) # ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel
    obs = state_dict #['observation'][0]
    # print('obs', obs, type(obs))
    start = time.time()
    done = False
    # let's save
    # s,a,r,s', timestep
    # trajs0 = []
    trajs = []
    timestep = 0
    
    scatter_x = []
    scatter_y = []
    rew_plot = []
    rew_sum = 0
    num_trajectories = 0

    for i in tqdm.tqdm(range(10000000)): #10M 10000000
        action = model.predict(obs, deterministic=True)[0]
        # , action.shape) # [[ 0.20949568,  0.13751644,  0.00859012,  0.15008973, -0.7435889 , 0.26740882,  0.1946652 ,  0.1315605 ]], dtype=float32)
        # print("env_test", env_test)
        next_obs, rew, done, info = env_test.step(action)
        rew_sum += rew
        if done: # term
            rew_plot.append(rew_sum)
            obs = env_test.reset()
            timestep = 0
            rew_sum = 0
            d = {}
            keys = ["obs", 'next_obs', "act", "rew", "term", "trunc"] #, "info"]
            d["obs"] = saved_obs.astype(np.float64)
            d["next_obs"] = saved_s_prime.astype(np.float64)
            d["act"] = saved_act.astype(np.float64)
            d["rew"] = saved_rew.astype(np.float64)
            d["term"] = saved_term.astype(np.float64)
            d["trunc"] = saved_trunc.astype(np.float64)
            write_trajectory(log_dir, i, d, keys)
            num_trajectories+=1

        else:
            if i % 1000 == 0: # 10000, 20000, etc
                scatter_x.append(action[0][0])
                scatter_y.append(action[0][1])
                rew_plot.append(rew)
            goal = True
            if goal:
                # then it's an ordered dict
                s = obs['observation']
                g = obs['desired_goal']
                s = np.concatenate([s.flatten(), g.flatten()])
                # acheived goal already part of s
            else:
                s = obs.flatten()
            a = action.flatten()
            r = np.array(rew)
                
            if goal:
                s_prime = next_obs['observation']
                g_prime = next_obs['desired_goal']
                s_prime = np.concatenate([s_prime.flatten(), g_prime.flatten()]) # g_prime should be the same
            else:
                s_prime = next_obs.flatten()
            t = np.array([timestep])

            trajs.append(np.concatenate([s, a, r, s_prime, t]))
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
        

    env_test.close()
    plt.scatter(scatter_x, scatter_y, color='blue', marker='o')
    # Add titles and labels (optional)
    plt.title('2D Scatter Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # Define the range for x and y axes
    plt.xlim(-1, 1)  # Set x-axis limits from 0 to 6
    plt.ylim(-1, 1) # Set y-axis limits from 0 to 12
    plt.savefig(log_dir + "/action_stats.png")
    plt.close()
    plot_fp = log_dir + "/rewards.png"
    save_rew(rew_plot, "Reward", plot_fp)


def write_trajectory(pth, tidx, d, keys): # (obs, act, rew, term, trunc, info) , trunc is always false, info is empty dictionary
    file_path = os.path.join(pth, 'trajectory_data/trajectory_data' + str(tidx) + '.hdf5')
    log_dir = os.path.join(pth, 'trajectory_data')
    os.makedirs(log_dir, exist_ok=True)
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

    # use_wandb = args.wandb
    # device = args.device
    # clear_prior_task_results = args.clear
    # train_env_model()
    # train_env_model('PointMaze')
    collect_env_data_trained_network(log_dir='/nfs/homes/air_hockey/PointMazeTrainedNetwork/', model_fp = 'baseline_models/PointMaze/PointMaze_7/model.zip')
    # collect_env_data_random_action(log_dir='baseline_models/PointMazeRandomAction/PointMaze_1')
    # collect_env_data_random_network(log_dir='/nfs/homes/air_hockey/PointMazeRandomNetwork')


    # python scripts/get_trained_agent_trajs.py --log_dir baseline_models/？
