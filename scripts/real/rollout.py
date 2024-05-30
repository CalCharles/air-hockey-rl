from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from airhockey import AirHockeyEnv
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from airhockey.renderers.render import AirHockeyRenderer
import argparse
import yaml
import os
import random
import wandb
import argparse
import shutil
import os
import yaml
import numpy as np
import sys
sys.path.append("/home/pearl/dilo_ar_inference")
from absl import app, flags
from ml_collections import config_flags
from dataclasses import dataclass
import torch
from models import  TwinQ, ValueFunction, TwinV, IDM, Discriminator
from dilo import RECOIL_V_ODICE
from bco import BCO
from smodice import SMODICE
from  policy import GaussianPolicy, DeterministicPolicy
from dilo_utils import DEFAULT_DEVICE

def modify_obs(obs):
    paddle_info = obs[:4]
    paddle_info[0] = paddle_info[0] - 1
    paddle_info = paddle_info.tolist()
    paddle_info[2] = 0
    paddle_info[3] = 0
    puck_info = [0,0,0] * 5
    return np.array(paddle_info + puck_info)

def modify_obs_hit(obs):
    for i in range(5):
        obs[4 + i * 3:4 + i * 3 + 2] = obs[4 + i * 3:4 + i * 3 + 2] - obs[:2]
    return obs


         
def run_policy(air_hockey_cfg, model, use_wandb=False, device='cpu', clear_prior_task_results=False, progress_bar=True):
    """
    Train an air hockey paddle model using stable baselines.

    This script loads the configuration file, creates an AirHockey2D environment,
    wraps the environment with necessary components, trains the model,
    and saves the trained model and environment statistics.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']

    if 'sac' == air_hockey_cfg['algorithm']:
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = True
        else:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    else:
        air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp['seed'] = 42
    air_hockey_params_cp['max_timesteps'] = 200

    eval_env = AirHockeyEnv(air_hockey_params_cp)
    state_dict = eval_env.simulator.get_current_state()
    # obs = eval_env.get_observation(state_dict)
    obs = get_observation_by_type(state_dict, obs_type='history', puck_history=state_dict["pucks"][0]["history"])
    obs_list = list()
    with NonBlockingConsole() as nbc:
        while True:
            obs = modify_obs_hit(obs)
            obs = torch.tensor(obs).unsqueeze(0).to(0).float()
            action = model.policy(obs)
            print("action", action, obs)
            # action = action.mean

            obs, reward, is_finished, truncated, info = eval_env.step(action.squeeze().detach().cpu().numpy())
            
            # for puck hitting observations
            state_dict = eval_env.simulator.get_current_state()
            obs = get_observation_by_type(state_dict, obs_type='history', puck_history=state_dict["pucks"][0]["history"])

            if nbc.get_data() == 'y':
                eval_env.reset(seed=None, write_traj = True)
            if nbc.get_data() == 'q':  
                eval_env.reset(seed=None, write_traj = False)

def load_model(pth):


    # agent_path = 'trained_models/dilo/rel_vel_low_temp_zero_vel_2_success.pth'
    agent_path = 'trained_models/puck_hitting/test_initial/dilo/best_model.pth'

    task = 'goal_obstacle_avoidance'

    obs_dim = 19 # TODO: change this to the correct observation dimension
    act_dim = 2
    
    if agent_path.find("dilo") != -1:
        agent = RECOIL_V_ODICE(qf=TwinQ(state_dim=obs_dim,act_dim=obs_dim),vf = ValueFunction(state_dim=obs_dim),policy=DeterministicPolicy(obs_dim,act_dim),
                                                        optimizer_factory=torch.optim.Adam,
                                                        tau=0.8, maximizer="smoothed_chi", gradient_type="full", beta=0.5,use_twinV=True, lr=3e-4, discount=0.99, alpha=0.005).to(DEFAULT_DEVICE)
    if agent_path.find("bco") != -1:
        agent = BCO(idm=IDM(state_dim=obs_dim,act_dim=act_dim),policy=DeterministicPolicy(obs_dim,act_dim),
                                                        optimizer_factory=torch.optim.Adam,
                                                        tau=0.8, maximizer="smoothed_chi", gradient_type="full", beta=0.5,use_twinV=True, lr=3e-4, discount=0.99, alpha=0.005).to(DEFAULT_DEVICE)
    if agent_path.find("smodice") != -1:
        agent = SMODICE(qf=TwinQ(state_dim=obs_dim,act_dim=obs_dim),vf = ValueFunction(state_dim=obs_dim),discriminator=Discriminator(obs_dim,0),policy=GaussianPolicy(obs_dim,act_dim),
                                                        optimizer_factory=torch.optim.Adam,
                                                        tau=0.8, maximizer="smoothed_chi", gradient_type="full", beta=0.5,use_twinV=True, lr=3e-4, discount=0.99, alpha=0.005).to(DEFAULT_DEVICE)

    agent.load_state_dict(torch.load(agent_path))
    return agent
    # import ipdb;ipdb.set_trace()
    observation = torch.randn(1,obs_dim).to(DEFAULT_DEVICE)
    action = agent.policy(observation)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--model', type=str, default=None, help='Path to the model file.')
    args = parser.parse_args()
    
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'configs/baseline_configs/paddle_pos_neg_regions_real_preset.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
            
    model = load_model(args.model)

    run_policy(air_hockey_cfg, model)
