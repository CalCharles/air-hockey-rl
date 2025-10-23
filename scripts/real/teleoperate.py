from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from airhockey import AirHockeyEnv
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
            
def run_teleop(air_hockey_cfg, use_wandb=False, device='cpu', clear_prior_task_results=False, progress_bar=True):
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
    with NonBlockingConsole() as nbc:
        while True:
            eval_env.step(np.array([0,0]))

            if nbc.get_data() == 'y':
                eval_env.reset(seed=None, write_traj = True)
            if nbc.get_data() == 'q':  
                eval_env.reset(seed=None, write_traj = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    # Note: You probably don't want this argument, only if you are retraining frequently
    # and task folder is getting too big
    args = parser.parse_args()
    
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'configs/baseline_configs/paddle_pos_neg_regions_real_preset.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
            
    run_teleop(air_hockey_cfg)
