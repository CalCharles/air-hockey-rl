from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from airhockey2d import AirHockey2D# , GoalConditionedAirHockey2D
import numpy as np
import argparse
import yaml
import os
import re


def train_air_hockey_model(air_hockey_cfg):
    """
    Train an air hockey paddle model using stable baselines.

    This script loads the configuration file, creates an AirHockey2D environment,
    wraps the environment with necessary components, trains the model,
    and saves the trained model and environment statistics.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    env = AirHockey2D.from_dict(air_hockey_params)
    # env = GoalConditionedAirHockey2D.from_dict(air_hockey_params)

    # check_env(env)
    def wrap_env(env):
        wrapped_env = Monitor(env) # needed for extracting eprewmean and eplenmean
        wrapped_env = DummyVecEnv([lambda: wrapped_env]) # Needed for all environments (e.g. used for multi-processing)
        wrapped_env = VecNormalize(wrapped_env) # probably something to try when tuning
        return wrapped_env

    env = wrap_env(env)
    
    tf_log = air_hockey_cfg['tb_log_dir']
    
    # if goal-conditioned use SAC
    if 'goal' in air_hockey_cfg['air_hockey']['reward_type']:
        # SAC hyperparams:
        # Create 4 artificial transitions per real transition
        n_sampled_goal = 4

        # SAC hyperparams:
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
            ),
            learning_starts=10000,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=512,
            # device='cuda',
            # device="cuda"
            # policy_kwargs=dict(net_arch=[64, 64]),
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, 
                tensorboard_log=air_hockey_cfg['tb_log_dir'], 
                device="cpu", # cpu is actually faster!
                gamma=air_hockey_cfg['gamma']) 
    
    model.learn(total_timesteps=air_hockey_cfg['n_training_steps'],
                tb_log_name=air_hockey_cfg['tb_log_name'], 
                progress_bar=True)
    
    log_dir = air_hockey_cfg['tb_log_dir']
    # get log dir ending with highest number
    subdirs = [x for x in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, x))]
    subdirs.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
    log_dir = os.path.join(log_dir, subdirs[-1])
    
    # let's save model and vec normalize here too
    model_filepath = os.path.join(log_dir, air_hockey_cfg['model_save_filepath'])
    env_filepath = os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath'])
    # copy cfg to same folder
    cfg_filepath = os.path.join(log_dir, 'model_cfg.yaml')
    with open(cfg_filepath, 'w') as f:
        yaml.dump(air_hockey_cfg, f)

    model.save(model_filepath)
    env.save(env_filepath)


if __name__ == "__main__":
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
    train_air_hockey_model(air_hockey_cfg)
