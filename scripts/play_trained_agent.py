from stable_baselines3 import PPO 
import argparse
import yaml
import os
from demonstrate import Demonstrator


def play_air_hockey_model(air_hockey_cfg):
    """
    Evaluate the performance of an air hockey model using Stable Baselines.

    This script loads a trained model and evaluates its performance in the air hockey environment.
    It uses a configuration file to specify the environment parameters and the file path of the trained model.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['num_paddles'] = 2
    air_hockey_params['gravity'] = 0
    model_fp = air_hockey_cfg['model_save_filepath']
    
    model = PPO.load(model_fp)

    demonstrator = Demonstrator(air_hockey_cfg)
    demonstrator.play_against_agent(model)

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
    
    play_air_hockey_model(air_hockey_cfg)
