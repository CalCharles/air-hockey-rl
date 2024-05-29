import os, copy
import h5py
import numpy as np
from airhockey.airhockey_base import get_observation_by_type
import argparse
import yaml
from airhockey import AirHockeyEnv


def get_observation(paddle, paddle_vel, puck, puck_history, obs_type):
    # uses the last puck history
    state_info = dict()
    state_info["paddles"] = dict()
    state_info["paddles"]["paddle_ego"] = {"position": paddle, "velocity": paddle_vel}
    state_info["pucks"] = dict()
    state_info["pucks"] = list()
    state_info["pucks"].append({"position": puck, "velocity": puck - puck_history[-2], "history": puck_history})
    observation = get_observation_by_type(state_info, obs_type=obs_type, puck_history=puck_history)
    return observation, state_info

def load_dataset(data_dir, obs_type, environment):
    # loads data into a dataset of observations based on the observation type
    dataset = dict()
    dataset["observations"] = list()
    dataset["actions"] = list()
    dataset["rewards"] = list()
    dataset["next_observations"] = list()
    dataset["terminals"] = list()
    dataset["image"] = list()

    for file in os.listdir(data_dir)[:1]:
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            try:
                paddle = f["pose"][:,:2]
                paddle_vel = f["speed"][:,:2]
                action = f["desired_pose"][:,:2] - paddle
                puck = f["puck"]
                image = f["image"]
            except Exception as e:
                print('Error in file:', file, e)
                continue
            print("added trajectory ", file)
            puck_history = [(-1,0,0) for i in range(5)]
            observation, state_info = get_observation(paddle[0], paddle_vel[0], puck[0], puck_history, obs_type=obs_type)
            observations = []
            next_observations = list()
            rewards = [environment.get_base_reward(state_info)]
            for pa, pav, pu in zip(paddle[1:], paddle_vel[1:], puck[1:]):
                next_observation, state_info = get_observation(pa, pav, pu, puck_history, obs_type=obs_type)
                observations.append(observation)
                rewards.append(environment.get_base_reward(state_info))
                next_observations.append(next_observation)
                observation = copy.deepcopy(next_observation)
                puck_history.append(pu)
            # next_observations.append(copy.deepcopy(next_observation)) # TODO: see if we want to append the same observation twice and use the terminal
            dataset["observations"].append(np.array(observations))
            dataset["actions"].append(action[:-1])
            dataset["rewards"].append(np.array(rewards))
            dataset["next_observations"].append(np.array(next_observations))
            terminals = np.zeros(len(action)-1)
            terminals[-1] = 1
            dataset["terminals"].append(terminals)
            dataset["image"].append(copy.deepcopy(image[:-1]))
    dataset["observations"] = np.concatenate(dataset["observations"], axis=0)
    dataset["actions"] = np.concatenate(dataset["actions"], axis=0)
    dataset["rewards"] = np.concatenate(dataset["rewards"], axis=0)
    dataset["next_observations"] = np.concatenate(dataset["next_observations"], axis=0)
    dataset["terminals"] = np.concatenate(dataset["terminals"], axis=0)
    dataset["image"] = np.concatenate(dataset["image"], axis=0)

    return dataset
# read_new_real_data("/datastor1/calebc/public/data/mouse/cleaned_new/")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')

    args = parser.parse_args()

    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'default_train_puck_vel.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)

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

    dataset = load_dataset("/datastor1/calebc/public/data/mouse/state_data_all/", "history", eval_env)
    print(dataset["observations"].shape)
    print(dataset["next_observations"].shape)
    print(dataset["actions"].shape)
    print(dataset["rewards"].shape)
    print(dataset["terminals"].shape)
