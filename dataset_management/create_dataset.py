import os, copy
import h5py
import numpy as np
from airhockey.airhockey_base import get_observation_by_type



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

    for file in os.listdir(data_dir):
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            try:
                paddle = f["pose"][:,:2]
                paddle_vel = f["speed"][:,:2]
                action = f["desired_pose"][:,:2]
                puck = f["puck"]
            except Exception as e:
                print('Error in file:', file, e)
                continue
            print("added trajectory ", file)
            puck_history = [(-2,0,0) for i in range(5)]
            observation, state_info = get_observation(paddle[0], paddle_vel[0], puck[0], puck_history, obs_type=obs_type)
            observations = []
            next_observations = list()
            rewards = [environment.get_reward(state_info)]
            for pa, pav, pu in zip(paddle[1:], paddle_vel[1:], puck[1:]):
                next_observation, state_info = get_observation(pa, pav, pu, puck_history, obs_type=obs_type)
                observations.append(observation)
                rewards.append(environment.get_reward(state_info))
                next_observations.append(next_observation)
                observation = copy.deepcopy(next_observation)
                puck_history.append(pu)
            next_observations.append(copy.deepcopy(next_observation)) # TODO: see if we want to append the same observation twice and use the terminal
            dataset["observations"].append(np.array(observations))
            dataset["actions"].append(action)
            dataset["rewards"].append(np.array(rewards))
            dataset["next_observations"].append(np.array(next_observations))
            terminals = np.zeros(len(action))
            terminals[-1] = 1
            dataset["terminals"].append(terminals)
    dataset["observations"] = np.concatenate(dataset["observations"], axis=0)
    dataset["actions"] = np.concatenate(dataset["actions"], axis=0)
    dataset["rewards"] = np.concatenate(dataset["rewards"], axis=0)
    dataset["next_observations"] = np.concatenate(dataset["next_observations"], axis=0)
    dataset["terminals"] = np.concatenate(dataset["terminals"], axis=0)

    return dataset
# read_new_real_data("/datastor1/calebc/public/data/mouse/cleaned_new/")
