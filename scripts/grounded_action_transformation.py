import argparse

import yaml
from airhockey import AirHockeyEnv
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""-------------------------------------- Buffers --------------------------------------"""
Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'rew', 'term', 'trunc', 'info'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def add(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        # print("in sample, current memory:", len(self.memory))
        return random.sample(self.memory, batch_size)
    
    def print_memory(self):
        print("Current memory:")
        for i, transition in enumerate(self.memory):
            print(f"Element {i}: {transition}")


"""-------------------------------------- Forward Model --------------------------------------"""
def compute_loss(predictions, targets):
    _loss = F.mse_loss(predictions, targets)
    return _loss

class ForwardKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ForwardKinematicsCNN, self).__init__()
        self.fc1 = nn.Linear(n_observations + n_actions, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_observations)
        self.double()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class InverseKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(InverseKinematicsCNN, self).__init__()
        self.fc1 = nn.Linear(n_observations * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)
        self.double()

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=0)   
        # print(x)
        # x = x.to(torch.double) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""-------------------------------------- GroundedActionTransformation --------------------------------------"""
    
def init_sim_buffer():
    return ReplayMemory(10000)

def init_real_buffer():
    return ReplayMemory(10000)

def init_forward_model(n_observations, n_actions):
    return ForwardKinematicsCNN(n_observations, n_actions)

def init_inverse_model(n_observations, n_actions):
    return InverseKinematicsCNN(n_observations, n_actions)

class GroundedActionTransformation():
    def __init__(self, args, data, sim_env, real_env, log_dir):

        self.sim_env = sim_env
        self.real_env = real_env

        self.sim_current_state = self.sim_env.simulator.get_current_state() # or self.sim_env.current_state
        self.real_current_state = self.real_env.simulator.get_current_state() # or self.real_env.current_state
        # print("self.sim_current_state", self.sim_current_state)
        # print("self.real_current_state", self.real_current_state)
        
        # print(type(self.sim_env), self.sim_current_state)
        state, info = self.sim_env.reset()
        self.n_observations = data[0][0].shape[0] # observations is paddle x y v (2dim) + puck last 5 x y (occ0/1), actions is dx dy
        self.n_actions =  data[0][2].shape[0]
        self.policy = "optimized_policy0"
        self.forward_model = init_forward_model(self.n_observations, self.n_actions)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.001)
        self.forward_batch_size = 1

        self.inverse_model = init_inverse_model(self.n_observations, self.n_actions)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=0.001)
        self.inverse_batch_size = 1
    
        # TODO: need to define the buffers, check with Michael Munje about how we are using buffers
        # for the current RL training code
        self.real_buffer = init_real_buffer()
        self.sim_buffer = init_sim_buffer()
        self.log_dir = log_dir
        # before we have trained anything, don't use the grounded transform
        self.first = True
        


    def grounded_transform(self, state, action):
        if self.first: return action
        return self.inverse_model(state, self.forward_model(state, action))
    
    def add_data(self, trajectories):
        # TODO: pretty sure you can't do this
        for trajectory in trajectories:
            self.real_buffer.add(*trajectory)
            self.sim_buffer.add(*trajectory) # proxy

    def rollout_real(self, num_frames):
        for i in range(num_frames):
            obs = self.real_env.get_observation(self.real_current_state)
            model = PPO.load(self.policy, env=self.real_env)
            act, _states = model.predict(obs) # apply policy to real_env
            # print(self.real_env.step(act))
            obs_next, reward, is_finished, truncated, info = self.real_env.step(act)
            self.real_current_state = self.real_env.simulator.get_current_state()
            traj = torch.tensor(obs), torch.tensor(obs_next), torch.tensor(act), torch.tensor(reward), torch.tensor(is_finished), torch.tensor(truncated), info
            self.real_buffer.add(*traj) # ('state', 'next_state', 'action', 'rew', 'term', 'trunc', 'info')
        # TODO: might need to make this actually work
    
    def train_sim(self, num_iters, i):
        # TODO: try to utilize the same train function as other components
        # TODO: train should automatically add to self.sim_buffer, so we don't need to keep
        # the trajectories
        # trajectories = self.train_ppo(self.sim_env, self.policy, self.sim_buffer, self.grounded_transform, num_iters)
        if i == 0:
            model = PPO("MlpPolicy", self.sim_env, verbose=1)
        else:
            model = PPO.load(self.policy, env=self.sim_env)
            obs = self.sim_env.get_observation(self.sim_current_state) # get current observation
            action, _states = model.predict(obs)
            grounded_action = self.grounded_transform(state=self.sim_current_state, action=action) # action transform
            print("\naction transform:", action, grounded_action, "\n")
            obs, reward, is_finished, truncated, info = self.sim_env.step(grounded_action)
            self.sim_current_state = self.sim_env.simulator.get_current_state() # update current state
            model = PPO.load(self.policy, env=self.sim_env) # env: the new environment to run the loaded model on
            
        model.learn(num_iters)
        self.policy = self.log_dir + "/optimized_policy" + str(i) # self.policy stores the current model name
        model.save(self.policy)


    def train_forward(self, num_iters):
        self.forward_model.train()
        for i in range(num_iters):
            data = self.real_buffer.sample(self.forward_batch_size)
            for i in range(self.inverse_batch_size):
                data = data[i]
                pred_next_state = self.forward_model(data.state, data.action)
                # TODO: define these functions generally so we can change them later
                loss = compute_loss(pred_next_state, data.next_state)
                # print("In forward model, loss:", loss, ", iter:", i)
                # self.forward_optimizer.step(loss.mean())
                loss.backward()  # Backward pass
                self.forward_optimizer.step()

    def train_inverse(self, num_iters):
        self.inverse_model.train()
        for i in range(num_iters):
            data = self.sim_buffer.sample(self.inverse_batch_size)
            # print("data", data)
            for i in range(self.inverse_batch_size):
                data = data[i]
                pred_action = self.inverse_model(data.state, data.next_state)
                # TODO: define these functions generally so we can change them later
                loss = compute_loss(pred_action, data.action)
                # print("In inverse model, loss:", loss, ", iter:", i)
                loss.backward()  # Backward pass
                self.forward_optimizer.step()

def load_dataset(dataset_pth): # TODO
    import h5py
    import os
    import ast
    keys = ['obs', 'act', 'rew', 'term', 'trunc'] #, 'info']
    file_path = os.path.join(dataset_pth)
    print("Reading h5py file from", file_path)
    
    with h5py.File(file_path, 'r') as hf:
        data = {}
        for key in keys:
            dataset = hf[key]
            data[key] = torch.tensor(dataset[:])
            print(f"Key: {key}")
            print(f"Data: {data[key][0]}")
            print(f"Shape: {dataset.shape}")
            print(f"Dtype: {dataset.dtype}")
            if key == 'term':
                n = dataset.shape[0]
        # info
        data['info'] = [{}] * n
        # print("data['info']", len(data['info']))
    # Create the list of tuples
    keys = ['obs', 'act', 'rew', 'term', 'trunc', 'info']
    result = []
    for i in range(n-1):
        # if i < n-1:
        next_state = data['obs'][i+1]
        # else:
        #     next_state = np.zeros(data['obs'][0].shape) # assume last state's next state is all zeros
        result.append((data['obs'][i], next_state, data['act'][i], data['rew'][i], data['term'][i], data['trunc'][i], data['info'][i]))
    # Print the result
    # for item in result:
    #     print(item)
    #     break

    return result
    

def train_GAT(args, data, sim_air_hockey_cfg, real_air_hockey_cfg):
    sim_env = AirHockeyEnv(sim_air_hockey_cfg['air_hockey'])
    real_env = AirHockeyEnv(real_air_hockey_cfg['air_hockey'])
    gat = GroundedActionTransformation(args, data, sim_env, real_env)
    gat.add_data(data)
    print("Starting train_sim...")
    gat.train_sim(args.initial_rl_training, i=0)
    print("train_sim finished...")
    print("Starting train_inverse...")
    gat.train_inverse(args.initial_inverse_training)
    print("train_inverse finished...")
    print("Starting train_forward...")
    gat.train_forward(args.initial_forward_training)
    for i in range(args.num_real_sim_iters):
        gat.train_sim(args.rl_iters, i)
        gat.rollout_real(args.num_real_steps)
        gat.train_inverse(args.inverse_iters)
        gat.train_forward(args.forward_iters)
    print("train_GAT finished,")
    gat.real_buffer.print_memory()
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--sim-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--real-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default=None, help='Path to the dataset file.') # real-cfg
    parser.add_argument('--initial_rl_training', type=int, default=10, help='initial_rl_training')
    parser.add_argument('--initial_inverse_training', type=int, default=10, help='initial_inverse_training')
    parser.add_argument('--initial_forward_training', type=int, default=10, help='initial_forward_training')
    parser.add_argument('--num_real_sim_iters', type=int, default=10, help='num_real_sim_iters')
    parser.add_argument('--num_real_steps', type=int, default=10, help='num_real_steps')
    parser.add_argument('--rl_iters', type=int, default=10, help='rl_iters')
    parser.add_argument('--inverse_iters', type=int, default=10, help='inverse_iters')
    parser.add_argument('--forward_iters', type=int, default=10, help='forward_iters')

    args = parser.parse_args()

    with open(args.sim_cfg, 'r') as f:
        sim_air_hockey_cfg = yaml.safe_load(f)

    with open(args.real_cfg, 'r') as f:
        real_air_hockey_cfg = yaml.safe_load(f)

    # TODO: write a loader for the dataset, which should load into a list of dicts with three keys: states, actions, dones
    # sorting is: trajectory->key->data
    data = load_dataset(args.dataset_pth) 
    log_dir = 'gat_log/'
    best_params, mean_params = train_GAT(args, data, sim_air_hockey_cfg, real_air_hockey_cfg, log_dir)
# python scripts/grounded_action_transformation.py --sim-cfg configs/gat/puck_height.yaml --real-cfg configs/gat/puch_height2.yaml --dataset_pth baseline_models/puck_height/air_hockey_agent_13/trajectory_data0.hdf5