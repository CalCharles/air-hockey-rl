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

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""-------------------------------------- Buffers --------------------------------------"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def add(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


"""-------------------------------------- Forward Model --------------------------------------"""
def compute_loss(predictions, targets):
    loss = F.mse_loss(predictions, targets)
    return loss

class ForwardKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ForwardKinematicsCNN, self).__init__()
        self.fc1 = nn.Linear(n_observations + n_actions, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_observations)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
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

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=1)        
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

def init_inverse_model(n_observations):
    return InverseKinematicsCNN(n_observations)

class GroundedActionTransformation():
    def __init__(self, args, data, sim_env, real_env):

        self.sim_env = sim_env
        self.real_env = real_env

        self.sim_current_state = self.sim_env.simulator.get_current_state() # or self.sim_env.current_state

        # print(type(self.sim_env), self.sim_current_state)
        self.n_actions = self.sim_env.action_space.n
        state, info = self.sim_env.reset()
        self.n_observations = 13 # observations is paddle x y v + puck last 5 xy (occ0/1), actions is dx dy
        self.n_actions = 2

        self.policy = "optimized_policy0"
        self.forward_model = init_forward_model(self.n_observations, self.n_actions)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.001)
        self.forward_batch_size = 1

        self.inverse_model = init_inverse_model(self.n_observations)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=0.001)
        self.inverse_batch_size = 1
    
        # TODO: need to define the buffers, check with Michael Munje about how we are using buffers
        # for the current RL training code
        self.real_buffer = init_real_buffer()
        self.sim_buffer = init_sim_buffer()

        # before we have trained anything, don't use the grounded transform
        self.first = True


    def grounded_transform(self, state, action):
        if self.first: return action
        return self.inverse_model(state, self.forward_model(state, action))
    
    def add_data(self, trajectories):
        # TODO: pretty sure you can't do this
        self.real_buffer.add(trajectories)

    def rollout_real(self, num_frames):
        obs = self.real_env.get_state()
        for i in range(num_frames):
            act = self.policy.act(obs)
            obs, rew, term, trunc, info = self.real_env.step()
            self.real_buffer.add((obs, act, rew, term, trunc, info))
        # TODO: might need to make this actually work
    
    def train_sim(self, num_iters, i):
        # TODO: try to utilize the same train function as other components
        # TODO: train should automatically add to self.sim_buffer, so we don't need to keep
        # the trajectories
        # trajectories = self.train_ppo(self.sim_env, self.policy, self.sim_buffer, self.grounded_transform, num_iters)
        if i == 0:
            model = PPO("MlpPolicy", self.sim_env, verbose=1)
        else:
            obs = self.sim_env.get_observation(self.sim_current_state) # get current observation
            action, _states = model.predict(obs)
            grounded_action = self.grounded_transform(state=self.sim_current_state, action=action) # action transform
            obs, rewards, dones, info = self.sim_env.step(grounded_action)
            self.sim_current_state = self.sim_env.simulator.get_current_state() # update current state
            model = PPO.load(self.policy, env=self.sim_env) # env: the new environment to run the loaded model on
            
        model.learn(num_iters)
        self.policy = "optimized_policy" + str(i) # self.policy stores the current model name
        model.save(self.policy)


    def train_forward(self, num_iters):
        for i in range(num_iters):
            data = self.real_buffer.sample(self.forward_batch_size)
            pred_next_state = self.forward_model(data.state, data.action)
            # TODO: define these functions generally so we can change them later
            loss = compute_loss(pred_next_state, data.next_state)
            self.forward_optimizer.step(loss.mean())

    def train_inverse(self, num_iters):
        for i in range(num_iters):
            data = self.sim_buffer.sample(self.inverse_batch_size)
            pred_action = self.inverse_model(data.state, data.next_state)
            # TODO: define these functions generally so we can change them later
            loss = compute_loss(pred_action, data.action)
            self.inverse_optimizer.step(loss.mean())

def load_dataset(dataset_pth): # TODO
    return

def train_GAT(args, data, sim_air_hockey_cfg, real_air_hockey_cfg):
    sim_env = AirHockeyEnv(sim_air_hockey_cfg)
    real_env = AirHockeyEnv(real_air_hockey_cfg)
    gat = GroundedActionTransformation(args, data, sim_env, real_env)
    gat.add_data(data)
    gat.train_sim(args.initial_rl_training, i=0)
    gat.train_inverse(args.initial_inverse_training)
    gat.train_forward(args.initial_forward_training)
    for i in range(args.num_real_sim_iters):
        gat.train_sim(args.rl_iters, i)
        gat.rollout_real(args.num_real_steps)
        gat.train_inverse(args.inverse_iters)
        gat.train_forward(args.forward_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--sim-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--real-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default=None, help='Path to the dataset file.') # real-cfg
    args = parser.parse_args()

    with open(args.sim_cfg, 'r') as f:
        sim_air_hockey_cfg = yaml.safe_load(f)

    with open(args.real_cfg, 'r') as f:
        real_air_hockey_cfg = yaml.safe_load(f)

    # TODO: write a loader for the dataset, which should load into a list of dicts with three keys: states, actions, dones
    # sorting is: trajectory->key->data
    data = load_dataset(args.dataset_pth) 
    best_params, mean_params = train_GAT(args, data, sim_air_hockey_cfg, real_air_hockey_cfg)

    # python grounded_action_transformation.py --sim-confg configs/baseline_configs/puck_height.yaml --real-cfg configs/gat/puch_height2.yaml