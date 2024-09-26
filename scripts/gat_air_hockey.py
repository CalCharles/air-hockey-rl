import argparse
import os
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
from dataset_management.create_dataset import load_dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x):
    original_min = np.min(x)
    original_max = np.max(x)
    return 2 * (x - original_min) / (original_max - original_min) - 1, original_min, original_max


def unnormalize(normalized_array, min_val, max_val):
    original_array = (normalized_array + 1) / 2 * (max_val - min_val) + min_val
    return original_array

"""-------------------------------------- Buffers --------------------------------------"""
Transition = namedtuple('Transition',
                        ('state', 'next_state', 'delta', 'action', 'rew', 'term', 'trunc', 'info'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.keys = ['state', 'next_state', 'delta', 'action', 'rew', 'term', 'trunc', 'info']
        self.memory = {
            'state': np.array([]),
            'next_state': np.array([]),
            'delta': np.array([]),
            'action': np.array([]),
            'rew': np.array([]),
            'term': np.array([]),
            'trunc': np.array([]),
            'info': np.array([])
        } #deque([], maxlen=capacity)
        self.current_index = 0
        self.delta_min = None
        self.delta_max = None
    
    def load_from_dict(self, data):
        # print("data['state'].shape[1]", data['state'].shape[1])
        self.delta_min = np.zeros(data['state'].shape[1]) # array of (4,) for point maze
        self.delta_max = np.zeros(data['state'].shape[1])
        for key in self.keys:
            self.memory[key] = data[key]
            # normalize delta and save min and max
            if key == 'delta':
                # print(self.memory[key].shape)
                for dim in range(self.memory[key].shape[1]):
                    cur_col = self.memory[key][:, dim]
                    normalized_delta, cur_delta_min, cur_delta_max = normalize(cur_col)
                    self.delta_min[dim] = cur_delta_min
                    self.delta_max[dim] = cur_delta_max
                    self.memory[key][:, dim] = normalized_delta
        # print("self.delta_min", self.delta_min, self.delta_max)
        # input()
        

    def add_from_dict(self, data):
        for key in self.keys:
            if key == 'delta':
                # print('before renormalization', np.max(self.memory[key]), np.min(self.memory[key]), self.memory[key].shape)
                # print('in deltax')
                old_memory = self.memory[key]
                self.memory[key] = np.vstack((self.memory[key], data[key])) # establish the shape as (N1 + N2) x 4
                for dim in range(self.memory[key].shape[1]): # fill in each (N1 + N2) x 1 column
                    # unnormalize the old memory
                    cur_col = old_memory[:, dim]
                    unnormalize_delta = unnormalize(cur_col, self.delta_min[dim], self.delta_max[dim])
                    # print(cur_col.shape, data[key][:, dim].shape)
                    combined_delta = np.concatenate((unnormalize_delta, data[key][:, dim]))
                    # print(cur_col.shape, data[key][:, dim].shape, combined_delta.shape)
                    # renormalize everything and update min and max
                    normalized_delta, cur_delta_min, cur_delta_max = normalize(combined_delta)
                    self.delta_min[dim] = cur_delta_min
                    self.delta_max[dim] = cur_delta_max
                    self.memory[key][:, dim] = normalized_delta
            else:
                self.memory[key] = np.vstack((self.memory[key], data[key]))


    def add(self, *args):
        """Save a transition"""
        for i in range(len(self.keys)):
            key = self.keys[i]
            # print("key", key)
            # print("before add:", Transition(*args)[i].shape, self.memory[key].shape)
            self.memory[key] = np.vstack((self.memory[key], Transition(*args)[i]))
            # print("after add:", self.memory[key].shape)
            
    def sample(self, batch_size) -> dict: # randomized
        batch = {}
        # print("self.memory['state'].shape[0]", self.memory['state'].shape[0])
        random_indices = np.random.randint(0, self.memory['state'].shape[0], batch_size)
        for key in self.keys:
            # batch[key] = self.memory[key][:batch_size]
            # print("key", key)
            # print(self.memory[key].shape)
            # Use the random indices to select samples from the array
            batch[key]  = self.memory[key][random_indices]
        # print(batch)
        return batch

    # def sample(self, batch_size) -> dict:
    #     batch = {}
    #     total_samples = self.memory['state'].shape[0]

    #     # Ensure the current index does not exceed the total number of samples
    #     if self.current_index + batch_size > total_samples:
    #         self.current_index = 0

    #     # Select the samples sequentially
    #     indices = np.arange(self.current_index, self.current_index + batch_size)
    #     for key in self.keys:
    #         batch[key] = self.memory[key][indices]

    #     self.current_index += batch_size
    #     return batch
    
    def print_memory(self):
        print("Current memory:")
        for i, transition in enumerate(self.memory):
            print(f"Element {i}: {transition}")



class GATWrapper(gym.Env):
    def __init__(self, env, current_state, forward_model, inverse_model, batch=None):
        if batch:
            self.s = batch['state']
            self.a = batch['action']
            self.s_prime = batch['next_state']

        self.env = env
        self.current_state = current_state
        self.forward_model = forward_model.eval()
        self.inverse_model = inverse_model.eval()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.error = deque(maxlen=1000)
    
    def reset(self, seed=-1):
        if seed == -1:
            return self.env.reset()
        return self.env.reset(seed=seed)
    
    def step(self, action):
        s = torch.tensor(self.current_state) # self.env.get_observation(self.env.simulator.get_current_state()))
        a = torch.tensor(action)
        
        # random_indices = np.random.randint(0, 1048, 1)[0]
        # print(random_indices)
        # s = torch.tensor(self.s)[random_indices]
        # a = torch.tensor(self.a)[random_indices]
        # s_prime = self.s_prime[random_indices]

        with torch.no_grad():
            predicted_normalized_delta = self.forward_model(torch.unsqueeze(s, dim=0), torch.unsqueeze(a, dim=0))
        # normalized_delta, _, _ = normalize(self.s_prime - self.s)
        # print('normalized_delta', normalized_delta) # next_state torch.Size([1, 8])
        # print("---------------------")
        # print("s", s.numpy())
        # print("predicted_normalized_delta", predicted_normalized_delta)
        # grounded_action_true = self.inverse_model(torch.unsqueeze(s, dim=0), torch.unsqueeze(torch.tensor(s_prime), dim=0))
        with torch.no_grad():
            # normalized_delta, _, _ = normalize_tensor(torch.unsqueeze(s, dim=0) - next_state)
            grounded_action = self.inverse_model(torch.unsqueeze(s, dim=0), predicted_normalized_delta)
        grounded_action = torch.squeeze(grounded_action, dim=0)
        # print("a", a.numpy(), "a'", grounded_action.detach().numpy())
        self.error.append((grounded_action - a).abs().sum().detach().numpy().item())
        # print("self.error", self.error[-1])

        state_dict, A, B, C, D = self.env.step(a)
        # print('state_dict', state_dict)
        self.current_state = state_dict
        return state_dict, A, B, C, D


"""-------------------------------------- Forward Model --------------------------------------"""
def compute_loss(predictions, targets):
    loss = F.mse_loss(predictions, targets)
    # _loss = (predictions - targets).square().mean()
    # print('loss type', _loss)
    return loss
    
def normalize_tensor(tensor, range_min=-1, range_max=1):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (range_max - range_min) + range_min
    return normalized_tensor, tensor_min, tensor_max

class ForwardKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ForwardKinematicsCNN, self).__init__()
        
        # Increase the size of the network by adding more layers and neurons
        self.fc1 = nn.Linear(n_observations + n_actions, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, n_observations)

        self.dropout = nn.Dropout(0.5)
        self.double()

    def forward(self, state, action):
        # print("dim", state.shape, action.shape)
        x = torch.cat((state, action), dim=1)
        # print("dim", x.shape)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        x = self.fc7(x) # prediction is normalized delta -> [-1, 1]
        # normalized_tensor, _, _ = normalize_tensor(x)  # Convert back to tensor
        return x #normalized_tensor


# need a new model for real data's format

class InverseKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(InverseKinematicsCNN, self).__init__()
        self.fc1 = nn.Linear(n_observations * 2, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, n_actions)
        self.double()

    def forward(self, state, state_delta):
        x = torch.cat((state, state_delta), dim=1)   
        # x = x.to(torch.double) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.fc7(x)  # prediction is action -> [-1, 1]
        x = torch.tanh(x)
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


def check_state_transition(data):
    """
    This function checks if every element in data['state'] minus the corresponding element in 
    data['next_state'] is according to the sign of data['action'].
    
    Parameters:
    data (dict): A dictionary containing 'state', 'next_state', and 'action'.
    
    Returns:
    bool: True if the condition is met for all elements, False otherwise.
    """
    state = data['state']
    next_state = data['next_state']
    action = data['action']

    for i in range(len(state)):
        difference = state[i] - next_state[i]
        if action[i%2] == 0:
            if difference != 0:
                return False
        elif (difference > 0 and action[i%2] < 0) or (difference < 0 and action[i%2] > 0):
            return False
    return True


class GroundedActionTransformation():
    def __init__(self, args, data, sim_env, real_env, log_dir):

        self.sim_env = sim_env
        self.real_env = real_env
        self.sim_current_state_dict, info = self.sim_env.reset()
        self.real_current_state_dict, info = self.real_env.reset()
        # self.sim_current_state = self.sim_env.simulator.get_current_state() # or self.sim_env.current_state
        # self.real_current_state = self.real_env.simulator.get_current_state() # or self.real_env.current_state
        # print("self.sim_current_state", self.sim_current_state)
        # print("self.real_current_state", self.real_current_state)
        self.sim_current_state = self.sim_env.simulator.get_current_state() # or self.sim_env.current_state
        self.real_current_state = self.real_env.simulator.get_current_state() # or self.real_env.current_state

        print("data['state']", data['state'].shape)
        print("data['action']", data['action'].shape)

        self.n_observations_sim = data['state'].shape[1] # observations is paddle x y vx vy + puck last 5 x y (occ0/1), actions is dx dy
        self.n_actions =  data['action'].shape[1]
        # print(self.n_observations, self.n_actions)
        self.forward_model = init_forward_model(self.n_observations_sim, self.n_actions) # for real
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.scheduler_forward = StepLR(self.forward_optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler
        # self.scheduler_forward = optim.lr_scheduler.ReduceLROnPlateau(self.forward_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.forward_batch_size = 1024

        
        self.inverse_model = init_inverse_model(self.n_observations_sim, self.n_actions)
        self.inverse_optimizer = optim.AdamW(self.inverse_model.parameters(), lr=0.0001, weight_decay=1e-5)
        # self.scheduler_inverse = StepLR(self.inverse_optimizer, step_size=100, gamma=0.9)
        self.scheduler_inverse = CosineAnnealingLR(self.inverse_optimizer, T_max=100)
        self.inverse_batch_size = 1024

        self.policy = "optimized_policy0"
        self.sim_batch_size = 2048

        # TODO: need to define the buffers, check with Michael Munje about how we are using buffers
        # for the current RL training code
        self.real_buffer = init_real_buffer()
        self.sim_buffer = init_sim_buffer()
        self.log_dir = log_dir
        # before we have trained anything, don't use the grounded transform
        self.first = True
    
    def load_inverse(self, pth):
        self.inverse_model.load_state_dict(torch.load(pth))
    
    def load_forward(self, pth):
        self.forward_model.load_state_dict(torch.load(pth))
    
    # def grounded_transform(self, state, action):
    #     if self.first: return action
    #     return self.inverse_model(state, self.forward_model(state, action))
    
    def load_buffer_from_dict(self, real_dict, sim_dict):
        self.real_buffer.load_from_dict(real_dict)
        self.sim_buffer.load_from_dict(sim_dict)
    
    def add_data(self, trajectories):
        # TODO: pretty sure you can't do this
        for trajectory in trajectories:
            self.real_buffer.add(*trajectory)
            self.sim_buffer.add(*trajectory) # proxy ( will be removed )

    def rollout_sim(self, num_frames):
        data = {
            'state': [],
            'next_state':  [],
            'delta': [],
            'action':  [],
            'rew':  [],
            'term': [],
            'trunc': [],
            'info':  [],
        }
        for i in range(num_frames):
            if (num_frames > 10 and i % 1000 == 0) or num_frames == 10:
                print("In rollout_sim, num_frames processed:", i)
            obs_dict = self.sim_current_state_dict # self.real_env.get_observation(self.real_current_state)
            obs = self.sim_env.get_observation(self.sim_current_state)
            model = PPO.load(self.policy, env=self.sim_env)
            act, _states = model.predict(obs_dict) # apply policy to sim_env
            # print(self.sim_env.step(act))
            obs_next, reward, is_finished, truncated, info = self.sim_env.step(act)
            self.sim_current_state = self.sim_env.simulator.get_current_state() # self.sim_env.simulator.get_current_state()
            self.sim_current_state_dict = obs_next
            data['state'].append(np.array(obs))
            data['next_state'].append(np.array(obs_next))
            data['delta'].append(np.array(obs - obs_next))
            data['action'].append(np.array(act))
            data['rew'].append(np.array([reward]))
            data['term'].append(np.array(np.array([int(is_finished)])))
            data['trunc'].append(np.array([int(truncated)]))
            data['info'].append(np.array([info]))
        data['state'] = np.array(data['state'])# not taking goal_x goal_y
        data['next_state'] = np.array(data['next_state'])
        data['delta'] = np.array(data['delta'])
        data['action'] = np.array(data['action'])
        data['rew'] = np.array(data['rew'])
        data['term'] = np.array(data['term'])
        data['trunc'] = np.array(data['trunc'])
        data['info'] = np.array(data['info'])
        self.sim_buffer.add_from_dict(data) # perform re-normalization
        self.inverse_model = init_inverse_model(self.n_observations_sim, self.n_actions)
        self.inverse_optimizer = optim.AdamW(self.inverse_model.parameters(), lr=0.0001, weight_decay=1e-6)
        # self.scheduler_inverse = StepLR(self.inverse_optimizer, step_size=100, gamma=0.9)
        self.scheduler_inverse = CosineAnnealingLR(self.inverse_optimizer, T_max=100)
        # self.scheduler_inverse = ReduceLROnPlateau(self.inverse_optimizer, mode='min', factor=0.5, patience=5)


    def rollout_real(self, num_frames):
        data = {
            'state': [],
            'next_state':  [],
            'delta': [],
            'action':  [],
            'rew':  [],
            'term': [],
            'trunc': [],
            'info':  [],
        }
        for i in range(num_frames):
            if (num_frames > 10 and i % 1000 == 0) or num_frames == 10:
                print("In rollout_real, num_frames processed:", i)
            obs_dict = self.real_current_state_dict # self.real_env.get_observation(self.real_current_state)
            obs = self.real_env.get_observation(self.real_current_state)
            model = PPO.load(self.policy, env=self.real_env)
            act, _states = model.predict(obs_dict) # apply policy to real_env
            # print(self.real_env.step(act))
            obs_next, reward, is_finished, truncated, info = self.real_env.step(act)
            self.real_current_state = self.real_env.simulator.get_current_state() # self.real_env.simulator.get_current_state()
            self.real_current_state_dict = obs_next
            # print(type(obs), obs, type(obs_next), obs_next)
            data['state'].append(np.array(obs))
            data['next_state'].append(np.array(obs_next))
            data['delta'].append(np.array(obs - obs_next))
            data['action'].append(np.array(act))
            data['rew'].append(np.array([reward]))
            data['term'].append(np.array(np.array([int(is_finished)])))
            data['trunc'].append(np.array([int(truncated)]))
            data['info'].append(np.array([info]))
            
        data['state'] = np.array(data['state'])# not taking goal_x goal_y
        data['next_state'] = np.array(data['next_state'])
        data['delta'] = np.array(data['delta'])
        data['action'] = np.array(data['action'])
        data['rew'] = np.array(data['rew'])
        data['term'] = np.array(data['term'])
        data['trunc'] = np.array(data['trunc'])
        data['info'] = np.array(data['info'])
        self.real_buffer.add_from_dict(data) # ('state', 'next_state', 'delta', 'action', 'rew', 'term', 'trunc', 'info')
        # TODO: might need to make this actually work
        # Initialize forward model
        self.forward_model = init_forward_model(self.n_observations_sim, self.n_actions) # for real
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.scheduler_forward = StepLR(self.forward_optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler
        # self.scheduler_forward = optim.lr_scheduler.ReduceLROnPlateau(self.forward_optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # TODO: might need to make this actually work
    
    def evaluate_on_dataset(self):
        """Sample a batch size of 500 once, then return the mean action_recovery_error"""
        batch_size = 500
        # Set the model to evaluation mode
        self.forward_model.eval()
        self.inverse_model.eval()
        # change this to real_buffer if evaluating on testing data
        data = self.real_buffer.sample(batch_size)
        s = torch.tensor(data['state'])
        a = torch.tensor(data['action'])
        with torch.no_grad():
            # print(s.shape, a.shape)
            predicted_normalized_delta = self.forward_model(s, a)
        # normalized_delta, _, _ = normalize(data['next_state'] - data['state'])
        normalized_delta = data['delta']
        # print(predicted_normalized_delta, torch.tensor(normalized_delta))
        forward_loss = compute_loss(predicted_normalized_delta, torch.tensor(normalized_delta))
        # print("In evaluate_on_train, forward model loss:", forward_loss.detach())
        with torch.no_grad():
            grounded_action = self.inverse_model(s, predicted_normalized_delta)
            inverse_loss = compute_loss(self.inverse_model(s, torch.tensor(normalized_delta)), a)
            # print("In evaluate_on_train, inverse model loss:", inverse_loss.detach())
        grounded_action = torch.squeeze(grounded_action, dim=0)
        # print("a", a.numpy(), "a'", grounded_action.detach().numpy())
        error = (grounded_action - a).abs().sum().detach().numpy().item()
        # print("Action recovery loss:", error)
        return forward_loss, inverse_loss, error / batch_size

    def train_sim(self, num_iters, i):
        # TODO: try to utilize the same train function as other components
        # TODO: train should automatically add to self.sim_buffer, so we don't need to keep
        if i == 0:
            model = PPO("MlpPolicy", self.sim_env, verbose=1)
            # model = PPO.load("baseline_models/PointMaze/PointMaze_1/model", self.sim_env)
            # model = PPO.load("gat_log/PointMaze/optimized_policy9", self.sim_env)
        else:
            # print("self.sim_current_state",self.sim_current_state)
            # print("current sim buffer index:", self.sim_buffer.current_index)
            state, info = self.sim_env.reset() # before training every iteration, reset simulator
            self.sim_current_state = self.sim_env.simulator.get_current_state() # udpate current state
            grounded_sim_env = GATWrapper(self.sim_env, self.sim_env.get_observation(self.sim_current_state), self.forward_model, self.inverse_model, self.sim_buffer.sample(self.sim_batch_size))
            model = PPO.load(self.policy, env=grounded_sim_env)
            model.learn(num_iters)
        if i != 0:
            print("Action transform error:", np.mean(grounded_sim_env.error))
        self.policy = self.log_dir + "/optimized_policy" + str(i) # self.policy stores the current model name
        model.save(self.policy)
        return
    

    def train_forward(self, num_iters, verbose=True):
        self.forward_model.train()
        loss_values = []
        for i in range(num_iters):
            data = self.real_buffer.sample(self.forward_batch_size)
            # pred_next_state = self.forward_model(data['state'], data['action'])
            normalized_delta_pred = self.forward_model(torch.tensor(data['state'], dtype=torch.double), torch.tensor(data['action'], dtype=torch.double))
            # print("before nomalize", data['next_state'] - data['state'])
            # normalized_delta, _, _ = normalize(data['next_state'] - data['state'])            
            # print("after nomalize", normalized_delta)
            # input()
            normalized_delta = data['delta']
            # print("nomalize", normalized_delta, normalized_delta.shape)
            loss = compute_loss(normalized_delta_pred, torch.tensor(normalized_delta, dtype=torch.double))
            # if i > 4000:
            #     if loss.detach() > 0.1:
            #         print('In forward', data['state'][:10], normalized_delta[:10], normalized_delta_pred[:10].detach().numpy(), data['action'][:10])
            if verbose and (num_iters > 10 and i % 100 == 0) or num_iters == 10:
                print("In forward model, loss:", loss.detach(), ", iter:", i)
            # self.forward_optimizer.step(loss.mean())
            loss.backward()  # Backward pass
            nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)
            self.forward_optimizer.step()
            self.scheduler_forward.step()
            loss_values.append(loss.item())
        return loss_values, self.forward_model
    

# forward(state, action) -> normalized_delta
# inverse(state, normalized_delta) -> action
    def train_inverse(self, num_iters, verbose=True):
        self.inverse_model.train()
        loss_values = []
        for i in range(num_iters):
            self.inverse_optimizer.zero_grad()
            # print("current sim buffer index:", self.sim_buffer.current_index)
            data = self.sim_buffer.sample(self.inverse_batch_size)
            normalized_delta = data['delta']
            # normalized_delta, _, _ = normalize(data['next_state'] - data['state'])
            # print(normalized_delta)
            # input()
            pred_action = self.inverse_model(torch.tensor(data['state'], dtype=torch.double), torch.tensor(normalized_delta, dtype=torch.double))
            loss = compute_loss(pred_action, torch.tensor(data['action'], dtype=torch.double))
            # print('In inverse', data['state'][:10], normalized_delta[:10], pred_action[:10].detach().numpy(), data['action'][:10])
            if verbose and (num_iters > 10 and i % 100 == 0) or num_iters == 10:
                print("In inverse model, loss:", loss.detach(), ", iter:", i)
                # print('pred',pred_action[0:30], data['action'][0:30])
            loss.backward()  # Backward pass
            nn.utils.clip_grad_norm_(self.inverse_model.parameters(), max_norm=1.0)
            self.inverse_optimizer.step()
            self.scheduler_inverse.step()
            loss_values.append(loss.item())
        return loss_values, self.inverse_model


def load_dataset0(dataset_pth): # TODO
    avg_delta = np.zeros(300) # len should be 300
    import h5py
    import os
    import ast
    keys = ['obs', 'next_obs', 'act', 'rew', 'term', 'trunc'] #, 'info']
    num_frame = 0 # frames
    data = {
            'state': [],
            'next_state':  [],
            'delta': [],
            'action':  [],
            'rew':  [],
            'term': [],
            'trunc': [],
            'info':  [],
        }
    key_map = {"obs": "state", "next_obs": "next_state", "act": "action", "rew": "rew", "term": "term", "trunc": "trunc"}
    file_idx = 0
    num_trajectories = 0
    for filename in os.listdir(dataset_pth): 
        if num_frame > 200000: #993564 #200000
            data['state'] = np.array(data['state'])
            data['next_state'] = np.array(data['next_state'])
            data['action'] = np.array(data['action'])
            data['rew'] = np.array(data['rew'])
            data['term'] = np.array(data['term'])
            data['trunc'] = np.array(data['trunc'])
            data['info'] = np.expand_dims(np.array(data['info']),axis=1)
            # delta_matrix = np.array(data['delta']) # convert list to np array
            # print("delta_matrix", delta_matrix.shape)
            # normalize each dimension
            # for dim in range(delta_matrix.shape[1]):
            #     normalized_delta, _, _ = normalize(delta_matrix[:, dim])
            #     delta_matrix[:, dim] = normalized_delta
                # print("normalized_delta at dim =", dim, delta_matrix[:, dim], normalized_delta, normalized_delta.shape)
                # input()
            data['delta'] = np.array(data['delta'])
            # print("data['info']", data['info'].shape)
            # produce histogram
            # mean_delta_matrix = np.mean(delta_matrix, axis=2)
            # print("delta_matrix", mean_delta_matrix.shape)
            # print("num_trajectories", num_trajectories)
            # print("mean_delta_matrix.shape[1]", mean_delta_matrix.shape[1])
            # # plt.scatter(time_steps, avg_delta/num_trajectories, color='blue', alpha=0.7, s=5)
            # for pic in range(mean_delta_matrix.shape[1]):
            #     cur_timestep = mean_delta_matrix[:, pic]
            #     print("cur_timestep", cur_timestep.shape)
            #     plt.hist(cur_timestep, bins=100, color='blue', alpha=0.7)
            #     plt.xlabel('avg delta')
            #     plt.ylabel('Frequency')
            #     plt.title('Histogram with 100 Bins')
            #     plt.savefig("gat_log/PointMazeRandom/histogram/histogram" + str(pic) + ".png")
            #     plt.close()
            # print("data['state']",data['state'].shape)
            # print("data['delta']", data['delta'].shape)
            # eee
            return data
        file_path = os.path.join(dataset_pth, filename)
        # print("Reading h5py file from", file_path)
        num_trajectories += 1
        with h5py.File(file_path, 'r') as hf:
            for key in keys:
                dataset = hf[key]
                data[key_map[key]].extend(dataset[:].tolist())
                # print('state', hf['obs'][:].tolist())
                # print(f"Key: {key}")
                # print(f"Data: {data[key][0]}")
                # print(f"Shape: {dataset.shape}")
                # print(f"Dtype: {dataset.dtype}")
                if key == 'term':
                    n = dataset.shape[0]
                    num_frame += n
            # info
            data['info'].extend([{}] * n)
            # append the trajectory's state deltas
            state_delta = np.array(hf['next_obs'][:]) - np.array(hf['obs'][:])
            # print("np.array(hf['obs'][:][-3:,:4])",  np.array(hf['obs'][:][-3:,:4])) # the last three timesteps
            # print("state_delta",  state_delta[:, 5], state_delta[:, 7]) # the last three timesteps
            # input()
            data['delta'].extend(state_delta)
            # print(hf['obs'][:], hf['next_obs'][:])
            # input()
            # avg_delta_old = avg_delta

            # print("state_delta", state_delta.shape)
            # avg_delta = avg_delta + np.mean(state_delta, axis=1)
            # print("avg_delta", avg_delta)
            # input()
            # avg_delta.append(np.mean(np.array(data['next_state']) - np.array(data['state'])))
        file_idx += 1
    return None


def real_data_key_mapping(real_dataset):
    real_keys = ["observations","actions","rewards","next_observations","terminals"]
    key_map = {"observations": "state", "next_observations": "next_state", "actions": "action", "rewards": "rew", "terminals": "term"}
    d = {}
    keys = key_map.keys()
    # print(keys)
    for key in keys:
        # print(key_map[key])
        d[key_map[key]] = real_dataset[key]
    d['trunc'] = d['term'] # proxy
    d['info'] = np.array([{}] * len(d['term']))
    return d

def save_loss(loss, label, plot_fp, title='Training Loss'):
    plt.plot(loss, label=label, color='b', linestyle='-', marker='o', markersize=1)
    # plt.plot(loss_values_f, label='Forward Model', color='r', linestyle='--', marker='x', markersize=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(plot_fp)
    plt.close()

def readDataset(args):
    dataset_pth = (args.dataset_pth)
    import h5py
    import os
    import ast
    keys = ['obs', 'next_obs', 'act', 'rew', 'term', 'trunc'] #, 'info']
    num_frame = 0 # frames
    data = {
            'state': [],
            'next_state':  [],
            'action':  [],
            'rew':  [],
            'term': [],
            'trunc': [],
            'info':  [],
        }
    key_map = {"obs": "state", "next_obs": "next_state", "act": "action", "rew": "rew", "term": "term", "trunc": "trunc"}
    for filename in os.listdir(dataset_pth):
        file_path = os.path.join(dataset_pth, filename)
        with h5py.File(file_path, 'r') as hf:
            for i in range(len(hf['obs'][:].tolist())):
                
                print('s', hf['obs'][:].tolist()[i])
                print("s'", hf['next_obs'][:].tolist()[i])
                print('a', hf['act'][:].tolist()[i])
                print('term', hf['term'][:].tolist()[i])
                # input()


def evaluate_GAT(args, mode,):
    train_forward = True
    train_inverse = True
    print('Loading sim_env and real_env from', args.sim_cfg, args.real_cfg)
    with open(args.sim_cfg, 'r') as f:
        sim_air_hockey_cfg = yaml.safe_load(f)

    with open(args.real_cfg, 'r') as f:
        real_air_hockey_cfg = yaml.safe_load(f)

    sim_env = AirHockeyEnv(sim_air_hockey_cfg['air_hockey'])
    real_env = AirHockeyEnv(real_air_hockey_cfg['air_hockey'])
    
    real_env.observation_space = sim_env.observation_space
    
    log_dir = None
    if mode == '11':
        log_dir = 'gat_log/AirHockey11/'
        print("loading sim dataset from", args.dataset_pth1)
        print("loading real dataset from", args.dataset_pth1)
        sim_dataset = load_dataset0(args.dataset_pth1)
        real_dataset = load_dataset0(args.dataset_pth1) 
        label = 'train on random action, test on random action'
    elif mode == '12':
        log_dir = 'gat_log/AirHockey12/'
        print("loading sim dataset from", args.dataset_pth1)
        print("loading real dataset from", args.dataset_pth2)
        sim_dataset = load_dataset0(args.dataset_pth1)
        real_dataset = load_dataset0(args.dataset_pth2)
        label = 'train on random action, test on random network' 
    elif mode == '13':
        log_dir = 'gat_log/AirHockey13/'
        print("loading sim dataset from", args.dataset_pth1)
        print("loading real dataset from", args.dataset_pth3)
        sim_dataset = load_dataset0(args.dataset_pth1)
        real_dataset = load_dataset0(args.dataset_pth3)
        label = 'train on random action, test on trained network'
    elif mode == '21':
        log_dir = 'gat_log/AirHockey21/'
        print("loading sim dataset from", args.dataset_pth2)
        print("loading real dataset from", args.dataset_pth1)
        sim_dataset = load_dataset0(args.dataset_pth2)
        real_dataset = load_dataset0(args.dataset_pth1)
        label = 'train on random network, test on random action'
    elif mode == '22':
        log_dir = 'gat_log/AirHockey22/'
        print("loading sim dataset from", args.dataset_pth2)
        print("loading real dataset from", args.dataset_pth2)
        sim_dataset = load_dataset0(args.dataset_pth2)
        real_dataset = load_dataset0(args.dataset_pth2)
        label = 'train on random network, test on random network'
    elif mode == '23':
        log_dir = 'gat_log/AirHockey23/'
        print("loading sim dataset from", args.dataset_pth2)
        print("loading real dataset from", args.dataset_pth3)
        sim_dataset = load_dataset0(args.dataset_pth2)
        real_dataset = load_dataset0(args.dataset_pth3)
        label = 'train on random network, test on trained network'
    elif mode == '31':
        log_dir = 'gat_log/AirHockey31/'
        print("loading sim dataset from", args.dataset_pth3)
        print("loading real dataset from", args.dataset_pth1)
        sim_dataset = load_dataset0(args.dataset_pth3)
        real_dataset = load_dataset0(args.dataset_pth1)
        label = 'train on trained network, test on random action'
    elif mode == '32':
        log_dir = 'gat_log/AirHockey32/'
        print("loading sim dataset from", args.dataset_pth3)
        print("loading real dataset from", args.dataset_pth2)
        sim_dataset = load_dataset0(args.dataset_pth3)
        real_dataset = load_dataset0(args.dataset_pth2)
        label = 'train on trained network, test on random network'
    elif mode == '33':
        log_dir = 'gat_log/AirHockey33/'
        print("loading sim dataset from", args.dataset_pth3)
        print("loading real dataset from", args.dataset_pth3)
        sim_dataset = load_dataset0(args.dataset_pth3)
        real_dataset = load_dataset0(args.dataset_pth3)   
        label = 'train on trained network, test on trained network'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + '/training_summary', exist_ok=True)
    #real_data_key_mapping(load_dataset("/nfs/homes/air_hockey/data", "history", real_env))

    gat = GroundedActionTransformation(args, sim_dataset, sim_env, real_env, log_dir)
    gat.load_buffer_from_dict(real_dict=real_dataset, sim_dict=sim_dataset)
    
    num_epochs = 6000
    eval_epoch_interval = 20
    action_recovery_error_list = []
    forward_testing_loss_list = []
    inverse_testing_loss_list = []
    loss_values_i = []
    loss_values_f = []
    for eval_freq in range(num_epochs//eval_epoch_interval): # eval every 20 epochs, repeat for 300 times
        loss_values_i0, inverse_model = gat.train_inverse(eval_epoch_interval, verbose=False)
        loss_values_i.extend(loss_values_i0)
        loss_values_f0, forward_model = gat.train_forward(eval_epoch_interval, verbose=False)
        loss_values_f.extend(loss_values_f0)
        forward_testing_loss, inverse_testing_loss, action_recovery_error = gat.evaluate_on_dataset() # sample 500 data for once
        if eval_freq % 100 == 0:
            print(str(eval_freq*20) + " epochs action_recovery_error:", action_recovery_error, '; forward and inverse loss:', forward_testing_loss, inverse_testing_loss)
        action_recovery_error_list.append(action_recovery_error)
        forward_testing_loss_list.append(forward_testing_loss)
        inverse_testing_loss_list.append(inverse_testing_loss)
    # save final inverse model
    torch.save(inverse_model.state_dict(), log_dir + 'inverse_model_pt.pth')
    plot_fp = log_dir + '/training_summary/training_summary_i0.png'
    save_loss(loss_values_i, 'Inverse Model', plot_fp)
    # save final forward model
    torch.save(forward_model.state_dict(), log_dir + 'forward_model_pt.pth')
    plot_fp = log_dir + '/training_summary/training_summary_f0.png'
    save_loss(loss_values_f, 'Forward Model', plot_fp)
    # save action recovery error
    x_axis = np.arange(0, num_epochs, eval_epoch_interval)
    plot_fp = log_dir + '/training_summary/testing_error.png'
    plt.plot(x_axis, action_recovery_error_list, label='Action Recovery Error', color='b', linestyle='-', marker='o', markersize=1)
    plt.plot(x_axis, forward_testing_loss_list, label='Forward Model Testing Error', color='r', linestyle='-', marker='x', markersize=1)
    plt.plot(x_axis, inverse_testing_loss_list, label='Inverse Model Testing Error', color='g', linestyle='-', marker='x', markersize=1)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(label)
    # plt.xticks(np.arange(0, 6000, 20))
    plt.legend()
    plt.savefig(plot_fp)
    plt.close()
    # plot_fp = log_dir + '/training_summary/action_recovery_error.png'
    # save_loss(action_recovery_error, 'Action Recovery Error', plot_fp)

def train_GAT(args, log_dir):
    with open(args.sim_cfg, 'r') as f:
        sim_air_hockey_cfg = yaml.safe_load(f)
    with open(args.real_cfg, 'r') as f:
        real_air_hockey_cfg = yaml.safe_load(f)
    train_forward = True
    train_inverse = True
    pth_dir_inverse = 'gat_log/AirHockey33/'
    pth_dir_forward = 'gat_log/AirHockey33/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + '/training_summary', exist_ok=True)
    sim_env = AirHockeyEnv(sim_air_hockey_cfg['air_hockey'])
    real_env = AirHockeyEnv(real_air_hockey_cfg['air_hockey'])
    real_env.observation_space = sim_env.observation_space

    sim_dataset = load_dataset0(args.dataset_pth3)
    real_dataset = load_dataset0(args.dataset_pth3) 
    #real_data_key_mapping(load_dataset("/nfs/homes/air_hockey/data", "history", real_env))

    gat = GroundedActionTransformation(args, sim_dataset, sim_env, real_env, log_dir)
    print("Adding data...")
    gat.load_buffer_from_dict(real_dict=real_dataset, sim_dict=sim_dataset)
    print("Starting train_sim...")
    gat.train_sim(args.initial_rl_training, i=0)
    print("train_sim finished...")

    # gat.rollout_sim(args.num_sim_steps) #need this before train_inversefor actual implementation
    if train_inverse:
        print("Starting train_inverse...")
        loss_values_i0, inverse_model = gat.train_inverse(args.initial_inverse_training)
        torch.save(inverse_model.state_dict(), log_dir + 'inverse_model_pt.pth')
        plot_fp = log_dir + '/training_summary/training_summary_i0.png'
        save_loss(loss_values_i0, 'Inverse Model', plot_fp)
        print("train_inverse finished...")
    else: #eval: load our pretrained model
        print("Loading pretrained inverse models from", pth_dir_inverse)
        gat.load_inverse(pth_dir_inverse + 'inverse_model_pt.pth')
    
    if train_forward:
        print("Starting train_forward...")
        loss_values_f0, forward_model = gat.train_forward(args.initial_forward_training)
        torch.save(forward_model.state_dict(), log_dir + 'forward_model_pt.pth')
        print('Pre-trained model saving to', log_dir + 'inverse_model_pt.pth', "and", log_dir + 'forward_model_pt.pth')
        plot_fp = log_dir + '/training_summary/training_summary_f0.png'
        save_loss(loss_values_f0, 'Forward Model', plot_fp)
    else:
        print("Loading pretrained forward models from", pth_dir_forward)
        gat.load_forward(pth_dir_forward + 'forward_model_pt.pth')    

    for i in range(1, args.num_real_sim_iters):
        print('Iteration', i, 'started.')
        gat.train_sim(args.n_rl_timesteps, i)
        print('rollout_real', i, 'started.')
        gat.rollout_real(args.num_real_steps)
        
        print('rollout_sim', i, 'started.')
        gat.rollout_sim(args.num_sim_steps)
        print('Iteration', i, 'training started.')
        if i > 3:
            loss_values_i, inverse_model_i = gat.train_inverse(10000)
        else:
            loss_values_i, inverse_model_i = gat.train_inverse(args.inverse_iters)
        loss_values_f, forward_model_i = gat.train_forward(args.forward_iters)
        print('Iteration', i, 'finished.')

        plot_fp = log_dir + 'training_summary/training_summary_i'+str(i)+'.png'
        save_loss(loss_values_i, 'Inverse Model', plot_fp)
        plot_fp = log_dir + 'training_summary/training_summary_f'+str(i)+'.png'
        save_loss(loss_values_f, 'Forward Model', plot_fp)

    print("train_GAT finished,")
    # save final dynamics_models
    os.makedirs(log_dir + 'ft_dynamics_model/', exist_ok=True)
    torch.save(inverse_model_i.state_dict(), log_dir + 'ft_dynamics_model/' + args.inverse_pt_pth)
    torch.save(forward_model_i.state_dict(), log_dir + 'ft_dynamics_model/' + args.forward_pt_pth)
    # gat.real_buffer.print_memory()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--sim-cfg', type=str, default='configs/gat/puck_height3.yaml', help='Path to the configuration file.')
    parser.add_argument('--real-cfg', type=str, default='configs/gat/puck_height_real.yaml', help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default=None, help='Path to the dataset file.') # real-cfg
    parser.add_argument('--dataset_pth1', type=str, default='/nfs/homes/air_hockey/cathy_air_hockey/AirHockeyRandomAction/trajectory_data', help='Path to the dataset file.')
    parser.add_argument('--dataset_pth2', type=str, default='/nfs/homes/air_hockey/cathy_air_hockey/AirHockeyRandomNetwork/trajectory_data', help='Path to the dataset file.')
    parser.add_argument('--dataset_pth3', type=str, default='/nfs/homes/air_hockey/cathy_air_hockey/AirHockeyTrainedNetwork/trajectory_data', help='Path to the dataset file.')
    parser.add_argument('--initial_rl_training', type=int, default=10000, help='initial_rl_training')
    parser.add_argument('--initial_inverse_training', type=int, default=6000, help='initial_inverse_training') #20000
    parser.add_argument('--initial_forward_training', type=int, default=6000, help='initial_forward_training') #22000
    parser.add_argument('--num_real_sim_iters', type=int, default=10, help='num_real_sim_iters')
    parser.add_argument('--num_real_steps', type=int, default=20000, help='num_real_steps') # 20000
    parser.add_argument('--num_sim_steps', type=int, default=20000, help='num_sim_steps')
    parser.add_argument('--n_rl_timesteps', type=int, default=100000, help='n_rl_timesteps') # 100000 in sim #10000 in real
    parser.add_argument('--inverse_iters', type=int, default=10000, help='inverse_iters') #3000
    parser.add_argument('--forward_iters', type=int, default=3000, help='forward_iters')
    parser.add_argument('--inverse_pt_pth', type=str, default='inverse_model.pth', help='forward_iters')
    parser.add_argument('--forward_pt_pth', type=str, default='forward_model.pth', help='forward_iters')

    args = parser.parse_args()

    # with open(args.sim_cfg, 'r') as f:
    #     sim_air_hockey_cfg = yaml.safe_load(f)

    # with open(args.real_cfg, 'r') as f:
    #     real_air_hockey_cfg = yaml.safe_load(f)

    # TODO: write a loader for the dataset, which should load into a list of dicts with three keys: states, actions, dones
    # sorting is: trajectory->key->data

    # input("cont?")
    log_dir = 'gat_log/AirHockey/'

    # print("len of data", len(data))
    # readDataset(args)

    train_GAT(args, log_dir)

    # evaluate_GAT(args, '11')
    # evaluate_GAT(args, '12')
    # evaluate_GAT(args, '13')
    # evaluate_GAT(args, '21')
    # evaluate_GAT(args, '22')
    # evaluate_GAT(args, '23')
    # evaluate_GAT(args, '31')
    # evaluate_GAT(args, '32')
    # evaluate_GAT(args, '33')

# python scripts/grounded_action_transformation.py --sim-cfg configs/gat/puck_height3.yaml --real-cfg configs/gat/puck_height3.yaml --dataset_pth baseline_models/puck_height/air_hockey_agent_20/trajectory_data
# python scripts/grounded_action_transformation.py --sim-cfg configs/gat/puck_height3.yaml --real-cfg configs/gat/puck_height_real.yaml --dataset_pth baseline_models/puck_height/air_hockey_agent_20/trajectory_data
# python scripts/gat_debug.py --sim-cfg configs/gat/puck_height3.yaml --real-cfg configs/gat/puck_height_real.yaml --dataset_pth baseline_models/PointMaze/PointMaze_1/trajectory_data

"""
retrain with real yaml and same final cnn models
do eval again

create pre-trained forward and inverse models with real data 
real data has model with diff obs dims
"""



# pointmaze remove obstacles except the outter wall (perfect inverse dyn)
# no visual obs, use only point position and velo
# see 0 train loss in inverse model


# 6 graphs: action statistics + reward
# Dataset 1: random action
# Dataset 2: random network
# Dataset 3: trained network
# Train on 1, test on 1
# Train on 1, test on 2
# Train on 1, test on 3
# Train on 2, test on 1
# Train on 2, test on 2
# Train on 2, test on 3
# Train on 3, test on 1
# Train on 3, test on 2
# Train on 3, test on 3
# training forward loss + training inverse loss + testing action recovery error = 3
# 3 * 9 = 27