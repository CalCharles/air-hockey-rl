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
        self.keys = ['state', 'next_state', 'action', 'rew', 'term', 'trunc', 'info']
        self.memory = {
            'state': np.array([]),
            'next_state': np.array([]),
            'action': np.array([]),
            'rew': np.array([]),
            'term': np.array([]),
            'trunc': np.array([]),
            'info': np.array([])
        } #deque([], maxlen=capacity)
    
    def load_from_dict(self, data):
        for key in self.keys:
            self.memory[key] = data[key]
        #     print(self.memory[key].shape)
        # print(data)
        

    def add(self, *args):
        """Save a transition"""
        for i in range(len(self.keys)):
            key = self.keys[i]
            # print("key", key)
            # print("before add:", Transition(*args)[i].shape, self.memory[key].shape)
            self.memory[key] = np.vstack((self.memory[key], Transition(*args)[i]))
            # print("after add:", self.memory[key].shape)
            
    def sample(self, batch_size) -> dict:
        batch = {}
        random_indices = np.random.randint(0, self.memory['state'].shape[0], batch_size)
        for key in self.keys:
            # batch[key] = self.memory[key][:batch_size]

            # Use the random indices to select samples from the array
            batch[key]  = self.memory[key][random_indices]
        # print(batch)
        return batch
    
    def print_memory(self):
        print("Current memory:")
        for i, transition in enumerate(self.memory):
            print(f"Element {i}: {transition}")



class GATWrapper(gym.Env):
    def __init__(self, env, forward_model, inverse_model, batch=None):
        if batch:
            self.s = batch['state']
            self.a = batch['action']
            self.s_prime = batch['next_state']

        self.env = env
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
        s = torch.tensor(self.env.get_observation(self.env.simulator.get_current_state()))
        a = torch.tensor(action)
        
        # random_indices = np.random.randint(0, 1048, 1)[0]
        # print(random_indices)
        # s = torch.tensor(self.s)[random_indices]
        # a = torch.tensor(self.a)[random_indices]
        # s_prime = self.s_prime[random_indices]
        with torch.no_grad():
            next_state = self.forward_model(torch.unsqueeze(s, dim=0), torch.unsqueeze(a, dim=0))
        # print('next_state', next_state.size()) # next_state torch.Size([1, 8])
        # grounded_action_true = self.inverse_model(torch.unsqueeze(s, dim=0), torch.unsqueeze(torch.tensor(s_prime), dim=0))
        with torch.no_grad():
            grounded_action = self.inverse_model(torch.unsqueeze(s, dim=0), next_state)
        grounded_action = torch.squeeze(grounded_action, dim=0)
        # print("s",s.numpy(), "a", a.numpy(), "a'", grounded_action.detach().numpy())
        self.error.append((grounded_action - a).abs().sum().detach().numpy())
        # print("self.error", self.error)
        # input()

        # debug
        # random_indices = np.random.randint(0, 1048, 1)[0]
        # print(random_indices)
        # s = torch.tensor(self.s)[random_indices]
        # a = torch.tensor(self.a)[random_indices]
        # s_prime = self.s_prime[random_indices]

        # next_state = self.forward_model(torch.unsqueeze(s, dim=0), torch.unsqueeze(a, dim=0))
        # # print('next_state', next_state.size()) # next_state torch.Size([1, 8])
        # grounded_action_true = self.inverse_model(torch.unsqueeze(s, dim=0), torch.unsqueeze(torch.tensor(s_prime), dim=0))
        
        # grounded_action = self.inverse_model(torch.unsqueeze(s, dim=0), next_state)
        # grounded_action = torch.squeeze(grounded_action, dim=0)
        # print("s",s.numpy(), "a", a.numpy(), grounded_action_true, "a'", grounded_action.detach().numpy(), "s'", s_prime, next_state)
        # # print("s",s.numpy(), "a", a.numpy(), "a'", grounded_action.detach().numpy())
        # # input()
        # self.error.append((grounded_action - a).abs().sum().detach().numpy())
        # print("self.error", self.error)
        # input()
        return self.env.step(grounded_action.detach())


"""-------------------------------------- Forward Model --------------------------------------"""
def compute_loss(predictions, targets):
    _loss = F.mse_loss(predictions, targets)
    # _loss = (predictions - targets).square().mean()
    # print('loss type', _loss)
    return _loss

class ForwardKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ForwardKinematicsCNN, self).__init__()
        self.fc1 = nn.Linear(n_observations + n_actions, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, n_observations)
        self.double()

    def forward(self, state, action):
        # print(state.size(), action.size())
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = torch.tanh(x)
        return x
    
# need a new model for real data's format

class InverseKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(InverseKinematicsCNN, self).__init__()
        self.fc1 = nn.Linear(n_observations * 2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, n_actions)
        self.double()

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=1)   
        # x = x.to(torch.double) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        x = torch.tanh(x)
        return x

# class InverseKinematicsCNN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(InverseKinematicsCNN, self).__init__()
#         self.fc1 = nn.Linear(n_observations * 2, n_observations)
        
#         # self.fc5 = nn.Linear(256, n_observations)
#         self.double()

#     def forward(self, state, next_state):
#         x = torch.cat((state, next_state), dim=1)   
#         # x = x.to(torch.double) 
        

#         x = self.fc1(x)
#         return x


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
        self.n_observations_sim = data['state'].shape[1] # observations is paddle x y vx vy + puck last 5 x y (occ0/1), actions is dx dy
        self.n_actions =  data['action'].shape[1]
        # print(self.n_observations, self.n_actions)
        self.policy = "optimized_policy0"
        self.forward_model = init_forward_model(self.n_observations_sim, self.n_actions) # for real
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=0.0001)
        self.forward_batch_size = 1024

        self.inverse_model = init_inverse_model(self.n_observations_sim, self.n_actions)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=0.0001)
        self.inverse_batch_size = 1024
    
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
        for i in range(num_frames):
            if (num_frames > 10 and i % 1000 == 0) or num_frames == 10:
                print("In rollout_sim, num_frames processed:", i)
            obs = self.sim_env.get_observation(self.sim_current_state)
            model = PPO.load(self.policy, env=self.sim_env)
            act, _states = model.predict(obs) # apply policy to sim_env
            # print(self.sim_env.step(act))
            obs_next, reward, is_finished, truncated, info = self.sim_env.step(act)
            self.sim_current_state = self.sim_env.simulator.get_current_state()
            traj = (obs, obs_next, act, np.array([reward]), np.array([int(is_finished)]), np.array([int(truncated)]), np.array([info]))
            self.sim_buffer.add(*traj)

    def rollout_real(self, num_frames):
        for i in range(num_frames):
            if (num_frames > 10 and i % 1000 == 0) or num_frames == 10:
                print("In rollout_real, num_frames processed:", i)
            obs = self.real_env.get_observation(self.real_current_state)
            
            model = PPO.load(self.policy, env=self.real_env)
            
            act, _states = model.predict(obs) # apply policy to real_env
            # print(self.real_env.step(act))
            obs_next, reward, is_finished, truncated, info = self.real_env.step(act)
            self.real_current_state = self.real_env.simulator.get_current_state()
            traj = (obs, obs_next, act, np.array([reward]), np.array([int(is_finished)]), np.array([int(truncated)]), np.array([info]))
            # print("traj", traj)
            self.real_buffer.add(*traj) # ('state', 'next_state', 'action', 'rew', 'term', 'trunc', 'info')
        # TODO: might need to make this actually work
    
    def train_sim(self, num_iters, i):
        # TODO: try to utilize the same train function as other components
        # TODO: train should automatically add to self.sim_buffer, so we don't need to keep
        # the trajectories
        # trajectories = self.train_ppo(self.sim_env, self.policy, self.sim_buffer, self.grounded_transform, num_iters)
        # print("i", i)
        if i == 0:
            # model = PPO("MlpPolicy", self.sim_env, verbose=1)
            model = PPO.load("baseline_models/puck_height/air_hockey_agent_20/model", self.sim_env)
        else:
            grounded_sim_env = GATWrapper(self.sim_env, self.forward_model, self.inverse_model, self.sim_buffer.sample(1048))
            model = PPO.load(self.policy, env=grounded_sim_env)
            model.learn(num_iters)
        if i != 0:
            print("Action transform error:", np.mean(grounded_sim_env.error))
        self.policy = self.log_dir + "/optimized_policy" + str(i) # self.policy stores the current model name
        model.save(self.policy)
        return


    def train_forward(self, num_iters):
        self.forward_model.train()
        loss_values = []
        for i in range(num_iters):
            data = self.real_buffer.sample(self.forward_batch_size)
            # pred_next_state = self.forward_model(data['state'], data['action'])
            pred_next_state = self.forward_model(torch.tensor(data['state'], dtype=torch.double), torch.tensor(data['action'], dtype=torch.double))
            # TODO: define these functions generally so we can change them later
            loss = compute_loss(pred_next_state,torch.tensor(data['next_state'], dtype=torch.double))
            if (num_iters > 10 and i % 100 == 0) or num_iters == 10:
                print("In forward model, loss:", loss.detach(), ", iter:", i)
            # self.forward_optimizer.step(loss.mean())
            loss.backward()  # Backward pass
            self.forward_optimizer.step()
            loss_values.append(loss.item())
        return loss_values, self.forward_model

    # def check(self, state, n_state, act, pred_act):
    #     second_col_array1 = state[:, 2]
    #     second_col_array2 = n_state[:, 2]
    #     difference = second_col_array2 - second_col_array1
    #     difference[np.abs(difference)<0.1] = 0
    #     print("difference", difference[:100])
    #     print("n_state", second_col_array2[:100])
    #     print("state", second_col_array1[:100])
    #     act = act[:,0]
    #     print(act.shape)
    #     act[np.abs(act)<0.1] = 0
    #     print("act", act[:100])

    #     difference = np.sign(difference)
    #     act = np.sign(act)

    #     res = np.where(difference == act)[0]
        
    #     print(res[:100])
    #     print("correct rate",len(res)/len(act))


    def train_inverse(self, num_iters):
        self.inverse_model.train()
        loss_values = []
        for i in range(num_iters):
            data = self.sim_buffer.sample(self.inverse_batch_size)
            
            pred_action = self.inverse_model(torch.tensor(data['state'], dtype=torch.double), torch.tensor(data['next_state'], dtype=torch.double))
            # TODO: define these functions generally so we can change them later
            # self.check(data['state'], data['next_state'], data['action'], pred_action)
            # pred_cur_state = self.inverse_model(torch.tensor(data['state'], dtype=torch.double), torch.tensor(data['next_state'], dtype=torch.double))
            # loss = compute_loss(pred_cur_state, torch.tensor(data['state'], dtype=torch.double))
            loss = compute_loss(pred_action, torch.tensor(data['action'], dtype=torch.double))
            if (num_iters > 10 and i % 100 == 0) or num_iters == 10:
                print("In inverse model, loss:", loss.detach(), ", iter:", i)
                # print('pred',pred_action[0:30], data['action'][0:30])
            try:
                loss.backward()  # Backward pass
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
            self.inverse_optimizer.step()
            loss_values.append(loss.item())
        return loss_values, self.inverse_model

def normalize(data):
    # Min-Max Normalization
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data

def load_dataset0(dataset_pth): # TODO
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
        if num_frame > 200000: #993564
            data['state'] = np.array(data['state']) 
            data['next_state'] = np.array(data['next_state'])
            data['action'] = np.array(data['action'])
            data['rew'] = np.array(data['rew'])
            data['term'] = np.array(data['term'])
            data['trunc'] = np.array(data['trunc'])

            # data['state'] = normalize(np.array(data['state']) )
            # data['next_state'] = normalize(np.array(data['next_state']))
            # data['action'] = normalize(np.array(data['action']))
            # data['rew'] = normalize(np.array(data['rew']))
            # data['term'] = normalize(np.array(data['term']))
            # data['trunc'] = normalize(np.array(data['trunc']))
            # print(data['info'])
            data['info'] = np.expand_dims(np.array(data['info']),axis=1)
            # print("data['info']", data['info'].shape)
            return data
        file_path = os.path.join(dataset_pth, filename)
        # print("Reading h5py file from", file_path)
        
        with h5py.File(file_path, 'r') as hf:
            for key in keys:
                dataset = hf[key]
                data[key_map[key]].extend(dataset[:].tolist())
                # print(key, data[key_map[key]])
                # print(f"Key: {key}")
                # print(f"Data: {data[key][0]}")
                # print(f"Shape: {dataset.shape}")
                # print(f"Dtype: {dataset.dtype}")
                if key == 'term':
                    n = dataset.shape[0]
                    num_frame += n
            # info
            data['info'].extend([{}] * n)
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

def save_loss(loss_values_i, loss_values_f, plot_fp):
    plt.plot(loss_values_i, label='Inverse Model', color='b', linestyle='-', marker='o', markersize=1)
    plt.plot(loss_values_f, label='Forward Model', color='r', linestyle='--', marker='x', markersize=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(plot_fp)

def train_GAT(args, sim_air_hockey_cfg, real_air_hockey_cfg, log_dir):
    train = False
    sim_env = AirHockeyEnv(sim_air_hockey_cfg['air_hockey'])
    real_env = AirHockeyEnv(real_air_hockey_cfg['air_hockey'])
    real_env.observation_space = sim_env.observation_space
    sim_dataset = load_dataset0(args.dataset_pth) 
    real_dataset = sim_dataset #real_data_key_mapping(load_dataset("/nfs/homes/air_hockey/data", "history", real_env))

    gat = GroundedActionTransformation(args, sim_dataset, sim_env, real_env, log_dir)
    print("Adding data...")
    gat.load_buffer_from_dict(real_dict=real_dataset, sim_dict=sim_dataset)
    print("Starting train_sim...")
    gat.train_sim(args.initial_rl_training, i=0)
    print("train_sim finished...")
    print("Starting train_inverse...")
    
    # gat.rollout_sim(args.num_sim_steps) #need this before train_inversefor actual implementation
    if train:
        loss_values_i0, inverse_model = gat.train_inverse(args.initial_inverse_training)
        torch.save(inverse_model.state_dict(), log_dir + 'inverse_model_pt.pth')
        print("train_inverse finished...")
        print("Starting train_forward...")
        loss_values_f0, forward_model = gat.train_forward(args.initial_forward_training)
        torch.save(forward_model.state_dict(), log_dir + 'forward_model_pt.pth')
        print('Pre-trained model saving to', log_dir + 'inverse_model_pt.pth', "and", log_dir + 'forward_model_pt.pth')
        plot_fp = log_dir + '/training_summary/training_summary0.png'
        save_loss(loss_values_i0, loss_values_f0, plot_fp)
    else: #eval: load our pretrained model
        gat.load_inverse(log_dir + 'inverse_model_pt.pth')
        gat.load_forward(log_dir + 'forward_model_pt.pth')
    
    for i in range(1, args.num_real_sim_iters):
        print('Iteration', i, 'started.')
        gat.train_sim(args.n_rl_timesteps, i)
        print('rollout_real', i, 'started.')
        gat.rollout_real(args.num_real_steps)
        print('rollout_sim', i, 'started.')
        gat.rollout_sim(args.num_sim_steps)
        loss_values_i, inverse_model_i = gat.train_inverse(args.inverse_iters)
        loss_values_f, forward_model_i = gat.train_forward(args.forward_iters)
        print('Iteration', i, 'finished.')

        plot_fp = log_dir + 'training_summary/training_summary'+str(i)+'.png'
        save_loss(loss_values_i, loss_values_f, plot_fp)
    print("train_GAT finished,")
    # save final dynamics_models
    os.makedirs(log_dir + 'ft_dynamics_model/', exist_ok=True)
    torch.save(inverse_model_i.state_dict(), log_dir + 'ft_dynamics_model/' + args.inverse_pt_pth)
    torch.save(forward_model_i.state_dict(), log_dir + 'ft_dynamics_model/' + args.forward_pt_pth)
    # gat.real_buffer.print_memory()
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--sim-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--real-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default=None, help='Path to the dataset file.') # real-cfg
    parser.add_argument('--initial_rl_training', type=int, default=10000, help='initial_rl_training')
    parser.add_argument('--initial_inverse_training', type=int, default=20000, help='initial_inverse_training')
    parser.add_argument('--initial_forward_training', type=int, default=22000, help='initial_forward_training')
    parser.add_argument('--num_real_sim_iters', type=int, default=10, help='num_real_sim_iters')
    parser.add_argument('--num_real_steps', type=int, default=1000, help='num_real_steps') # 20000
    parser.add_argument('--num_sim_steps', type=int, default=1000, help='num_sim_steps')
    parser.add_argument('--n_rl_timesteps', type=int, default=100000, help='n_rl_timesteps') # 100000 in sim #10000 in real
    parser.add_argument('--inverse_iters', type=int, default=3000, help='inverse_iters') #3000
    parser.add_argument('--forward_iters', type=int, default=3000, help='forward_iters')
    parser.add_argument('--inverse_pt_pth', type=str, default='inverse_model.pth', help='forward_iters')
    parser.add_argument('--forward_pt_pth', type=str, default='forward_model.pth', help='forward_iters')

    args = parser.parse_args()

    with open(args.sim_cfg, 'r') as f:
        sim_air_hockey_cfg = yaml.safe_load(f)

    with open(args.real_cfg, 'r') as f:
        real_air_hockey_cfg = yaml.safe_load(f)

    # TODO: write a loader for the dataset, which should load into a list of dicts with three keys: states, actions, dones
    # sorting is: trajectory->key->data

    # input("cont?")
    log_dir = 'gat_log/puck_height_real/'
    # print("len of data", len(data))
    train_GAT(args, sim_air_hockey_cfg, real_air_hockey_cfg, log_dir)
# python scripts/grounded_action_transformation.py --sim-cfg configs/gat/puck_height3.yaml --real-cfg configs/gat/puck_height3.yaml --dataset_pth baseline_models/puck_height/air_hockey_agent_20/trajectory_data
# python scripts/grounded_action_transformation.py --sim-cfg configs/gat/puck_height3.yaml --real-cfg configs/gat/puck_height_real.yaml --dataset_pth baseline_models/puck_height/air_hockey_agent_20/trajectory_data


"""
retrain with real yaml and same final cnn models
do eval again

create pre-trained forward and inverse models with real data 
real data has model with diff obs dims
"""


