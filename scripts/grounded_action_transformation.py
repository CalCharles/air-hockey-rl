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

"""-------------------------------------- DQN --------------------------------------"""
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_Train():        
    def train(self, env, policy_net, sim_buffer, grounded_transform, num_episodes):
        def select_action(state):
            global steps_done
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

        def optimize_model():
            if len(memory) < BATCH_SIZE:
                return
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()
        
        def plot_durations(show_result=False):
            plt.figure(1)
            durations_t = torch.tensor(episode_durations, dtype=torch.float)
            if show_result:
                plt.title('Result')
            else:
                plt.clf()
                plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.plot(durations_t.numpy())
            # Take 100 episode averages and plot them too
            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())

            plt.pause(0.001)  # pause a bit so that plots are updated
            if is_ipython:
                if not show_result:
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                else:
                    display.display(plt.gcf())

        BATCH_SIZE = 128
        GAMMA = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        TAU = 0.005
        LR = 1e-4

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        steps_done = 0
        episode_durations = []

        trajectories = []

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            trajectories.append(state) # add to trajectories
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    trajectories.append(next_staxte) # add to trajectories

                # Store the transition in memory
                sim_buffer.push(state, action, next_state, reward) # traj too

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(policy_net, target_net, optimizer)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

        print('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()

        return trajectories



"""-------------------------------------- Forward Model --------------------------------------"""
def compute_loss(predictions, targets):
    loss = F.mse_loss(predictions, targets)
    return loss

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ForwardKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions): # observations is paddle x y v + puck last 5 xy (occ0/1), actions is dx dy
        super(ForwardKinematicsCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_observations + n_actions, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512) # a few linear with activation
        self.fc2 = nn.Linear(512, n_observations)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class InverseKinematicsCNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(InverseKinematicsCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_observations * 2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, state, next_state):
        x = torch.cat((state, next_state), dim=1)        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    def __init__(self, args, data, sim_air_hockey_cfg, real_air_hockey_cfg):
        self.sim_air_hockey_cfg = sim_air_hockey_cfg
        self.real_air_hockey_cfg = real_air_hockey_cfg
        self.sim_env = AirHockeyEnv(sim_air_hockey_cfg)
        self.real_env = AirHockeyEnv(real_air_hockey_cfg)

        self.sim_current_state = self.sim_env.simulator.get_current_state() # or self.sim_env.current_state

        # print(type(self.sim_env), self.sim_current_state)
        self.n_actions = self.sim_env.action_space.n
        state, info = self.sim_env.reset()
        self.n_observations = len(state)

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

    gat = GroundedActionTransformation(args, sim_air_hockey_cfg, real_air_hockey_cfg)
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