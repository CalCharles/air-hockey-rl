import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import yaml
from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
import cv2
import gymnasium as gym
import tqdm
import imageio

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DeterministicPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)
    
@torch.no_grad()
def eval_actor(env: gym.Env, actor: nn.Module, dir, device: str, n_episodes: int, seed: int, renderer: AirHockeyRenderer = None, best_rew_so_far=-100000) -> np.ndarray:
    # env.seed(seed)
    actor.eval()
    episode_rewards = []
    frames = []
    for _ in range(n_episodes):
        # import ipdb;ipdb.set_trace()
        state, done = env.reset()[0], False
        episode_reward = 0.0
        while not done:
            if renderer is not None:
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                frames.append(frame)
            if env.goal_conditioned:
                s = state['observation'].flatten()
                g = state['desired_goal'].flatten()
                state = np.concatenate([s, g])  
            state = torch.tensor(state, dtype=torch.float32).to(device)
            action = actor(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)
    if renderer is not None:
        def fps_to_duration(fps):
            return int(1000 * 1/fps)
        if np.mean(episode_rewards) > best_rew_so_far:
            print('Best reward so far:', np.mean(episode_rewards))
            save_fp = os.path.join(dir, 'best_bc.gif')
            imageio.mimsave(save_fp, frames, format='GIF', loop=0, duration=fps_to_duration(20))
    actor.train()
    return np.asarray(episode_rewards)

def train_bc(dataset, actor, optimizer, dir, env, renderer, epochs=10, batch_size=64):
    best_rew = float('-inf')
    states = torch.tensor(dataset["observations"], dtype=torch.float32)
    actions = torch.tensor(dataset["actions"], dtype=torch.float32)
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in tqdm.tqdm(range(epochs)):
        for states, actions in dataloader:
            optimizer.zero_grad()
            predictions = actor(states)
            loss = nn.MSELoss()(predictions, actions)
            loss.backward()
            optimizer.step()
        rews = eval_actor(env, actor, dir, "cpu", 5, epoch, renderer=renderer, best_rew_so_far=best_rew)
        avg_rew = np.mean(rews)
        best_rew = max(best_rew, avg_rew)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

if __name__ == "__main__":
    # Load dataset and environment configurations
    log_dir = 'baseline_models/Hit Goal/air_hockey_agent_1'
    air_hockey_cfg_fp = os.path.join(log_dir, 'model_cfg.yaml')
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    air_hockey_params = air_hockey_cfg['air_hockey']

    env = AirHockeyEnv.from_dict(air_hockey_params)
    renderer = AirHockeyRenderer(env)
    if env.goal_conditioned:
        state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset_fp = os.path.join(log_dir, 'trajs.npy')
    assert os.path.exists(dataset_fp)
    np_dataset = np.load(dataset_fp, allow_pickle=True)
    dataset = {"observations": np_dataset[:,:state_dim], "actions": np_dataset[:,state_dim:state_dim+action_dim]}

    actor = DeterministicPolicy(state_dim, action_dim).to("cpu")
    optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    train_bc(dataset, actor, optimizer, log_dir, env, renderer)
