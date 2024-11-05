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
from stable_baselines3 import PPO
import argparse
from scripts.train import init_params

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

def train_bc(dataset, actor, optimizer, dir, env, renderer, epochs=500, batch_size=1024, eval_freq=50, eval_actor=True, device='cpu'):
    best_rew = float('-inf')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mse = nn.MSELoss()
    for epoch in tqdm.tqdm(range(epochs)):
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            predictions = actor.policy(states, deterministic=True)[0]
            loss = mse(predictions, actions)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        if epoch % eval_freq == 0:
            model_fp = os.path.join(dir, f"model_finetune_{epoch}.zip")
            PPO.save(actor, model_fp)
            if eval_actor:
                rews = eval_actor(env, actor, dir, "cpu", 5, epoch, renderer=renderer, best_rew_so_far=best_rew)
                avg_rew = np.mean(rews)
                best_rew = max(best_rew, avg_rew)
        
if __name__ == "__main__":
    # Load dataset and environment configurations
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--log_dir', type=str, help='Directory to save the logs.')
    parser.add_argument('--load_agent_chkpt', type=str, default='/datastor1/calebc/public/data/box2d_models/puck_strike/model.zip', help='Path to the agent checkpoint.')
    parser.add_argument('--config_path', type=str, default='configs/baseline_configs/box2d/strike.yaml', help='Path to the bc config.')
    parser.add_argument('--data_dir', type=str, default='/datastor1/calebc/public/data/mouse/state_data_all_new/puck_strike_dataset.pkl', help='Path to the data config.')
    parser.add_argument('--eval_actor', type=bool, default=False, help='Whether to evaluate the actor.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the training on.')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            air_hockey_cfg = yaml.safe_load(f)
        air_hockey_params = init_params(air_hockey_cfg)
        env = AirHockeyEnv(air_hockey_params)
        renderer = AirHockeyRenderer(env)
        if env.goal_conditioned:
            state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        else:
            state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

    from dataset_management.create_dataset import load_dataset
    import pickle
    if args.data_dir.endswith('pkl'):
        with open(args.data_dir, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = load_dataset(args.data_dir, "pos", env, num_trajectories=-1, save_dir=os.path.join(args.data_dir, 'puck_strike_dataset.pkl'))
    states = torch.tensor(np.vstack([obs[:-1] for obs in dataset["observations"]]), dtype=torch.float32)
    actions = torch.tensor(np.vstack(dataset["actions"]), dtype=torch.float32)
    dataset = TensorDataset(states, actions)

    if args.load_agent_chkpt is not None:
        actor = PPO.load(args.load_agent_chkpt, device=args.device)
    else:
        actor = DeterministicPolicy(state_dim, action_dim).to(args.device)
    optimizer = optim.Adam(actor.policy.parameters(), lr=3e-4)

    train_bc(dataset, actor, optimizer, args.log_dir, env, renderer, eval_actor=args.eval_actor, device=args.device)
