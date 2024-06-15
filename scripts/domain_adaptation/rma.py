from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs, VecEnvObs
from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvObs,
    VecEnvStepReturn,
)
import argparse
import yaml
import os
import random
import wandb
import argparse
import shutil
import os
import yaml
from utils import EvalCallback, save_evaluation_gifs, save_tensorboard_plots
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
from typing import Callable
# from curriculum.classifier_curriculum import CurriculumCallback


# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from gymnasium import Env
from gymnasium.spaces import Space
from copy import deepcopy
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from numpy.typing import NDArray

import pyrallis
import cv2
import imageio

priv_keys = ["puck_density", "puck_damping", "gravity"]

# priv_keys = ["puck_density"]

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    run_name: str = "test"
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    cfg: str = None
    wandb: bool = False
    device: str = 'cpu'
    progress_bar: bool = False
    clear: bool = False

    history_len: int = 10
    phase: int = 1
    phase_2_batch_size: int = 256
    phase_2_data_size: int = 4000
    checkpoint: str = None



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SubprocVecEnv_domain_random(SubprocVecEnv):
    def reset(self, seed=None, **kwargs) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space), self.reset_infos

class Agent(nn.Module):
    def __init__(self, envs, phase=1, extrinsics_dim=6, history_len=10):
        super().__init__()

        self.shared_priv_encoder = nn.Sequential(
            layer_init(nn.Linear(len(priv_keys), 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, extrinsics_dim)),
            nn.Tanh(),
        )


        self.shared_history_encoder = nn.Sequential(
            nn.Linear((np.array(envs.observation_space.shape).prod() + np.prod(envs.action_space.shape)), 32),
            nn.Tanh(),
            nn.Conv1d(in_channels=history_len, out_channels=128, kernel_size=(3,), stride=(1,)),
            nn.Tanh(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(2,), stride=(1,)),
            nn.Tanh(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(2,), stride=(1,)),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(128 * 28, extrinsics_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod() + 6 + np.prod(envs.action_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod() + 6 + np.prod(envs.action_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

        self.phase = phase
        self.extrinsics_dim = extrinsics_dim
        self.history_len = history_len

    def get_priv_extrinsics(self, priv_info):
        return self.shared_priv_encoder(priv_info)
    
    def get_history_extrinsics(self, old_obs, old_action):
        obs_action = torch.cat([old_obs, old_action], dim=2)
        return self.shared_history_encoder(obs_action)
        

    def get_value(self, obs, priv_info, last_action, old_obs, old_action,):
        if self.phase == 1 or self.phase == 2:
            extrinsics = self.shared_priv_encoder(priv_info)
        elif self.phase == 3:
            extrinsics = self.shared_history_encoder(torch.cat([old_obs, old_action], dim=2))

        obs = torch.cat([obs.view(obs.size(0), -1), extrinsics, last_action], dim=1)
        return self.critic(obs)

    def get_action(self, obs, priv_info, last_action, old_obs, old_action,):
        return self.get_action_and_value(obs, priv_info, last_action, old_obs, old_action)[0]

    def get_action_and_value(self, obs, priv_info, last_action, old_obs, old_action, action=None,):
        if self.phase == 1 or self.phase == 2:
            extrinsics = self.shared_priv_encoder(priv_info)
        elif self.phase == 3:
            extrinsics = self.shared_history_encoder(torch.cat([old_obs, old_action], dim=2))
        
        # print(type(obs))
        # print(type(extrinsics))
        # print(type(last_action))
        # import pdb; pdb.set_trace()
        obs = torch.cat([obs.view(obs.size(0), -1), extrinsics, last_action], dim=1)
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def get_intereact_history(next_done, next_obs, action, old_obs, old_action, last_action, history_len):
    """
    Get the interaction history from the terminations.
    """
    
    for i in range(len(next_done)):
        if next_done[i]:
            old_obs[i] = torch.zeros_like(old_obs[i])
            old_action[i] = torch.zeros_like(old_action[i])
            last_action[i] = torch.zeros_like(last_action[i])
        else:
            old_obs[i] = torch.cat([old_obs[i][1:], next_obs[i].unsqueeze(0)], dim=0)
            old_action[i] = torch.cat([old_action[i][1:], action[i].unsqueeze(0)], dim=0)
            last_action[i] = action[i]
    
    return old_obs, old_action, last_action

def get_priv_info(infos, device):
    """
    Get the private information from the environment.
    """

    priv_info = torch.zeros(len(infos), len(priv_keys))
    for i in range(len(infos)):
        for j, key in enumerate(priv_keys):
            priv_info[i, j] = infos[i][key]

    priv_info = priv_info.to(device)
    return priv_info

class SubprocVecEnv_domain_random_eval(SubprocVecEnv_domain_random):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.finished_envs = [False] * len(env_fns)
        self.results = [None] * len(env_fns)

    def step_async(self, actions: np.ndarray) -> None:
        for index, (remote, action) in enumerate(zip(self.remotes, actions)):
            if not self.finished_envs[index]:
                remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        for index, remote in enumerate(self.remotes):
            if not self.finished_envs[index]:
                self.results[index] = remote.recv()
                if self.results[index][2]:
                    self.finished_envs[index] = True
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*self.results)  # type: ignore[assignment]
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]
        

def evaluate(
    agent: torch.nn.Module,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
    air_hockey_cfg: Dict[str, Any] = None,
    args: Dict[str, Any] = None,
    writer: wandb = None,
    global_step: int = 0,
    gif_savepath = "./rma_gifs"
):
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    curr_seed = [0]
    def get_airhockey_env_for_parallel():
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            air_hockey_params['seed'] = curr_seed[0]
            curr_seed[0] += 1
            # Note: With this seed, an individual rng is created for each env
            # It does not affect the global rng!
            env = AirHockeyEnv(air_hockey_params)
            return Monitor(env)
        return _init()
	
    envs = SubprocVecEnv_domain_random_eval([get_airhockey_env_for_parallel for _ in range(eval_episodes)])

    agent.eval()

    next_obs, infos = envs.reset(seed=args.seed)
    # infos = sort_info(infos)
    next_priv_info = get_priv_info(infos, device)
    next_obs = torch.Tensor(next_obs).to(device)
    old_obs = torch.zeros((eval_episodes,) + (args.history_len,) + envs.observation_space.shape).to(device)
    old_action = torch.zeros((eval_episodes,) + (args.history_len,) + envs.action_space.shape).to(device)
    last_action = torch.zeros((eval_episodes,) + envs.action_space.shape).to(device)

    episodic_returns = []
    # if all end, then break and count the number of successful episodes
    while True:
        # import pdb; pdb.set_trace()
        action, _, _, _ = agent.get_action_and_value(obs = next_obs,
                                                    priv_info = next_priv_info,
                                                    last_action = last_action,
                                                    old_obs = old_obs,
                                                    old_action = old_action,)

        step_ret = envs.step(action.cpu().numpy())

        next_obs = torch.Tensor(step_ret[0]).to(device)
        reward = step_ret[1]
        terminations = step_ret[2]
        truncations = np.copy(terminations)
        infos = step_ret[3]
        # import pdb; pdb.set_trace()
        next_priv_info = get_priv_info(infos, device)
        next_done = np.logical_or(terminations, truncations)
        old_obs, old_action, last_action = get_intereact_history(step_ret[2], next_obs, action, old_obs, old_action, last_action, args.history_len)
        

        if np.sum(envs.finished_envs) == eval_episodes:
            success_num = 0
            # count the number of successful episodes
            for info in infos:
                if info["success"]:
                    success_num += 1
            
            success_rate = success_num / len(infos)
            print(f"global_step={global_step}, success rate={success_rate}")
            writer.add_scalar("task/success_rate", success_rate, global_step)
            break

        # print("here !!!!!!!!!")
        # print("done: ", np.sum(envs.finished_envs))
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']


    # Set the return_goal_obs parameter for sac
    air_hockey_cfg['air_hockey']['return_goal_obs'] = 'goal' in air_hockey_cfg['air_hockey']['task'] and 'sac' == air_hockey_cfg['algorithm']
    
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
            
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    air_hockey_params['seed'] = args.seed

    env_test = AirHockeyEnv(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)
    env_test = DummyVecEnv([lambda: env_test])

    # get frames
    frames, robosuite_frames = get_frames(agent, 5, device, air_hockey_cfg, args, env_test, renderer)

    os.makedirs(gif_savepath, exist_ok=True)
    
    # save gif
    assert len(frames) > 0

    def fps_to_duration(fps):
        return int(1000 * 1/fps)
    
    fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
    imageio.mimsave(f"{gif_savepath}/eval.gif", frames, format='GIF', loop=0, duration=fps_to_duration(fps))
    if len(robosuite_frames) > 0:
        for key in robosuite_frames:
            frames = robosuite_frames[key]
            gif_savepath = os.path.join(gif_savepath, f'feval_robosuite_{key}.gif')
            imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))

    # import pdb; pdb.set_trace()
    agent.train()
    return success_rate

def get_frames(
    agent: torch.nn.Module,
    eval_episodes: int,
    device: torch.device,
    air_hockey_cfg: Dict[str, Any],
    args: Dict[str, Any],
    env_test: DummyVecEnv,
    renderer: AirHockeyRenderer,
) -> Tuple[List, Dict]:
    frames = []
    robosuite_frames = {}
    
    for ep_idx in range(eval_episodes):

        next_obs = env_test.reset()
        next_done = False
        next_priv_info = torch.zeros((1, 3)).to(device)
        next_obs = torch.Tensor(next_obs).to(device)
        old_obs = torch.zeros((1, args.history_len,) + env_test.observation_space.shape).to(device)
        old_action = torch.zeros((1, args.history_len,) + env_test.action_space.shape).to(device)
        last_action = torch.zeros((1,) + env_test.action_space.shape).to(device)
        while not next_done:
            # import pdb; pdb.set_trace()
            frame = renderer.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # decrease width to 160 but keep aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
            frames.append(frame)
            if air_hockey_cfg['air_hockey']['simulator'] == 'robosuite':
                for key in env_test.envs[0].current_state:
                    if 'image' not in key:
                        continue
                    current_img = env_test.envs[0].current_state[key]
                    # flip upside down
                    current_img = cv2.flip(current_img, 0)
                    # concatenate with frame
                    current_img = cv2.resize(current_img, (160, int(160 / aspect_ratio)))
                    current_img = np.concatenate([frame, current_img], axis=1)
                    if key not in robosuite_frames:
                        robosuite_frames[key] = [current_img]
                    else:
                        robosuite_frames[key].append(current_img)
            action, _, _, _ = agent.get_action_and_value(obs = next_obs,
                                                        priv_info = next_priv_info,
                                                        last_action = last_action,
                                                        old_obs = old_obs,
                                                        old_action = old_action,)

            obs, rew, done, info = env_test.step(action.cpu().numpy())
            # import pdb; pdb.set_trace()
            next_priv_info = get_priv_info(info, device)
            next_done = torch.Tensor([done[0] or info[0]['TimeLimit.truncated']]).to(device)
            next_obs = torch.Tensor(obs).to(device)
            old_obs, old_action, last_action = get_intereact_history(torch.Tensor(done[0]).to(device), next_obs, action, old_obs, old_action, last_action, args.history_len)
    
    return frames, robosuite_frames

# regress shared_history_encoder to shared_priv_encoder
def phase_2_train(envs, writer, run_name, args, air_hockey_cfg, device):
    agent = Agent(envs, args.phase)
    agent.load_state_dict(torch.load(args.checkpoint))
    agent = agent.to(device)
    optimizer = optim.Adam(agent.shared_history_encoder.parameters(), lr=args.learning_rate, eps=1e-5)
    global_step = 0

    


    for epoch in range(0, 1000):
        

        next_obs, infos = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        old_obs = torch.zeros((args.num_envs,) + (args.history_len,) + envs.observation_space.shape).to(device)
        old_action = torch.zeros((args.num_envs,) + (args.history_len,) + envs.action_space.shape).to(device)
        last_action = torch.zeros((args.num_envs,) + envs.action_space.shape).to(device)

        
        data_step = int(args.phase_2_data_size / args.num_envs)

        priv_extrinsics_data = torch.zeros((data_step, args.num_envs, agent.extrinsics_dim)).to(device)
        history_extrinsics_data = torch.zeros((data_step, args.num_envs, agent.extrinsics_dim)).to(device)
        next_priv_info_data = torch.zeros((data_step, args.num_envs, len(priv_keys))).to(device)
        old_obs_data = torch.zeros((data_step, args.num_envs, args.history_len) + envs.observation_space.shape).to(device)
        old_action_data = torch.zeros((data_step, args.num_envs, args.history_len) + envs.action_space.shape).to(device)
        
        for step in range(0, data_step):
            global_step += args.num_envs

            next_priv_info = get_priv_info(infos, device)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs = next_obs,
                                                                priv_info = next_priv_info,
                                                                last_action = last_action,
                                                                old_obs = old_obs,
                                                                old_action = old_action,)

            next_priv_info_data[step] = next_priv_info
            old_obs_data[step] = old_obs
            old_action_data[step] = old_action
                
            step_ret = envs.step(action.cpu().numpy())
            old_obs, old_action, last_action = get_intereact_history(step_ret[2], next_obs, action, old_obs, old_action, last_action, args.history_len)

            next_obs = step_ret[0] 
            reward = step_ret[1]
            terminations = step_ret[2]
            truncations = np.copy(terminations)
            infos = step_ret[3]
            next_priv_info = get_priv_info(infos, device)
            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)


        next_priv_info_data = next_priv_info_data.reshape((-1, len(priv_keys)))
        old_obs_data = old_obs_data.reshape((-1,) + (args.history_len,) + envs.observation_space.shape)
        old_action_data = old_action_data.reshape((-1,) + (args.history_len,) + envs.action_space.shape)

        total_loss = []
        for i in range(0, int(args.phase_2_data_size / args.phase_2_batch_size)):
            # sample a batch and learn
            idx = np.random.choice(next_priv_info_data.shape[0], args.phase_2_batch_size, replace=False)
            priv_extrinsics_data_sample = agent.get_priv_extrinsics(next_priv_info_data[idx])
            history_extrinsics_data_sample = agent.get_history_extrinsics(old_obs_data[idx], old_action_data[idx])
            loss = ((priv_extrinsics_data_sample - history_extrinsics_data_sample) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss.append(loss.item())

        mean_loss = np.mean(total_loss)
        print(f"epoch={epoch + 1}, loss={mean_loss}")
        writer.add_scalar("losses/mean_loss", mean_loss, epoch)
        # log mean loss
        # writer.add_scalar("losses/phase_2_loss", total_loss.mean(), global_step)

        if (epoch + 1) % 10 == 0:
            
            # save the model
            # agent.phase = 3
            # evaluate the model
            agent.phase = 3
            evaluate(agent=agent, eval_episodes=10, device=device, air_hockey_cfg=air_hockey_cfg, args=args, writer=writer, global_step=epoch)
            agent.phase = 2

            # if args.save_model and global_step % 1000 == 0:
            if args.save_model:
                model_path = f"runs/{run_name}/phase_{args.phase}/epoch_{epoch}.pth"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")



def train(envs, writer, run_name, args, air_hockey_cfg, device):
    if args.phase == 1:
        agent = Agent(envs, args.phase).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.phase == 3:
        agent = Agent(envs, args.phase)
        agent.load_state_dict(torch.load(args.checkpoint))
        agent = agent.to(device)
        optimizer = optim.Adam(agent.shared_history_encoder.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    obs_history = torch.zeros((args.num_steps, args.num_envs, args.history_len) + envs.observation_space.shape).to(device)
    actions_history = torch.zeros((args.num_steps, args.num_envs, args.history_len) + envs.action_space.shape).to(device)
    last_action_history = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    priv_info = torch.zeros((args.num_steps, args.num_envs, len(priv_keys))).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_priv_info = get_priv_info(infos, device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    old_obs = torch.zeros((args.num_envs,) + (args.history_len,) + envs.observation_space.shape).to(device)
    old_action = torch.zeros((args.num_envs,) + (args.history_len,) + envs.action_space.shape).to(device)
    last_action = torch.zeros((args.num_envs,) + envs.action_space.shape).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            obs_history[step] = old_obs
            actions_history[step] = old_action
            last_action_history[step] = last_action
            priv_info[step] = next_priv_info


            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs = next_obs,
                                                                       priv_info = next_priv_info,
                                                                       last_action = last_action,
                                                                       old_obs = old_obs,
                                                                       old_action = old_action,)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob


            # TRY NOT TO MODIFY: execute the game and log data.
            # next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            step_ret = envs.step(action.cpu().numpy())

            old_obs, old_action, last_action = get_intereact_history(step_ret[2], next_obs, action, old_obs, old_action, last_action, args.history_len)

            next_obs = step_ret[0] 
            reward = step_ret[1]
            terminations = step_ret[2]
            truncations = np.copy(terminations)
            infos = step_ret[3]
            next_priv_info = get_priv_info(infos, device)

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(obs = next_obs,
                                         priv_info = next_priv_info,
                                         last_action = last_action,
                                         old_obs = old_obs,
                                         old_action = old_action,
                                         ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_obs_history = obs_history.reshape((-1,) + (args.history_len,) + envs.observation_space.shape)
        b_actions_history = actions_history.reshape((-1,) + (args.history_len,) + envs.action_space.shape)
        b_last_action_history = last_action_history.reshape((-1,) + envs.action_space.shape)
        b_priv_info = priv_info.reshape((-1, len(priv_keys)))


        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs = b_obs[mb_inds],
                                                                            priv_info = b_priv_info[mb_inds],
                                                                            last_action = b_last_action_history[mb_inds], 
                                                                            action = b_actions[mb_inds],
                                                                            old_action = b_actions_history[mb_inds],
                                                                            old_obs = b_obs_history[mb_inds],
                                                                            )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        print("global_step: ", global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        evaluate(agent=agent, eval_episodes=10, device=device, air_hockey_cfg=air_hockey_cfg, args=args, writer=writer)

        if args.save_model and global_step % (args.num_envs * args.num_steps * 4) == 0:
            model_path = f"runs/{run_name}/phase_{args.phase}/global_step_{global_step}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            

    envs.close()
    writer.close()
            
def train_air_hockey_model(air_hockey_cfg, use_wandb=False, device='cpu', clear_prior_task_results=False, progress_bar=False):
    """
    Train an air hockey paddle model using stable baselines.

    This script loads the configuration file, creates an AirHockey2D environment,
    wraps the environment with necessary components, trains the model,
    and saves the trained model and environment statistics.
    """
    
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
    
    if type(air_hockey_cfg['seed']) is not list:
        seeds = [int(air_hockey_cfg['seed'])]
    else:
        seeds = [int(s) for s in air_hockey_cfg['seed']]
        del air_hockey_cfg['seed'] # otherwise it will be saved in the model cfg when copied over
        
    # Train different seeds. If one seed in config, this is just one iteration.
    for seed in seeds:
        air_hockey_cfg['seed'] = seed # since it it used as training seed
        air_hockey_params['seed'] = seed # and environment seed
        
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project="air-hockey",
                config=air_hockey_cfg,
                sync_tensorboard=True,
                save_code=True)
        
        def get_airhockey_env_for_parallel():
            """
            Utility function for multiprocessed env.

            :param env_id: (str) the environment ID
            :param num_env: (int) the number of environments you wish to have in subprocesses
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
            """
            def _init():
                curr_seed = random.randint(0, int(1e8))
                air_hockey_params['seed'] = curr_seed
                # Note: With this seed, an individual rng is created for each env
                # It does not affect the global rng!
                env = AirHockeyEnv(air_hockey_params)
                return Monitor(env)
            return _init()

        # set seed for reproducibility
        seed = air_hockey_params['seed']
        random.seed(seed)
        # get number of threads
        n_threads = air_hockey_cfg['n_threads']
        env = SubprocVecEnv_domain_random([get_airhockey_env_for_parallel for _ in range(n_threads)])

        os.makedirs(air_hockey_cfg['tb_log_dir'], exist_ok=True)
        log_parent_dir = os.path.join(air_hockey_cfg['tb_log_dir'], air_hockey_cfg['air_hockey']['task'])
        if clear_prior_task_results and os.path.exists(log_parent_dir):
            shutil.rmtree(log_parent_dir)
        os.makedirs(log_parent_dir, exist_ok=True)
        
        # determine the actual log dir
        subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
        subdir_nums = [int(x.split(air_hockey_cfg['tb_log_name'] + '_')[1]) for x in subdirs]
        next_num = max(subdir_nums) + 1 if subdir_nums else 1
        log_dir = os.path.join(log_parent_dir, air_hockey_cfg['tb_log_name'] + f'_{next_num}')
    
    return env, eval_env, log_dir, air_hockey_cfg, wandb_run

@pyrallis.wrap()
def main(args: Args):
    # args = tyro.cli(Args)

    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'default_train_puck_vel.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
        
    assert 'n_threads' in air_hockey_cfg, "Please specify the number of threads to use for training."
    assert 'algorithm' in air_hockey_cfg, "Please specify the algorithm to use for training."

    air_hockey_cfg["air_hockey"]["domain_random"] = True
    use_wandb = args.wandb
    device = args.device
    clear_prior_task_results = args.clear
    progress_bar = args.progress_bar
    envs, eval_env, log_dir, air_hockey_cfg, wandb_run = train_air_hockey_model(air_hockey_cfg, use_wandb, device, clear_prior_task_results, progress_bar)
    
    # train prep
    # args = tyro.cli(Args)
    args.num_envs = air_hockey_cfg['n_threads']
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = args.run_name
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    if args.save_model:
        os.makedirs(f"runs/{run_name}/phase_{args.phase}", exist_ok=True)
    writer_dir = run_name + "_wandb"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    
    writer = SummaryWriter(f"runs/{writer_dir}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    if args.phase == 1 or args.phase == 3:
        train(envs, writer, run_name, args, air_hockey_cfg, device)

    elif args.phase == 2:
        phase_2_train(envs, writer, run_name, args, air_hockey_cfg, device)

if __name__ == "__main__":
    main()