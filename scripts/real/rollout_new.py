import torch
import argparse
import yaml
import os
from airhockey import AirHockeyEnv
import numpy as np
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole

### ================================ ###
### PORTED AGENT CODE (TO AVOID ERRORS) ###
### ================================ ###

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

# default constants
LOG_STD_MIN = -1
LOG_STD_MAX = 1
EPS = 1e-6

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Simple MLP Gaussian Policy + Critic for PPO (actions clipped into some pre-determined range)
class Agent(nn.Module):
    def __init__(self, envs, action_scale=0.2, action_bias=0.0, hidden_size=64): # preliminary calculation
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor_mean_head = layer_init(nn.Linear(hidden_size, act_dim), std=0.01) # action initially close to 0, exploration guided by logstd
        # self.actor_logstd_head = layer_init(nn.Linear(64, act_dim), std=3)

        # forget about per-state logstd, just use a fixed one for now
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.register_buffer("action_scale", torch.tensor(action_scale))
        self.register_buffer("action_bias", torch.tensor(action_bias))

        self.register_buffer("LOG_STD_MIN", torch.tensor(LOG_STD_MIN))
        self.register_buffer("LOG_STD_MAX", torch.tensor(LOG_STD_MAX))
        self.register_buffer("EPS", torch.tensor(EPS))

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_mean_and_logstd(self, x):
        x = self.actor(x)
        mean = self.actor_mean_head(x)

        # logstd = torch.tanh(self.actor_logstd_head(x))
        # logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)

        logstd = torch.tanh(self.actor_logstd.expand_as(mean))
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)

        return mean, logstd

    def forward(self, x):
        with torch.no_grad(): # only for evaluation
            # check if x is a tensor, and if need unsqueeze
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            return self.get_action_and_value(x)[0] # just return the action, as a tensor

    def get_action_and_value(self, x, action=None):
        action_mean, action_logstd = self.get_action_mean_and_logstd(x)
        action_std = torch.exp(action_logstd)
        normal = Normal(action_mean, action_std)

        # # get actions
        if action is None:
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            y_t = (action - self.action_bias) / self.action_scale
            x_t = torch.atanh(torch.clamp(y_t, -1 + self.EPS, 1 - self.EPS)) # reverse tanh 
        
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + self.EPS)
        log_prob = log_prob.sum(1)
        mean = torch.tanh(action_mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, self.critic(x)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rollout')

    # optional arguments if use-parent-log-dir is False
    parser.add_argument('--model', type=str, default="ex_model/model.pth", help='Path to the model to evaluate.')
    parser.add_argument('--config-path', type=str, default="configs/real_configs/rollout_config.yaml", help='Path to the config file.')
    parser.add_argument('--action-scale', type=float, default=0.2, help='action scale')
    parser.add_argument('--agent-hidden-size', type=int, default=128, help='agent size')

    args = parser.parse_args()
    
    air_hockey_cfg = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    air_hockey_params = air_hockey_cfg['air_hockey']
    
    print("action scale: ", args.action_scale)

    # processing to avoid bugs
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
    
    eval_air_hockey_params = air_hockey_params_cp.copy()
    
    # Create environment factory function
    def make_eval_env():
        env = AirHockeyEnv(eval_air_hockey_params)
        return env
    
    eval_env = make_eval_env()
    model = Agent(eval_env, action_scale=args.action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size)
    state_dict = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(state_dict)
    model = model.to(device="cuda:0")

    print("model action scale: ", model.action_scale)
    model.action_scale = torch.tensor(0.2) # something went wrong during saving
    state_dict = eval_env.simulator.get_current_state()
    obs_type = "history"
    
    obs = get_observation_by_type(state_dict, obs_type=obs_type, puck_history=state_dict["pucks"][0]["history"], paddle_history=state_dict['paddles']['paddle_ego']['history'])
    obs_list = list()
    with NonBlockingConsole() as nbc:
        delay_counter = 0
        while True:
            obs = torch.tensor(obs).unsqueeze(0).to(device="cuda:0").float()
            action = model(obs).cpu().numpy().squeeze()
            if delay_counter < 10 and delay_counter >= 0:
                # action = model.policy(obs)
                # action = action.mean
                action = action * 0.0
            else:
                # action = model.policy(obs)
                # action = action.mean
                action = action.clip(-1,1)
                # action[0,0] = action[0,0] * 0.5
            delay_counter += 1
            print("action", action, obs)
            # action = action * 0.0
            # action[0,0] = 0
            # action[0,1] = 0

            obs, reward, is_finished, truncated, info = eval_env.step(action)

            # for puck hitting observations
            state_dict = eval_env.simulator.get_current_state()
            obs = get_observation_by_type(state_dict, obs_type=obs_type, puck_history=state_dict["pucks"][0]["history"], paddle_history=state_dict['paddles']['paddle_ego']['history'])

            # Store the keypress to avoid calling get_data() twice
            key = nbc.get_data()
            if key == 'y':
                print("Saving trajectory and resetting...")
                eval_env.reset(seed=None, write_traj = True)
                delay_counter = 0
            elif key == 'q':
                print("Resetting without saving...")
                eval_env.reset(seed=None, write_traj = False)
                delay_counter = 0
            elif key == 'x':
                print("Exiting...")
                break




