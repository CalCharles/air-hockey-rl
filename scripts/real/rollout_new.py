import torch
import argparse
import yaml
import os
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
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

DEFAULT_RMA_FIXED_LATENT = [
    -0.13727053,
    0.73891377,
    -0.22653317,
    -0.61562562,
    -0.61218202,
    0.17801657,
    -0.09171214,
    0.04772795,
    0.14299890,
    -0.99860352,
    -0.70962828,
    0.70619726,
]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def parse_fixed_latent_from_args(args):
    if args.rma_fixed_latent_path is not None:
        if not os.path.exists(args.rma_fixed_latent_path):
            raise FileNotFoundError(
                f"Requested --rma-fixed-latent-path does not exist: {args.rma_fixed_latent_path}"
            )
        if args.rma_fixed_latent_path.endswith(".npy"):
            arr = np.load(args.rma_fixed_latent_path).astype(np.float32).reshape(-1)
        else:
            raw = yaml.load(open(args.rma_fixed_latent_path, "r"), Loader=yaml.FullLoader)
            if isinstance(raw, dict) and "latent_centroid" in raw:
                raw = raw["latent_centroid"]
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    elif args.rma_fixed_latent is not None and len(args.rma_fixed_latent) > 0:
        arr = np.asarray(args.rma_fixed_latent, dtype=np.float32).reshape(-1)
    else:
        arr = np.asarray(DEFAULT_RMA_FIXED_LATENT, dtype=np.float32)
    return arr


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot coerce value '{value}' to bool.")


def load_rma_settings_from_model_folder(model_path):
    """
    Read RMA-relevant settings from config files next to model checkpoint.
    Priority:
    1) args.yaml
    2) config.yaml
    """
    model_dir = Path(model_path).resolve().parent
    candidates = [model_dir / "args.yaml", model_dir / "config.yaml"]

    settings = {
        "env_latent_dim": None,
        "use_last_action_in_policy_state": None,
        "sources": {},
    }

    for p in candidates:
        if not p.exists():
            continue
        with open(p, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(cfg, dict):
            continue

        if settings["env_latent_dim"] is None and "env_latent_dim" in cfg:
            settings["env_latent_dim"] = int(cfg["env_latent_dim"])
            settings["sources"]["env_latent_dim"] = str(p)

        if settings["use_last_action_in_policy_state"] is None and "use_last_action_in_policy_state" in cfg:
            settings["use_last_action_in_policy_state"] = _coerce_bool(cfg["use_last_action_in_policy_state"])
            settings["sources"]["use_last_action_in_policy_state"] = str(p)

    return settings


def build_policy_env_view(policy_obs_dim, action_dim):
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(policy_obs_dim),),
            dtype=np.float32,
        ),
        single_action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        ),
    )


def infer_policy_dims_from_state_dict(state_dict):
    actor_input_dim = int(state_dict["actor.0.weight"].shape[1])
    action_dim = int(state_dict["actor_mean_head.weight"].shape[0])
    return actor_input_dim, action_dim


def maybe_generate_gifs_for_saved_trajectory(
    simulator,
    auto_gif: bool,
    gif_fps: int,
    gif_max_frames_per_file: int,
):
    if not auto_gif:
        return

    saved_idx = simulator.tidx - 1
    if saved_idx < 0:
        print("No saved trajectory index found for GIF generation.")
        return

    hdf5_path = (Path(simulator.save_path) / f"trajectory_data{saved_idx}.hdf5").resolve()
    if not hdf5_path.exists():
        print(f"Skipping GIF generation; missing file: {hdf5_path}")
        return

    output_dir = hdf5_path.parent / f"{hdf5_path.stem}_gifs"
    try:
        from visualize_saved_trajectory import generate_gifs_from_hdf5
    except ModuleNotFoundError as exc:
        print(f"Skipping GIF generation; dependency missing: {exc}")
        return

    try:
        outputs = generate_gifs_from_hdf5(
            input_hdf5=hdf5_path,
            output_dir=output_dir,
            fps=gif_fps,
            max_frames_per_gif=gif_max_frames_per_file,
        )
    except Exception as exc:
        print(f"GIF generation failed for {hdf5_path}: {exc}")
        return

    if outputs:
        print(f"Generated {len(outputs)} GIF(s) in {output_dir}")
    else:
        print(f"No GIF frames produced for {hdf5_path}")

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
    parser.add_argument('--save-path', type=str, default=None, help='Override trajectory save path (defaults to config value).')
    parser.add_argument('--action-scale', type=float, default=0.2, help='action scale')
    parser.add_argument('--agent-hidden-size', type=int, default=128, help='agent size')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='torch device')
    parser.add_argument('--use-rma', action='store_true', help='Use fixed-latent RMA policy input construction.')
    parser.add_argument('--rma-env-latent-dim', type=int, default=12, help='RMA latent dimension expected by policy.')
    parser.add_argument('--rma-use-last-action', type=int, choices=[0, 1], default=1, help='Whether to append previous action to policy observation in RMA mode.')
    parser.add_argument('--rma-fixed-latent', type=float, nargs='*', default=None, help='Optional fixed latent vector values (space-separated).')
    parser.add_argument('--rma-fixed-latent-path', type=str, default=None, help='Optional path to .npy or YAML containing fixed latent.')
    parser.add_argument('--auto-gif', action='store_true', help='Generate GIF visualization(s) after each saved trajectory.')
    parser.add_argument('--gif-fps', type=int, default=20, help='GIF playback FPS used when --auto-gif is enabled.')
    parser.add_argument('--gif-max-frames-per-file', type=int, default=250, help='Maximum rendered frames per GIF when --auto-gif is enabled.')

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

    if args.save_path is not None:
        air_hockey_params_cp['simulator_params']['save_path'] = args.save_path
    
    eval_air_hockey_params = air_hockey_params_cp.copy()
    print("trajectory save path:", eval_air_hockey_params['simulator_params']['save_path'])
    
    # Create environment factory function
    def make_eval_env():
        env = AirHockeyEnv(eval_air_hockey_params)
        return env
    
    eval_env = make_eval_env()
    device = torch.device(args.device)
    state_dict = torch.load(args.model, map_location=device)

    if args.use_rma:
        rma_cfg = load_rma_settings_from_model_folder(args.model)
        base_obs_dim = int(np.prod(eval_env.single_observation_space.shape))
        model_obs_dim, model_action_dim = infer_policy_dims_from_state_dict(state_dict)
        fixed_latent_np = parse_fixed_latent_from_args(args)

        if rma_cfg["env_latent_dim"] is not None:
            requested_latent_dim = int(rma_cfg["env_latent_dim"])
            print(
                "Using env_latent_dim from model config:",
                requested_latent_dim,
                f"(source: {rma_cfg['sources'].get('env_latent_dim', 'unknown')})",
            )
        else:
            requested_latent_dim = int(args.rma_env_latent_dim)
            print("No env_latent_dim found in model folder config; using CLI value:", requested_latent_dim)

        if fixed_latent_np.shape[0] != requested_latent_dim:
            raise ValueError(
                f"Fixed latent length ({fixed_latent_np.shape[0]}) does not match "
                f"--rma-env-latent-dim ({requested_latent_dim})."
            )

        if rma_cfg["use_last_action_in_policy_state"] is not None:
            use_last_action = bool(rma_cfg["use_last_action_in_policy_state"])
            print(
                "Using use_last_action_in_policy_state from model config:",
                use_last_action,
                f"(source: {rma_cfg['sources'].get('use_last_action_in_policy_state', 'unknown')})",
            )
        else:
            use_last_action = bool(args.rma_use_last_action)
            print("No use_last_action_in_policy_state found in model folder config; using CLI value:", use_last_action)

        expected_obs_dim = base_obs_dim + (model_action_dim if use_last_action else 0) + requested_latent_dim
        if expected_obs_dim != model_obs_dim:
            raise ValueError(
                "RMA policy observation dimension mismatch. "
                f"Computed={expected_obs_dim} (base={base_obs_dim}, "
                f"last_action={model_action_dim if use_last_action else 0}, latent={requested_latent_dim}), "
                f"model expects {model_obs_dim}."
            )

        policy_env_view = build_policy_env_view(policy_obs_dim=expected_obs_dim, action_dim=model_action_dim)
        model = Agent(policy_env_view, action_scale=args.action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size)
        fixed_latent_t = torch.tensor(fixed_latent_np, dtype=torch.float32, device=device).unsqueeze(0)
        last_action_for_policy = torch.zeros((1, model_action_dim), dtype=torch.float32, device=device)
        args.rma_use_last_action = int(use_last_action)
        print("Using RMA mode with fixed latent:", fixed_latent_np.tolist())
    else:
        model_obs_dim, model_action_dim = infer_policy_dims_from_state_dict(state_dict)
        policy_env_view = build_policy_env_view(policy_obs_dim=model_obs_dim, action_dim=model_action_dim)
        model = Agent(policy_env_view, action_scale=args.action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size)
        fixed_latent_t = None
        use_last_action = model_obs_dim > eval_env.single_observation_space.shape[0]
        last_action_for_policy = torch.zeros((1, model_action_dim), dtype=torch.float32, device=device)

    model.load_state_dict(state_dict)
    model = model.to(device=device)

    print("model action scale: ", model.action_scale)
    # model.action_scale = torch.tensor(0.2) # manually scaling just for testing a model
    
    state_dict = eval_env.simulator.get_current_state()
    obs_type = "history"
    
    obs = get_observation_by_type(state_dict, obs_type=obs_type, puck_history=state_dict["pucks"][0]["history"], paddle_history=state_dict['paddles']['paddle_ego']['history'])
    obs_list = list()
    with NonBlockingConsole() as nbc:
        delay_counter = 0
        while True:
            obs_t = torch.tensor(obs).unsqueeze(0).to(device=device).float()
            policy_obs = obs_t
            if use_last_action:
                policy_obs = torch.cat([policy_obs, last_action_for_policy], dim=-1)
            if args.use_rma and fixed_latent_t is not None:
                policy_obs = torch.cat([policy_obs, fixed_latent_t], dim=-1)
            action = model(policy_obs).cpu().numpy().squeeze()
            if delay_counter < 10 and delay_counter >= 0:
                # action = model.policy(obs)
                # action = action.mean
                action = action * 0.0
            else:
                # action = model.policy(obs)
                # action = action.mean
                action = action / model.action_scale.item() # normalizes to [-1, 1]
                # action[0,0] = action[0,0] * 0.5
            if use_last_action:
                last_action_for_policy = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
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
                maybe_generate_gifs_for_saved_trajectory(
                    simulator=eval_env.simulator,
                    auto_gif=args.auto_gif,
                    gif_fps=args.gif_fps,
                    gif_max_frames_per_file=args.gif_max_frames_per_file,
                )
                delay_counter = 0
                if use_last_action:
                    last_action_for_policy.zero_()
            elif key == 'q':
                print("Resetting without saving...")
                eval_env.reset(seed=None, write_traj = False)
                delay_counter = 0
                if use_last_action:
                    last_action_for_policy.zero_()
            elif key == 'x':
                print("Exiting...")
                break




