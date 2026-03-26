"""
PPO training using Stable Baselines3 with optional motion rewards.

Replaces the from-scratch PPO with SB3's PPO while preserving:
  - 5-component motion reward (stand-still, temporal alignment, axis alignment,
    velocity, jerk) via a gym.Wrapper
  - Last-action observation augmentation via a gym.Wrapper
  - Periodic eval GIF generation and checkpointing via a Callback

Run with:
  python scripts/smooth_policy/amp_history/amp_training/ppo/amp_training_ppo_sb3.py \
    --args-file <path-to-yaml>
"""

import os
import random
from dataclasses import dataclass
from datetime import datetime

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.agent import ResidualMLPTrunk, layer_init
from scripts.utils import save_tensorboard_plots


# ---------------------------------------------------------------------------
# Motion reward helpers (numpy, single-env)
# ---------------------------------------------------------------------------

def _velocity_reward(vel_mag, vel_at_one, vel_at_zero):
    denom = max(vel_at_zero - vel_at_one, 1e-6)
    return min(1.0 - (vel_mag - vel_at_one) / denom, 1.0)


def _jerk_reward(jerk_mag, jerk_at_one, jerk_at_zero):
    denom = max(jerk_at_zero - jerk_at_one, 1e-6)
    return 1.0 - (jerk_mag - jerk_at_one) / denom


def _extract_paddle_pos(obs):
    return obs[12:14] if len(obs) >= 30 else obs[0:2]


def _extract_puck_pos(obs):
    if len(obs) >= 30:
        return obs[27:29]
    if len(obs) >= 8:
        return obs[4:6]
    return obs[2:4]


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class MotionRewardWrapper(gym.Wrapper):
    """Computes 5-component motion reward and blends with task reward."""

    def __init__(
        self, env, *,
        task_reward_weight=1.0, motion_reward_weight=0.0,
        stand_still_weight=0.5, temporal_alignment_weight=0.5,
        axis_alignment_weight=0.5, velocity_weight=0.5, jerk_weight=0.5,
        stand_still_threshold=0.015, temporal_horizon=4,
        velocity_at_one=0.3, velocity_at_zero=0.6,
        jerk_at_one=10.0, jerk_at_zero=23.0,
    ):
        super().__init__(env)
        self.task_rw = task_reward_weight
        self.motion_rw = motion_reward_weight
        self.sw = stand_still_weight
        self.tw = temporal_alignment_weight
        self.aw = axis_alignment_weight
        self.vw = velocity_weight
        self.jw = jerk_weight
        self.ss_thresh = stand_still_threshold
        self.th = temporal_horizon
        self.v1, self.v0 = velocity_at_one, velocity_at_zero
        self.j1, self.j0 = jerk_at_one, jerk_at_zero

        self._paddle_hist = np.zeros((temporal_horizon + 1, 2))
        self._puck_hist = np.zeros((temporal_horizon + 1, 2))
        self._steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._paddle_hist[:] = 0
        self._paddle_hist[-1] = _extract_paddle_pos(obs)
        self._puck_hist[:] = 0
        self._puck_hist[-1] = _extract_puck_pos(obs)
        self._steps = 0
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        # Update position histories
        self._paddle_hist = np.roll(self._paddle_hist, -1, axis=0)
        self._paddle_hist[-1] = _extract_paddle_pos(obs)
        self._puck_hist = np.roll(self._puck_hist, -1, axis=0)
        self._puck_hist[-1] = _extract_puck_pos(obs)

        done = term or trunc
        self._steps = 0 if done else self._steps + 1

        info["task_reward"] = reward

        if self.motion_rw > 0:
            mr = self._compute_motion_reward(info)
            info["motion_reward"] = mr
            combined = self.task_rw * reward + self.motion_rw * mr
        else:
            combined = self.task_rw * reward
            info["motion_reward"] = 0.0

        return obs, combined, term, trunc, info

    def _compute_motion_reward(self, info):
        eps = 1e-8
        movement = self._paddle_hist[-1] - self._paddle_hist[0]
        move_norm = np.linalg.norm(movement)
        valid = self._steps >= self.th

        # 1. Stand-still
        ss = float(move_norm <= self.ss_thresh and valid)

        # 2. Temporal alignment
        if valid and move_norm > eps:
            target = self._puck_hist[0] - self._paddle_hist[0]
            target_norm = max(np.linalg.norm(target), eps)
            cosine = np.dot(movement, target) / (move_norm * target_norm)
            ta = float(np.clip((cosine + 1) * 0.5, 0, 1))
        else:
            ta = 0.0
        if ss > 0.5:
            ta = 1.0

        # 3. Axis alignment
        if valid and move_norm > eps:
            unit = movement / move_norm
            max_cos = max(abs(unit[0]), abs(unit[1]))
            min_cos = 1.0 / np.sqrt(2.0)
            aa = float(np.clip((max_cos - min_cos) / (1.0 - min_cos + eps), 0, 1))
        else:
            aa = 0.0
        if ss > 0.5:
            aa = 1.0

        # 4. Velocity
        vel_mag = info.get("paddle_velocity_mag", 0.0)
        vr = _velocity_reward(vel_mag, self.v1, self.v0)

        # 5. Jerk
        jerk_mag = info.get("paddle_jerk_mag", 0.0)
        jr = _jerk_reward(jerk_mag, self.j1, self.j0)

        return self.sw * ss + self.tw * ta + self.aw * aa + self.vw * vr + self.jw * jr


class LastActionObsWrapper(gym.Wrapper):
    """Concatenates last action to observation."""

    def __init__(self, env):
        super().__init__(env)
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim + act_dim,), dtype=np.float32,
        )
        self._last_action = np.zeros(act_dim, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_action[:] = 0
        return np.concatenate([obs, self._last_action]), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._last_action = np.asarray(action, dtype=np.float32).copy()
        aug_obs = np.concatenate([obs, self._last_action])
        if term or trunc:
            self._last_action[:] = 0
        return aug_obs, reward, term, trunc, info


# ---------------------------------------------------------------------------
# Custom SB3 policy with residual architecture
# ---------------------------------------------------------------------------

class ResidualMlpExtractor(nn.Module):
    """Drop-in replacement for SB3's MlpExtractor using ResidualMLPTrunk."""

    def __init__(self, feature_dim, hidden_layer_size=64, num_hidden_layers=5):
        super().__init__()
        self.policy_net = ResidualMLPTrunk(
            feature_dim, hidden_layer_size, num_hidden_layers,
        )
        self.value_net = ResidualMLPTrunk(
            feature_dim, hidden_layer_size, num_hidden_layers,
        )
        self.latent_dim_pi = hidden_layer_size
        self.latent_dim_vf = hidden_layer_size

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class ResidualActorCriticPolicy(PPO.policy_aliases["MlpPolicy"]):
    """ActorCriticPolicy with ResidualMLPTrunk as the backbone."""

    def __init__(self, *args, hidden_layer_size=64, num_hidden_layers=5, **kwargs):
        self._custom_hidden_size = hidden_layer_size
        self._custom_num_layers = num_hidden_layers
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ResidualMlpExtractor(
            self.features_dim,
            hidden_layer_size=self._custom_hidden_size,
            num_hidden_layers=self._custom_num_layers,
        )


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TrainingCallback(BaseCallback):
    """Handles periodic eval GIFs, checkpointing, and motion reward logging."""

    def __init__(
        self, env_config, log_dir, *,
        eval_freq=5000, checkpoint_freq=5000,
        n_eval_eps=4, n_gifs=1,
        use_last_action=False,
        verbose=0,
    ):
        super().__init__(verbose)
        self.env_config = env_config
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_eps = n_eval_eps
        self.n_gifs = n_gifs
        self.use_last_action = use_last_action
        self._last_eval_step = 0
        self._last_ckpt_step = 0
        self._episode_returns = []
        self._success_rates = []
        self._collision_count = 0.0
        self._interval_steps = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        self._interval_steps += self.training_env.num_envs

        for info in infos:
            if info is None:
                continue
            self._collision_count += info.get("paddle_puck_collision_count", 0)
            if "episode" in info:
                self._episode_returns.append(info["episode"]["r"])
            if "success" in info:
                self._success_rates.append(float(info["success"]))

        step = self.num_timesteps

        if step - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = step
            self._log_interval_stats(step)

        if step - self._last_ckpt_step >= self.checkpoint_freq:
            self._last_ckpt_step = step
            self._save_checkpoint(step)

        return True

    def _log_interval_stats(self, step):
        if self._episode_returns:
            avg_ret = np.mean(self._episode_returns)
            avg_suc = np.mean(self._success_rates) if self._success_rates else 0.0
            self.logger.record("charts/avg_episodic_return", avg_ret)
            self.logger.record("charts/avg_success_rate", avg_suc)
            print(
                f"Step {step}: Avg Return: {avg_ret:.2f}, "
                f"Success: {avg_suc:.2f}, Episodes: {len(self._episode_returns)}"
            )
            self._episode_returns.clear()
            self._success_rates.clear()
        else:
            print(f"Step {step}: No episodes completed in last interval")

        col_rate = self._collision_count / max(self._interval_steps, 1)
        self.logger.record("contacts/paddle_puck_collisions_per_step", col_rate)
        self._collision_count = 0.0
        self._interval_steps = 0

    def _save_checkpoint(self, step):
        ckpt_dir = os.path.join(self.log_dir, f"checkpoint_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save(os.path.join(ckpt_dir, "model"))

        try:
            self._eval_gif(ckpt_dir)
        except Exception as e:
            print(f"Eval GIF failed: {e}")

        print(f"\nCheckpoint saved at step {step}")

    def _eval_gif(self, save_dir):
        env = AirHockeyEnv(self.env_config)
        renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)
        act_dim = env.action_space.shape[0]

        for gif_idx in range(self.n_gifs):
            frames = []
            for _ in tqdm.tqdm(range(self.n_eval_eps), desc="eval"):
                obs, _ = env.reset()
                last_action = np.zeros(act_dim, dtype=np.float32)
                done = False
                rew = 0.0
                cum_rew = 0.0
                while not done:
                    frame = renderer.get_frame()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ar = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(frame, (160, int(160 / ar)))
                    cv2.putText(
                        frame, f"Reward: {rew:.2f}",
                        (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                    )
                    cv2.putText(
                        frame, f"Return: {cum_rew:.2f}",
                        (frame.shape[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                    )
                    frames.append(frame)

                    if self.use_last_action:
                        policy_obs = np.concatenate([obs, last_action])
                    else:
                        policy_obs = obs
                    action, _ = self.model.predict(policy_obs, deterministic=True)
                    obs, rew, term, trunc, info = env.step(action)
                    done = term or trunc
                    cum_rew += rew
                    last_action = action.astype(np.float32).copy()
                    if done:
                        last_action[:] = 0

            path = os.path.join(save_dir, f"eval_{gif_idx}.gif")
            imageio.mimsave(path, frames, format="GIF", loop=0, duration=50)

    def on_training_end(self):
        self._save_checkpoint(self.num_timesteps)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # PPO core
    num_envs: int = 8
    num_steps: int = 512
    learning_rate: float = 1e-4
    num_iterations: int = 5000
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    norm_adv: bool = True

    # Reward weighting
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 0.0

    # Motion reward component weights
    stand_still_reward_weight: float = 0.5
    temporal_alignment_reward_weight: float = 0.5
    axis_alignment_reward_weight: float = 0.5
    velocity_reward_weight: float = 0.5
    jerk_reward_weight: float = 0.5

    # Motion reward calibration
    stand_still_threshold: float = 0.015
    temporal_alignment_horizon: int = 4
    velocity_at_one: float = 0.3
    velocity_at_zero: float = 0.6
    jerk_at_one: float = 10.0
    jerk_at_zero: float = 23.0

    # Policy state
    use_last_action_in_policy_state: bool = True

    # Paths
    config: str = "scripts/smooth_policy/amp_history/configs/td3/puck_touch_verify/puck_touch_config.yaml"
    args_file: str = None
    model_path: str = None
    log_parent_dir: str = None
    run_name: str = "ppo_motion"

    # Runtime
    seed: int = 0
    device: str = "cuda:0"
    action_scale: float = 1

    # Agent architecture
    agent_hidden_layer_size: int = 64
    agent_num_hidden_layers: int = 5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config["air_hockey"]

    # Logging directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = env_config.get("task")
    log_dir = args.log_parent_dir or f"runs/ppo_training/{task_name}/{args.run_name}_{timestamp}"
    if os.path.exists(log_dir):
        base = log_dir
        i = 1
        while os.path.exists(log_dir):
            log_dir = f"{base}r{i}"
            i += 1
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    if env_config.get("use_pid", False):
        action_scale = 1
    else:
        action_scale = args.action_scale

    # Env factory with wrappers
    motion_kwargs = dict(
        task_reward_weight=args.task_reward_weight,
        motion_reward_weight=args.motion_reward_weight,
        stand_still_weight=args.stand_still_reward_weight,
        temporal_alignment_weight=args.temporal_alignment_reward_weight,
        axis_alignment_weight=args.axis_alignment_reward_weight,
        velocity_weight=args.velocity_reward_weight,
        jerk_weight=args.jerk_reward_weight,
        stand_still_threshold=args.stand_still_threshold,
        temporal_horizon=args.temporal_alignment_horizon,
        velocity_at_one=args.velocity_at_one,
        velocity_at_zero=args.velocity_at_zero,
        jerk_at_one=args.jerk_at_one,
        jerk_at_zero=args.jerk_at_zero,
    )

    def make_env():
        def _thunk():
            seed = random.randint(0, int(1e8))
            cfg = dict(env_config)
            cfg["seed"] = seed
            env = AirHockeyEnv(cfg)
            env = MotionRewardWrapper(env, **motion_kwargs)
            if args.use_last_action_in_policy_state:
                env = LastActionObsWrapper(env)
            return env
        return _thunk

    envs = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    envs = VecMonitor(envs)

    # Total timesteps: num_iterations * num_envs * num_steps
    total_timesteps = args.num_iterations * args.num_envs * args.num_steps
    batch_size = args.num_envs * args.num_steps

    print(f"\n{'='*80}")
    print(f"Starting SB3 PPO training with motion rewards")
    print(f"  task_reward_weight={args.task_reward_weight}")
    print(f"  motion_reward_weight={args.motion_reward_weight}")
    print(f"  num_envs={args.num_envs}, num_steps={args.num_steps}")
    print(f"  total_timesteps={total_timesteps}")
    print(f"  log_dir={log_dir}")
    print(f"{'='*80}\n")

    policy_kwargs = dict(
        hidden_layer_size=args.agent_hidden_layer_size,
        num_hidden_layers=args.agent_num_hidden_layers,
    )

    model = PPO(
        policy=ResidualActorCriticPolicy,
        env=envs,
        learning_rate=args.learning_rate,
        n_steps=args.num_steps,
        batch_size=args.minibatch_size,
        n_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_coef,
        clip_range_vf=args.clip_coef if args.clip_vloss else None,
        normalize_advantage=args.norm_adv,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        seed=args.seed,
        device=args.device,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # Load pretrained weights if provided
    if args.model_path is not None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"Loading pre-trained model from {args.model_path}")
        pretrained = PPO.load(args.model_path, env=envs, device=args.device)
        model.policy.load_state_dict(pretrained.policy.state_dict())
        print("Model loaded successfully")

    callback = TrainingCallback(
        env_config=env_config,
        log_dir=log_dir,
        eval_freq=500,
        checkpoint_freq=5000,
        n_eval_eps=4,
        n_gifs=1,
        use_last_action=args.use_last_action_in_policy_state,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    # Final save
    envs.close()
    model.save(os.path.join(log_dir, "model"))

    metrics = [
        "charts/avg_episodic_return",
        "charts/avg_success_rate",
        "contacts/paddle_puck_collisions_per_step",
    ]
    save_tensorboard_plots(log_dir, config, metrics=metrics)

    print(f"\nTraining complete. Results saved to {log_dir}")
