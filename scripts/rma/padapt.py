# --------------------------------------------------------
# ProprioAdapt (RMA Stage 2) — freeze μ+π, train φ with L2.
# Ported from hora/hora/algo/padapt/padapt.py for air-hockey-rl-rma.
# --------------------------------------------------------

import os
import time

import torch

from scripts.rma.misc import AverageScalarMeter, tprint
from scripts.rma.models import ActorCritic
from scripts.rma.running_mean_std import RunningMeanStd
from scripts.rma.wandb_utils import wandb_log

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter


def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default)


class ProprioAdapt(object):
    def __init__(self, env, output_dir: str, cfg, device: str):
        self.device = device
        self.cfg = cfg
        # ---- build environment ----
        self.env = env
        self.num_actors = env.num_envs
        cfg_num_actors = _cfg_get(cfg, 'num_actors', None)
        if cfg_num_actors is not None:
            assert int(cfg_num_actors) == self.num_actors, (
                f'cfg.num_actors ({cfg_num_actors}) must equal env.num_envs ({self.num_actors})'
            )
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        # ---- Priv Info ----
        self.priv_info = _cfg_get(cfg, 'priv_info', True)
        self.priv_info_dim = cfg.priv_info_dim
        self.proprio_adapt = _cfg_get(cfg, 'proprio_adapt', True)
        self.proprio_hist_dim = env.prop_hist_len
        proprio_hist_input_dim = _cfg_get(cfg, 'proprio_hist_input_dim', None)
        if proprio_hist_input_dim is None:
            proprio_hist_input_dim = env.proprio_hist_entry_dim
        self.proprio_hist_input_dim = int(proprio_hist_input_dim)
        # ---- Model ----
        net_config = {
            'actor_units': list(cfg.actor_units),
            'priv_mlp_units': list(cfg.priv_mlp_units),
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'proprio_hist_input_dim': self.proprio_hist_input_dim,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd(
            (self.proprio_hist_dim, self.proprio_hist_input_dim)
        ).to(self.device)
        self.sa_mean_std.train()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        self.max_agent_steps = int(
            _cfg_get(cfg, 'adaptation_max_agent_steps', None)
            or _cfg_get(cfg, 'max_agent_steps', int(1e9))
        )
        # Offset so phase-2 wandb steps continue after phase 1.
        self.wandb_step_offset = int(_cfg_get(cfg, 'wandb_step_offset', 0))
        # ---- Optim ----
        adapt_params = []
        for name, p in self.model.named_parameters():
            if 'adapt_tconv' in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        adaptation_lr = float(_cfg_get(cfg, 'adaptation_lr', 3e-4))
        self.optim = torch.optim.Adam(adapt_params, lr=adaptation_lr)
        # ---- Training Misc ----
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= self.max_agent_steps:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']).detach(),
                'priv_info': obs_dict['priv_info'],
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
            }
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
            loss = ((e - e_gt.detach()) ** 2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.direct_info['adaptation_loss'] = float(loss.detach().item())
            self.log_tensorboard()

            save_every = int(_cfg_get(self.cfg, 'adaptation_save_interval', 50_000))
            # This check allows for more frequent checkpointing
            if save_every > 0 and self.agent_steps % save_every < self.batch_size:
                self.save(os.path.join(self.nn_dir, f'step_{int(self.agent_steps)}'))
                last_path = os.path.join(self.nn_dir, 'model_last')
                self.save(last_path)
                self._maybe_eval_checkpoint(f'{last_path}.ckpt')

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                best_path = os.path.join(self.nn_dir, 'model_best')
                self.save(best_path)
                self.best_rewards = mean_rewards
                self._maybe_eval_checkpoint(f'{best_path}.ckpt')

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | '
                f'Last FPS: {last_fps:.1f} | '
                f'AdaptLoss: {loss.detach().item():.5f} | '
                f'Current Best: {self.best_rewards:.2f}'
            )
            tprint(info_string)

        # Final snapshot.
        self.save(os.path.join(self.nn_dir, 'model_last'))

    def _maybe_eval_checkpoint(self, checkpoint_path: str):
        cb = _cfg_get(self.cfg, 'eval_callback', None)
        if cb is None:
            return
        try:
            cb(checkpoint_path=checkpoint_path, stage='phase2')
        except Exception as exc:
            print(f'[ProprioAdapt] eval_callback failed (continuing): {exc}')

    def log_tensorboard(self):
        mean_rew = self.mean_eps_reward.get_mean()
        mean_len = self.mean_eps_length.get_mean()
        self.writer.add_scalar('episode_rewards/step', mean_rew, self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', mean_len, self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

        metrics = {
            'episode_rewards/step': mean_rew,
            'episode_lengths/step': mean_len,
            'charts/best_episode_reward': self.best_rewards,
        }
        metrics.update({f'{k}/frame': v for k, v in self.direct_info.items()})
        wandb_log(
            metrics,
            step=self.agent_steps + self.wandb_step_offset,
            prefix='phase2/',
        )

    def restore_train(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        print('careful, using non-strict matching')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def load_phase1_checkpoint(self, path):
        """Alias for restore_train — load Stage-1 weights into Stage-2 model."""
        return self.restore_train(path)

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights['sa_mean_std'] = self.sa_mean_std.state_dict()
        torch.save(weights, f'{name}.ckpt')
