# --------------------------------------------------------
# PPO (RMA Stage 1) — jointly train μ + π with privileged info.
# Ported from hora/hora/algo/ppo/ppo.py for air-hockey-rl-rma.
# --------------------------------------------------------

import os
import time

import torch

from scripts.rma.experience import ExperienceBuffer
from scripts.rma.misc import AverageScalarMeter
from scripts.rma.models import ActorCritic
from scripts.rma.running_mean_std import RunningMeanStd
from scripts.rma.wandb_utils import wandb_log

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter


def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default)


class PPO(object):
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
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Priv Info ----
        self.priv_info_dim = cfg.priv_info_dim
        self.priv_info = _cfg_get(cfg, 'priv_info', True)
        self.proprio_adapt = _cfg_get(cfg, 'proprio_adapt', False)
        # ---- Model ----
        net_config = {
            'actor_units': list(cfg.actor_units),
            'priv_mlp_units': list(cfg.priv_mlp_units),
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage1_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(cfg.learning_rate)
        self.weight_decay = float(_cfg_get(cfg, 'weight_decay', 0.0))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.last_lr, weight_decay=self.weight_decay
        )
        # ---- PPO Train Param ----
        self.e_clip = cfg.e_clip
        self.clip_value = cfg.clip_value
        self.entropy_coef = cfg.entropy_coef
        self.critic_coef = cfg.critic_coef
        self.bounds_loss_coef = cfg.bounds_loss_coef
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.truncate_grads = cfg.truncate_grads
        self.grad_norm = cfg.grad_norm
        self.value_bootstrap = cfg.value_bootstrap
        self.normalize_advantage = cfg.normalize_advantage
        self.normalize_input = cfg.normalize_input
        self.normalize_value = cfg.normalize_value
        self.reward_scale = float(_cfg_get(cfg, 'reward_scale', 1.0))
        # ---- PPO Collect Param ----
        self.horizon_length = cfg.horizon_length
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = cfg.minibatch_size
        self.mini_epochs_num = cfg.mini_epochs
        assert self.batch_size % self.minibatch_size == 0, (
            f'batch_size ({self.batch_size}) must be divisible by '
            f'minibatch_size ({self.minibatch_size})'
        )
        # ---- scheduler ----
        self.kl_threshold = cfg.kl_threshold
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot ----
        self.save_freq = cfg.save_frequency
        self.save_best_after = cfg.save_best_after
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        self.writer = SummaryWriter(self.tb_dir)

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_shape[0],
            self.actions_num,
            self.priv_info_dim,
            self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = cfg.max_agent_steps
        self.best_rewards = -10000
        # ---- Timing ----
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        actor_loss = torch.mean(torch.stack(a_losses)).item()
        critic_loss = torch.mean(torch.stack(c_losses)).item()
        bounds_loss = torch.mean(torch.stack(b_losses)).item() if b_losses else 0.0
        entropy = torch.mean(torch.stack(entropies)).item()
        kl = torch.mean(torch.stack(kls)).item()
        rl_fps = self.agent_steps / max(self.rl_train_time, 1e-8)
        env_fps = self.agent_steps / max(self.data_collect_time, 1e-8)

        self.writer.add_scalar('performance/RLTrainFPS', rl_fps, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', env_fps, self.agent_steps)
        self.writer.add_scalar('losses/actor_loss', actor_loss, self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', bounds_loss, self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', critic_loss, self.agent_steps)
        self.writer.add_scalar('losses/entropy', entropy, self.agent_steps)
        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', kl, self.agent_steps)

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

        metrics = {
            'performance/RLTrainFPS': rl_fps,
            'performance/EnvStepFPS': env_fps,
            'losses/actor_loss': actor_loss,
            'losses/bounds_loss': bounds_loss,
            'losses/critic_loss': critic_loss,
            'losses/entropy': entropy,
            'info/last_lr': self.last_lr,
            'info/e_clip': self.e_clip,
            'info/kl': kl,
        }
        metrics.update({str(k): v for k, v in self.extra_info.items()})
        wandb_log(metrics, step=self.agent_steps, prefix='phase1/')


    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'priv_info': obs_dict['priv_info'],
        }
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | '
                f'Last FPS: {last_fps:.1f} | '
                f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                f'Current Best: {self.best_rewards:.2f}'
            )
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            wandb_log(
                {
                    'episode_rewards/step': mean_rewards,
                    'episode_lengths/step': mean_lengths,
                    'charts/best_episode_reward': self.best_rewards,
                },
                step=self.agent_steps,
                prefix='phase1/',
            )
            checkpoint_name = (
                f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'
            )

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    last_path = os.path.join(self.nn_dir, 'last')
                    self.save(last_path)
                    self._maybe_eval_checkpoint(f'{last_path}.pth')

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                self.best_rewards = mean_rewards
                best_path = os.path.join(self.nn_dir, 'best')
                self.save(best_path)
                self._maybe_eval_checkpoint(f'{best_path}.pth')

        # Always persist a final snapshot.
        final_path = os.path.join(self.nn_dir, 'last')
        self.save(final_path)
        print('max steps achieved')

    def _maybe_eval_checkpoint(self, checkpoint_path: str):
        cb = _cfg_get(self.cfg, 'eval_callback', None)
        if cb is None:
            return
        try:
            cb(checkpoint_path=checkpoint_path, stage='phase1')
        except Exception as exc:
            print(f'[PPO] eval_callback failed (continuing): {exc}')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': obs_dict['priv_info'],
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info = self.storage[i]

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros((), device=self.device)
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (
                    isinstance(v, torch.Tensor) and len(v.shape) == 0
                ):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps += self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
