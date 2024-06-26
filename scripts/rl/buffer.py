from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from types import Optional, Union, Generator
import numpy as np
from gymnasium import spaces
import torch as th


class RolloutRMABufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    obs_history: th.Tensor
    actions_history: th.Tensor
    last_action_history: th.Tensor
    priv_info: th.Tensor

class RolloutRMABuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        history_len: int = 1,
        priv_info_dim: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.history_len = history_len
        self.priv_info_dim = priv_info_dim
        self.reset()

    def reset(self) -> None:
        super().reset()

        self.obs_history = np.zeros((self.buffer_size, self.n_envs, self.history_len) + envs.observation_space.shape)
        self.actions_history = np.zeros((self.buffer_size, self.n_envs, self.history_len) + envs.action_space.shape)
        # TODO: change var name
        self.last_action_history = np.zeros((self.buffer_size, self.n_envs) + envs.action_space.shape)
        self.priv_info = np.zeros((self.buffer_size, self.n_envs, self.priv_info_dim))
        


    # def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
    #     """
    #     Post-processing step: compute the lambda-return (TD(lambda) estimate)
    #     and GAE(lambda) advantage.

    #     Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    #     to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
    #     where R is the sum of discounted reward with value bootstrap
    #     (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

    #     The TD(lambda) estimator has also two special cases:
    #     - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
    #     - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

    #     For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

    #     :param last_values: state value estimation for the last step (one for each env)
    #     :param dones: if the last step was a terminal step (one bool for each env).
    #     """
    #     # Convert to numpy
    #     last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

    #     last_gae_lam = 0
    #     for step in reversed(range(self.buffer_size)):
    #         if step == self.buffer_size - 1:
    #             next_non_terminal = 1.0 - dones.astype(np.float32)
    #             next_values = last_values
    #         else:
    #             next_non_terminal = 1.0 - self.episode_starts[step + 1]
    #             next_values = self.values[step + 1]
    #         delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
    #         last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
    #         self.advantages[step] = last_gae_lam
    #     # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
    #     # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
    #     self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        priv_info: np.ndarray=None,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        super().add(obs,
                action,
                reward,
                episode_start,
                value,
                log_prob,)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        self.pos -= 1

        for i in range(len(episode_start)):
            if episode_start[i]:
                self.obs_history[self.pos,i] = np.zeros(self.obs_history[self.pos][i].shape)
                self.actions_history[self.pos,i] = np.zeros(self.actions_history[self.pos][i].shape)
                self.last_action_history[self.pos,i] = np.zeros(self.last_action_history[self.pos][i].shape)
            else:
                self.obs_history[self.pos,i] = np.concatenate([self.obs_history[self.pos, i][1:], np.expand_dims(obs[i], axis=0)], axis=0)
                self.actions_history[self.pos,i] = np.concatenate([self.actions_history[self.pos, i][1:], np.expand_dims(action[i], axis=0)], axis=0)
                self.last_action_history[self.pos,i] = self.actions[self.pos - 1,i]

        if priv_info is None:
            self.priv_info[self.pos] = np.zeros(self.priv_info_dim)
        else:
            self.priv_info[self.pos] = priv_info

        self.pos += 1

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutRMABufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "obs_history",
                "actions_history",
                "last_action_history",
                "priv_info",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutRMABufferSamples:
        data = (
            self.observations[batch_inds].reshape((-1,) + self.obs_shape),
            self.actions[batch_inds].reshape((-1,) + self.action_dim),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.obs_history[batch_inds].reshape((-1,) + (self.history_len,) + self.obs_shape),
            self.actions_history[batch_inds].reshape((-1,) + (self.history_len,) + self.action_dim),
            self.last_action_history[batch_inds].reshape((-1,) + self.action_dim),
            self.priv_info[batch_inds].reshape((-1,) + self.priv_info_dim),
        )
        return RolloutRMABufferSamples(*tuple(map(self.to_torch, data)))

    def get_curr(self,):
        data = (
            self.observations[self.pos - 1],
            self.actions[self.pos - 1],
            self.values[self.pos - 1],
            self.log_probs[self.pos - 1],
            self.advantages[self.pos - 1],
            self.returns[self.pos - 1],
            self.obs_history[self.pos - 1],
            self.actions_history[self.pos - 1],
            self.last_action_history[self.pos - 1],
            self.priv_info[self.pos - 1],
        )
        return RolloutRMABufferSamples(*tuple(map(self.to_torch, data)))

