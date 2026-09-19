"""Environment plumbing for RMA: expose the per-episode privileged factors e_t.

The Box2D env re-samples ``random_variables`` (paddle_density, puck_damping,
gravity for the canonical ±25 % DR config) on every ``reset()`` when
``domain_random: true`` and keeps them in ``env.simulator_params``.  RMA
needs that vector as the privileged input e_t of the encoder mu in phase 1
and as the regression target (through mu) in phase 2.

``RMAEnvVector`` is the single-env vector wrapper the phase-1 trainer uses.
It wraps ``scripts.td3.td3_training.SingleEnvVector`` (same autoreset
semantics; ``GoalEnvVector`` for goal-conditioned tasks, whose flat obs is
``[observation, desired_goal]``) and **appends the normalised e_t to every
observation**::

    obs_aug = [ x_t (obs_dim) | e_t normalised to [-1, 1] (env_param_dim) ]

so the unmodified replay buffer / update engine carry e_t with each
transition.  Normalisation uses the DR ranges of the sim config: the sysid
centre maps to 0 and the DR bounds to ±1; out-of-distribution physics maps
outside [-1, 1].
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gymnasium as gym
import numpy as np

from scripts.td3.td3_training import SingleEnvVector


class EnvParamNormalizer:
    """Affine map raw physics params <-> [-1, 1] over the DR ranges."""

    def __init__(self, random_variables: Sequence[str], random_variable_ranges: Dict[str, Sequence[float]]):
        self.random_variables: List[str] = [str(v) for v in random_variables]
        if not self.random_variables:
            raise ValueError("RMA needs at least one randomized variable (`random_variables` in the sim config).")
        lows, highs = [], []
        for var in self.random_variables:
            if var not in random_variable_ranges:
                raise KeyError(f"random_variables lists {var!r} but random_variable_ranges has no entry for it")
            lo, hi = random_variable_ranges[var]
            lows.append(float(lo))
            highs.append(float(hi))
        self.low = np.asarray(lows, dtype=np.float32)
        self.high = np.asarray(highs, dtype=np.float32)
        if np.any(self.high <= self.low):
            raise ValueError(f"random_variable_ranges must have high > low, got low={self.low} high={self.high}")

    @property
    def dim(self) -> int:
        return len(self.random_variables)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        return 2.0 * (raw - self.low) / (self.high - self.low) - 1.0

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized, dtype=np.float32)
        return self.low + (normalized + 1.0) * 0.5 * (self.high - self.low)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "random_variables": list(self.random_variables),
            "random_variable_ranges": {
                var: [float(lo), float(hi)] for var, lo, hi in zip(self.random_variables, self.low, self.high)
            },
        }

    @classmethod
    def from_air_hockey_config(cls, air_hockey_params: Dict[str, Any]) -> "EnvParamNormalizer":
        return cls(
            list(air_hockey_params.get("random_variables", [])),
            dict(air_hockey_params.get("random_variable_ranges", {})),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvParamNormalizer":
        return cls(d["random_variables"], d["random_variable_ranges"])


def raw_env_params(env, random_variables: Sequence[str]) -> np.ndarray:
    """Read the current physics parameters of a (reset) Box2D env."""
    sim_params = getattr(env, "simulator_params", None)
    if sim_params is None:
        env = getattr(env, "unwrapped", env)
        sim_params = getattr(env, "simulator_params", None)
    if sim_params is None:
        raise AttributeError("env has no `simulator_params`; cannot read privileged env factors")
    return np.asarray([float(getattr(sim_params, var)) for var in random_variables], dtype=np.float32)


def make_inner_vector(env_fn, goal: bool):
    """The single-env vector the RMA wrappers build on: ``SingleEnvVector`` (flat obs) or, for
    goal-conditioned tasks (``return_goal_obs: true``), ``GoalEnvVector`` (flat ``[observation,
    desired_goal]`` obs + the ``achieved_goal`` / ``hard_terminal`` / ``final_achieved_goal`` infos
    the HER relabeler needs)."""
    if goal:
        from scripts.td3.helper.td3_her import GoalEnvVector

        return GoalEnvVector(env_fn)
    return SingleEnvVector(env_fn)


class _AugmentedEnvVector:
    """Shared plumbing: an inner single-env vector whose observations get a suffix appended.

    ``goal=True`` makes the inner vector a ``GoalEnvVector`` (goal-conditioned tasks); the flat
    observation is then ``[x_t | desired_goal | suffix]`` and the inner infos (``achieved_goal``,
    ``hard_terminal``, ``final_achieved_goal``) are passed through, so the HER relabeler can
    rewrite the goal slice of the stored observations while the suffix stays intact.
    """

    def __init__(self, env_fn, goal: bool = False) -> None:
        self.inner = make_inner_vector(env_fn, goal)
        self.goal = bool(goal)
        self.env = self.inner.env
        self.envs = [self.env]
        self.num_envs = 1
        base = self.inner.single_observation_space
        self.raw_obs_dim = int(np.prod(base.shape))
        self.single_action_space = self.inner.single_action_space
        self.action_space = self.single_action_space
        self._base_low = np.asarray(base.low, dtype=np.float64).reshape(-1)
        self._base_high = np.asarray(base.high, dtype=np.float64).reshape(-1)

    def _set_obs_space(self, suffix_dim: int) -> None:
        low = np.concatenate([self._base_low, -np.ones(suffix_dim) * np.inf])
        high = np.concatenate([self._base_high, np.ones(suffix_dim) * np.inf])
        self.single_observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        self.observation_space = self.single_observation_space

    @property
    def goal_dim(self) -> int:
        return int(getattr(self.inner, "goal_dim", 0))

    @property
    def observation_dim(self) -> int:
        """The task observation without the goal (30 for the canonical history obs)."""
        return int(getattr(self.inner, "observation_dim", self.raw_obs_dim))

    def close(self):
        self.inner.close()


class RMAEnvVector(_AugmentedEnvVector):
    """Single-env vector wrapper that appends normalised e_t to the observation."""

    def __init__(self, env_fn, normalizer: EnvParamNormalizer, goal: bool = False) -> None:
        super().__init__(env_fn, goal=goal)
        self.normalizer = normalizer
        self.env_param_dim = normalizer.dim
        self._set_obs_space(self.env_param_dim)
        self._current_env_params_norm = np.zeros(self.env_param_dim, dtype=np.float32)
        self._current_env_params_raw = np.zeros(self.env_param_dim, dtype=np.float32)

    # ---------------------------------------------------------------- params
    def _refresh_env_params(self) -> None:
        self._current_env_params_raw = raw_env_params(self.env, self.normalizer.random_variables)
        self._current_env_params_norm = self.normalizer.normalize(self._current_env_params_raw)

    @property
    def current_env_params(self) -> np.ndarray:
        """Normalised e_t of the episode currently in progress."""
        return self._current_env_params_norm.copy()

    @property
    def current_env_params_raw(self) -> np.ndarray:
        return self._current_env_params_raw.copy()

    def _augment(self, obs) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float64).reshape(-1), self._current_env_params_norm.astype(np.float64)])

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        self._refresh_env_params()
        return self._augment(obs[0])[None].copy(), info

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.inner.step(actions)
        if "final_observation" in infos:
            # The final observation belongs to the episode that just ended:
            # augment it with the OLD e_t before reading the re-sampled physics.
            infos["final_observation"] = [self._augment(infos["final_observation"][0])]
            self._refresh_env_params()
        return self._augment(obs[0])[None].copy(), reward, terminated, truncated, infos


class HistoryEnvVector(_AugmentedEnvVector):
    """Single-env vector wrapper that appends the flattened H-step (x, a) window.

    Used by the long-history TD3 control baseline (``HistoryActor``): the
    observation becomes ``[x_t | window_t]`` where ``window_t`` holds the H
    (state-features, action) pairs *before* step t, padded exactly like the
    RMA adaptation module's history (``scripts.rma.history``).  No privileged
    information is exposed.  For goal tasks x_t = ``[observation, desired_goal]``;
    the per-step features are taken from the observation part.
    """

    def __init__(self, env_fn, history_len: int, step_features: str, goal: bool = False) -> None:
        from scripts.rma.history import StepHistoryBuffer
        from scripts.rma.networks import step_feature_dim, step_state_features

        super().__init__(env_fn, goal=goal)
        self.history_len = int(history_len)
        self.step_features = str(step_features)
        self.act_dim = int(np.prod(self.single_action_space.shape))
        self.feature_dim = step_feature_dim(self.raw_obs_dim, self.act_dim, self.step_features)
        self.window_dim = self.history_len * self.feature_dim
        self._features = step_state_features
        self.history = StepHistoryBuffer(self.history_len, self.feature_dim, self.act_dim)
        self._set_obs_space(self.window_dim)
        self._prev_raw_obs = None

    def _state_features(self, obs) -> np.ndarray:
        import torch

        return self._features(torch.as_tensor(np.asarray(obs, dtype=np.float32)), self.step_features).numpy()

    def _augment(self, obs) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float64).reshape(-1), self.history.window().reshape(-1).astype(np.float64)])

    def _start_episode(self, obs) -> None:
        self.history.reset(self._state_features(obs))
        self._prev_raw_obs = np.asarray(obs, dtype=np.float32).copy()

    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        self._start_episode(obs[0])
        return self._augment(obs[0])[None].copy(), info

    def step(self, actions):
        action = np.asarray(actions[0], dtype=np.float32).reshape(-1)
        self.history.push(self._state_features(self._prev_raw_obs), action)
        obs, reward, terminated, truncated, infos = self.inner.step(actions)
        if "final_observation" in infos:
            infos["final_observation"] = [self._augment(infos["final_observation"][0])]
            self._start_episode(obs[0])
        else:
            self._prev_raw_obs = np.asarray(obs[0], dtype=np.float32).copy()
        return self._augment(obs[0])[None].copy(), reward, terminated, truncated, infos
