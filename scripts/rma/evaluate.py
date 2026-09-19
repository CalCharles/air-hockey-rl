"""Evaluation of RMA policies on the fixed DR eval-env set.

Mirrors ``scripts.td3.td3_training_dr._evaluate_agent_multi_env`` (same
``eval_param_seed`` -> same ``eval_n_envs`` physics overrides, same
per-checkpoint episode-seed shift, same ``multi_env_eval.json`` aggregate
schema so ``run_experiments.py --summarise-only`` reads RMA runs too) but
takes *agents* instead of a bare actor, because RMA policies carry state:

* ``PrivilegedRMAAgent`` — pi(x, a_prev, mu(e)) with the true e_t of the
  env (phase-1 policy; sim-only "expert"/oracle).
* ``AdaptedRMAAgent``   — pi(x, a_prev, phi(history)) — the deployable
  policy (phase 2).  Also logs the online latent error |z_hat - mu(e)|^2.
* ``NominalLatentAgent``— pi(x, a_prev, mu(0)): the base policy run at the
  sysid-centre latent, i.e. RMA without adaptation.
* ``PlainTD3Agent``     — a canonical DR-trained ``DeterministicAgent``
  (the paper's parameter-randomization baseline) on the same envs.
* ``HistoryAgent``      — the long-history TD3 control (``HistoryActor``: the
  deployable RMA architecture trained end-to-end without privileged input).

Every agent is run on the same env seeds, so comparisons are paired.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import imageio
import numpy as np
import torch

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.rma.env_wrapper import EnvParamNormalizer, raw_env_params
from scripts.rma.history import StepHistoryBuffer
from scripts.rma.networks import AdaptationModule, HistoryActor, RMAActor, step_feature_dim, step_state_features
from scripts.td3.eval_utils import build_policy_env_view, load_policy_for_evaluation

# The DR-trainer helpers below are imported lazily by the multi-env functions
# that use them. Importing the trainer at module scope would make the agent
# classes — which deployment loads to run a single policy — depend on the whole
# training stack.

GIF_WIDTH = 160
GIF_FPS = 20


# ---------------------------------------------------------------------- agents
class EvalAgent:
    """reset(obs, env_params_norm) at episode start; act(obs, last_action) -> action."""

    name = "agent"

    def reset(self, obs: np.ndarray, env_params_norm: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def act(self, obs: np.ndarray, last_action: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def episode_extras(self) -> Dict[str, float]:
        return {}


class PrivilegedRMAAgent(EvalAgent):
    name = "privileged"

    def __init__(self, actor: RMAActor):
        self.actor = actor.eval()
        self._env_params = None

    def reset(self, obs, env_params_norm):
        self._env_params = torch.as_tensor(env_params_norm, dtype=torch.float32).reshape(1, -1)

    @torch.no_grad()
    def act(self, obs, last_action):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        parts = [obs_t, self._env_params]
        if self.actor.use_last_action:
            parts.append(torch.as_tensor(last_action, dtype=torch.float32).reshape(1, -1))
        return self.actor.get_action(torch.cat(parts, dim=-1)).numpy().reshape(-1)


class NominalLatentAgent(EvalAgent):
    """Base policy with the latent of the DR centre (normalised e = 0)."""

    name = "nominal"

    def __init__(self, actor: RMAActor):
        self.actor = actor.eval()
        with torch.no_grad():
            self._latent = actor.encoder(torch.zeros(1, actor.env_param_dim))

    def reset(self, obs, env_params_norm):
        pass

    @torch.no_grad()
    def act(self, obs, last_action):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        prev = torch.as_tensor(last_action, dtype=torch.float32).reshape(1, -1) if self.actor.use_last_action else None
        return self.actor.get_action_from_latent(obs_t, prev, self._latent).numpy().reshape(-1)


class AdaptedRMAAgent(EvalAgent):
    name = "adapted"

    def __init__(self, actor: RMAActor, adaptation: AdaptationModule, step_features: str, action_noise: float = 0.0, rng: Optional[np.random.Generator] = None):
        self.actor = actor.eval()
        self.adaptation = adaptation.eval()
        self.step_features = str(step_features)
        self.feature_dim = step_feature_dim(actor.obs_dim, actor.act_dim, self.step_features)
        if self.feature_dim != adaptation.feature_dim:
            raise ValueError(
                f"adaptation module feature_dim={adaptation.feature_dim} but step_features={step_features!r} gives {self.feature_dim}"
            )
        self.history = StepHistoryBuffer(adaptation.history_len, self.feature_dim, actor.act_dim)
        self.action_noise = float(action_noise)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._target_latent = None
        self._sq_err: List[float] = []
        self._latest_latent = None

    def reset(self, obs, env_params_norm=None):
        # ``env_params_norm=None`` is the deployment case (real robot): the true
        # physics are unknown, so there is no target latent to compare against
        # and ``latent_mse`` is simply not reported. ``act`` already guards on
        # ``_target_latent is None``.
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        self.history.reset(step_state_features(obs_t, self.step_features).numpy())
        if env_params_norm is None:
            self._target_latent = None
        else:
            with torch.no_grad():
                self._target_latent = self.actor.encoder(torch.as_tensor(env_params_norm, dtype=torch.float32).reshape(1, -1))
        self._sq_err = []

    @torch.no_grad()
    def act(self, obs, last_action):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        window = torch.from_numpy(self.history.window()).unsqueeze(0)
        latent = self.adaptation(window)
        self._latest_latent = latent
        if self._target_latent is not None:
            self._sq_err.append(float(((latent - self._target_latent) ** 2).mean()))
        prev = torch.as_tensor(last_action, dtype=torch.float32).reshape(1, -1) if self.actor.use_last_action else None
        action = self.actor.get_action_from_latent(obs_t, prev, latent).numpy().reshape(-1)
        if self.action_noise > 0.0:
            action = np.clip(action + self.rng.normal(0.0, self.action_noise, size=action.shape), -1.0, 1.0)
        self.history.push(step_state_features(obs_t[0], self.step_features).numpy(), action)
        return action.astype(np.float32)

    def episode_extras(self):
        if not self._sq_err:
            return {}
        err = np.asarray(self._sq_err, dtype=np.float64)
        return {
            "latent_mse": float(err.mean()),
            "latent_mse_first10": float(err[:10].mean()),
            "latent_mse_after50": float(err[50:].mean()) if err.shape[0] > 50 else float("nan"),
        }


class HistoryAgent(EvalAgent):
    """Long-history TD3 baseline: pi(x, a_prev, phi(history)) with phi trained by RL (no privileged input)."""

    name = "history"

    def __init__(self, actor: HistoryActor, step_features: str | None = None):
        self.actor = actor.eval()
        self.step_features = str(step_features or getattr(actor, "step_features", "latest_frame"))
        self.history = StepHistoryBuffer(actor.history_len, actor.feature_dim, actor.act_dim)

    def reset(self, obs, env_params_norm=None):
        self.history.reset(step_state_features(torch.as_tensor(obs, dtype=torch.float32), self.step_features).numpy())

    @torch.no_grad()
    def act(self, obs, last_action):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        window = torch.from_numpy(self.history.window()).unsqueeze(0)
        prev = torch.as_tensor(last_action, dtype=torch.float32).reshape(1, -1) if self.actor.use_last_action else None
        action = self.actor.get_action_from_latent(obs_t, prev, self.actor.encoder(window)).numpy().reshape(-1)
        self.history.push(step_state_features(obs_t[0], self.step_features).numpy(), action)
        return action.astype(np.float32)


class PlainTD3Agent(EvalAgent):
    """Canonical DR-trained TD3 actor (no privileged input) — the DR baseline."""

    name = "td3_dr"

    def __init__(self, model_path: str, obs_dim: int, act_dim: int, use_last_action: bool, hidden_layer_size: int, num_hidden_layers: int):
        from types import SimpleNamespace

        import gymnasium as gym

        env_view = SimpleNamespace(
            single_observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            single_action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
        )
        self.use_last_action = bool(use_last_action)
        self.model = load_policy_for_evaluation(
            model_path=model_path,
            policy_env_view=build_policy_env_view(env_view, self.use_last_action),
            action_scale=1,
            agent_hidden_layer_size=hidden_layer_size,
            agent_num_hidden_layers=num_hidden_layers,
        )

    def reset(self, obs, env_params_norm):
        pass

    @torch.no_grad()
    def act(self, obs, last_action):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        if self.use_last_action:
            obs_t = torch.cat([obs_t, torch.as_tensor(last_action, dtype=torch.float32).reshape(1, -1)], dim=-1)
        return self.model(obs_t).numpy().reshape(-1)


# --------------------------------------------------------------------- rollout
def _gif_frame(renderer, reward: float, cum_reward: float) -> np.ndarray:
    frame = renderer.get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    aspect_ratio = frame.shape[1] / frame.shape[0]
    frame = cv2.resize(frame, (GIF_WIDTH, int(GIF_WIDTH / aspect_ratio)))
    cv2.putText(frame, f"R: {reward:.2f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, f"G: {cum_reward:.2f}", (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame


class FlatGoalEnv:
    """``AirHockeyEnv`` of a goal-conditioned task with flat ``[observation, desired_goal]`` observations.

    The RMA trainers see exactly this layout (``GoalEnvVector``), so the evaluation agents get the
    same x_t. ``simulator_params`` / ``unwrapped`` are forwarded for the privileged-factor readout
    and the renderer."""

    def __init__(self, air_hockey_params: Dict[str, Any]) -> None:
        import gymnasium as gym

        from scripts.td3.helper.td3_her import flatten_goal_observation

        params = dict(air_hockey_params)
        params["return_goal_obs"] = True
        self.env = AirHockeyEnv(params)
        self._flatten = flatten_goal_observation
        space = self.env.observation_space
        self.observation_dim = int(np.prod(space["observation"].shape))
        self.goal_dim = int(np.prod(space["desired_goal"].shape))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim + self.goal_dim,), dtype=np.float32)
        self.action_space = self.env.action_space

    @property
    def unwrapped(self):
        return self.env

    @property
    def simulator_params(self):
        return self.env.simulator_params

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), float(np.asarray(reward, dtype=np.float64).reshape(-1)[0]), terminated, truncated, info

    def close(self):
        self.env.close()


def make_eval_env(air_hockey_params: Dict[str, Any]):
    """A flat-observation env for rollouts: ``FlatGoalEnv`` when the config is goal-conditioned."""
    if air_hockey_params.get("return_goal_obs", False):
        return FlatGoalEnv(air_hockey_params)
    return AirHockeyEnv(air_hockey_params)


def rollout_episodes(
    air_hockey_params: Dict[str, Any],
    agent: EvalAgent,
    normalizer: EnvParamNormalizer,
    n_eps: int,
    gif_path: Optional[str] = None,
    gif_episodes: int = 1,
) -> Dict[str, Any]:
    """Run ``n_eps`` episodes of ``agent`` in a fresh env built from ``air_hockey_params``."""
    env = make_eval_env(air_hockey_params)
    renderer = None
    frames: List[np.ndarray] = []
    if gif_path is not None:
        renderer = AirHockeyRenderer(getattr(env, "unwrapped", env), show_target_position=True, show_acceleration_arrow=False)
    act_dim = int(np.prod(env.action_space.shape))

    returns, successes, lengths, extras = [], [], [], []
    for ep in range(int(n_eps)):
        obs, _ = env.reset()
        env_params = normalizer.normalize(raw_env_params(env, normalizer.random_variables))
        agent.reset(np.asarray(obs, dtype=np.float32), env_params)
        last_action = np.zeros(act_dim, dtype=np.float32)
        done = False
        cum_rew, rew, steps = 0.0, 0.0, 0
        record = renderer is not None and ep < gif_episodes
        info = {}
        while not done:
            if record:
                frames.append(_gif_frame(renderer, rew, cum_rew))
            action = agent.act(np.asarray(obs, dtype=np.float32), last_action)
            obs, rew, term, trunc, info = env.step(action)
            rew = float(np.asarray(rew, dtype=np.float64).reshape(-1)[0])
            cum_rew += rew
            steps += 1
            done = bool(term or trunc)
            last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        returns.append(cum_rew)
        successes.append(int(bool(info.get("success", False))) if info else 0)
        lengths.append(steps)
        extras.append(agent.episode_extras())
    env.close()
    if gif_path is not None and frames:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        imageio.mimsave(gif_path, frames, format="GIF", loop=0, duration=int(1000 / GIF_FPS))

    out: Dict[str, Any] = {
        "returns": returns,
        "successes": successes,
        "episode_lengths": lengths,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_success_rate": float(np.mean(successes)) if successes else float("nan"),
        "mean_episode_length": float(np.mean(lengths)) if lengths else float("nan"),
    }
    extra_keys = sorted({k for e in extras for k in e})
    for k in extra_keys:
        vals = [e[k] for e in extras if k in e and np.isfinite(e[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


# ------------------------------------------------------------------ multi-env
def eval_env_overrides(air_hockey_params: Dict[str, Any], eval_param_seed: int, n_envs: int, ranges_key: str = "random_variable_ranges") -> List[Dict[str, float]]:
    random_variables = list(air_hockey_params.get("random_variables", []))
    ranges = dict(air_hockey_params.get(ranges_key, {}))
    if not random_variables or not ranges:
        raise ValueError(f"air_hockey config needs `random_variables` and `{ranges_key}` to sample eval envs")
    from scripts.td3.td3_training_dr import _sample_eval_env_overrides

    return _sample_eval_env_overrides(seed=int(eval_param_seed), n_envs=int(n_envs), random_variable_ranges=ranges, random_variables=random_variables)


def _aggregate(per_env: List[Dict[str, Any]], eps_per_env: int) -> Dict[str, Any]:
    valid = [r for r in per_env if "error" not in r]
    agg: Dict[str, Any] = {
        "n_envs_used": len(valid),
        "eps_per_env": int(eps_per_env),
        "mean_return_across_envs": float(np.mean([r["mean_return"] for r in valid])) if valid else float("nan"),
        "mean_success_across_envs": float(np.mean([r["mean_success_rate"] for r in valid])) if valid else float("nan"),
        "mean_ep_length_across_envs": float(np.mean([r["mean_episode_length"] for r in valid])) if valid else float("nan"),
        "per_env_mean_return": [r.get("mean_return", float("nan")) for r in per_env],
        "per_env_mean_success": [r.get("mean_success_rate", float("nan")) for r in per_env],
    }
    all_returns = [x for r in valid for x in r["returns"]]
    agg["return_std_across_episodes"] = float(np.std(all_returns)) if all_returns else float("nan")
    agg["return_sem_across_episodes"] = float(np.std(all_returns) / np.sqrt(len(all_returns))) if all_returns else float("nan")
    for k in ("latent_mse", "latent_mse_first10", "latent_mse_after50"):
        vals = [r[k] for r in valid if k in r and np.isfinite(r[k])]
        if vals:
            agg[k] = float(np.mean(vals))
    return agg


def evaluate_multi_env(
    air_hockey_params: Dict[str, Any],
    agents: Dict[str, Callable[[], EvalAgent]],
    normalizer: EnvParamNormalizer,
    *,
    eval_param_seed: int,
    n_envs: int,
    eps_per_env: int,
    call_index: int = 1,
    save_dir: Optional[str] = None,
    primary: Optional[str] = None,
    gif_agents: Sequence[str] = (),
    ranges_key: str = "random_variable_ranges",
    json_name: str = "multi_env_eval.json",
    log_prefix: str = "[rma]",
) -> Dict[str, Any]:
    """Roll every agent on the same ``n_envs`` fixed physics overrides.

    Returns ``{"overrides": [...], "agents": {name: {"aggregate", "per_env"}}}``
    and, if ``save_dir`` is given, writes ``json_name`` there with the
    ``primary`` agent's aggregate/per_env at the top level (canonical DR-eval
    schema) plus the full ``agents`` block.  ``call_index`` shifts the
    episode seeds exactly like ``td3_training_dr`` does per checkpoint.
    """
    from scripts.td3.td3_training_dr import _apply_overrides_to_air_hockey_params

    overrides = eval_env_overrides(air_hockey_params, eval_param_seed, n_envs, ranges_key=ranges_key)
    primary = primary or next(iter(agents))
    results: Dict[str, Dict[str, Any]] = {name: {"per_env": []} for name in agents}
    for env_idx, override in enumerate(overrides):
        eval_cfg = _apply_overrides_to_air_hockey_params(air_hockey_params, override)
        eval_cfg["seed"] = int(int(eval_param_seed) * 100000 + env_idx * 1000 + int(call_index))
        for name, factory in agents.items():
            gif_path = None
            if save_dir is not None and env_idx == 0 and name in gif_agents:
                gif_path = os.path.join(save_dir, f"eval_{name}_env0.gif")
            try:
                stats = rollout_episodes(copy.deepcopy(eval_cfg), factory(), normalizer, eps_per_env, gif_path=gif_path)
                stats["env_idx"] = env_idx
                stats["override"] = override
            except Exception as exc:  # keep the other envs / agents going
                print(f"{log_prefix} env{env_idx} agent={name} rollout failed (continuing): {exc}")
                stats = {"env_idx": env_idx, "override": override, "error": str(exc)}
            results[name]["per_env"].append(stats)
    for name in agents:
        results[name]["aggregate"] = _aggregate(results[name]["per_env"], eps_per_env)
        agg = results[name]["aggregate"]
        per_env_str = ", ".join(f"{x:.1f}" for x in agg["per_env_mean_return"])
        latent = f", latent_mse={agg['latent_mse']:.4f}" if "latent_mse" in agg else ""
        print(
            f"{log_prefix} Multi-env eval [{name}] (n_envs={n_envs}, eps/env={eps_per_env}, ranges={ranges_key}): "
            f"mean_return={agg['mean_return_across_envs']:.2f}, mean_success={agg['mean_success_across_envs']:.3f}, "
            f"per_env_returns=[{per_env_str}]{latent}",
            flush=True,
        )
    payload = {
        "aggregate": results[primary]["aggregate"],
        "per_env": results[primary]["per_env"],
        "primary_agent": primary,
        "eval_param_seed": int(eval_param_seed),
        "call_index": int(call_index),
        "ranges_key": ranges_key,
        "overrides": overrides,
        "agents": results,
    }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, json_name), "w") as f:
            json.dump(payload, f, indent=2)
    return payload
