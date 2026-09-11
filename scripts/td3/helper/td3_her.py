"""Hindsight experience replay (HER) for the goal-conditioned TD3 trainer.

Used by ``scripts/td3/td3_training_her.py``.  Three pieces:

``GoalEnvVector``
    Single-env vector wrapper over an ``AirHockeyGoalEnv`` built with
    ``return_goal_obs: true``.  The env returns the gym ``GoalEnv`` dict
    (``observation`` / ``achieved_goal`` / ``desired_goal``); this flattens it
    to ``[observation, desired_goal]`` for the networks and forwards the
    achieved goal of the *next* state through ``infos`` (from the final,
    pre-reset state on episode end).  Same autoreset surface as
    ``td3_training.SingleEnvVector``.

``HERRelabeler``
    Pure-numpy hindsight relabelling of one finished episode (Andrychowicz
    et al. 2017).  For every transition ``t`` it draws ``k`` replacement goals
    from the achieved goals of later steps (``future`` strategy; ``final`` and
    ``episode`` are also available), rewrites the goal slice of ``obs`` /
    ``next_obs``, recomputes the reward with the task's own
    ``compute_reward`` and marks the copy terminal when the relabelled goal
    is met (the env ends the episode on goal arrival, so a relabelled
    transition that reaches its goal has to be terminal too) or when the
    original step ended the episode for a non-goal reason (puck hit the
    bottom, passed the paddle, ...).  Only *self-consistent* achieved states
    are used as goals — those that satisfy the task's success test against
    themselves (e.g. a puck-position goal counts only while the puck is
    moving up the table, so a falling puck's position is never proposed as a
    goal: it could never be rewarded).

``HEREpisodeTrajectory``
    ``EpisodeTrajectory`` plus the per-step achieved goals / hard-terminal
    flags HER needs, and ``flush_to_buffer`` that writes the original
    transitions followed by the relabelled copies.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from scripts.td3.helper.td3_episode_collection import EpisodeTrajectory

GoalFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ------------------------------------------------------------------ env side
def flatten_goal_observation(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs_dict["observation"], dtype=np.float32).reshape(-1),
         np.asarray(obs_dict["desired_goal"], dtype=np.float32).reshape(-1)]
    )


def make_goal_functions(env) -> Tuple[GoalFn, GoalFn, Optional[Callable[[np.ndarray], np.ndarray]], Optional[Callable[[np.ndarray], np.ndarray]]]:
    """``(compute_reward, goal_met, goal_in_distribution, achieved_to_desired)`` batched over ``(N, dim)``.

    ``compute_reward`` is the task's own (scaled by ``base_reward_scaling`` as
    ``env.step`` does).  ``goal_met`` uses the reward class's ``goal_met`` when
    it has one, else ``reward > 0`` (sparse tasks pay a positive reward only
    on success).  ``goal_in_distribution`` is the env's method of that name
    (desired goals -> bool: inside the goal-sampling region) or None.
    """
    reward_obj = env.reward
    scaling = float(getattr(env, "base_reward_scaling", 1.0))

    def compute_reward(achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
        achieved = np.asarray(achieved, dtype=np.float64).reshape(len(achieved), -1)
        desired = np.asarray(desired, dtype=np.float64).reshape(len(desired), -1)
        r = np.asarray(env.compute_reward(achieved, desired, {}), dtype=np.float64).reshape(-1)
        return r * scaling

    if hasattr(reward_obj, "goal_met"):
        def goal_met(achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
            achieved = np.asarray(achieved, dtype=np.float64).reshape(len(achieved), -1)
            desired = np.asarray(desired, dtype=np.float64).reshape(len(desired), -1)
            return np.asarray(reward_obj.goal_met(achieved, desired), dtype=bool).reshape(-1)
    else:
        def goal_met(achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
            return compute_reward(achieved, desired) > 0.0

    goal_in_distribution = None
    if hasattr(env, "goal_in_distribution"):
        def goal_in_distribution(desired: np.ndarray) -> np.ndarray:
            desired = np.asarray(desired, dtype=np.float64).reshape(len(desired), -1)
            return np.asarray(env.goal_in_distribution(desired), dtype=bool).reshape(-1)

    achieved_to_desired = None
    if hasattr(env, "achieved_to_desired"):
        def achieved_to_desired(achieved: np.ndarray) -> np.ndarray:
            achieved = np.asarray(achieved, dtype=np.float64).reshape(len(achieved), -1)
            return np.asarray(env.achieved_to_desired(achieved), dtype=np.float64).reshape(len(achieved), -1)

    return compute_reward, goal_met, goal_in_distribution, achieved_to_desired


class GoalEnvVector:
    """One ``GoalEnv``-dict env with the vector-env surface the trainer uses.

    ``step`` returns the flat ``[observation, desired_goal]`` obs and puts in
    ``infos``:
      * ``achieved_goal``  — achieved goal of the returned (next) state, shape (1, A)
      * ``hard_terminal``  — the step terminated the episode for a reason other
        than reaching the goal, shape (1,)
      * on episode end: ``final_observation`` (flat), ``final_achieved_goal``,
        ``final_info`` and ``_final_observation`` like ``SyncVectorEnv``.
    """

    def __init__(self, env_fn) -> None:
        self.env = env_fn()
        self.envs = [self.env]
        self.num_envs = 1
        space = self.env.observation_space
        if not isinstance(space, gym.spaces.Dict) or "desired_goal" not in space.spaces:
            raise TypeError(
                "GoalEnvVector needs a GoalEnv dict observation space; set "
                "`return_goal_obs: true` in the air_hockey config."
            )
        self.observation_dim = int(np.prod(space["observation"].shape))
        self.goal_dim = int(np.prod(space["desired_goal"].shape))
        self.achieved_dim = int(np.prod(space["achieved_goal"].shape))
        flat_dim = self.observation_dim + self.goal_dim
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.single_action_space = self.env.action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    @property
    def goal_slice(self) -> slice:
        return slice(self.observation_dim, self.observation_dim + self.goal_dim)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options) if options is not None else self.env.reset(seed=seed)
        return flatten_goal_observation(obs)[None].copy(), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        reward = float(np.asarray(reward, dtype=np.float64).reshape(-1)[0])
        reasons = info.get("termination_reasons", []) or []
        hard_terminal = bool(terminated) and (
            len(reasons) == 0 or any(r != "goal_reached" for r in reasons)
        )
        infos: Dict[str, Any] = {
            "paddle_puck_collision_count": np.array([info.get("paddle_puck_collision_count", 0)]),
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32)[None].copy(),
            "hard_terminal": np.array([hard_terminal]),
        }
        if terminated or truncated:
            final_flat = flatten_goal_observation(obs)
            obs, _ = self.env.reset()
            infos["final_observation"] = [final_flat]
            infos["final_achieved_goal"] = [np.asarray(infos["achieved_goal"][0]).copy()]
            infos["final_info"] = [info]
            infos["_final_observation"] = np.array([True])
        return (
            flatten_goal_observation(obs)[None].copy(),
            np.array([reward], dtype=np.float64),
            np.array([bool(terminated)]),
            np.array([bool(truncated)]),
            infos,
        )

    def close(self):
        self.env.close()


# ------------------------------------------------------------ relabelling
class HERRelabeler:
    """Hindsight relabelling of one episode's transitions (numpy, CPU)."""

    STRATEGIES = ("future", "final", "episode")

    def __init__(
        self,
        *,
        observation_dim: int,
        goal_dim: int,
        compute_reward: GoalFn,
        goal_met: GoalFn,
        k: int = 4,
        strategy: str = "future",
        done_on_success: bool = True,
        seed: int = 0,
        goal_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        achieved_to_goal: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"her_strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.observation_dim = int(observation_dim)
        self.goal_dim = int(goal_dim)
        self.goal_slice = slice(self.observation_dim, self.observation_dim + self.goal_dim)
        self.compute_reward = compute_reward
        self.goal_met = goal_met
        self.k = int(k)
        self.strategy = strategy
        self.done_on_success = bool(done_on_success)
        # Optional extra test on candidate goals (desired-goal space -> bool),
        # e.g. "inside the task's goal-sampling region".
        self.goal_filter = goal_filter
        # Achieved goal -> desired-goal space (default: first goal_dim columns).
        self.achieved_to_goal = achieved_to_goal
        self.rng = np.random.default_rng(int(seed))

    def relabel(
        self,
        *,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        prev_actions: np.ndarray,
        achieved_next: np.ndarray,
        hard_terminal: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return the relabelled copies of an episode, or None if there are none.

        ``achieved_next[t]`` is the achieved goal of ``next_obs[t]``; a goal
        proposed from step ``t'`` is ``achieved_next[t'][:goal_dim]``.
        """
        T = int(obs.shape[0])
        if self.k <= 0 or T == 0:
            return None
        ag = np.asarray(achieved_next, dtype=np.float64).reshape(T, -1)
        hard_terminal = np.asarray(hard_terminal, dtype=bool).reshape(T)
        ag_goal = (
            np.asarray(self.achieved_to_goal(ag), dtype=np.float64).reshape(T, self.goal_dim)
            if self.achieved_to_goal is not None else ag[:, : self.goal_dim]
        )
        # Achieved states that satisfy the task's own success test against
        # themselves: the only ones that can ever be rewarded as goals.
        valid = np.asarray(self.goal_met(ag, ag_goal), dtype=bool).reshape(T)
        if self.goal_filter is not None:
            valid &= np.asarray(self.goal_filter(ag_goal), dtype=bool).reshape(T)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return None

        t = np.arange(T)
        if self.strategy == "future":
            start = np.searchsorted(valid_idx, t)          # first valid index >= t
            n_avail = valid_idx.size - start
            rows = np.repeat(t, self.k)
            starts = np.repeat(start, self.k)
            avail = np.repeat(n_avail, self.k)
            keep = avail > 0
            rows, starts, avail = rows[keep], starts[keep], avail[keep]
            if rows.size == 0:
                return None
            pick = starts + np.floor(self.rng.random(rows.size) * avail).astype(np.int64)
            pick = np.minimum(pick, valid_idx.size - 1)
            source = valid_idx[pick]
        elif self.strategy == "final":
            keep = t <= valid_idx[-1]
            rows = t[keep]
            source = np.full(rows.size, valid_idx[-1], dtype=np.int64)
        else:  # episode: any valid achieved state of the episode
            rows = np.repeat(t, self.k)
            source = valid_idx[np.floor(self.rng.random(rows.size) * valid_idx.size).astype(np.int64)]

        new_goal = ag_goal[source]
        new_obs = np.array(obs[rows], dtype=np.float32, copy=True)
        new_obs[:, self.goal_slice] = new_goal
        new_next_obs = np.array(next_obs[rows], dtype=np.float32, copy=True)
        new_next_obs[:, self.goal_slice] = new_goal
        new_rewards = np.asarray(self.compute_reward(ag[rows], new_goal), dtype=np.float32).reshape(-1)
        success = np.asarray(self.goal_met(ag[rows], new_goal), dtype=bool).reshape(-1)
        new_dones = hard_terminal[rows]
        if self.done_on_success:
            new_dones = new_dones | success
        return {
            "obs": new_obs,
            "next_obs": new_next_obs,
            "actions": np.asarray(actions[rows], dtype=np.float32),
            "prev_actions": np.asarray(prev_actions[rows], dtype=np.float32),
            "rewards": new_rewards,
            "dones": new_dones.astype(np.float32),
            "success": success,
            "n_valid": int(valid_idx.size),
        }


# ------------------------------------------------------------- trajectory
class HEREpisodeTrajectory(EpisodeTrajectory):
    """Episode staging with the achieved goals and hard-terminal flags HER needs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.achieved_goals: List[np.ndarray] = []
        self.hard_terminals: List[bool] = []

    @staticmethod
    def empty() -> "HEREpisodeTrajectory":
        base = EpisodeTrajectory.empty()
        return HEREpisodeTrajectory(
            observations=base.observations,
            next_observations=base.next_observations,
            actions=base.actions,
            rewards=base.rewards,
            dones=base.dones,
            bootstrap_terminals=base.bootstrap_terminals,
            prev_actions=base.prev_actions,
            episode_return=0.0,
        )

    def append_step(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        prev_action: torch.Tensor,
        achieved_goal: np.ndarray,
        hard_terminal: bool,
        bootstrap_terminal: torch.Tensor | None = None,
    ) -> None:
        super().append_step(
            obs=obs, next_obs=next_obs, action=action, reward=reward, done=done,
            prev_action=prev_action, bootstrap_terminal=bootstrap_terminal,
        )
        self.achieved_goals.append(np.asarray(achieved_goal, dtype=np.float32).reshape(-1).copy())
        self.hard_terminals.append(bool(hard_terminal))

    def reset(self) -> None:
        super().reset()
        self.achieved_goals.clear()
        self.hard_terminals.clear()

    def flush_to_buffer(self, replay_buffer, relabeler: HERRelabeler | None = None) -> Dict[str, int]:  # type: ignore[override]
        """Write the episode (original transitions, then HER copies) to replay.

        Returns ``{"original": n, "relabeled": m, "valid": v}``.
        """
        n = len(self.observations)
        if n == 0:
            return {"original": 0, "relabeled": 0, "valid": 0}
        obs = torch.stack(self.observations, dim=0)
        next_obs = torch.stack(self.next_observations, dim=0)
        actions = torch.stack(self.actions, dim=0)
        prev_actions = torch.stack(self.prev_actions, dim=0)
        replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=torch.stack(self.rewards, dim=0).view(-1),
            dones=torch.stack(self.dones, dim=0).view(-1),
            prev_action=prev_actions,
        )
        stats = {"original": n, "relabeled": 0, "valid": 0}
        if relabeler is not None and relabeler.k > 0:
            out = relabeler.relabel(
                obs=obs.cpu().numpy(),
                next_obs=next_obs.cpu().numpy(),
                actions=actions.cpu().numpy(),
                prev_actions=prev_actions.cpu().numpy(),
                achieved_next=np.stack(self.achieved_goals, axis=0),
                hard_terminal=np.asarray(self.hard_terminals, dtype=bool),
            )
            if out is not None:
                replay_buffer.add(
                    obs=torch.from_numpy(out["obs"]),
                    next_obs=torch.from_numpy(out["next_obs"]),
                    actions=torch.from_numpy(out["actions"]),
                    rewards=torch.from_numpy(out["rewards"]),
                    dones=torch.from_numpy(out["dones"]),
                    prev_action=torch.from_numpy(out["prev_actions"]),
                )
                stats["relabeled"] = int(out["obs"].shape[0])
                stats["valid"] = int(out["n_valid"])
        self.reset()
        return stats

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["achieved_goals"] = [np.array(a, copy=True) for a in self.achieved_goals]
        state["hard_terminals"] = [bool(h) for h in self.hard_terminals]
        return state

    @classmethod
    def from_state_dict(cls, state_dict: Any, device: str) -> "HEREpisodeTrajectory":
        base = EpisodeTrajectory.from_state_dict(state_dict, device=device)
        traj = cls.empty()
        for attr in ("observations", "next_observations", "actions", "rewards", "dones",
                     "bootstrap_terminals", "prev_actions"):
            setattr(traj, attr, getattr(base, attr))
        traj.episode_return = base.episode_return
        if isinstance(state_dict, dict):
            traj.achieved_goals = [np.asarray(a, dtype=np.float32) for a in state_dict.get("achieved_goals", [])]
            traj.hard_terminals = [bool(h) for h in state_dict.get("hard_terminals", [])]
        n = len(traj.observations)
        if len(traj.achieved_goals) != n or len(traj.hard_terminals) != n:
            # Inconsistent partial episode (older checkpoint): drop it rather
            # than relabel garbage.
            traj.reset()
        return traj


def finalize_her_episode_if_done(
    episode_done: bool,
    episode_trajectory: HEREpisodeTrajectory,
    recent_episode_returns: deque,
    success_top_fraction: float,
    episode_return_success_threshold: float,
    success_rb,
    failure_rb,
    relabeler: HERRelabeler | None,
    single_buffer: bool = True,
) -> Tuple[float, Dict[str, int]]:
    """HER counterpart of ``td3_episode_collection.finalize_episode_if_done``.

    Same success-threshold bookkeeping and buffer routing; the flush writes
    the original episode plus its hindsight copies.
    """
    if not episode_done:
        return float(episode_return_success_threshold), {"original": 0, "relabeled": 0, "valid": 0}
    episode_return = float(episode_trajectory.episode_return)
    recent_episode_returns.append(episode_return)
    if len(recent_episode_returns) > 0:
        quantile = 1.0 - float(success_top_fraction)
        episode_return_success_threshold = float(
            np.quantile(np.asarray(recent_episode_returns, dtype=np.float32), quantile)
        )
    if single_buffer:
        target = success_rb
    else:
        target = success_rb if episode_return >= episode_return_success_threshold else failure_rb
    stats = episode_trajectory.flush_to_buffer(target, relabeler)
    return float(episode_return_success_threshold), stats
