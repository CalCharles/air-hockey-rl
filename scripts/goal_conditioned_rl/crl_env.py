"""The new sim for contrastive RL: env construction, policy state and the true-puck goal view.

The sim config is used unchanged (``configs/new_juggle/sysid_v2_hist4_realplant.yaml``: noise, occlusion,
25 ms puck lag all on), with only ``max_timesteps`` set by the caller and 2 control steps of action latency
applied outside the sim (``scripts/rma/action_latency.py``), as the new-sim TD3 recipe does. No DR.

Goal view = the puck's (x, y, vx, vy) read from the Box2D body, i.e. noise-free and undelayed. Note that
``env.current_state`` (and ``scripts.rma.env_wrapper.true_state_features``) is NOT that: it is the
simulator's observation state, which carries the puck lag and noise.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
NEW_SIM = REPO / "configs/new_juggle/sysid_v2_hist4_realplant.yaml"
GOAL_DIM = 4
OBS_DIM = 30
ACTION_DIM = 2
STATE_DIM = OBS_DIM + ACTION_DIM  # history observation + the policy's own last action


def load_sim_config(path=NEW_SIM, max_timesteps: int = 7000) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())["air_hockey"]
    cfg = copy.deepcopy(cfg)
    cfg["max_timesteps"] = int(max_timesteps)
    return cfg


def make_env(cfg: dict, seed: int, action_latency: int = 2):
    from airhockey import AirHockeyEnv
    from scripts.rma.action_latency import install_action_latency

    params = copy.deepcopy(cfg)
    params["seed"] = int(seed)
    params["simulator_params"]["seed"] = int(seed)
    env = AirHockeyEnv(params)
    return install_action_latency(env, action_latency)


def puck_goal(env) -> np.ndarray:
    """True puck (x, y, vx, vy) in the env frame: Box2D (bx, by) -> env (-by, bx)."""
    puck = next(iter(env.simulator.pucks.values()))
    return np.array([-puck.position[1], puck.position[0], -puck.linearVelocity[1], puck.linearVelocity[0]],
                    dtype=np.float32)


def policy_state(obs, last_action) -> np.ndarray:
    return np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1),
                           np.asarray(last_action, dtype=np.float32).reshape(-1)])


# ----------------------------------------------------------------------------- TD3-style extra inputs (opt-in)
# The best new-sim TD3 recipe (configs/ceiling/scratch_newsim.yaml, rma_mode history_privileged) feeds its actor
# a 50-step window and its critic the window plus the simulator state. The same inputs can be switched on for
# CRL (CRLConfig.history_len / critic_privileged / actor_privileged); with all of them off the state is exactly
# ``policy_state`` above. Layout of the state vector s:
#     [ obs (30) | a_{t-1} (2) | privileged (8, if used) | window (history_len x 8, if used) ]
# Both extras are built with the TD3 code itself: the window is scripts.rma.history.StepHistoryBuffer over the
# 'latest_frame' step features (latest paddle [x, y, valid] + latest puck [x, y, valid] + the action taken),
# padded with (x_0, a = 0) at the episode start; the privileged block is scripts.rma.env_wrapper
# .true_state_features (env.current_state: puck position with the 25 ms lag, noise and occlusion hold, puck
# velocity with the lag, paddle position and velocity exact).
WINDOW_FEATURE_DIM = 6 + ACTION_DIM
PRIVILEGED_DIM = 8
_LATEST_PADDLE, _LATEST_PUCK = slice(12, 15), slice(27, 30)  # = scripts.rma.networks step features


def state_dim(history_len: int = 0, privileged: bool = False) -> int:
    return STATE_DIM + (PRIVILEGED_DIM if privileged else 0) + int(history_len) * WINDOW_FEATURE_DIM


def window_step_features(obs) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.concatenate([obs[_LATEST_PADDLE], obs[_LATEST_PUCK]])


class StateBuilder:
    """Per-env policy state with the optional TD3 inputs. One instance per env, used in step order:

        builder.reset(env, obs)              # after env.reset
        s = builder.state(env, obs, last)    # before choosing a_t (reads the privileged state now)
        builder.push(obs, a_t)               # once a_t is chosen, before env.step
    """

    def __init__(self, history_len: int = 0, privileged: bool = False):
        self.history_len = int(history_len)
        self.privileged = bool(privileged)
        self.dim = state_dim(self.history_len, self.privileged)
        self._hist = None
        if self.history_len > 0:
            from scripts.rma.history import StepHistoryBuffer

            self._hist = StepHistoryBuffer(self.history_len, WINDOW_FEATURE_DIM, ACTION_DIM)

    @classmethod
    def for_agent_config(cls, cfg) -> "StateBuilder":
        return cls(int(getattr(cfg, "history_len", 0)),
                   bool(getattr(cfg, "critic_privileged", False) or getattr(cfg, "actor_privileged", False)))

    def reset(self, env, obs) -> None:
        if self._hist is not None:
            self._hist.reset(window_step_features(obs))

    def state(self, env, obs, last_action) -> np.ndarray:
        parts = [policy_state(obs, last_action)]
        if self.privileged:
            from scripts.rma.env_wrapper import true_state_features

            parts.append(np.asarray(true_state_features(env), dtype=np.float32))
        if self._hist is not None:
            parts.append(self._hist.window().reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def push(self, obs, action) -> None:
        if self._hist is not None:
            self._hist.push(window_step_features(obs), np.asarray(action, dtype=np.float32).reshape(-1))
