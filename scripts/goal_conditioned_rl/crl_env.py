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
