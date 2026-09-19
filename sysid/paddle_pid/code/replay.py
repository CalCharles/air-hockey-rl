"""Replay recorded paddle trials through the Box2D paddle plant and score them.

The sim is driven exactly the way a policy drives it: reset to the real paddle state of row
0, then ``env.step(actions[k])`` for ``k = 0 .. N-2`` and compare the sim paddle position after
step ``k`` with the recorded ``pose[k+1]``. The puck is parked at the far end with gravity off,
noise / occlusion / observation delay / terminations are disabled, and the workspace, edge
and move limits plus ``hist_len`` of the sim are set from the recording so the PID *target*
sequence matches the robot's (verified to < 1 mm on the 2026-09-09 session).

Frames: the recording is in the robot frame; the sim's base frame is the robot frame shifted
by ``center_offset_constant`` in x (``_clip_limits`` in ``airhockey/sims/airhockey_box2d.py``).
Everything reported here is converted back to the robot frame.

Metric: **per-step position error** — for one trial the mean over ``k = 1 .. N-1`` of
``||sim_pose[k] − pose[k]||`` in mm; for a set of trials the mean of the per-trial means
(all trials of a session have the same length, so this equals the global per-step mean).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import yaml

from .dataset import PaddleTrial

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = _REPO_ROOT / "configs/new_juggle/sysid_best_params_hist2.yaml"

_NOISE_KEYS = ("puck_noise", "enable_random_occlusions", "enable_observation_delay",
               "enable_action_delay", "enable_puck_delay_interpolation",
               "enable_action_force_attenuation")
_PUCK_PARK_BASE = (-0.9, 0.0)     # far (opponent) end of the table, base frame


@dataclass
class PlantParams:
    """The PID gains being identified; ``paddle_density`` (mass) is fixed during the fit."""
    kp: float
    ki: float
    kd: float
    paddle_density: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


def load_base_config(path=DEFAULT_BASE_CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_replay_sim_config(base_config: dict, session_attrs: dict, hist_len: Optional[int] = None,
                            time_frequency: Optional[float] = None) -> dict:
    """The ``air_hockey`` section of a sim config prepared for a pure paddle replay.

    ``session_attrs``: ``hist_len`` / ``move_lims`` / ``workspace_lims`` / ``edge_lims`` of the
    recording (``dataset.session_attrs``). ``hist_len`` overrides the recording's value."""
    cfg = copy.deepcopy(base_config["air_hockey"] if "air_hockey" in base_config else base_config)
    sp = cfg.setdefault("simulator_params", {})
    for k in _NOISE_KEYS:
        sp[k] = False
    for k in list(cfg):
        if k.startswith("terminate_on"):
            cfg[k] = False
    sp["gravity"] = 0.0                       # the puck must not move into the paddle
    hl = hist_len if hist_len is not None else session_attrs.get("hist_len")
    if hl is not None:
        sp["hist_len"] = int(hl)
    if time_frequency is not None:
        sp["time_frequency"] = float(time_frequency)
    ml = session_attrs.get("move_lims")
    if ml is not None:
        sp["rmax_x"], sp["rmax_y"] = float(ml[0]), float(ml[1])
    wl = session_attrs.get("workspace_lims")
    if wl is not None:
        sp["x_min_lim"], sp["x_max_lim"], sp["y_min"], sp["y_max"] = [float(v) for v in wl]
    el = session_attrs.get("edge_lims")
    if el is not None:
        sp["top_abs"], sp["bot_abs"], sp["max_bias_p"], sp["max_bias_m"] = [float(v) for v in el]
    cfg["max_timesteps"] = max(int(cfg.get("max_timesteps", 250)), 1000)
    return cfg


@dataclass
class ReplayResult:
    sim_pose: np.ndarray      # (N, 2) robot frame; row 0 == real pose[0]
    target: np.ndarray        # (N-1, 2) PID target used at step k (robot frame, after smoothing)
    pos_err_mm: np.ndarray    # (N,) ||sim − real|| per step (row 0 is 0 by construction)


class PaddleReplayer:
    """One Box2D env reused for every trial; gains / density can be changed between replays."""

    def __init__(self, sim_cfg: dict):
        from airhockey import AirHockeyEnv           # local import: heavy
        self.sim_cfg = copy.deepcopy(sim_cfg)
        self.env = AirHockeyEnv(copy.deepcopy(sim_cfg))
        self.sim = self.env.simulator
        self.offset = np.array([float(self.sim.center_offset_constant), 0.0])
        self.dt = float(self.sim.time_per_step)

    # -- parameters ---------------------------------------------------------------------
    def set_params(self, params: PlantParams) -> None:
        pid = self.sim.pid_controller
        pid.Kp, pid.Ki, pid.Kd = float(params.kp), float(params.ki), float(params.kd)
        if params.paddle_density is not None:
            # spawn_paddle() reads self.paddle_density at every reset, so this takes effect
            # on the next replay without rebuilding the env.
            self.sim.paddle_density = float(params.paddle_density)
            self.sim._paddle_density_base = float(params.paddle_density)

    def current_params(self) -> PlantParams:
        pid = self.sim.pid_controller
        return PlantParams(float(pid.Kp), float(pid.Ki), float(pid.Kd), float(self.sim.paddle_density))

    @property
    def paddle_mass(self) -> float:
        return float(self.sim.paddle_density * np.pi * self.sim.paddle_radius ** 2)

    # -- frames -------------------------------------------------------------------------
    def _paddle_pose_robot(self) -> np.ndarray:
        body = self.sim.paddles["paddle_ego"]
        return self.sim._box2d_to_base_coords(body.position) - self.offset

    def target_from(self, pose_robot, action) -> np.ndarray:
        """The (unsmoothed) PID target the sim derives from a robot-frame pose + action."""
        tgt = self.sim._compute_pid_target_pos(
            self.sim._base_to_box2d_coords(np.asarray(pose_robot, float) + self.offset),
            self.sim.convert_to_box2d_coords(np.asarray(action, float)))
        return self.sim._box2d_to_base_coords(tgt) - self.offset

    # -- replay -------------------------------------------------------------------------
    def reset_to(self, pose_robot, speed_robot) -> None:
        state = np.concatenate([np.asarray(pose_robot, float) + self.offset, np.asarray(speed_robot, float),
                                np.asarray(_PUCK_PARK_BASE, float), np.zeros(2)])
        self.env.reset_from_state(state, seed=0)

    def _park_puck(self) -> None:
        park = self.sim.base_coord_to_box2d(_PUCK_PARK_BASE)
        for puck in self.sim.pucks.values():
            puck.position = park
            puck.linearVelocity = (0.0, 0.0)
            puck.angularVelocity = 0.0

    def replay(self, trial: PaddleTrial, action_delay_steps: int = 0) -> ReplayResult:
        """Drive the sim with the trial's actions from its initial state.

        ``action_delay_steps = d > 0`` applies ``actions[k-d]`` at step ``k`` (zeros before) —
        a diagnostic for the robot's command latency, off by default."""
        n = trial.n_steps
        acts = trial.actions
        if action_delay_steps > 0:
            acts = np.vstack([np.zeros((action_delay_steps, 2)), acts[:n - action_delay_steps]])
        self.reset_to(trial.pose[0], trial.speed[0])
        sim_pose = np.zeros((n, 2))
        target = np.zeros((max(n - 1, 0), 2))
        sim_pose[0] = self._paddle_pose_robot()
        for k in range(n - 1):
            self.env.step(acts[k])
            self._park_puck()
            sim_pose[k + 1] = self._paddle_pose_robot()
            if self.sim.last_target_position is not None:
                target[k] = np.asarray(self.sim.last_target_position, float) - self.offset
        err = np.linalg.norm(sim_pose - trial.pose, axis=1) * 1000.0
        return ReplayResult(sim_pose=sim_pose, target=target, pos_err_mm=err)


# -- metrics ----------------------------------------------------------------------------
def trial_metrics(trial: PaddleTrial, result: ReplayResult) -> dict:
    """Per-step position error statistics of one replayed trial (mm), rows ``k >= 1``."""
    e = result.pos_err_mm[1:]
    d_real = np.diff(trial.pose, axis=0)
    d_sim = np.diff(result.sim_pose, axis=0)
    delta_err = np.linalg.norm(d_sim - d_real, axis=1) * 1000.0
    active = ~trial.settle[1:]
    return {
        "mean_pos_err_mm": float(e.mean()) if e.size else float("nan"),
        "rms_pos_err_mm": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
        "max_pos_err_mm": float(e.max()) if e.size else float("nan"),
        "final_pos_err_mm": float(e[-1]) if e.size else float("nan"),
        "active_mean_pos_err_mm": float(e[active].mean()) if active.any() else float("nan"),
        "mean_delta_err_mm": float(delta_err.mean()) if delta_err.size else float("nan"),
        "n_steps": int(e.size),
    }


def evaluate_trials(replayer: PaddleReplayer, trials: Iterable[PaddleTrial], params: PlantParams,
                    action_delay_steps: int = 0, keep_trajectories: bool = False) -> dict:
    """Replay every trial with ``params``; aggregate = mean of per-trial mean position errors."""
    replayer.set_params(params)
    per_trial = []
    traj = {}
    for t in trials:
        r = replayer.replay(t, action_delay_steps=action_delay_steps)
        m = trial_metrics(t, r)
        m.update(name=t.name, condition=t.condition, repeat=t.repeat, trial_type=t.trial_type)
        per_trial.append(m)
        if keep_trajectories:
            traj[t.name] = r
    per_cond: dict[str, dict] = {}
    for m in per_trial:
        per_cond.setdefault(m["condition"], []).append(m["mean_pos_err_mm"])
    per_cond = {c: {"mean_pos_err_mm": float(np.mean(v)), "n": len(v)} for c, v in per_cond.items()}
    agg = {k: float(np.mean([m[k] for m in per_trial])) for k in
           ("mean_pos_err_mm", "rms_pos_err_mm", "max_pos_err_mm", "final_pos_err_mm",
            "active_mean_pos_err_mm", "mean_delta_err_mm")} if per_trial else {}
    out = {"params": replayer.current_params().as_dict(), "action_delay_steps": int(action_delay_steps),
           "n_trials": len(per_trial), **agg, "per_condition": per_cond, "per_trial": per_trial}
    if keep_trajectories:
        out["trajectories"] = traj
    return out
