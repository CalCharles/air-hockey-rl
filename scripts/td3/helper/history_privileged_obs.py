"""Online fine-tuning support for the RMA ``history_privileged`` policies (real robot + sim dry run).

The policies in ``data/newsim_scratch_cap7000_swa/`` were trained by
``scripts/rma/train_base_policy.py`` with ``rma_mode: history_privileged``:

* **actor** ``PrivilegedHistoryActor`` = pi(x_t, a_{t-1}, phi(window_t)): the 30-d
  history observation, the last action, and a 50-step window of
  (latest paddle [x, y, flag], latest puck [x, y, flag], action) pairs before step t;
* **critics** Q(o_t, a_t) with o_t = ``[x_t (30) | window_t (400) | s_t (8)]`` (438 + 2 = 440 inputs),
  where s_t = ``[puck x, y, vx, vy, paddle x, y, vx, vy]`` is the noise-free Box2D state
  (``scripts.rma.env_wrapper.true_state_features``);
* **reward** = task reward + the training-only strike shaping (``scripts/rma/motion_shaping.py``:
  edge penalty, weak-strike penalty, puck-out bonus, ... and the edge termination).

On the robot there is no true state. ``HistoryPrivilegedObs`` builds o_t from what
is observed, step by step, exactly as the sim wrapper (``PrivilegedHistoryEnvVector``)
builds it from the true state:

* positions: the latest frame of the observation (paddle ``obs[12:14]``, puck ``obs[27:29]``);
  an occluded puck (flag ``obs[29]`` = 1) keeps its last seen position;
* velocities: (current − previous position) / dt (0 for the puck if either frame is occluded,
  0 at the first step of an episode);
* the window is pushed with the action actually sent to the env (after exploration
  noise / clamping / transition holds), as in sim training;
* shaping: ``motion_shaping.shaping_step`` (the same code as the sim trainer) with the
  paddle x / puck speed from those positions and the contact flag from
  ``info["paddle_puck_collision_count"]`` (the real env infers it from geometry);
  a contact within ``shape_edge_term_margin`` of the top edge ends the episode
  (``terminated``, end reason ``edge_termination``).

The observation frame is the same in sim and real: robot-frame paddle x = obs x −
``center_offset_constant`` (checked against ``motion_shaping._paddle_x`` in the sim).

``HistoryPrivilegedPolicyEnv`` wraps the env handed to the policy runner only (the
reset runner keeps the raw env): ``step`` returns o_{t+1} and the shaped reward,
``begin_episode(raw_obs)`` starts a new window. Everything else is delegated.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch

from scripts.rma.history import StepHistoryBuffer
from scripts.rma.motion_shaping import MotionShaping, shaping_step
from scripts.rma.networks import PrivilegedHistoryActor, step_feature_dim, step_state_features

TRUE_STATE_DIM = 8  # scripts.rma.env_wrapper.TRUE_STATE_DIM
_PADDLE_POS = slice(12, 14)
_PUCK_POS = slice(27, 29)
_PUCK_FLAG = 29
RMA_META_FILENAME = "rma_meta.json"
EDGE_TERMINATION_REASON = "edge_termination"


@dataclass
class HistoryPrivilegedSpec:
    """Architecture + training settings of one ``history_privileged`` checkpoint."""

    raw_obs_dim: int
    act_dim: int
    history_len: int
    step_features: str
    latent_dim: int
    embed_dim: int
    conv_channels: int
    hidden_layer_size: int
    num_hidden_layers: int
    use_last_action: bool
    gamma: float
    shaping: MotionShaping
    source_dir: Path

    @property
    def feature_dim(self) -> int:
        return step_feature_dim(self.raw_obs_dim, self.act_dim, self.step_features)

    @property
    def window_dim(self) -> int:
        return self.history_len * self.feature_dim

    @property
    def obs_dim(self) -> int:
        """Replay / critic observation: [x_t | window | s_t]."""
        return self.raw_obs_dim + self.window_dim + TRUE_STATE_DIM

    def build_actor(self, device: torch.device | str) -> PrivilegedHistoryActor:
        actor = PrivilegedHistoryActor(
            privileged_dim=TRUE_STATE_DIM,
            obs_dim=self.raw_obs_dim,
            act_dim=self.act_dim,
            history_len=self.history_len,
            feature_dim=self.feature_dim,
            latent_dim=self.latent_dim,
            use_last_action=self.use_last_action,
            hidden_layer_size=self.hidden_layer_size,
            num_hidden_layers=self.num_hidden_layers,
            embed_dim=self.embed_dim,
            conv_channels=self.conv_channels,
            action_scale=1.0,
            action_bias=0.0,
        )
        actor.step_features = self.step_features
        return actor.to(device)

    def write_bundle_meta(self, checkpoint_dir: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
        """Copy ``rma_meta.json`` next to an online checkpoint's ``model.pth`` so the
        real eval (``async_td3_real_eval.py --agent history --model-path <dir>``) can load it.
        ``extra`` entries are merged into the copy (residual fine-tuning adds
        ``{"residual": {"scale": ...}}``, see helper/residual_finetune.py)."""
        src = self.source_dir / RMA_META_FILENAME
        if not src.exists():
            return
        dst = Path(checkpoint_dir) / RMA_META_FILENAME
        if not extra:
            shutil.copyfile(src, dst)
            return
        meta = json.loads(src.read_text())
        meta.update(extra)
        dst.write_text(json.dumps(meta, indent=2))


def history_privileged_spec(
    checkpoint: Dict[str, Any], checkpoint_path: str, raw_obs_dim: int, act_dim: int
) -> Optional[HistoryPrivilegedSpec]:
    """The spec of an RMA ``history_privileged`` checkpoint; None for a plain TD3 checkpoint."""
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
    mode = ckpt_args.get("rma_mode")
    if mode is None:
        return None
    if mode != "history_privileged":
        raise ValueError(
            f"Online fine-tuning supports plain TD3 and rma_mode=history_privileged checkpoints, got rma_mode={mode!r}."
        )
    if bool(ckpt_args.get("goal", False)):
        raise ValueError("Goal-conditioned history policies are not supported.")
    spec = HistoryPrivilegedSpec(
        raw_obs_dim=int(raw_obs_dim),
        act_dim=int(act_dim),
        history_len=int(ckpt_args["history_len"]),
        step_features=str(ckpt_args.get("step_features", "latest_frame")),
        latent_dim=int(ckpt_args["rma_latent_dim"]),
        embed_dim=int(ckpt_args["history_embed_dim"]),
        conv_channels=int(ckpt_args["history_conv_channels"]),
        hidden_layer_size=int(ckpt_args["agent_hidden_layer_size"]),
        num_hidden_layers=int(ckpt_args["agent_num_hidden_layers"]),
        use_last_action=bool(ckpt_args["use_last_action_in_policy_state"]),
        gamma=float(ckpt_args["gamma"]),
        shaping=MotionShaping.from_args(SimpleNamespace(**ckpt_args)),
        source_dir=Path(checkpoint_path).resolve().parent,
    )
    meta_path = spec.source_dir / RMA_META_FILENAME
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for key, value in (("history_len", spec.history_len), ("latent_dim", spec.latent_dim),
                           ("embed_dim", spec.embed_dim), ("conv_channels", spec.conv_channels),
                           ("obs_dim", spec.raw_obs_dim), ("feature_dim", spec.feature_dim)):
            if key in meta and int(meta[key]) != int(value):
                raise ValueError(f"{meta_path}: {key}={meta[key]} but the checkpoint args give {value}.")
    return spec


class HistoryPrivilegedObs:
    """Builds the critic observation [x_t | window_t | s_t] and the shaped reward, step by step."""

    def __init__(self, spec: HistoryPrivilegedSpec, *, dt: float, center_offset_constant: float, x_min_lim: float):
        self.spec = spec
        self.dt = float(dt)
        self.center_offset_constant = float(center_offset_constant)
        self.x_min_lim = float(x_min_lim)
        self.history = StepHistoryBuffer(spec.history_len, spec.feature_dim, spec.act_dim)
        self._prev_raw: np.ndarray | None = None
        self._paddle_prev: np.ndarray | None = None
        self._puck_prev: np.ndarray | None = None  # last seen puck position
        self._puck_prev_visible = False
        self._xs: list[float] = []
        self.last_state = np.zeros(TRUE_STATE_DIM)
        self.last_bonus = 0.0
        self._reset_episode_stats()

    def _reset_episode_stats(self) -> None:
        self.episode_task_return = 0.0
        self.episode_shaping_return = 0.0
        self.episode_edge_terminated = False
        self.episode_steps = 0

    def episode_stats(self) -> Dict[str, float]:
        """Per-episode sums since the last ``start`` (the orchestrator's stats keys)."""
        return {
            "last_episode_task_return": float(self.episode_task_return),
            "last_episode_shaping_return": float(self.episode_shaping_return),
            "last_episode_edge_termination": float(self.episode_edge_terminated),
        }

    @classmethod
    def for_env(cls, spec: HistoryPrivilegedSpec, env) -> "HistoryPrivilegedObs":
        sim = env.simulator
        return cls(spec, dt=float(sim.time_per_step), center_offset_constant=float(sim.center_offset_constant),
                   x_min_lim=float(sim.x_min_lim))

    def _features(self, raw: np.ndarray) -> np.ndarray:
        return step_state_features(torch.as_tensor(raw, dtype=torch.float32), self.spec.step_features).numpy()

    def _robot_x(self, paddle_xy: np.ndarray) -> float:
        return float(paddle_xy[0]) - self.center_offset_constant

    def _state(self, raw: np.ndarray, first: bool) -> np.ndarray:
        paddle = np.asarray(raw[_PADDLE_POS], dtype=np.float64)
        puck_visible = float(raw[_PUCK_FLAG]) < 0.5
        puck_obs = np.asarray(raw[_PUCK_POS], dtype=np.float64)
        if first or self._paddle_prev is None:
            paddle_vel = np.zeros(2)
        else:
            paddle_vel = (paddle - self._paddle_prev) / self.dt
        if puck_visible:
            puck = puck_obs
            if not first and self._puck_prev_visible and self._puck_prev is not None:
                puck_vel = (puck - self._puck_prev) / self.dt
            else:
                puck_vel = np.zeros(2)
        else:
            puck = self._puck_prev if self._puck_prev is not None else puck_obs
            puck_vel = np.zeros(2)
        self._paddle_prev = paddle
        if puck_visible or self._puck_prev is None:
            self._puck_prev = puck
        self._puck_prev_visible = puck_visible
        return np.concatenate([puck, puck_vel, paddle, paddle_vel])

    def _augment(self, raw: np.ndarray, state: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.asarray(raw, dtype=np.float64).reshape(-1),
            self.history.window().reshape(-1).astype(np.float64),
            state,
        ]).astype(np.float32)

    def start(self, raw_obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        if raw.shape[0] != self.spec.raw_obs_dim:
            raise ValueError(f"expected a {self.spec.raw_obs_dim}-d raw observation, got {raw.shape[0]}")
        self.history.reset(self._features(raw))
        self._prev_raw = raw.copy()
        self._paddle_prev = None
        self._puck_prev = None
        self._puck_prev_visible = False
        state = self._state(raw, first=True)
        self._xs = [self._robot_x(state[4:6])]
        self.last_state = state
        self.last_bonus = 0.0
        self._reset_episode_stats()
        return self._augment(raw, state)

    def step(self, raw_next_obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, info: dict):
        """Advance by one env step. Returns (o_{t+1}, shaped reward, terminated, info)."""
        if self._prev_raw is None:
            raise RuntimeError("HistoryPrivilegedObs.step before start()")
        raw = np.asarray(raw_next_obs, dtype=np.float32).reshape(-1)
        self.history.push(self._features(self._prev_raw), np.asarray(action, dtype=np.float32).reshape(-1))
        self._prev_raw = raw.copy()
        state = self._state(raw, first=False)
        x = self._robot_x(state[4:6])
        prev_x = self._xs[-1] if self._xs else x
        self._xs = (self._xs + [x])[-(self.spec.shaping.cock_window + 1):]
        bonus, edge_terminate = 0.0, False
        if self.spec.shaping.enabled:
            bonus, edge_terminate = shaping_step(
                self.spec.shaping,
                x=x,
                prev_x=prev_x,
                xs=self._xs,
                x_min_lim=self.x_min_lim,
                dt=self.dt,
                contact=int((info or {}).get("paddle_puck_collision_count", 0) or 0) > 0,
                puck_speed_fn=lambda: float(np.linalg.norm(state[2:4])),
            )
        info = dict(info) if isinstance(info, dict) else {}
        info["motion_shaping"] = float(bonus)
        info["task_reward"] = float(reward)
        if edge_terminate and not terminated:
            terminated = True
            info["episode_end_reason"] = EDGE_TERMINATION_REASON
            info["episode_end_reasons"] = [EDGE_TERMINATION_REASON]
            info["episode_end_type"] = "terminated"
            info["edge_termination"] = True
            self.episode_edge_terminated = True
        self.episode_task_return += float(reward)
        self.episode_shaping_return += float(bonus)
        self.episode_steps += 1
        self.last_state = state
        self.last_bonus = float(bonus)
        return self._augment(raw, state), float(reward) + float(bonus), bool(terminated), info


class HistoryPrivilegedPolicyEnv:
    """The env as the policy runner sees it: [x_t | window | s_t] observations and the shaped reward."""

    def __init__(self, env, builder: HistoryPrivilegedObs):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_builder", builder)

    @property
    def unwrapped_env(self):
        return self._env

    def begin_episode(self, raw_obs: np.ndarray) -> np.ndarray:
        return self._builder.start(raw_obs)

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        aug, reward, terminated, info = self._builder.step(next_obs, action, reward, terminated, info)
        return aug, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        setattr(self._env, name, value)
