"""Load a scripted paddle–puck collision session (``puck_collision`` recordings).

A session (``robot_data_collection_puck_collision_<stamp>/``) is a directory of
``collision_<idx>_h<height>_delta<d>_y<offset>.hdf5`` files. Every trial is one *condition* —
the height the puck is released from (``top`` / ``3/4`` / ``1/2`` of the table, which sets the
puck's approach speed) × the paddle action magnitude ``delta`` (0.00 / 0.33 / 0.66 / 1.00,
which sets the paddle speed) — and the protocol aims for three clean repeats per condition.
Extra files under a condition are re-takes after a bad collision (glancing hit, puck carried
by the paddle, tracker loss); the operator did not delete the bad ones, so the *best three per
condition* have to be selected from the data (``speeds.select_canonical``).

Per trial the file holds

* ``arm_puck_track`` (K, 4) — ``[frame_time_s, puck_x_obs, puck_y_obs, puck_occluded]`` camera
  frames while the robot waited for the puck to slide in (the incoming free flight);
* ``train_vals`` (N, 35) — the canonical real-world row for the ``action_steps`` scripted
  action steps + ``post_steps`` zero-action steps at ~20 Hz: ``cur_time`` (loop time),
  ``pose_x/y`` + ``speed_x/y`` (robot frame, metres / m/s), ``puck_x/y`` + ``puck_occluded``
  (the latest camera frame at that loop time);
* attrs — the condition (``puck_height_label``, ``action_delta``, ``y_offset``), the trigger
  (``trigger_x_obs``, ``trigger_approach_speed_obs_m_s``), the controller settings
  (``hist_len``, ``move_lims``, ``workspace_lims``, ``edge_lims``, ``center_offset_constant``)
  and the outcome flags (``protective_stop``, ``aborted_at_step``).

Frames. Puck positions are in the observation frame (robot end at ``x ≈ +0.9``, the puck
*accelerates* towards ``+x`` down the tilted table). The paddle is logged in the robot frame;
``pose_x + center_offset_constant`` (1.2) puts it in the same frame as the puck (verified by the
stationary-paddle trials: contact happens at ``r_paddle + r_puck`` from the shifted pose). The
camera lags the robot clock by ~0.1 s (``speeds.calibrate_camera_lag``).

``manifest.json`` only lists the trials of the *last* (re)start of a session, so everything is
read from the HDF5 attrs instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np

_CANONICAL_COLUMNS = {"cur_time": 0, "pose_x": 5, "pose_y": 6, "speed_x": 11, "speed_y": 12,
                      "desired_pose_x": 26, "desired_pose_y": 27, "puck_x": 32, "puck_y": 33, "puck_occluded": 34}
_NAME_RE = re.compile(r"^collision_(\d+)_h([^_]+)_delta([0-9.]+)_y(-?[0-9.]+)$")
_HEIGHT_LABELS = {"top": "top", "3/4": "3-4", "1/2": "1-2"}


@dataclass
class CollisionTrial:
    name: str
    path: str
    index: int
    height: str                     # 'top' | '3/4' | '1/2' (release position → approach speed)
    action_delta: float             # paddle action magnitude → paddle speed
    y_offset: float                 # commanded paddle y (robot frame)
    condition: str                  # 'h<top|3-4|1-2>_delta<d.dd>'
    # puck track (camera frames: arm-wait frames followed by the step frames), observation frame
    t: np.ndarray                   # (N,) unix seconds
    puck_xy: np.ndarray             # (N, 2) m
    puck_valid: np.ndarray          # (N,) bool — not occluded and not a stale repeat
    n_arm: int                      # rows that came from arm_puck_track
    # paddle track (robot rows, shifted to the observation frame)
    pad_t: np.ndarray               # (M,)
    pad_xy: np.ndarray              # (M, 2)
    pad_v: np.ndarray               # (M, 2) m/s
    actions: np.ndarray             # (M, 2)
    attrs: dict = field(default_factory=dict)
    repeat: int = 0                 # rank within the condition (by trial index), 1-based, set by load_session

    @property
    def t0(self) -> float:
        """Loop time of the first scripted step (the trigger fires ~20 ms before)."""
        return float(self.pad_t[0])

    @property
    def n_valid(self) -> int:
        return int(self.puck_valid.sum())

    @property
    def paddle_offset_x(self) -> float:
        return float(self.attrs.get("center_offset_constant", 1.2))


def parse_trial_name(name: str) -> tuple[int, str, float, float]:
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"trial name {name!r} does not match collision_<idx>_h<height>_delta<d>_y<offset>")
    return int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))


def condition_name(height_label: str, action_delta: float) -> str:
    return f"h{_HEIGHT_LABELS.get(height_label, height_label.replace('/', '-'))}_delta{float(action_delta):.2f}"


def _column_indices(f: h5py.File) -> dict:
    names = f.attrs.get("vals_column_names")
    if names is None:
        return dict(_CANONICAL_COLUMNS)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return {n: i for i, n in enumerate(names)}


def _attr(f: h5py.File, key: str, default=None):
    if key not in f.attrs:
        return default
    v = f.attrs[key]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def mark_valid(puck_xy: np.ndarray, occluded: np.ndarray, stale_eps: float = 1e-9) -> np.ndarray:
    """Usable camera samples: not flagged occluded and not an exact repeat of the previous
    sample (the tracker holds the last position while it loses the puck, and the 20 Hz loop
    sometimes reads the same 30 Hz frame twice)."""
    valid = ~np.asarray(occluded, bool)
    same = np.r_[False, np.all(np.abs(np.diff(puck_xy, axis=0)) < stale_eps, axis=1)]
    return valid & ~same


def load_trial(path) -> CollisionTrial:
    path = Path(path)
    with h5py.File(path, "r") as f:
        tv = np.asarray(f["train_vals"][()], dtype=np.float64)
        col = _column_indices(f)
        arm = np.asarray(f["arm_puck_track"][()], dtype=np.float64) if "arm_puck_track" in f else np.zeros((0, 4))
        if arm.ndim != 2 or arm.shape[1] < 4:
            arm = np.zeros((0, 4))
        actions = np.asarray(f["actions"][()], dtype=np.float64) if "actions" in f else np.zeros((tv.shape[0], 2))
        attrs = {k: _attr(f, k) for k in (
            "hist_len", "move_lims", "workspace_lims", "edge_lims", "center_offset_constant", "block_time",
            "puck_height_label", "action_delta", "y_offset", "trigger_x_obs", "trigger_approach_speed_obs_m_s",
            "trigger_time", "trigger_to_action_s", "protective_stop", "aborted_at_step", "session_start_iso",
            "action_steps", "post_steps", "direction_key")}
        name = str(_attr(f, "trial_name", path.stem))
    index, h_from_name, d_from_name, y_from_name = parse_trial_name(name if _NAME_RE.match(name) else path.stem)
    height = str(attrs.get("puck_height_label") or {"3-4": "3/4", "1-2": "1/2"}.get(h_from_name, h_from_name))
    delta = float(attrs["action_delta"]) if attrs.get("action_delta") is not None else d_from_name
    y_off = float(attrs["y_offset"]) if attrs.get("y_offset") is not None else y_from_name
    offset = float(attrs.get("center_offset_constant") or 1.2)

    t = np.concatenate([arm[:, 0], tv[:, col["cur_time"]]])
    xy = np.vstack([arm[:, 1:3], tv[:, [col["puck_x"], col["puck_y"]]]])
    occ = np.concatenate([arm[:, 3], tv[:, col["puck_occluded"]]]) > 0.5
    order = np.argsort(t, kind="stable")
    t, xy, occ = t[order], xy[order], occ[order]
    valid = mark_valid(xy, occ)
    return CollisionTrial(
        name=name, path=str(path), index=index, height=height, action_delta=delta, y_offset=y_off,
        condition=condition_name(height, delta), t=t, puck_xy=xy, puck_valid=valid, n_arm=int(arm.shape[0]),
        pad_t=tv[:, col["cur_time"]].copy(),
        pad_xy=np.column_stack([tv[:, col["pose_x"]] + offset, tv[:, col["pose_y"]]]),
        pad_v=tv[:, [col["speed_x"], col["speed_y"]]].copy(), actions=actions, attrs=attrs)


def load_session(input_dir, pattern: str = "collision_*.hdf5", skip_aborted: bool = True) -> list[CollisionTrial]:
    """All trials of a session, ordered by index; protective-stopped / aborted ones dropped."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {input_dir}")
    trials = [load_trial(p) for p in files]
    if skip_aborted:
        trials = [t for t in trials
                  if not t.attrs.get("protective_stop") and t.attrs.get("aborted_at_step", -1) in (None, -1)]
    trials.sort(key=lambda t: t.index)
    for group in group_by_condition(trials).values():
        for k, t in enumerate(group, start=1):
            t.repeat = k
    return trials


def group_by_condition(trials: Iterable[CollisionTrial]) -> dict[str, list[CollisionTrial]]:
    groups: dict[str, list[CollisionTrial]] = {}
    for t in trials:
        groups.setdefault(t.condition, []).append(t)
    for g in groups.values():
        g.sort(key=lambda t: t.index)
    return groups


def split_train_val(trials: list[CollisionTrial], seed: int = 0,
                    val_repeat: Optional[int] = None) -> tuple[list[CollisionTrial], list[CollisionTrial], dict]:
    """Hold out one trial of every condition (``val_repeat`` = fixed rank, else random per condition)."""
    rng = np.random.RandomState(seed)
    train, val, info = [], [], {"seed": seed, "val_repeat": val_repeat, "conditions": {}}
    for cond, group in group_by_condition(trials).items():
        if len(group) < 2:
            train.extend(group)
            info["conditions"][cond] = {"train": [t.name for t in group], "val": []}
            continue
        if val_repeat is not None:
            cand = [t for t in group if t.repeat == val_repeat]
            held = cand[0] if cand else group[-1]
        else:
            held = group[int(rng.randint(len(group)))]
        val.append(held)
        rest = [t for t in group if t is not held]
        train.extend(rest)
        info["conditions"][cond] = {"train": [t.name for t in rest], "val": [held.name]}
    info["n_train"], info["n_val"] = len(train), len(val)
    return train, val, info


def session_attrs(trials: list[CollisionTrial]) -> dict:
    """Controller settings shared by the session (first trial's; the replay mirrors them)."""
    keys = ("hist_len", "move_lims", "workspace_lims", "edge_lims")
    return {k: trials[0].attrs.get(k) for k in keys}
