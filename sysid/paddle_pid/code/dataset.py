"""Load scripted paddle-motion recordings and split them into train / validation.

A recording session (``configs/robot_data_collection/paddle_motion_config.yaml`` on the robot
machine) is a directory of trial HDF5 files plus ``manifest.json``. Two layouts exist:

* **paddle_motion** (2026-09-09): ``traj_<idx>_<condition>_trial<k>.hdf5``; every trial is one
  *condition* (a straight line ``<axis><sign>_delta<d>`` or an arc ``arc_<aspect>_v<speed>``)
  repeated ``k = 1..3`` times: ``settle_steps`` zero actions, then ``action_steps`` scripted actions.
* **reversal_jerk** (2026-09-10): ``jerk_<idx>_<cond>_delta<d>_out<n>_slow<m>.hdf5``; the paddle
  is driven ``out`` steps in one direction, reversed for ``back`` steps and slowed down over the
  last ``slow`` steps (``slowdown_scales``). The condition is everything after ``jerk_<idx>_``;
  the repeat comes from the ``repeat`` attr or, when missing, the rank within the condition.

``load_session`` takes one directory or a list of them (trials of several sessions are pooled;
``PaddleTrial.session`` records the origin). Aborted / protective-stop trials are detected from
the HDF5 attrs (the manifest only lists the last restart of a session).

Per-step layout of a trial file (all arrays share the step axis, row ``i`` = state observed at
the start of step ``i`` and the action executed during step ``i``):

* ``train_vals`` (N, 35) — the canonical 35-column real-world row; ``pose_x/y`` (cols 5:7) is
  the paddle position in the **robot frame** (metres), ``speed_x/y`` (11:13) its velocity,
  ``desired_pose_x/y`` (26:28) the commanded target. ``desired[i] = clip(pose[i] +
  actions[i] * move_lims)`` holds to < 1 mm.
* ``actions`` (N, 2) — the normalised actions in [-1, 1].
* ``is_settle_step`` (N,) — 1 for the settle steps at the start.
* attrs ``hist_len``, ``move_lims``, ``workspace_lims``, ``edge_lims`` — the real controller
  settings the recording was made with; the replay mirrors them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np

# Canonical column indices of the 35-column train_vals row (see
# airhockey/sims/real/proprioceptive_state.py); used when a file has no column names.
_CANONICAL_COLUMNS = {"cur_time": 0, "pose_x": 5, "pose_y": 6, "speed_x": 11, "speed_y": 12,
                      "desired_pose_x": 26, "desired_pose_y": 27, "puck_x": 32, "puck_y": 33}

_TRIAL_NAME_RE = re.compile(r"^traj_(\d+)_(.+)_trial(\d+)$")
_JERK_NAME_RE = re.compile(r"^jerk_(\d+)_(.+)$")


@dataclass
class PaddleTrial:
    name: str
    path: str
    index: int
    condition: str
    repeat: int
    trial_type: str                      # "line" | "curve" | "reversal_jerk" | "unknown"
    time: np.ndarray                     # (N,) unix seconds
    pose: np.ndarray                     # (N, 2) robot frame, m
    speed: np.ndarray                    # (N, 2) robot frame, m/s
    desired: np.ndarray                  # (N, 2) robot frame, m
    actions: np.ndarray                  # (N, 2) normalised
    settle: np.ndarray                   # (N,) bool
    attrs: dict = field(default_factory=dict)
    session: str = ""                    # name of the session directory the trial came from

    @property
    def n_steps(self) -> int:
        return int(self.pose.shape[0])

    @property
    def dt(self) -> float:
        d = np.diff(self.time)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.mean(d)) if d.size else float("nan")


def parse_trial_name(name: str) -> tuple[int, str, int]:
    """``traj_<idx>_<condition>_trial<k>`` → (idx, condition, k); ``jerk_<idx>_<condition>`` → (idx, condition, 0)
    (the repeat of a jerk trial comes from its attrs or its rank within the condition, see ``load_session``)."""
    m = _TRIAL_NAME_RE.match(name)
    if m:
        return int(m.group(1)), m.group(2), int(m.group(3))
    m = _JERK_NAME_RE.match(name)
    if m:
        return int(m.group(1)), m.group(2), 0
    raise ValueError(f"trial name {name!r} matches neither traj_<idx>_<condition>_trial<k> nor jerk_<idx>_<condition>")


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


def load_trial(path) -> PaddleTrial:
    path = Path(path)
    with h5py.File(path, "r") as f:
        tv = np.asarray(f["train_vals"][()], dtype=np.float64)
        col = _column_indices(f)
        n = tv.shape[0]
        if "actions" in f:
            actions = np.asarray(f["actions"][()], dtype=np.float64)
        else:                                   # older recordings: invert the target
            move = np.asarray(_attr(f, "move_lims", [0.26, 0.12]), dtype=float)
            actions = (tv[:, [col["desired_pose_x"], col["desired_pose_y"]]]
                       - tv[:, [col["pose_x"], col["pose_y"]]]) / move
            actions = np.clip(actions, -1.0, 1.0)
        if actions.shape[0] != n:
            raise ValueError(f"{path.name}: actions has {actions.shape[0]} rows, train_vals {n}")
        if "is_settle_step" in f:
            settle = np.asarray(f["is_settle_step"][()]).astype(bool)
        else:
            settle = np.zeros(n, dtype=bool)
        attrs = {k: _attr(f, k) for k in ("hist_len", "move_lims", "workspace_lims", "edge_lims",
                                          "block_time", "trial_type", "direction_key", "action_delta",
                                          "control_type", "condition_key", "out_steps", "back_steps",
                                          "slowdown_steps", "saturates_at_step", "aborted_at_step",
                                          "protective_stop", "repeat")}
        name = str(_attr(f, "trial_name", path.stem))
    try:
        index, condition, repeat = parse_trial_name(name)
    except ValueError:
        index, condition, repeat = parse_trial_name(path.stem)
    if attrs.get("repeat") is not None:
        repeat = int(attrs["repeat"])
    trial_type = attrs.get("trial_type") or ("curve" if condition.startswith("arc") else "line")
    return PaddleTrial(
        name=name, path=str(path), index=index, condition=condition, repeat=repeat,
        trial_type=str(trial_type), time=tv[:, col["cur_time"]],
        pose=tv[:, [col["pose_x"], col["pose_y"]]].copy(),
        speed=tv[:, [col["speed_x"], col["speed_y"]]].copy(),
        desired=tv[:, [col["desired_pose_x"], col["desired_pose_y"]]].copy(),
        actions=actions, settle=settle, attrs=attrs, session=path.parent.name,
    )


def trial_aborted(trial: PaddleTrial) -> bool:
    """Protective stop / abort recorded in the file's own attrs (the manifest only covers the last restart)."""
    ps = trial.attrs.get("protective_stop")
    ab = trial.attrs.get("aborted_at_step")
    return bool(ps) or (ab is not None and int(ab) not in (-1,))


def _load_one_session(input_dir: Path, pattern: str, skip_aborted: bool) -> list[PaddleTrial]:
    files = [p for p in sorted(input_dir.glob(pattern)) if p.name.startswith(("traj_", "jerk_"))]
    if not files:
        raise FileNotFoundError(f"no traj_*/jerk_* trial files matching {pattern} under {input_dir}")
    bad = set()
    manifest = input_dir / "manifest.json"
    if skip_aborted and manifest.exists():
        with open(manifest) as fh:
            for t in json.load(fh).get("trials", []):
                if t.get("protective_stop") or t.get("aborted_at_step") not in (None, -1):
                    bad.add(t["trial_name"])
    trials = [load_trial(p) for p in files]
    if skip_aborted:
        trials = [t for t in trials if t.name not in bad and not trial_aborted(t)]
    trials.sort(key=lambda t: t.index)
    # Repeats of jerk trials that carry no ``repeat`` attr: rank within the condition (by index).
    seen: dict[str, int] = {}
    for t in trials:
        if t.repeat <= 0:
            seen[t.condition] = seen.get(t.condition, 0) + 1
            t.repeat = seen[t.condition]
    return trials


def load_session(input_dir, pattern: str = "*.hdf5", skip_aborted: bool = True) -> list[PaddleTrial]:
    """All trials of one session directory — or of several (a list / tuple of directories), pooled.

    Within a session trials are ordered by index; sessions keep the given order. Trials whose
    attrs (or the manifest) mark them protective-stopped / aborted are dropped when
    ``skip_aborted`` (they are not a clean response to the scripted actions). Only ``traj_*`` and
    ``jerk_*`` files are trial files."""
    dirs = [Path(d) for d in (input_dir if isinstance(input_dir, (list, tuple)) else [input_dir])]
    trials: list[PaddleTrial] = []
    for d in dirs:
        trials.extend(_load_one_session(d, pattern, skip_aborted))
    names = [t.name for t in trials]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate trial names across sessions: {dup[:5]}")
    return trials


def sessions_of(trials: Iterable[PaddleTrial]) -> list[str]:
    out: list[str] = []
    for t in trials:
        if t.session not in out:
            out.append(t.session)
    return out


def group_by_condition(trials: Iterable[PaddleTrial]) -> dict[str, list[PaddleTrial]]:
    groups: dict[str, list[PaddleTrial]] = {}
    for t in trials:
        groups.setdefault(t.condition, []).append(t)
    for g in groups.values():
        g.sort(key=lambda t: t.repeat)
    return groups


def split_train_val(trials: list[PaddleTrial], seed: int = 0,
                    val_repeat: Optional[int] = None) -> tuple[list[PaddleTrial], list[PaddleTrial], dict]:
    """Hold out exactly one trial of every condition for validation.

    ``val_repeat`` fixes which repeat is held out everywhere (e.g. 3); ``None`` draws the
    held-out repeat per condition with ``seed``. Conditions with a single trial go to
    training only. Returns (train, val, split_info)."""
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


def session_attrs(trials: list[PaddleTrial]) -> dict:
    """Controller settings shared by the session (asserts they agree across trials)."""
    keys = ("hist_len", "move_lims", "workspace_lims", "edge_lims")
    ref = {k: trials[0].attrs.get(k) for k in keys}
    for t in trials[1:]:
        for k in keys:
            if json.dumps(t.attrs.get(k)) != json.dumps(ref[k]):
                raise ValueError(f"{t.name}: attr {k}={t.attrs.get(k)} differs from {trials[0].name}: {ref[k]}")
    return ref
