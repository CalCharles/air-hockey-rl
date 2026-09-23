"""Automatic segmentation of real air-hockey puck trajectories.

Chunks a split-schema HDF5 recording (``puck``, ``pose``, ``cur_time`` ...)
into contiguous, non-overlapping segments labelled

    free_fall         puck moving under gravity + damping only (usable for the
                      puck grid search / kinematic fits)
    wall_collision    puck bounced off a wall (``meta["wall"]`` = x+/x-/y+/y-)
    paddle_collision  puck hit / pushed by the paddle
    opponent_hit      velocity jump on the opponent half (x > 0) away from the
                      robot paddle — a human hitting / catching the puck
    unknown_impulse   velocity jump on the robot half that is neither near a
                      wall nor the paddle (tracking glitch, hand, ...)
    rest              puck essentially stationary
    occluded          long occlusion gap that could not be bridged

Method
------
The puck tracker runs asynchronously from the 20 Hz control loop, so raw
finite-difference velocities alternate between ~1x and ~2x the true value
(one or two camera frames elapsed) and occasionally repeat a stale sample.
Per-frame accelerations are therefore useless. Instead, for every candidate
split point ``s`` we fit the damped free-flight model

    v(t) = (v0 + g/γ) e^{-γ t} - g/γ

separately to the ``fit_window`` usable frames *before* ``s`` and the
``fit_window`` usable frames *from* ``s`` on (linear least squares in
(p0, v0) for fixed g, γ), then evaluate both fits at ``t_s``. In free flight
the two fits agree; a collision shows up as a jump in velocity (``dv``) and /
or position (``dp``). Peaks of ``dv`` above ``dv_threshold`` become events,
which are classified by geometry (near a wall with a reversed normal
component -> wall; within contact distance of the paddle -> paddle). Occlusion
gaps are handled naturally: the pre/post windows straddle the gap and the
event (if any) spans it.

Stale samples (identical consecutive puck readings while the puck is moving)
are excluded from all fits; genuine rests (>= ``stale_run_is_rest`` identical
readings) are kept.

Frame convention: everything is in the *sim frame* — x is the long table
axis, the robot paddle lives on the ``x < 0`` end and gravity pulls the puck
towards ``x < 0`` (``gravity_x`` < 0, as in ``sysid_best_params*.yaml``).
The raw recordings do not share one convention. The puck tracker logs the
puck with the robot at ``x > 0`` (measured free-flight acceleration ≈ +0.7),
so the puck is mirrored (``puck_x_sign = -1``). The paddle ``pose`` field is
worse: in older recordings (``sysid/wall_collision_teleop``, curated collision
clips) it is the physical paddle position in the raw puck frame, while in
``shared/mouse_state_data_all_new_*`` it is ``robot_x - 1.2`` and the physical
paddle sits at ≈ ``robot_x - 0.1`` in the raw puck frame (verified against the
camera images). ``estimate_axis_transforms`` therefore calibrates the paddle
mapping ``x_sim = paddle_x_sign * pose_x + paddle_x_offset`` from the data
itself: it picks the (sign, offset) under which puck velocity jumps on the
table coincide with puck–paddle contact and free flight does not pass through
the paddle. The puck sign comes from the free-flight acceleration sign. All
three numbers are recorded in every output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

LABELS = (
    "free_fall",
    "wall_collision",
    "paddle_collision",
    "opponent_hit",
    "unknown_impulse",
    "rest",
    "occluded",
)

EVENT_LABELS = ("wall_collision", "paddle_collision", "opponent_hit", "unknown_impulse")

# Datasets copied into per-segment HDF5 slices (image is opt-in, it is large).
SPLIT_DATASETS = (
    "cur_time", "tidx", "i", "estop", "safety", "pose", "speed", "force",
    "acc", "desired_pose", "puck", "action", "paddle",
)
# Minimum keys for ``load_trajectory`` (older ``train_vals`` dumps lack these).
_REQUIRED_RECORDING_KEYS = ("puck", "pose", "cur_time")


def is_split_schema_recording(path: str | Path) -> bool:
    """True if ``path`` is a split-schema episode HDF5 (has ``puck`` / ``pose`` / ``cur_time``)."""
    try:
        with h5py.File(path, "r") as f:
            return all(k in f for k in _REQUIRED_RECORDING_KEYS)
    except OSError:
        return False


def list_split_schema_recordings(root: str | Path) -> list[Path]:
    """Recursive ``*.hdf5`` under ``root``, skipping non-split-schema files (e.g. old ``train_vals``)."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*.hdf5") if is_split_schema_recording(p))
    return files


@dataclass
class SegmentationConfig:
    # Puck physics (sysid best fit) used by the free-flight model.
    gravity_x: float = -0.661
    gravity_y: float = 0.0
    damping: float = 0.178
    # Table geometry (metres, table frame centred on the table).
    table_length: float = 1.9304
    table_width: float = 0.8636
    puck_radius: float = 0.03175
    paddle_radius: float = 0.0508
    # Split-fit windows.
    fit_window: int = 4            # usable frames on each side of a split
    min_fit_points: int = 3        # fewer -> split not evaluated
    max_window_span: int = 14      # frames; pre window may not reach further back
    # Event detection.
    dv_threshold: float = 0.35     # m/s velocity jump between pre/post fits
    dp_threshold: float = 0.06     # m position jump between pre/post fits
    min_event_separation: int = 2  # frames; closer peaks merge into one event
    merge_unknown_within: int = 4  # frames; an unknown impulse this close to a classified event is absorbed by it
    # Event classification.
    wall_proximity: float = 0.08   # m from wall-contact line (puck centre)
    wall_min_approach: float = 0.10  # m/s normal speed into the wall (pre)
    wall_min_reversal: float = 0.20  # m/s drop of the normal component
    paddle_contact_slack: float = 0.06  # m added to (r_paddle + r_puck); generous because the tracker lags the pose ~1 frame
    opponent_x_min: float = 0.0    # m; impulses with puck x beyond this are 'opponent_hit'
    opponent_gain_ratio: float = 1.15  # speed_post/speed_pre above this near a far wall = human hit
    # Rest / occlusion / stale handling.
    rest_speed: float = 0.08       # m/s
    min_rest_run: int = 3          # frames
    max_occlusion_gap: int = 10    # frames; longer gaps stay 'occluded'
    stale_eps: float = 1e-6        # m; identical readings below this are stale
    stale_run_is_rest: int = 4     # >= this many identical readings = real rest
    # Segment layout.
    collision_pad: int = 1         # frames added on each side of an event

    @property
    def x_wall(self) -> float:
        return 0.5 * self.table_length - self.puck_radius

    @property
    def y_wall(self) -> float:
        return 0.5 * self.table_width - self.puck_radius

    @property
    def contact_distance(self) -> float:
        return self.paddle_radius + self.puck_radius + self.paddle_contact_slack


@dataclass
class Trajectory:
    path: Path
    times: np.ndarray        # (N,) absolute seconds
    t_rel: np.ndarray        # (N,) seconds from first frame
    puck_xy: np.ndarray      # (N, 2)
    valid: np.ndarray        # (N,) bool — puck not occluded
    fresh: np.ndarray        # (N,) bool — reading is not a stale duplicate
    paddle_xy: np.ndarray    # (N, 2) table frame
    paddle_v: np.ndarray     # (N, 2)
    has_image: bool
    puck_x_sign: int = 1     # -1 if the logged puck x was mirrored into the sim frame
    paddle_x_sign: int = 1   # sim paddle x = paddle_x_sign * pose_x + paddle_x_offset
    paddle_x_offset: float = 0.0

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def usable(self) -> np.ndarray:
        return self.valid & self.fresh


@dataclass
class Event:
    split: int               # first post-impact usable frame (b)
    pre_end: int             # last pre-impact usable frame (a)
    label: str               # wall_collision / paddle_collision / unknown_impulse
    dv: float
    dp: float
    v_pre: list[float]
    v_post: list[float]
    speed_pre: float
    speed_post: float
    speed_ratio: float
    gap_frames: int          # occluded/stale frames between a and b
    wall: Optional[str] = None
    wall_normal_pre: Optional[float] = None
    wall_normal_post: Optional[float] = None
    paddle_min_dist: Optional[float] = None
    paddle_approach_speed: Optional[float] = None


@dataclass
class Segment:
    label: str
    start: int               # inclusive frame index
    end: int                 # inclusive frame index
    n_frames: int
    duration_s: float
    valid_fraction: float
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _mark_stale(puck_xy: np.ndarray, valid: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """A valid frame whose reading equals the previous valid frame's is stale,
    unless it is part of a run of >= stale_run_is_rest identical readings
    (a genuine rest)."""
    n = len(puck_xy)
    fresh = np.ones(n, dtype=bool)
    same = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if valid[t] and valid[t - 1]:
            same[t] = np.linalg.norm(puck_xy[t] - puck_xy[t - 1]) < cfg.stale_eps
    t = 1
    while t < n:
        if same[t]:
            u = t
            while u < n and same[u]:
                u += 1
            run = u - t
            if run < cfg.stale_run_is_rest:
                fresh[t:u] = False
            t = u
        else:
            t += 1
    return fresh


def load_trajectory(path: str | Path, cfg: SegmentationConfig, puck_x_sign: int = 1,
                    paddle_x_sign: int = 1, paddle_x_offset: float = 0.0) -> Trajectory:
    """Load a split-schema HDF5 into the sim frame:
    ``puck_x = puck_x_sign * puck_x_logged`` and
    ``paddle_x = paddle_x_sign * pose_x + paddle_x_offset``
    (see module docstring and ``estimate_axis_transforms``)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        puck = np.asarray(f["puck"][:], dtype=np.float64)
        times = np.asarray(f["cur_time"][:], dtype=np.float64).ravel()
        pose = np.asarray(f["pose"][:, :2], dtype=np.float64)
        speed = np.asarray(f["speed"][:, :2], dtype=np.float64) if "speed" in f else np.zeros_like(pose)
        has_image = "image" in f
    valid = puck[:, 2] == 0
    puck_xy = puck[:, :2].copy()
    if puck_x_sign == -1:
        puck_xy[:, 0] *= -1.0
    pose = pose.copy(); speed = speed.copy()
    pose[:, 0] = paddle_x_sign * pose[:, 0] + paddle_x_offset
    speed[:, 0] *= paddle_x_sign
    fresh = _mark_stale(puck_xy, valid, cfg)
    return Trajectory(
        path=path, times=times, t_rel=times - times[0], puck_xy=puck_xy,
        valid=valid, fresh=fresh, paddle_xy=pose, paddle_v=speed, has_image=has_image,
        puck_x_sign=int(puck_x_sign), paddle_x_sign=int(paddle_x_sign),
        paddle_x_offset=float(paddle_x_offset),
    )


def measure_x_acceleration(traj: Trajectory, window: int = 10, max_rms: float = 0.015) -> tuple[float, int]:
    """Median raw x-acceleration over sliding windows of consecutive usable
    frames whose quadratic fit is clean (rms < max_rms). Returns (median, n)."""
    idx = np.flatnonzero(traj.usable)
    accs = []
    for k in range(0, len(idx) - window + 1):
        w = idx[k:k + window]
        if w[-1] - w[0] != window - 1:
            continue
        t = traj.t_rel[w] - traj.t_rel[w[0]]
        A = np.column_stack([np.ones_like(t), t, 0.5 * t ** 2])
        c, *_ = np.linalg.lstsq(A, traj.puck_xy[w, 0], rcond=None)
        rms = float(np.sqrt(np.mean((A @ c - traj.puck_xy[w, 0]) ** 2)))
        if rms < max_rms:
            accs.append(c[2])
    if not accs:
        return float("nan"), 0
    return float(np.median(accs)), len(accs)


def estimate_puck_x_sign(paths: list[str | Path], cfg: SegmentationConfig, min_abs_acc: float = 0.2) -> dict:
    """Sign of the measured free-flight x-acceleration vs ``cfg.gravity_x``.
    Falls back to +1 (no mirror) when the acceleration is too small to call."""
    accs, n_tot = [], 0
    for p in paths:
        a, n = measure_x_acceleration(load_trajectory(p, cfg))
        if n > 0:
            accs.append(a); n_tot += n
    out = {"puck_x_sign": 1, "acc_measured": float("nan"), "n_windows": n_tot, "puck_decided": False}
    if accs:
        acc = float(np.median(accs))
        out["acc_measured"] = acc
        out["puck_decided"] = abs(acc) >= min_abs_acc and abs(cfg.gravity_x) >= min_abs_acc
        if out["puck_decided"] and np.sign(acc) != np.sign(cfg.gravity_x):
            out["puck_x_sign"] = -1
    return out


def estimate_paddle_transform(paths: list[str | Path], cfg: SegmentationConfig, puck_x_sign: int,
                              offsets: Optional[np.ndarray] = None, min_hits: int = 3) -> dict:
    """Calibrate ``paddle_x = sign * pose_x + offset`` from puck–paddle interactions.

    Impulse frames (split-fit ``dv`` above threshold, away from walls) should
    be within contact distance of the paddle; clean free-flight frames should
    not lie inside the paddle. score = hits - 0.5 * pass_throughs, maximised
    over sign ∈ {+1, -1} and a grid of offsets. Falls back to the sign that
    puts the median paddle on the robot half with offset 0 when fewer than
    ``min_hits`` impulses can be explained by any mapping.
    """
    if offsets is None:
        offsets = np.arange(-1.6, 1.6001, 0.02)
    ev_puck, ev_pose, fr_puck, fr_pose = [], [], [], []
    pose_med = []
    for p in paths:
        tr = load_trajectory(p, cfg, puck_x_sign=puck_x_sign)
        st = split_fit_stats(tr, cfg)
        pose_med.append(float(np.median(tr.paddle_xy[:, 0])))
        dv = np.nan_to_num(st["dv"])
        sp = np.nan_to_num(st["speed_local"])
        for s_ in range(1, tr.n - 1):
            if not tr.usable[s_]:
                continue
            wd = min(cfg.x_wall - abs(tr.puck_xy[s_, 0]), cfg.y_wall - abs(tr.puck_xy[s_, 1]))
            if dv[s_] > cfg.dv_threshold and wd > cfg.wall_proximity:
                lo, hi = max(0, s_ - 2), min(tr.n, s_ + 2)
                m = tr.usable[lo:hi]
                ev_puck.append(tr.puck_xy[lo:hi][m]); ev_pose.append(tr.paddle_xy[lo:hi][m])
            elif dv[s_] < 0.5 * cfg.dv_threshold and sp[s_] > 0.3:
                fr_puck.append(tr.puck_xy[s_]); fr_pose.append(tr.paddle_xy[s_])
    fr_puck = np.asarray(fr_puck).reshape(-1, 2); fr_pose = np.asarray(fr_pose).reshape(-1, 2)
    contact = cfg.paddle_radius + cfg.puck_radius + 0.02
    table = []
    for sign in (1, -1):
        for off in offsets:
            hits = 0
            for pu, po in zip(ev_puck, ev_pose):
                px = np.column_stack([sign * po[:, 0] + off, po[:, 1]])
                if np.min(np.linalg.norm(pu - px, axis=1)) <= contact:
                    hits += 1
            if len(fr_puck):
                px = np.column_stack([sign * fr_pose[:, 0] + off, fr_pose[:, 1]])
                passes = int(np.sum(np.linalg.norm(fr_puck - px, axis=1) < contact - 0.03))
            else:
                passes = 0
            table.append((hits - 0.5 * passes, sign, float(off), hits, passes))
    table.sort(key=lambda r: -r[0])
    best = table[0]
    out = {"paddle_x_sign": int(best[1]), "paddle_x_offset": round(best[2], 3),
           "hits": int(best[3]), "pass_throughs": int(best[4]), "n_impulses": len(ev_puck),
           "n_free_frames": int(len(fr_puck)), "decided": best[3] >= min_hits,
           "runner_up": [(round(r[0], 1), r[1], round(r[2], 2), r[3], r[4]) for r in table[1:4]]}
    if not out["decided"]:
        med = float(np.median(pose_med)) if pose_med else 0.0
        out["paddle_x_sign"] = -1 if med > 0 else 1
        out["paddle_x_offset"] = 0.0
    return out


def estimate_axis_transforms(paths: list[str | Path], cfg: SegmentationConfig) -> dict:
    """Puck sign (from gravity) + paddle (sign, offset) (from interactions)."""
    out = estimate_puck_x_sign(paths, cfg)
    out.update(estimate_paddle_transform(paths, cfg, out["puck_x_sign"]))
    return out


# ---------------------------------------------------------------------------
# Damped free-flight model
# ---------------------------------------------------------------------------

def fit_damped(t: np.ndarray, pos: np.ndarray, cfg: SegmentationConfig) -> dict:
    """LSQ fit of the damped free-flight model  a = g - γ v  per axis:

        v(t) = (v0 - g/γ) e^{-γt} + g/γ
        p(t) = p0 + (v0 - g/γ)(1 - e^{-γt})/γ + (g/γ) t

    which is linear in (p0, u = v0 - g/γ) for fixed g, γ. Returns dict with
    p0 (2,), u (2,), rms (float). ``t`` is relative to an arbitrary origin;
    ``model_state`` evaluates the fit at times relative to that origin.

    Note: ``sysid/puck_grid_search.py`` & co. use the opposite sign for g
    (their model acceleration is ``-gx``) on the *raw* puck frame; here
    ``gravity_x`` is the physical acceleration in the sim frame.
    """
    g = np.array([cfg.gravity_x, cfg.gravity_y])
    gam = cfg.damping
    if gam < 1e-8:
        A = np.column_stack([np.ones_like(t), t])
        target = pos - 0.5 * g[None, :] * (t ** 2)[:, None]
        coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
        p0, u = coeffs[0], coeffs[1]
        pred = A @ coeffs + 0.5 * g[None, :] * (t ** 2)[:, None]
    else:
        basis = (1.0 - np.exp(-gam * t)) / gam
        A = np.column_stack([np.ones_like(t), basis])
        target = pos - (g / gam)[None, :] * t[:, None]
        coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
        p0, u = coeffs[0], coeffs[1]
        pred = A @ coeffs + (g / gam)[None, :] * t[:, None]
    rms = float(np.sqrt(np.mean(np.sum((pred - pos) ** 2, axis=1))))
    return {"p0": p0, "u": u, "rms": rms}


def model_state(fit: dict, tau: float, cfg: SegmentationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of a fit at relative time ``tau``."""
    g = np.array([cfg.gravity_x, cfg.gravity_y])
    gam = cfg.damping
    if gam < 1e-8:
        v = fit["u"] + g * tau
        p = fit["p0"] + fit["u"] * tau + 0.5 * g * tau ** 2
    else:
        e = np.exp(-gam * tau)
        v = fit["u"] * e + g / gam
        p = fit["p0"] + fit["u"] * (1.0 - e) / gam + (g / gam) * tau
    return p, v


# ---------------------------------------------------------------------------
# Split-fit statistics
# ---------------------------------------------------------------------------

def split_fit_stats(traj: Trajectory, cfg: SegmentationConfig) -> dict:
    """For every usable frame s compute pre/post fits around the split (s-1 | s).

    Returns arrays of length N (NaN where not evaluated):
        dv, dp          velocity / position jump magnitudes at t_s
        v_pre, v_post   (N, 2) fitted velocities at t_s
        p_pre, p_post   (N, 2) fitted positions at t_s
        pre_end         index of the last usable pre frame (-1 if none)
        speed_local     |v_post| — local free-flight speed estimate at s
    """
    n = traj.n
    usable = traj.usable
    idx_usable = np.flatnonzero(usable)
    K = cfg.fit_window
    dv = np.full(n, np.nan)
    dp = np.full(n, np.nan)
    v_pre = np.full((n, 2), np.nan)
    v_post = np.full((n, 2), np.nan)
    p_pre = np.full((n, 2), np.nan)
    p_post = np.full((n, 2), np.nan)
    pre_end = np.full(n, -1, dtype=int)
    speed_local = np.full(n, np.nan)
    pre_fits: dict[int, dict] = {}
    post_fits: dict[int, dict] = {}

    for k, s in enumerate(idx_usable):
        post_idx = idx_usable[k:k + K]
        pre_idx = idx_usable[max(0, k - K):k]
        pre_idx = pre_idx[pre_idx >= s - cfg.max_window_span]
        if len(post_idx) >= cfg.min_fit_points:
            t0 = traj.t_rel[s]
            fpost = fit_damped(traj.t_rel[post_idx] - t0, traj.puck_xy[post_idx], cfg)
            post_fits[s] = fpost
            pp, vp = model_state(fpost, 0.0, cfg)
            p_post[s], v_post[s] = pp, vp
            speed_local[s] = float(np.linalg.norm(vp))
        if len(pre_idx) >= cfg.min_fit_points and s in post_fits:
            t0 = traj.t_rel[s]
            fpre = fit_damped(traj.t_rel[pre_idx] - t0, traj.puck_xy[pre_idx], cfg)
            pre_fits[s] = fpre
            pq, vq = model_state(fpre, 0.0, cfg)
            p_pre[s], v_pre[s] = pq, vq
            pre_end[s] = int(pre_idx[-1])
            dv[s] = float(np.linalg.norm(v_post[s] - vq))
            dp[s] = float(np.linalg.norm(p_post[s] - pq))
    return {
        "dv": dv, "dp": dp, "v_pre": v_pre, "v_post": v_post, "p_pre": p_pre,
        "p_post": p_post, "pre_end": pre_end, "speed_local": speed_local,
        "pre_fits": pre_fits, "post_fits": post_fits,
    }


# ---------------------------------------------------------------------------
# Event detection + classification
# ---------------------------------------------------------------------------

def _wall_normals(cfg: SegmentationConfig) -> dict[str, tuple[np.ndarray, float]]:
    """side -> (outward unit normal, signed distance function offset)."""
    return {
        "x+": (np.array([1.0, 0.0]), cfg.x_wall),
        "x-": (np.array([-1.0, 0.0]), cfg.x_wall),
        "y+": (np.array([0.0, 1.0]), cfg.y_wall),
        "y-": (np.array([0.0, -1.0]), cfg.y_wall),
    }


def _classify_event(traj: Trajectory, stats: dict, s: int, cfg: SegmentationConfig) -> Event:
    a = int(stats["pre_end"][s])
    b = int(s)
    v_pre = stats["v_pre"][s]
    v_post = stats["v_post"][s]
    fpre = stats["pre_fits"][s]
    fpost = stats["post_fits"][s]
    t_s = traj.t_rel[s]

    # Puck positions across the impact window [a, b] (measured where usable,
    # otherwise extrapolated from the nearer fit).
    frames = list(range(a, b + 1))
    mid = 0.5 * (traj.t_rel[a] + traj.t_rel[b])
    puck_path = []
    for f in frames:
        if traj.usable[f]:
            puck_path.append(traj.puck_xy[f])
        else:
            fit = fpre if traj.t_rel[f] <= mid else fpost
            puck_path.append(model_state(fit, traj.t_rel[f] - t_s, cfg)[0])
    puck_path = np.asarray(puck_path)

    # Paddle test.
    pad = traj.paddle_xy[a:b + 1]
    dists = np.linalg.norm(puck_path - pad, axis=1)
    j = int(np.argmin(dists))
    paddle_min = float(dists[j])
    normal = puck_path[j] - pad[j]
    nn = np.linalg.norm(normal)
    normal = normal / nn if nn > 1e-6 else np.array([1.0, 0.0])
    v_rel = v_pre - traj.paddle_v[frames[j]]
    approach = -float(np.dot(v_rel, normal))   # >0: closing along the contact normal
    paddle_hit = paddle_min <= cfg.contact_distance

    # Wall test — best side by normal-component reversal.
    best_wall = None
    for side, (nvec, lim) in _wall_normals(cfg).items():
        dist_to_wall = lim - float(np.max(puck_path @ nvec))
        if dist_to_wall > cfg.wall_proximity:
            continue
        n_pre = float(v_pre @ nvec)
        n_post = float(v_post @ nvec)
        if n_pre < cfg.wall_min_approach:
            continue
        if n_pre - n_post < cfg.wall_min_reversal:
            continue
        score = n_pre - n_post
        if best_wall is None or score > best_wall[1]:
            best_wall = (side, score, n_pre, n_post)

    speed_pre = float(np.linalg.norm(v_pre))
    speed_post = float(np.linalg.norm(v_post))
    on_opponent_half = float(np.mean(puck_path[:, 0])) > cfg.opponent_x_min
    gained_energy = speed_post > cfg.opponent_gain_ratio * speed_pre
    if paddle_hit and (best_wall is None or approach > 0.0):
        label = "paddle_collision"
    elif best_wall is not None and not (on_opponent_half and gained_energy):
        label = "wall_collision"
    elif on_opponent_half:
        label = "opponent_hit"
    else:
        label = "unknown_impulse"

    return Event(
        split=b, pre_end=a, label=label,
        dv=float(stats["dv"][s]), dp=float(stats["dp"][s]),
        v_pre=[float(x) for x in v_pre], v_post=[float(x) for x in v_post],
        speed_pre=speed_pre, speed_post=speed_post,
        speed_ratio=speed_post / speed_pre if speed_pre > 1e-6 else float("nan"),
        gap_frames=b - a - 1,
        wall=best_wall[0] if best_wall else None,
        wall_normal_pre=best_wall[2] if best_wall else None,
        wall_normal_post=best_wall[3] if best_wall else None,
        paddle_min_dist=paddle_min,
        paddle_approach_speed=approach,
    )


def detect_events(traj: Trajectory, stats: dict, cfg: SegmentationConfig) -> list[Event]:
    dv, dp = stats["dv"], stats["dp"]
    cand = np.flatnonzero((np.nan_to_num(dv) > cfg.dv_threshold) | (np.nan_to_num(dp) > cfg.dp_threshold))
    events: list[Event] = []
    if len(cand) == 0:
        return events
    # Group candidates closer than min_event_separation, keep the peak of each.
    groups: list[list[int]] = [[int(cand[0])]]
    for s in cand[1:]:
        if s - groups[-1][-1] <= cfg.min_event_separation:
            groups[-1].append(int(s))
        else:
            groups.append([int(s)])
    for g in groups:
        score = [np.nan_to_num(dv[s]) + np.nan_to_num(dp[s]) / 0.05 for s in g]
        s_star = g[int(np.argmax(score))]
        events.append(_classify_event(traj, stats, s_star, cfg))
    return _absorb_unknowns(events, cfg)


def _absorb_unknowns(events: list[Event], cfg: SegmentationConfig) -> list[Event]:
    """Because the puck tracker lags the paddle pose by ~1 frame, a paddle /
    wall impact often shows up as an 'unknown' velocity jump a couple of frames
    before the geometric contact is seen. Merge such an unknown event into the
    neighbouring classified event (extending its window)."""
    events = sorted(events, key=lambda e: e.split)
    keep: list[Event] = []
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.label == "unknown_impulse":
            nxt = events[i + 1] if i + 1 < len(events) else None
            prv = keep[-1] if keep else None
            if nxt is not None and nxt.label != "unknown_impulse" and nxt.pre_end - ev.split <= cfg.merge_unknown_within:
                nxt.pre_end = ev.pre_end
                nxt.gap_frames = nxt.split - nxt.pre_end - 1
                nxt.v_pre, nxt.speed_pre = ev.v_pre, ev.speed_pre
                nxt.speed_ratio = nxt.speed_post / nxt.speed_pre if nxt.speed_pre > 1e-6 else float("nan")
                nxt.dv = max(nxt.dv, ev.dv)
                i += 1
                continue
            if prv is not None and prv.label != "unknown_impulse" and ev.pre_end - prv.split <= cfg.merge_unknown_within:
                prv.v_post, prv.speed_post = ev.v_post, ev.speed_post
                prv.speed_ratio = prv.speed_post / prv.speed_pre if prv.speed_pre > 1e-6 else float("nan")
                prv.dv = max(prv.dv, ev.dv)
                prv.split = ev.split
                prv.gap_frames = prv.split - prv.pre_end - 1
                i += 1
                continue
        keep.append(ev)
        i += 1
    return keep


# ---------------------------------------------------------------------------
# Frame labelling + segment construction
# ---------------------------------------------------------------------------

def label_frames(traj: Trajectory, stats: dict, events: list[Event], cfg: SegmentationConfig) -> np.ndarray:
    n = traj.n
    labels = np.array(["free_fall"] * n, dtype=object)
    labels[~traj.valid] = "occluded"

    # Bridge short occlusion gaps (events spanning a gap overwrite below).
    t = 0
    while t < n:
        if not traj.valid[t]:
            u = t
            while u < n and not traj.valid[u]:
                u += 1
            if (u - t) <= cfg.max_occlusion_gap and t > 0 and u < n:
                labels[t:u] = "free_fall"
            t = u
        else:
            t += 1

    # Rest: runs of low local speed.
    sp = stats["speed_local"]
    low = np.zeros(n, dtype=bool)
    for t in range(n):
        if traj.usable[t] and np.isfinite(sp[t]) and sp[t] < cfg.rest_speed:
            low[t] = True
    t = 0
    while t < n:
        if low[t]:
            u = t
            while u < n and (low[u] or not traj.usable[u]):
                u += 1
            # trim trailing non-usable frames
            while u - 1 > t and not low[u - 1]:
                u -= 1
            if (u - t) >= cfg.min_rest_run:
                labels[t:u] = np.where(labels[t:u] == "free_fall", "rest", labels[t:u])
            t = max(u, t + 1)
        else:
            t += 1

    # Events: [a - pad, b + pad]; overlapping windows split at the midpoint.
    windows = []
    for ev in sorted(events, key=lambda e: e.split):
        lo = max(0, ev.pre_end - cfg.collision_pad)
        hi = min(n - 1, ev.split + cfg.collision_pad)
        windows.append([lo, hi, ev.label])
    for i in range(1, len(windows)):
        if windows[i][0] <= windows[i - 1][1]:
            mid = (windows[i - 1][1] + windows[i][0]) // 2
            windows[i - 1][1] = mid
            windows[i][0] = mid + 1
    for lo, hi, lab in windows:
        if hi >= lo:
            labels[lo:hi + 1] = lab
    return labels


def build_segments(traj: Trajectory, labels: np.ndarray, events: list[Event],
                   stats: dict, cfg: SegmentationConfig) -> list[Segment]:
    n = traj.n
    segments: list[Segment] = []
    ev_by_split = {}
    for ev in events:
        ev_by_split.setdefault(ev.label, []).append(ev)
    t = 0
    while t < n:
        u = t
        while u < n and labels[u] == labels[t]:
            u += 1
        lab = str(labels[t])
        start, end = t, u - 1
        meta: dict = {}
        if lab in EVENT_LABELS:
            inside = [e for e in events if e.label == lab and start <= e.split <= end]
            if inside:
                meta = {"events": [asdict(e) for e in inside]}
        elif lab in ("free_fall", "rest"):
            idx = np.flatnonzero(traj.usable[start:end + 1]) + start
            if len(idx) >= cfg.min_fit_points:
                t0 = traj.t_rel[idx[0]]
                fit = fit_damped(traj.t_rel[idx] - t0, traj.puck_xy[idx], cfg)
                _, v0 = model_state(fit, 0.0, cfg)
                meta = {
                    "fit_rms_m": fit["rms"],
                    "v0": [float(x) for x in v0],
                    "speed0": float(np.linalg.norm(v0)),
                    "n_usable": int(len(idx)),
                }
        segments.append(Segment(
            label=lab, start=int(start), end=int(end), n_frames=int(end - start + 1),
            duration_s=float(traj.times[end] - traj.times[start]),
            valid_fraction=float(np.mean(traj.valid[start:end + 1])),
            meta=meta,
        ))
        t = u
    return segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    trajectory: Trajectory
    stats: dict
    events: list[Event]
    labels: np.ndarray
    segments: list[Segment]
    config: SegmentationConfig

    def counts(self) -> dict[str, int]:
        return {lab: sum(1 for s in self.segments if s.label == lab) for lab in LABELS}

    def frame_counts(self) -> dict[str, int]:
        return {lab: int(np.sum(self.labels == lab)) for lab in LABELS}

    def to_json_dict(self) -> dict:
        return {
            "source": str(self.trajectory.path),
            "n_frames": int(self.trajectory.n),
            "puck_x_sign": int(self.trajectory.puck_x_sign),
            "paddle_x_sign": int(self.trajectory.paddle_x_sign),
            "paddle_x_offset": float(self.trajectory.paddle_x_offset),
            "config": asdict(self.config),
            "segment_counts": self.counts(),
            "frame_counts": self.frame_counts(),
            "segments": [asdict(s) for s in self.segments],
            "events": [asdict(e) for e in self.events],
        }


def segment_trajectory(path: str | Path, cfg: Optional[SegmentationConfig] = None,
                       puck_x_sign: int = 1, paddle_x_sign: int = 1,
                       paddle_x_offset: float = 0.0) -> SegmentationResult:
    cfg = cfg or SegmentationConfig()
    traj = load_trajectory(path, cfg, puck_x_sign=puck_x_sign, paddle_x_sign=paddle_x_sign,
                           paddle_x_offset=paddle_x_offset)
    stats = split_fit_stats(traj, cfg)
    events = detect_events(traj, stats, cfg)
    labels = label_frames(traj, stats, events, cfg)
    segments = build_segments(traj, labels, events, stats, cfg)
    return SegmentationResult(traj, stats, events, labels, segments, cfg)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_segments_json(result: SegmentationResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.to_json_dict(), f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_segment_hdf5s(result: SegmentationResult, out_dir: str | Path,
                        include_images: bool = False,
                        labels: tuple[str, ...] = ("free_fall", "wall_collision", "paddle_collision")) -> list[Path]:
    """Slice the source HDF5 into one split-schema file per segment.

    Files are named ``<label>_<start>_<end>.hdf5`` (frame indices are
    inclusive and refer to the source file). Each file carries ``segment``
    attrs: label / start / end / source.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with h5py.File(result.trajectory.path, "r") as src:
        keys = [k for k in SPLIT_DATASETS if k in src]
        if include_images and "image" in src:
            keys.append("image")
        for seg in result.segments:
            if seg.label not in labels:
                continue
            dst_path = out_dir / f"{seg.label}_{seg.start}_{seg.end}.hdf5"
            with h5py.File(dst_path, "w") as dst:
                for k in keys:
                    dst.create_dataset(k, data=src[k][seg.start:seg.end + 1])
                dst.attrs["label"] = seg.label
                dst.attrs["start"] = seg.start
                dst.attrs["end"] = seg.end
                dst.attrs["source"] = str(result.trajectory.path)
                dst.attrs["puck_x_sign"] = int(result.trajectory.puck_x_sign)
                dst.attrs["paddle_x_sign"] = int(result.trajectory.paddle_x_sign)
                dst.attrs["paddle_x_offset"] = float(result.trajectory.paddle_x_offset)
                dst.attrs["meta"] = json.dumps(seg.meta, default=_json_default)
            written.append(dst_path)
    return written
