"""Puck incoming / outgoing speed at the paddle collision, and the best-three-per-condition selection.

The 30 Hz tracker is sampled by the 20 Hz control loop, so consecutive-sample differences alias
badly (at 2 m/s a 0–33 ms timestamp jitter is ±7 cm) and the puck is often occluded right at
the impact (the arm crosses the camera's view). Speeds are therefore taken from **model fits**,
exactly as ``sysid/common/trajectory_segmentation.py`` does for its split fits: the
damped free-flight model ``a = g − γ v`` (``fit_damped``, with the sysid gravity / damping, in
the observation frame where the puck accelerates towards ``+x``) is fitted to the usable frames
*before* and *after* a candidate split, the split that minimises the combined residual is the
collision, the two model trajectories are intersected to get the contact time ``t_c`` and both
models are evaluated at ``t_c``:

    v_in  = pre-model velocity at t_c        (towards the paddle, +x)
    v_out = post-model velocity at t_c       (away from the paddle, −x)

The paddle speed ``u_p`` is the robot's own ``speed_x`` (already a velocity) interpolated at
``t_c − camera_lag``; the lag is calibrated from the moving-paddle trials by requiring the shifted
paddle pose to be exactly ``r_paddle + r_puck`` from the puck at contact
(``calibrate_camera_lag``).

Quality / validity (``estimate_collision``): a trial is *valid* when a reversal was found with
enough frames on both sides, the outgoing direction is within ``max_out_angle_deg`` of head-on
and the puck actually separates from the paddle (``vx_out_away − u_p ≥ min_separation``).
Among valid trials the *quality score* (lower is better) ranks how head-on and how clean a
collision was — ``angle_out_deg + 500·|Δy| + 200·(rms_pre + rms_post)`` — and
``select_canonical`` keeps the best ``per_condition`` (3) trials of every condition.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from sysid.common.trajectory_segmentation import SegmentationConfig, fit_damped, model_state
from .dataset import CollisionTrial, group_by_condition


@dataclass
class SpeedConfig:
    gravity_x: float = 0.661          # observation frame: the puck accelerates towards +x (the robot)
    damping: float = 0.178
    paddle_radius: float = 0.0508
    puck_radius: float = 0.03175
    pre_frames: int = 6               # usable frames fitted on each side of the split
    post_frames: int = 6
    min_fit_points: int = 3
    max_pre_span_s: float = 0.6       # the pre window may not reach further back than this
    max_post_span_s: float = 0.8
    min_reversal_x: float = 0.30      # the collision must happen at x above this (near the robot end)
    far_wall_x: float = -0.80         # post frames beyond these are a wall bounce → not used
    side_wall_y: float = 0.34
    max_out_angle_deg: float = 40.0   # validity gate: outgoing direction vs head-on
    min_separation: float = 0.10      # m/s; vx_out_away − u_p must exceed this (else the paddle carried the puck)
    min_speed_fraction: float = 0.5   # moving-paddle trials: u_p must be at least this fraction of the paddle's plateau speed
    recontact_margin: float = 0.03    # m; paddle closer than contact_distance − margin to the post model = re-contact (3 cm tolerates the 0–33 ms frame-time jitter)
    camera_lag_s: float = 0.0         # paddle rows are read at t_c − lag; see calibrate_camera_lag
    oblique: bool = False             # offset collisions: separation / re-contact gates use the 2-D contact geometry (normal from the centres) instead of x only

    @property
    def contact_distance(self) -> float:
        return self.paddle_radius + self.puck_radius

    def seg_cfg(self) -> SegmentationConfig:
        return SegmentationConfig(gravity_x=self.gravity_x, gravity_y=0.0, damping=self.damping,
                                  puck_radius=self.puck_radius, paddle_radius=self.paddle_radius)


@dataclass
class CollisionMeasurement:
    name: str
    condition: str
    height: str
    action_delta: float
    repeat: int
    valid: bool
    reason: str                       # '' when valid
    t_c: float = float("nan")         # s, relative to the first scripted step (camera clock)
    x_c: float = float("nan")         # contact position of the puck centre (obs frame)
    y_c: float = float("nan")
    speed_in: float = float("nan")    # |v_in|
    speed_out: float = float("nan")   # |v_out|
    vx_in: float = float("nan")       # towards the paddle (+x)
    vy_in: float = float("nan")
    vx_out_away: float = float("nan") # away from the paddle (= −vx of the post model)
    vy_out: float = float("nan")
    angle_in_deg: float = float("nan")
    angle_out_deg: float = float("nan")
    u_p: float = float("nan")         # paddle speed towards the puck at t_c − lag (m/s)
    u_p_y: float = float("nan")
    u_p_plateau: float = float("nan") # the paddle's peak speed during the action phase (what the condition intended)
    pad_x: float = float("nan")       # paddle centre at t_c − lag (obs frame)
    pad_y: float = float("nan")
    contact_gap: float = float("nan") # pad_x − x_c (should be r_paddle + r_puck)
    dy: float = float("nan")          # y_c − pad_y (lateral offset at contact; 0 = head-on)
    approach_speed: float = float("nan")   # u_p + vx_in
    gain: float = float("nan")        # (vx_out_away + vx_in) / (u_p + vx_in) = (1+e)·r/(r+1) for a free paddle
    rms_pre_mm: float = float("nan")
    rms_post_mm: float = float("nan")
    n_pre: int = 0
    n_post: int = 0
    quality: float = float("inf")
    selected: bool = False
    rank: int = 0
    split: int = -1
    pre_idx: list = field(default_factory=list, repr=False)
    post_idx: list = field(default_factory=list, repr=False)
    pre_fit: dict = field(default_factory=dict, repr=False)
    post_fit: dict = field(default_factory=dict, repr=False)

    def row(self) -> dict:
        d = asdict(self)
        for k in ("pre_idx", "post_idx", "pre_fit", "post_fit"):
            d.pop(k)
        return d


CSV_COLUMNS = ["name", "condition", "height", "action_delta", "repeat", "selected", "rank", "valid", "reason", "quality",
               "t_c", "speed_in", "speed_out", "u_p", "u_p_plateau", "approach_speed", "gain", "vx_in", "vy_in", "vx_out_away", "vy_out",
               "angle_in_deg", "angle_out_deg", "x_c", "y_c", "pad_x", "pad_y", "contact_gap", "dy",
               "rms_pre_mm", "rms_post_mm", "n_pre", "n_post", "split"]


def _paddle_at(trial: CollisionTrial, t_abs: float) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([np.interp(t_abs, trial.pad_t, trial.pad_xy[:, 0]), np.interp(t_abs, trial.pad_t, trial.pad_xy[:, 1])])
    v = np.array([np.interp(t_abs, trial.pad_t, trial.pad_v[:, 0]), np.interp(t_abs, trial.pad_t, trial.pad_v[:, 1])])
    return xy, v


def _find_split(trial: CollisionTrial, cfg: SpeedConfig):
    """Change-point search: the usable-frame split with the lowest pre + post residual."""
    seg = cfg.seg_cfg()
    t, xy, valid = trial.t, trial.puck_xy, trial.puck_valid
    idx = np.flatnonzero(valid)
    best = None
    for k in range(cfg.min_fit_points, len(idx) - cfg.min_fit_points + 1):
        s = idx[k]
        pre = idx[max(0, k - cfg.pre_frames):k]
        pre = pre[t[pre] >= t[s] - cfg.max_pre_span_s]
        post = idx[k:k + cfg.post_frames]
        post = post[(xy[post, 0] > cfg.far_wall_x) & (np.abs(xy[post, 1]) < cfg.side_wall_y) & (t[post] <= t[s] + cfg.max_post_span_s)]
        if len(pre) < cfg.min_fit_points or len(post) < cfg.min_fit_points:
            continue
        if xy[pre[-1], 0] <= xy[pre[0], 0] or xy[pre[-1], 0] < cfg.min_reversal_x:
            continue                                     # pre must approach the paddle (+x) near the robot end
        if xy[post[-1], 0] >= xy[post[0], 0]:
            continue                                     # post must move away (−x)
        t0 = t[s]
        fpre = fit_damped(t[pre] - t0, xy[pre], seg)
        fpost = fit_damped(t[post] - t0, xy[post], seg)
        cost = fpre["rms"] + fpost["rms"]
        if best is None or cost < best[0]:
            best = (cost, int(s), pre, post, fpre, fpost)
    return best


def estimate_collision(trial: CollisionTrial, cfg: SpeedConfig) -> CollisionMeasurement:
    m = CollisionMeasurement(name=trial.name, condition=trial.condition, height=trial.height,
                             action_delta=trial.action_delta, repeat=trial.repeat, valid=False, reason="no reversal found")
    found = _find_split(trial, cfg)
    if found is None:
        return m
    _, s, pre, post, fpre, fpost = found
    seg = cfg.seg_cfg()
    t = trial.t
    t0 = t[s]
    # contact time: where the pre and post model trajectories cross (between the last pre frame and the split)
    taus = np.linspace(t[pre[-1]] - t0, 0.0, 241)
    gap = np.array([model_state(fpre, tau, seg)[0][0] - model_state(fpost, tau, seg)[0][0] for tau in taus])
    tau_c = float(taus[int(np.argmin(np.abs(gap)))])
    p_c, v_in = model_state(fpre, tau_c, seg)
    _, v_out = model_state(fpost, tau_c, seg)
    t_c_abs = t0 + tau_c
    pad_xy, pad_v = _paddle_at(trial, t_c_abs - cfg.camera_lag_s)

    m.t_c, m.x_c, m.y_c = float(t_c_abs - trial.t0), float(p_c[0]), float(p_c[1])
    m.vx_in, m.vy_in = float(v_in[0]), float(v_in[1])
    m.vx_out_away, m.vy_out = float(-v_out[0]), float(v_out[1])
    m.speed_in, m.speed_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
    m.angle_in_deg = float(np.degrees(np.arctan2(abs(v_in[1]), abs(v_in[0]))))
    m.angle_out_deg = float(np.degrees(np.arctan2(abs(v_out[1]), abs(v_out[0]))))
    m.u_p, m.u_p_y = float(-pad_v[0]), float(pad_v[1])
    m.u_p_plateau = float(np.max(-trial.pad_v[:, 0])) if trial.pad_v.size else float("nan")
    m.pad_x, m.pad_y = float(pad_xy[0]), float(pad_xy[1])
    m.contact_gap = float(pad_xy[0] - p_c[0])
    m.dy = float(p_c[1] - pad_xy[1])
    m.approach_speed = m.u_p + m.vx_in
    m.gain = (m.vx_out_away + m.vx_in) / m.approach_speed if m.approach_speed > 1e-6 else float("nan")
    m.rms_pre_mm, m.rms_post_mm = 1000.0 * fpre["rms"], 1000.0 * fpost["rms"]
    m.n_pre, m.n_post, m.split = int(len(pre)), int(len(post)), int(s)
    m.pre_idx, m.post_idx = [int(i) for i in pre], [int(i) for i in post]
    m.pre_fit = {"p0": fpre["p0"].tolist(), "u": fpre["u"].tolist(), "t0": float(t0)}
    m.post_fit = {"p0": fpost["p0"].tolist(), "u": fpost["u"].tolist(), "t0": float(t0)}

    reasons = []
    if m.angle_out_deg > cfg.max_out_angle_deg:
        reasons.append(f"outgoing angle {m.angle_out_deg:.0f}° > {cfg.max_out_angle_deg:.0f}°")
    t_post = t[post]
    xy_model = np.array([model_state(fpost, tt - t0, seg)[0] for tt in t_post])
    pad_post = np.column_stack([np.interp(t_post - cfg.camera_lag_s, trial.pad_t, trial.pad_xy[:, 0]),
                                np.interp(t_post - cfg.camera_lag_s, trial.pad_t, trial.pad_xy[:, 1])])
    if cfg.oblique:
        # contact normal from the centres at contact (paddle → puck); the puck must leave along it
        if abs(m.dy) >= cfg.contact_distance:
            reasons.append(f"no contact possible (|dy| {100 * abs(m.dy):.1f} cm ≥ contact distance {100 * cfg.contact_distance:.1f} cm)")
        n = np.array([-1.0, m.dy / cfg.contact_distance])            # puck sits at −x of the paddle, offset dy in y
        n[0] = -float(np.sqrt(max(0.0, 1.0 - n[1] ** 2)))
        v_rel = np.array([-m.vx_out_away, m.vy_out]) - np.array([-m.u_p, m.u_p_y])
        sep = float(v_rel @ n)
        if sep < cfg.min_separation:
            reasons.append(f"puck not separating along the contact normal ({sep:.2f} m/s)")
        # re-contact: 2-D centre distance between the lag-corrected paddle and the post model
        dist = np.linalg.norm(pad_post - xy_model, axis=1)
        if np.any(dist < cfg.contact_distance - cfg.recontact_margin):
            reasons.append("paddle re-contacts the puck inside the post window")
    else:
        if m.vx_out_away - m.u_p < cfg.min_separation:
            reasons.append(f"puck not separating (vx_out {m.vx_out_away:.2f} vs paddle {m.u_p:.2f} m/s)")
        if m.vx_out_away <= 0:
            reasons.append("puck did not rebound")
        # the paddle must not catch the puck again inside the post window (lag-corrected pose vs post model, x only)
        gaps = pad_post[:, 0] - xy_model[:, 0]
        if np.any(gaps < cfg.contact_distance - cfg.recontact_margin):
            reasons.append("paddle re-contacts the puck inside the post window")
    if trial.action_delta > 0 and np.isfinite(m.u_p_plateau) and m.u_p < cfg.min_speed_fraction * m.u_p_plateau:
        reasons.append(f"hit before the paddle reached speed (u_p {m.u_p:.2f} vs plateau {m.u_p_plateau:.2f} m/s)")
    m.valid = not reasons
    m.reason = "; ".join(reasons)
    m.quality = quality_score(m) if m.valid else float("inf")
    return m


def quality_score(m: CollisionMeasurement) -> float:
    """Lower = more head-on and cleaner: outgoing angle (deg) + 5 per cm of lateral offset + 2 per 10 mm of
    fit residual + up to 50 for a hit while the paddle was still accelerating (u_p below its plateau)."""
    accel = 0.0
    if m.action_delta > 0 and np.isfinite(m.u_p_plateau) and m.u_p_plateau > 1e-6:
        accel = 50.0 * max(0.0, 1.0 - m.u_p / m.u_p_plateau)
    return float(m.angle_out_deg + 500.0 * abs(m.dy) + 0.2 * (m.rms_pre_mm + m.rms_post_mm) + accel)


def measure_all(trials: Iterable[CollisionTrial], cfg: SpeedConfig) -> list[CollisionMeasurement]:
    return [estimate_collision(t, cfg) for t in trials]


def calibrate_camera_lag(trials: list[CollisionTrial], measurements: list[CollisionMeasurement], cfg: SpeedConfig,
                         min_paddle_speed: float = 0.2, lag_range=(-0.05, 0.30)) -> dict:
    """Per moving-paddle trial, the lag ``L`` for which the paddle pose at ``t_c − L`` is exactly
    ``contact_distance`` from the puck at contact; returns the median (the calibrated lag), the
    per-trial values, and the contact gap seen by the stationary-paddle trials (a check of the
    paddle frame offset — it should equal ``contact_distance``)."""
    by_name = {t.name: t for t in trials}
    grid = np.linspace(lag_range[0], lag_range[1], int(round((lag_range[1] - lag_range[0]) / 0.0005)) + 1)
    per_trial, static_gaps = {}, {}
    for m in measurements:
        if not m.valid or not np.isfinite(m.t_c):
            continue
        tr = by_name[m.name]
        t_abs = tr.t0 + m.t_c
        if abs(m.u_p) < min_paddle_speed and tr.action_delta == 0.0:
            static_gaps[m.name] = float(np.interp(t_abs, tr.pad_t, tr.pad_xy[:, 0]) - m.x_c)
            continue
        if abs(m.u_p) < min_paddle_speed:
            continue
        gaps = np.interp(t_abs - grid, tr.pad_t, tr.pad_xy[:, 0]) - m.x_c - cfg.contact_distance
        sign_change = np.flatnonzero(np.diff(np.sign(gaps)) != 0)
        i = int(sign_change[0]) if sign_change.size else int(np.argmin(np.abs(gaps)))
        per_trial[m.name] = float(grid[i])
    lags = np.array(list(per_trial.values()))
    out = {"lag_s": float(np.median(lags)) if lags.size else 0.0, "lag_mean_s": float(lags.mean()) if lags.size else float("nan"),
           "lag_std_s": float(lags.std()) if lags.size else float("nan"), "n_trials": int(lags.size), "per_trial": per_trial,
           "static_contact_gap_mean": float(np.mean(list(static_gaps.values()))) if static_gaps else float("nan"),
           "static_contact_gap_std": float(np.std(list(static_gaps.values()))) if static_gaps else float("nan"),
           "static_trials": static_gaps, "expected_contact_distance": cfg.contact_distance}
    return out


def select_canonical(measurements: list[CollisionMeasurement], per_condition: int = 3) -> dict:
    """Mark the ``per_condition`` best valid trials of every condition as selected (in place).
    Returns {condition: {"selected": [...], "rejected": [...], "n_valid": int}}."""
    groups: dict[str, list[CollisionMeasurement]] = {}
    for m in measurements:
        groups.setdefault(m.condition, []).append(m)
    info = {}
    for cond, ms in groups.items():
        valid = sorted([m for m in ms if m.valid], key=lambda m: m.quality)
        for r, m in enumerate(valid, start=1):
            m.rank = r
            m.selected = r <= per_condition
        for m in ms:
            if not m.valid:
                m.rank, m.selected = 0, False
        info[cond] = {"selected": [m.name for m in valid[:per_condition]],
                      "rejected": [m.name for m in ms if not m.selected], "n_valid": len(valid), "n_files": len(ms)}
    return info


def selected_trials(trials: list[CollisionTrial], measurements: list[CollisionMeasurement]) -> list[CollisionTrial]:
    keep = {m.name for m in measurements if m.selected}
    out = [t for t in trials if t.name in keep]
    for group in group_by_condition(out).values():          # re-rank repeats among the selected only
        for k, t in enumerate(group, start=1):
            t.repeat = k
    return out


def write_measurements_csv(measurements: list[CollisionMeasurement], path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for m in measurements:
            r = m.row()
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items() if k in CSV_COLUMNS})


def measurements_table(measurements: list[CollisionMeasurement]) -> list[str]:
    """Markdown rows, one per trial (all files, selection flag included)."""
    lines = ["| trial | condition | sel | valid / reason | t_c (s) | v_in (m/s) | v_out (m/s) | u_p (m/s) | gain | out angle (°) | Δy (cm) | rms pre / post (mm) | quality |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in measurements:
        f = lambda v, p=2: f"{v:.{p}f}" if np.isfinite(v) else "–"
        lines.append(f"| {m.name.replace('collision_', '')} | {m.condition} | {'**yes**' if m.selected else ''} | "
                     f"{'ok' if m.valid else m.reason} | {f(m.t_c, 3)} | {f(m.speed_in)} | {f(m.speed_out)} | {f(m.u_p)} | {f(m.gain)} | "
                     f"{f(m.angle_out_deg, 0)} | {f(100 * m.dy, 1)} | {f(m.rms_pre_mm, 0)} / {f(m.rms_post_mm, 0)} | "
                     f"{f(m.quality, 1) if np.isfinite(m.quality) else '–'} |")
    return lines
