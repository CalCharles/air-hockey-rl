"""Wall restitution sysid by replaying real bounces in Box2D.

For every ``WallBounce`` the puck is placed in the simulator at the real
pre-impact state — position and velocity of the damped-model fit (with the
identified g, γ) evaluated at the last clean frame before impact — and the
sim is stepped with a static paddle until the puck leaves the wall. The exit
velocity is compared with the real post-impact velocity (post-window fit
evaluated at the first clean frame after impact). Side walls (y±) use
``side_wall_restitution`` and end walls (x±) ``end_wall_restitution``; each is
swept independently on the train bounces and evaluated on the held-out ones.

Four error metrics are computed per sweep value (mean over the bounces the
sim reproduced):

* ``speed_err``       |exit speed sim − real|                       [m/s]
* ``normal_err``      |normal exit speed sim − real|                [m/s]
* ``speed_rel_err``   speed_err  / real exit speed                  [fraction]
* ``normal_rel_err``  normal_err / real exit speed                  [fraction]
* ``angle_err``       |exit direction sim − real|                   [deg]

The relative metrics weigh fast and slow bounces equally; ``speed_rel_err``
is the default selection objective. Both relative errors are normalised by
the *total* real exit speed (not the normal component) so glancing bounces
cannot blow up the ratio.

How the simulator applies restitution (``CollisionForceListener`` in
``airhockey/sims/airhockey_box2d.py``): a puck–wall contact uses the *wall
fixture's* restitution directly (not Box2D's mixing rule and not the puck's
``puck_restitution``); PostSolve then adds an impulse so that the outgoing
normal speed equals ``incoming_normal_speed × wall_restitution`` (for incoming
normal speeds ≥ ``puck_wall_restitution_threshold_speed`` = 0.25 m/s; below
that a fixed 0.1 m/s rebound is enforced). So the exit-speed error is a direct
function of the wall value being swept.

Frames: bounces come in the sim frame (x long axis, robot at x < 0); the
Box2D env's ``reset_from_state`` takes "base" coordinates (robot at x > 0),
i.e. ``x_base = −x_sim``.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airhockey import AirHockeyEnv  # noqa: E402

from sysid.common.fit_validation import percentile_report  # noqa: E402
from sysid.common.sysid_dataset import WallBounce  # noqa: E402
from sysid.common.trajectory_segmentation import SegmentationConfig, fit_damped, model_state  # noqa: E402

_NOISE_SIM_KEYS = ("puck_noise", "enable_random_occlusions", "enable_observation_delay",
                   "enable_action_delay", "enable_action_force_attenuation",
                   "enable_puck_delay_interpolation")
_TERMINATION_KEYS = ("terminate_on_enemy_goal", "terminate_on_puck_hit_bottom",
                     "terminate_on_puck_pass_paddle", "terminate_on_puck_stop",
                     "terminate_on_out_of_bounds", "terminate_on_puck_hit_paddle")

SIDE_WALLS = ("y+", "y-")
END_WALLS = ("x+", "x-")

WALL_METRICS = ("speed_err", "normal_err", "speed_rel_err", "normal_rel_err", "angle_err")
WALL_METRIC_LABEL = {"speed_err": "exit speed err", "normal_err": "normal exit speed err",
                     "speed_rel_err": "exit speed err / real speed", "normal_rel_err": "normal exit speed err / real speed",
                     "angle_err": "exit angle err"}
WALL_METRIC_UNIT = {"speed_err": "m/s", "normal_err": "m/s", "speed_rel_err": "fraction", "normal_rel_err": "fraction", "angle_err": "deg"}


def load_clean_sim_config(sim_config_path: str | Path, overrides: dict | None = None) -> dict:
    """Load a sim YAML with noise / termination off and simulator_params overrides applied."""
    with open(sim_config_path) as f:
        cfg = yaml.safe_load(f)
    ah = cfg["air_hockey"]
    sim = ah.setdefault("simulator_params", {})
    for k in _NOISE_SIM_KEYS:
        if k in sim:
            sim[k] = False
    sim["random_occlusion_rate"] = 0.0
    for k in _TERMINATION_KEYS:
        if k in ah:
            ah[k] = False
    ah["max_timesteps"] = 10_000
    for k, v in (overrides or {}).items():
        sim[k] = v
    return ah


def build_env(sim_config_path, overrides: dict | None = None) -> AirHockeyEnv:
    return AirHockeyEnv(copy.deepcopy(load_clean_sim_config(sim_config_path, overrides)))


@dataclass
class BounceState:
    """Real pre / post impact state of one bounce under the identified (g, γ).

    The sim is started from the fitted state at the *first* pre-window frame
    (``p_start``, ``v_start``) rather than at the last one, so Box2D gets a
    run-up and a real puck position that already sits past the sim's wall
    line (camera scale / table-width mismatch) does not start inside the wall.
    """
    wall: str
    p_a: np.ndarray       # fitted position at t_a, last clean pre-impact frame (sim frame)
    v_a: np.ndarray       # fitted velocity at t_a
    v_b: np.ndarray       # fitted velocity at t_b, first clean post-impact frame (exit)
    p_start: np.ndarray   # fitted state at the first pre-window frame
    v_start: np.ndarray
    steps_to_b: int       # frames from the start frame to b
    n_steps: int          # frames between a and b
    dt_real: float
    normal: np.ndarray    # outward wall normal (sim frame)
    rms_pre_cm: float
    rms_post_cm: float
    apex: float           # furthest measured puck-centre coordinate along the normal near impact

    @property
    def normal_in(self) -> float:
        return float(self.v_a @ self.normal)

    @property
    def normal_out_real(self) -> float:
        return float(-(self.v_b @ self.normal))

    @property
    def speed_out_real(self) -> float:
        return float(np.linalg.norm(self.v_b))


_NORMALS = {"x+": np.array([1.0, 0.0]), "x-": np.array([-1.0, 0.0]),
            "y+": np.array([0.0, 1.0]), "y-": np.array([0.0, -1.0])}


def bounce_state(bounce: WallBounce, cfg: SegmentationConfig, max_side_frames: int = 10) -> BounceState:
    pre = bounce.pre_idx[-max_side_frames:]
    post = bounce.post_idx[:max_side_frames]
    t_a, t_b = bounce.t[pre[-1]], bounce.t[post[0]]
    fpre = fit_damped(bounce.t[pre] - t_a, bounce.xy[pre], cfg)
    fpost = fit_damped(bounce.t[post] - t_b, bounce.xy[post], cfg)
    p_a, v_a = model_state(fpre, 0.0, cfg)
    _, v_b = model_state(fpost, 0.0, cfg)
    t_start = bounce.t[pre[0]]
    p_s, v_s = model_state(fpre, t_start - t_a, cfg)
    normal = _NORMALS[bounce.wall]
    near = np.concatenate([pre[-2:], post[:2]])
    apex = float(np.max(bounce.xy[near] @ normal))
    return BounceState(bounce.wall, p_a, v_a, v_b, p_s, v_s,
                       int(bounce.usable_idx[post[0]] - bounce.usable_idx[pre[0]]), int(bounce.b - bounce.a),
                       float(t_b - t_a), normal, 100 * fpre["rms"], 100 * fpost["rms"], apex)


def _to_base(xy: np.ndarray) -> np.ndarray:
    return np.array([-xy[0], xy[1]], dtype=np.float64)


def replay_bounce(env: AirHockeyEnv, st: BounceState, paddle_xy_sim: np.ndarray, extra_steps: int = 6) -> dict:
    """Place the puck at the real (fitted) state of the first pre-window frame
    and step with a static paddle until it leaves the wall.

    Returns exit velocity (sim frame) at the first step after the normal
    component reverses, the step index of that reversal, and the velocity at
    the step matching the real post frame for reference. ``bounced`` is False
    if no reversal happened within steps_to_b + extra_steps. A start position
    already past the sim wall line is pulled back to 5 mm inside it.
    """
    sim = env.simulator
    lim = (0.5 * sim.length if st.wall[0] == "x" else 0.5 * sim.width) - sim.puck_radius - 0.005
    p0 = st.p_start.copy()
    along = float(p0 @ st.normal)
    if along > lim:
        p0 = p0 - (along - lim) * st.normal
    state0 = np.concatenate([_to_base(paddle_xy_sim), np.zeros(2), _to_base(p0), _to_base(st.v_start)])
    env.reset_from_state(state0)
    zero = np.zeros(2, dtype=np.float32)
    vels, poss = [], []
    for _ in range(st.steps_to_b + extra_steps):
        env.step(zero)
        pk = env.current_state["pucks"][0]
        v = np.asarray(pk["velocity"][:2], dtype=np.float64)
        p = np.asarray(pk["position"][:2], dtype=np.float64)
        vels.append(np.array([-v[0], v[1]])); poss.append(np.array([-p[0], p[1]]))
    vels = np.asarray(vels); poss = np.asarray(poss)
    n_comp = vels @ st.normal
    k = next((i for i in range(len(vels)) if n_comp[i] < 0), None)
    out = {"bounced": k is not None, "bounce_step": k, "vels": vels, "poss": poss,
           "v_exit": vels[k] if k is not None else vels[-1],
           "v_at_b": vels[min(st.steps_to_b, len(vels)) - 1]}
    return out


@dataclass
class WallSweepResult:
    wall_kind: str                 # "side" or "end"
    param: str                     # config key
    objective: str                 # metric minimised on train (one of WALL_METRICS)
    values: np.ndarray
    train: dict                    # metric -> array over values (mean over reproduced train bounces)
    val: dict
    best: float
    canonical: float
    n_train: int
    n_val: int
    n_reproduced_train: int        # bounces the sim reproduced at the best value
    per_bounce: dict = field(default_factory=dict)   # value -> train rows (real_speed_out, sim_speed_out, real_n_out, sim_n_out, bounced, bounce_step, n_steps, real_angle, sim_angle)
    validation: dict = field(default_factory=dict)   # metric -> PercentileReport over the val curve
    per_bounce_val: dict = field(default_factory=dict)   # value -> val rows, same columns

    @property
    def best_index(self) -> int:
        return int(np.argmin(np.abs(self.values - self.best)))

    @property
    def canonical_index(self) -> int:
        return int(np.argmin(np.abs(self.values - self.canonical)))

    def to_json(self) -> dict:
        return {"wall_kind": self.wall_kind, "param": self.param, "objective": self.objective, "values": self.values.tolist(),
                "train": {m: a.tolist() for m, a in self.train.items()}, "val": {m: a.tolist() for m, a in self.val.items()},
                "best": self.best, "canonical": self.canonical, "n_train": self.n_train, "n_val": self.n_val,
                "n_reproduced_train": self.n_reproduced_train, "best_index": self.best_index, "canonical_index": self.canonical_index,
                "validation": {m: r.to_json() for m, r in self.validation.items()}}


def _eval_env(env, states: list[BounceState], paddles: list[np.ndarray]) -> tuple[dict, np.ndarray]:
    """Replay every bounce in ``env``; returns ({metric: mean over reproduced
    bounces}, rows)."""
    rows = []
    for st, pad in zip(states, paddles):
        r = replay_bounce(env, st, pad)
        v = r["v_exit"]
        rows.append((st.speed_out_real, float(np.linalg.norm(v)), st.normal_out_real, float(-(v @ st.normal)), r["bounced"], r["bounce_step"], st.n_steps,
                     float(np.arctan2(st.v_b[1], st.v_b[0])), float(np.arctan2(v[1], v[0]))))
    rows = np.array(rows, dtype=object)
    nan = float("nan")
    if len(rows) == 0:
        return {m: nan for m in WALL_METRICS}, rows
    ok = np.array([bool(x) for x in rows[:, 4]])
    if ok.sum() == 0:
        return {m: nan for m in WALL_METRICS}, rows
    real_s, sim_s = rows[ok, 0].astype(float), rows[ok, 1].astype(float)
    real_n, sim_n = rows[ok, 2].astype(float), rows[ok, 3].astype(float)
    denom = np.maximum(real_s, 1e-3)
    dang = np.abs(_wrap_angle(rows[ok, 8].astype(float) - rows[ok, 7].astype(float)))
    return {"speed_err": float(np.mean(np.abs(real_s - sim_s))),
            "normal_err": float(np.mean(np.abs(real_n - sim_n))),
            "speed_rel_err": float(np.mean(np.abs(real_s - sim_s) / denom)),
            "normal_rel_err": float(np.mean(np.abs(real_n - sim_n) / denom)),
            "angle_err": float(np.degrees(np.mean(dang)))}, rows


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def sweep_wall(kind: str, sim_config_path, base_overrides: dict,
               train: list[tuple[BounceState, np.ndarray]], val: list[tuple[BounceState, np.ndarray]],
               values: np.ndarray, objective: str = "speed_rel_err") -> WallSweepResult:
    """Sweep the wall restitution over ``values``; the train minimum of
    ``objective`` is selected, every metric is recorded on train and val, and
    the val curve of every metric gets a percentile report."""
    assert objective in WALL_METRICS, objective
    param = "side_wall_restitution" if kind == "side" else "end_wall_restitution"
    canonical = load_clean_sim_config(sim_config_path)["simulator_params"][param]
    tr = {m: [] for m in WALL_METRICS}; va = {m: [] for m in WALL_METRICS}
    per, per_val = {}, {}
    for v in values:
        env = build_env(sim_config_path, {**base_overrides, param: float(v)})
        e, rows = _eval_env(env, [s for s, _ in train], [p for _, p in train])
        for m in WALL_METRICS:
            tr[m].append(e[m])
        per[float(v)] = rows
        e, rows_v = _eval_env(env, [s for s, _ in val], [p for _, p in val]) if val else ({m: float("nan") for m in WALL_METRICS}, np.zeros((0, 9), dtype=object))
        per_val[float(v)] = rows_v
        for m in WALL_METRICS:
            va[m].append(e[m])
        env.close() if hasattr(env, "close") else None
    tr = {m: np.array(a) for m, a in tr.items()}; va = {m: np.array(a) for m, a in va.items()}
    ib = int(np.nanargmin(tr[objective])); best = float(values[ib])
    n_rep = int(sum(bool(x) for x in per[best][:, 4])) if len(per[best]) else 0
    res = WallSweepResult(kind, param, objective, np.asarray(values, dtype=float), tr, va, best, float(canonical), len(train), len(val), n_rep, per,
                          per_bounce_val=per_val)
    if val:
        ic = int(np.argmin(np.abs(values - canonical)))
        for m in WALL_METRICS:
            res.validation[m] = percentile_report(f"{kind} walls {WALL_METRIC_LABEL[m]}", WALL_METRIC_UNIT[m], va[m], float(va[m][ib]),
                                                  lambda k: {param: float(values[k])}, canonical_err=float(va[m][ic]),
                                                  train_errors=tr[m], selected_train_err=float(tr[m][ib]), canonical_train_err=float(tr[m][ic]))
    return res


def plot_sweeps(results: list[WallSweepResult], out_path):
    """Per wall kind, 2 × 3 panels. Top: absolute sweep curves (train / val,
    speed + normal), sim-vs-real exit-speed scatter at the selected value,
    exit-angle sweep curve. Bottom: relative sweep curves, per-bounce relative
    error vs real exit speed, sim-vs-real exit-angle scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(2, 3 * n, figsize=(16.5 * n, 8.6), squeeze=False)

    def curve(ax, r, m_a, m_b, unit, title):
        ax.plot(r.values, r.train[m_a], "o-", color="tab:blue", label=f"train {WALL_METRIC_LABEL[m_a]} (n={r.n_train})")
        ax.plot(r.values, r.val[m_a], "s--", color="tab:blue", alpha=0.7, label=f"val {WALL_METRIC_LABEL[m_a]} (n={r.n_val})")
        if m_b is not None:
            ax.plot(r.values, r.train[m_b], "o-", color="tab:green", ms=3, lw=0.8, label=f"train {WALL_METRIC_LABEL[m_b]}")
            ax.plot(r.values, r.val[m_b], "s--", color="tab:green", ms=3, lw=0.8, alpha=0.7, label=f"val {WALL_METRIC_LABEL[m_b]}")
        ax.axvline(r.best, color="r", ls=":", label=f"selected {r.best:.3f} ({r.objective} on train)")
        ax.axvline(r.canonical, color="gray", ls=":", label=f"canonical {r.canonical:.3f}")
        ax.set_xlabel(r.param); ax.set_ylabel(f"mean |sim − real| [{unit}]"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10)

    for k, r in enumerate(results):
        c = 3 * k
        curve(axes[0, c], r, "speed_err", "normal_err", "m/s", f"{r.wall_kind} walls — absolute speed error")
        curve(axes[1, c], r, "speed_rel_err", "normal_rel_err", "fraction of real exit speed", f"{r.wall_kind} walls — relative speed error")
        curve(axes[0, c + 2], r, "angle_err", None, "deg", f"{r.wall_kind} walls — exit angle error")
        rows = r.per_bounce[float(r.best)]
        ok = np.array([bool(x) for x in rows[:, 4]]) if len(rows) else np.zeros(0, bool)
        real_s, sim_s = rows[ok, 0].astype(float), rows[ok, 1].astype(float)
        real_n, sim_n = rows[ok, 2].astype(float), rows[ok, 3].astype(float)
        real_a, sim_a = np.degrees(rows[ok, 7].astype(float)), np.degrees(rows[ok, 8].astype(float))
        ax2 = axes[0, c + 1]
        ax2.scatter(real_s, sim_s, s=16, label="exit speed")
        ax2.scatter(real_n, sim_n, s=16, marker="x", label="normal exit speed")
        lim = max(1e-3, float(np.nanmax(real_s)) if len(real_s) else 1.0) * 1.1
        ax2.plot([0, lim], [0, lim], "k:", lw=0.8)
        ax2.set_xlabel("real [m/s]"); ax2.set_ylabel(f"sim @ {r.param}={r.best:.3f} [m/s]"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
        ax2.set_title(f"{r.wall_kind} walls, train bounces ({ok.sum()}/{len(rows)} reproduced)", fontsize=10)
        ax3 = axes[1, c + 1]
        if len(real_s):
            ax3.scatter(real_s, (sim_s - real_s) / np.maximum(real_s, 1e-3), s=16, label="(sim − real) / real, exit speed")
            ax3.scatter(real_s, (sim_n - real_n) / np.maximum(real_s, 1e-3), s=16, marker="x", label="(sim − real) / real, normal component")
        ax3.axhline(0, color="k", lw=0.8, ls=":")
        ax3.set_xlabel("real exit speed [m/s]"); ax3.set_ylabel("relative error"); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)
        ax3.set_title(f"{r.wall_kind} walls, per-bounce relative error @ {r.best:.3f}", fontsize=10)
        ax4 = axes[1, c + 2]
        if len(real_a):
            ax4.scatter(real_a, sim_a, s=16)
            ax4.plot([-180, 180], [-180, 180], "k:", lw=0.8)
        ax4.set_xlabel("real exit angle [deg, sim frame]"); ax4.set_ylabel(f"sim exit angle @ {r.best:.3f} [deg]"); ax4.grid(alpha=0.3)
        ax4.set_title(f"{r.wall_kind} walls, exit direction (train)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def _ok_rows(rows: np.ndarray) -> np.ndarray:
    if rows is None or len(rows) == 0:
        return np.zeros((0, 9), dtype=object)
    ok = np.array([bool(x) for x in rows[:, 4]])
    return rows[ok]


def plot_paper_figures(results: list[WallSweepResult], out_dir, kind: str = "side", stem: str = "wall_side") -> list:
    """The individual figures kept for the paper (PNG + PDF each), for one
    wall kind (default the side walls):
    ``<stem>_exit_speed_sweep``    mean |exit speed sim − real| vs restitution, train and val;
    ``<stem>_exit_speed_scatter``  sim vs real exit speed at the selected value, train and val bounces;
    ``<stem>_exit_angle_scatter``  sim vs real exit angle [deg] at the selected value, train and val."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    r = next((x for x in results if x.wall_kind == kind), None)
    if r is None:
        return []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=150)
        plt.close(fig); written.append(out_dir / f"{name}.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(r.values, r.train["speed_err"], "o-", color="tab:blue", label=f"train ({r.n_train} bounces)")
    ax.plot(r.values, r.val["speed_err"], "s--", color="tab:red", label=f"validation ({r.n_val} bounces)")
    ax.axvline(r.best, color="k", ls=":", lw=1, label=f"selected {r.best:.3f}")
    ax.axvline(r.canonical, color="gray", ls="--", lw=1, label=f"canonical {r.canonical:.3f}")
    ax.set_xlabel(r.param); ax.set_ylabel("mean |exit speed sim − real| [m/s]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{kind} walls — exit-speed error vs restitution", fontsize=10)
    save(fig, f"{stem}_exit_speed_sweep")

    tr = _ok_rows(r.per_bounce.get(float(r.best))); va = _ok_rows(r.per_bounce_val.get(float(r.best)))
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    if len(tr):
        ax.scatter(tr[:, 0].astype(float), tr[:, 1].astype(float), s=18, color="tab:blue", label=f"train ({len(tr)})")
    if len(va):
        ax.scatter(va[:, 0].astype(float), va[:, 1].astype(float), s=22, marker="s", color="tab:red", label=f"validation ({len(va)})")
    allr = np.concatenate([tr[:, 0].astype(float), va[:, 0].astype(float)]) if len(tr) + len(va) else np.array([1.0])
    lim = float(np.nanmax(allr)) * 1.1
    ax.plot([0, lim], [0, lim], "k:", lw=0.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("real exit speed [m/s]"); ax.set_ylabel(f"sim exit speed @ {r.param}={r.best:.3f} [m/s]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{kind} walls — exit speed, sim vs real", fontsize=10)
    save(fig, f"{stem}_exit_speed_scatter")

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    if len(tr):
        ax.scatter(np.degrees(tr[:, 7].astype(float)), np.degrees(tr[:, 8].astype(float)), s=18, color="tab:blue", label=f"train ({len(tr)})")
    if len(va):
        ax.scatter(np.degrees(va[:, 7].astype(float)), np.degrees(va[:, 8].astype(float)), s=22, marker="s", color="tab:red", label=f"validation ({len(va)})")
    ax.plot([-180, 180], [-180, 180], "k:", lw=0.8); ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
    ax.set_xlabel("real exit angle [deg]"); ax.set_ylabel(f"sim exit angle @ {r.param}={r.best:.3f} [deg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{kind} walls — exit angle, sim vs real", fontsize=10)
    save(fig, f"{stem}_exit_angle_scatter")
    return written
