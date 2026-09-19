"""Orchestrate a head-on paddle–puck collision in the Box2D env at prescribed speeds.

The identified parameters are the paddle–puck restitution ``e`` (the sim applies
``max(puck_restitution, paddle_restitution)`` in ``CollisionForceListener._presolve_paddle_puck``,
so **both** fixtures are set to ``e``) and the mass ratio ``r = m_paddle / m_puck``. The paddle
density stays at its fitted value (it is the PID plant inertia — see
``notes/scratch/experiments/2026-09-10_02-49_paddle-puck-mass-ratio.md``); the ratio is realised
through ``puck_density = m_paddle / (r · π · r_puck²)``.

Replication of one real collision (``HeadOnCollider.run``): the paddle starts at the real start
pose, already moving at the measured paddle speed ``u_p`` and driven by the sim's own PID with
the constant action that holds that speed (``action_for_speed``, calibrated once per plant); the
puck is placed in its path moving at the measured incoming speed ``u_k`` so that contact happens a
few ms into the next 50 ms env step; the env is stepped through the contact and a short
post-window, and the puck velocity at the end of the window is the sim's outgoing velocity.
Gravity and the puck's damping are off so the launch speed is read exactly; the paddle keeps
its damping (part of the plant). Secondary contacts (a light paddle that recoils and is pushed
back into the puck by the PID) are included by construction.

Closed form for a single impulse with a free paddle (what the sim's listener implements):

    v_out = −u_k + (1 + e) · (u_p + u_k) · r / (r + 1)

so head-on data only determines the product ``(1 + e) · r / (r + 1)`` — ``e`` and ``r`` are
degenerate along that ridge unless the paddle's post-impact motion is also matched.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from sysid.paddle_pid.code.replay import DEFAULT_BASE_CONFIG, build_replay_sim_config, load_base_config  # noqa: F401

_PUCK_PARK_BASE = (-0.9, 0.0)


@dataclass
class CollisionParams:
    restitution: float          # paddle–puck restitution e (applied to both fixtures)
    mass_ratio: float           # m_paddle / m_puck

    def as_dict(self) -> dict:
        return asdict(self)

    def gain(self) -> float:
        """Closed-form head-on speed gain (1 + e) · r / (r + 1)."""
        return (1.0 + self.restitution) * self.mass_ratio / (self.mass_ratio + 1.0)


def closed_form_out_speed(params: CollisionParams, u_p: float, u_k: float) -> float:
    """Outgoing puck speed for a single reduced-mass impulse between a free paddle (speed ``u_p``
    towards the puck) and the puck (speed ``u_k`` towards the paddle)."""
    return -u_k + params.gain() * (u_p + u_k)


def params_from_config(sim_cfg: dict) -> CollisionParams:
    """The (e, r) the env would use for a sim config's simulator_params."""
    sp = sim_cfg["air_hockey"]["simulator_params"] if "air_hockey" in sim_cfg else sim_cfg.get("simulator_params", sim_cfg)
    e = max(float(sp.get("puck_restitution", 1.0)), float(sp.get("paddle_restitution", 1.0)))
    r = (float(sp["paddle_density"]) / float(sp["puck_density"])) * (float(sp["paddle_radius"]) / float(sp["puck_radius"])) ** 2
    return CollisionParams(restitution=e, mass_ratio=r)


def puck_density_for_ratio(mass_ratio: float, paddle_density: float, paddle_radius: float, puck_radius: float) -> float:
    return float(paddle_density) * (float(paddle_radius) / float(puck_radius)) ** 2 / float(mass_ratio)


def build_collision_sim_config(base_config: dict, session_attrs: dict, hist_len: Optional[int] = None,
                               x_min_lim: float = -1.3, min_rmax_x: float = 0.4, x_max_lim: Optional[float] = None,
                               gravity: Optional[float] = None) -> dict:
    """``build_replay_sim_config`` (noise / delays / terminations off, gravity 0, controller limits
    from the recording) with two changes that only matter for this orchestrated collision: the
    paddle workspace is extended down the table (``x_min_lim``) so the paddle is not clamped by
    the robot's limit inside the short post-contact window, and the per-step x move limit is
    raised to at least ``min_rmax_x`` because the sim plant only reaches ~0.83 m/s with the
    robot's 0.26 m lead while the real strikes reach 1 m/s (0.4 m → ~1.27 m/s; the paddle must
    stay in the robot's half, so a larger lead runs out of table) (the action → speed map is
    calibrated by ``HeadOnCollider.action_for_speed``, so the plant dynamics are unchanged)."""
    cfg = build_replay_sim_config(base_config, session_attrs, hist_len=hist_len)
    sp = cfg["simulator_params"]
    sp["x_min_lim"] = float(x_min_lim)
    sp["rmax_x"] = max(float(sp.get("rmax_x", 0.26)), float(min_rmax_x))
    if x_max_lim is not None:                   # rendering only: let the paddle start beyond the robot's x_max
        sp["x_max_lim"] = float(x_max_lim)
    if gravity is not None:                     # rendering only: real table slope (the fit runs with gravity 0)
        sp["gravity"] = float(gravity)
    return cfg


class HeadOnCollider:
    """One Box2D env reused for every collision; (e, r) switchable between runs."""

    def __init__(self, sim_cfg: dict, paddle_start_x: float = 0.766, y: float = 0.0, pre_roll_steps: int = 2,
                 post_steps: int = 5, contact_tau: float = 0.02, settle_steps: int = 6):
        from airhockey import AirHockeyEnv           # local import: heavy
        self.sim_cfg = copy.deepcopy(sim_cfg)
        self.env = AirHockeyEnv(copy.deepcopy(sim_cfg))
        self.sim = self.env.simulator
        self.dt = float(self.sim.time_per_step)
        self.paddle_start_x, self.y = float(paddle_start_x), float(y)
        self.pre_roll_steps, self.post_steps = int(pre_roll_steps), int(post_steps)
        self.contact_tau, self.settle_steps = float(contact_tau), int(settle_steps)
        self.paddle_density = float(self.sim.paddle_density)
        self.paddle_radius, self.puck_radius = float(self.sim.paddle_radius), float(self.sim.puck_radius)
        self.paddle_mass = self.paddle_density * np.pi * self.paddle_radius ** 2
        self._action_cache: dict[float, float] = {}
        self._max_speed: Optional[float] = None
        self._params: Optional[CollisionParams] = None
        self.set_params(params_from_config(self.sim_cfg))

    # -- parameters ---------------------------------------------------------------------
    def set_params(self, params: CollisionParams) -> None:
        e, r = float(params.restitution), float(params.mass_ratio)
        self.sim.puck_restitution = e
        self.sim.paddle_restitution = e
        self.sim.puck_density = puck_density_for_ratio(r, self.paddle_density, self.paddle_radius, self.puck_radius)
        self.sim.puck_mass = self.sim.puck_density * np.pi * self.puck_radius ** 2
        self._params = CollisionParams(e, r)

    def current_params(self) -> CollisionParams:
        return CollisionParams(self._params.restitution, self._params.mass_ratio)

    @property
    def contact_distance(self) -> float:
        return self.paddle_radius + self.puck_radius

    # -- frames -------------------------------------------------------------------------
    def _paddle(self):
        return self.sim.paddles["paddle_ego"]

    def _puck(self):
        return self.sim.pucks["puck"] if "puck" in self.sim.pucks else next(iter(self.sim.pucks.values()))

    def _base(self, coord) -> np.ndarray:
        return self.sim._box2d_to_base_coords(coord)

    def _reset(self, u_p: float, paddle_start_x: Optional[float] = None, y: Optional[float] = None,
               puck_physics: bool = False) -> None:
        x0 = self.paddle_start_x if paddle_start_x is None else float(paddle_start_x)
        y0 = self.y if y is None else float(y)
        state = np.array([x0, y0, -u_p, 0.0, _PUCK_PARK_BASE[0], _PUCK_PARK_BASE[1], 0.0, 0.0])
        self.env.reset_from_state(state, seed=0)
        puck = self._puck()
        if not puck_physics:                    # the fit reads the launch speed exactly: no decay on the puck
            puck.linearDamping = 0.0
            puck.gravityScale = 0.0
        puck.linearVelocity = (0.0, 0.0)

    def _park_puck(self) -> None:
        puck = self._puck()
        puck.position = self.sim.base_coord_to_box2d(_PUCK_PARK_BASE)
        puck.linearVelocity = (0.0, 0.0)

    # -- paddle speed ↔ action --------------------------------------------------------
    def steady_speed(self, action_mag: float) -> float:
        """Paddle speed (towards −x, m/s) after ``settle_steps`` of the constant action from rest."""
        self._reset(0.0)
        for _ in range(self.settle_steps):
            self.env.step(np.array([-action_mag, 0.0]))
            self._park_puck()
        v = self._base(self._paddle().linearVelocity)
        return float(-v[0])

    def max_speed(self) -> float:
        if self._max_speed is None:
            self._max_speed = self.steady_speed(1.0)
        return self._max_speed

    def action_for_speed(self, u_p: float, tol: float = 2e-3) -> float:
        """Constant action magnitude whose steady paddle speed is ``u_p`` (bisection, cached)."""
        key = round(float(u_p), 3)
        if key in self._action_cache:
            return self._action_cache[key]
        if u_p <= 1e-6:
            self._action_cache[key] = 0.0
            return 0.0
        if u_p >= self.max_speed():
            self._action_cache[key] = 1.0
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            v = self.steady_speed(mid)
            if abs(v - u_p) < tol:
                lo = hi = mid
                break
            if v < u_p:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
        self._action_cache[key] = a
        return a

    # -- the collision ------------------------------------------------------------------
    def run(self, u_p: float, u_k: float, pre_roll_steps: Optional[int] = None, post_steps: Optional[int] = None,
            paddle_start_x: Optional[float] = None, y: Optional[float] = None, puck_launch: str = "teleport",
            puck_physics: bool = False, on_frame=None, dy: float = 0.0) -> dict:
        """Head-on collision: paddle at ``u_p`` towards the puck, puck at ``u_k`` towards the paddle.
        ``dy`` = lateral offset of the puck's lane from the paddle centre (base-frame y, m): the puck
        still travels straight along x and meets the paddle when the centres are ``contact_distance``
        apart, i.e. at an x-gap of ``sqrt(d² − dy²)``; the contact is then oblique (exit angle).
        Returns the outgoing puck velocity (base frame; ``speed_out`` = |v|, ``vx_out_away`` =
        the component away from the paddle), the paddle velocity at the end, contact count.

        The defaults are what the fit uses. ``pre_roll_steps`` / ``post_steps`` / ``paddle_start_x`` / ``y``
        override the timeline and lane (rendering: align the contact with a real recording);
        ``puck_launch="free"`` releases the puck at the start of the pre-roll, already moving at
        ``u_k``, at the distance that makes contact ``contact_tau`` into the step after the pre-roll
        (nothing acts on the puck before contact, so the collision itself is unchanged);
        ``puck_physics=True`` keeps the puck's damping and the world gravity (rendering with the real
        table slope; the free launch is then back-integrated so the puck still arrives at ``u_k``, and
        ``speed_out`` is read at the end of the contact step, i.e. up to one step of decay later);
        ``on_frame(j, paddle_xy, puck_xy)`` is called with base-frame positions after every step
        (``j = 0`` = the initial state)."""
        u_p, u_k = float(u_p), float(u_k)
        dy = float(dy)
        if abs(dy) >= self.contact_distance:
            raise ValueError(f"lateral offset {dy:.3f} m ≥ contact distance {self.contact_distance:.3f} m: no contact possible")
        gap_x = float(np.sqrt(self.contact_distance ** 2 - dy ** 2))     # centre x-gap at contact for this offset
        pre_roll = self.pre_roll_steps if pre_roll_steps is None else int(pre_roll_steps)
        post = self.post_steps if post_steps is None else int(post_steps)
        a = self.action_for_speed(u_p)
        act = np.array([-a, 0.0])
        self._reset(u_p, paddle_start_x, y, puck_physics)
        puck = self._puck()
        if puck_launch == "free":
            pad0 = self._base(self._paddle().position)
            T = pre_roll * self.dt + self.contact_tau
            x_c = pad0[0] - gap_x - u_p * T                           # puck centre x at contact (paddle surface ahead by gap_x)
            v0, travel = u_k, u_k * T
            if puck_physics:                                          # a = g − γ v back-integrated over T
                g, gam = -float(self.sim.gravity), float(self.sim.puck_damping)
                if gam > 1e-8:
                    v0 = (u_k - g / gam) * np.exp(gam * T) + g / gam
                    travel = (v0 - g / gam) * (1.0 - np.exp(-gam * T)) / gam + (g / gam) * T
                else:
                    v0 = u_k - g * T
                    travel = v0 * T + 0.5 * g * T ** 2
            puck.position = self.sim.base_coord_to_box2d((x_c - travel, pad0[1] + dy))
            puck.linearVelocity = self.sim.base_coord_to_box2d((v0, 0.0))
        if on_frame is not None:
            on_frame(0, self._base(self._paddle().position), self._base(puck.position))
        for j in range(pre_roll):
            self.env.step(act)
            if puck_launch != "free":
                self._park_puck()
            if on_frame is not None:
                on_frame(j + 1, self._base(self._paddle().position), self._base(puck.position))
        pad_xy = self._base(self._paddle().position)
        pad_v = self._base(self._paddle().linearVelocity)
        u_p_actual = float(-pad_v[0])
        u_k_actual = float(self._base(puck.linearVelocity)[0]) if puck_launch == "free" else u_k
        if puck_launch != "free":
            x_puck = pad_xy[0] - gap_x - (u_p_actual + u_k) * self.contact_tau
            puck.position = self.sim.base_coord_to_box2d((x_puck, pad_xy[1] + dy))
            puck.linearVelocity = self.sim.base_coord_to_box2d((u_k, 0.0))
            puck.angularVelocity = 0.0
        stats = self.sim.collision_listener._episode_stats["paddle"]
        n0 = sum(b["count"] for b in stats.values())
        n_prev = n0
        first_contact_step = -1
        v_after_contact = None
        for k in range(post):
            self.env.step(act)
            n_now = sum(b["count"] for b in self.sim.collision_listener._episode_stats["paddle"].values())
            if first_contact_step < 0 and n_now > n0:
                first_contact_step = k
            if n_now > n0 and (v_after_contact is None or n_now > n_prev):
                v_after_contact = self._base(puck.linearVelocity)      # end of the latest contact step
            n_prev = n_now
            if on_frame is not None:
                on_frame(pre_roll + k + 1, self._base(self._paddle().position), self._base(puck.position))
        v_puck = self._base(puck.linearVelocity)
        if puck_physics and v_after_contact is not None:
            v_puck = v_after_contact
        v_pad = self._base(self._paddle().linearVelocity)
        n_contacts = sum(b["count"] for b in self.sim.collision_listener._episode_stats["paddle"].values()) - n0
        return {"vx_out_away": float(-v_puck[0]), "vy_out": float(v_puck[1]), "speed_out": float(np.linalg.norm(v_puck)),
                "paddle_v_post": float(-v_pad[0]), "u_p_actual": u_p_actual, "u_k_actual": u_k_actual, "action": a,
                "n_contacts": int(n_contacts), "first_contact_step": int(first_contact_step),
                "pre_roll_steps": pre_roll, "post_steps": post, "dy": dy,
                "angle_out_deg": float(np.degrees(np.arctan2(v_puck[1], -v_puck[0]))),
                "paddle_vy_post": float(v_pad[1])}


# -- evaluation against measured collisions ----------------------------------------------
def evaluate_measurements(collider: HeadOnCollider, measurements, params: CollisionParams,
                          keep_details: bool = False) -> dict:
    """Replay every measurement's (u_p, speed_in) and score the outgoing speed.

    Aggregate = RMS over trials of (speed_out_sim − speed_out_real) in m/s; the mean signed
    error, the mean absolute relative error and per-condition RMS are reported alongside."""
    collider.set_params(params)
    per_trial = []
    for m in measurements:
        r = collider.run(m.u_p, m.speed_in)
        err = r["speed_out"] - m.speed_out
        row = {"name": m.name, "condition": m.condition, "repeat": m.repeat, "u_p": m.u_p, "speed_in": m.speed_in,
               "speed_out_real": m.speed_out, "speed_out_sim": r["speed_out"], "err": float(err),
               "rel_err": float(err / m.speed_out) if m.speed_out > 1e-6 else float("nan"),
               "closed_form": closed_form_out_speed(params, m.u_p, m.speed_in),
               "n_contacts": r["n_contacts"], "paddle_v_post_sim": r["paddle_v_post"]}
        if keep_details:
            row.update({k: r[k] for k in ("vx_out_away", "vy_out", "u_p_actual", "action", "first_contact_step")})
        per_trial.append(row)
    errs = np.array([p["err"] for p in per_trial]) if per_trial else np.zeros(0)
    per_cond: dict[str, list] = {}
    for p in per_trial:
        per_cond.setdefault(p["condition"], []).append(p["err"])
    per_cond = {c: {"rms": float(np.sqrt(np.mean(np.square(v)))), "mean_err": float(np.mean(v)), "n": len(v)} for c, v in per_cond.items()}
    return {"params": params.as_dict(), "gain": params.gain(), "n_trials": len(per_trial),
            "rms_err": float(np.sqrt(np.mean(errs ** 2))) if errs.size else float("nan"),
            "mean_err": float(errs.mean()) if errs.size else float("nan"),
            "mean_abs_rel_err": float(np.nanmean([abs(p["rel_err"]) for p in per_trial])) if per_trial else float("nan"),
            "max_abs_err": float(np.abs(errs).max()) if errs.size else float("nan"),
            "per_condition": per_cond, "per_trial": per_trial}
