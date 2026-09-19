"""Shared trial plan for the paddle-motion data-collection batteries.

Both ``collect_paddle_motion.py`` (real UR5) and ``collect_paddle_motion_sim.py``
(Box2D) build their trial list from here, so a sim session and a real session with
the same flags produce the same conditions in the same order --- which is what
makes the two sets of trajectories comparable.

Frame conventions (ROBOT frame, matching AirHockeyReal):

    +x  toward the robot base  = "bottom" of the vertical render (x_max_lim)
    -x  toward table centre    = "top"    (x_min_lim), the strike direction
    +y  = "right" of the vertical render (y_max)
    -y  = "left"  (y_min)

A direction is a 2D unit-ish vector in ACTION space, and the trial's action is
just ``delta * vec``. Single-axis directions leave the other component at zero;
the diagonals drive both axes at the same delta.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from airhockey.sims.real.coordinate_transform import (
    CORNER_CUT_Y_EXTENT,
    corner_cut_biases,
    effective_x_max,
)

AXIS_INDEX = {"x": 0, "y": 1}


@dataclass(frozen=True)
class Direction:
    """One travel direction: a label plus a sign vector in action space."""

    key: str
    vec: tuple[int, int]
    description: str

    @property
    def axis(self) -> str:
        """"x" / "y" for single-axis directions, "xy" for the diagonals."""
        if self.vec[0] and self.vec[1]:
            return "xy"
        return "x" if self.vec[0] else "y"

    @property
    def sign(self) -> int:
        """Sign for single-axis directions; 0 for diagonals (see ``vec``)."""
        return 0 if self.axis == "xy" else (self.vec[0] or self.vec[1])


# Registry. The two diagonals both travel toward the top of the table, and both
# start from the bottom (see start_pose_for), which is what makes them
# "bottom-left -> top-right" and "bottom-right -> top-left".
DIRECTIONS = {
    "xpos": Direction("xpos", (1, 0), "vertical, toward the robot base (bottom)"),
    "xneg": Direction("xneg", (-1, 0), "vertical, toward table centre (top) -- the strike direction"),
    "ypos": Direction("ypos", (0, 1), "horizontal, toward +y (right)"),
    "yneg": Direction("yneg", (0, -1), "horizontal, toward -y (left)"),
    "diagpos": Direction("diagpos", (-1, 1), "positive diagonal: bottom-left -> top-right"),
    "diagneg": Direction("diagneg", (-1, -1), "negative diagonal: bottom-right -> top-left"),
}

# Accepted --directions tokens. The signed-axis spellings are the originals; the
# diagonals take either a name or an explicit component pair.
DIRECTION_ALIASES = {
    "+x": "xpos", "x": "xpos", "xpos": "xpos",
    "-x": "xneg", "xneg": "xneg",
    "+y": "ypos", "y": "ypos", "ypos": "ypos",
    "-y": "yneg", "yneg": "yneg",
    "diagpos": "diagpos", "diag+": "diagpos", "+diag": "diagpos",
    "-x+y": "diagpos", "+y-x": "diagpos",
    "diagneg": "diagneg", "diag-": "diagneg", "-diag": "diagneg",
    "-x-y": "diagneg", "-y-x": "diagneg",
}

DELTA_MODES = ("action", "workspace")
# delta IS the action: the magnitude put on every axis the direction travels.
DEFAULT_DELTA_MODE = "action"

@dataclass(frozen=True)
class Curve:
    """A half-arc trajectory: a semi-ellipse whose chord lies along the bottom edge.

    ``half_width_m`` (a) is the semi-axis along y, ``height_m`` (b) the bulge along
    -x (up the table). The paddle starts at ``(x_base, yc - a)``, peaks at
    ``(x_base - b, yc)`` and finishes at ``(x_base, yc + a)`` --- so it sweeps left
    to right along the bottom of the table with one smooth rise and fall.

    ``b / a`` is the character of the curve: << 1 is a wide mild bow, 1 a rounded
    半 circle, >> 1 a tall narrow loop. None of the three shipped shapes is near a
    straight line in either direction.
    """

    key: str
    half_width_m: float
    height_m: float
    description: str
    # Speed sweep for THIS shape: the PATH speed the paddle is asked to hold along
    # the arc, in m/s. Per-shape rather than global because the axes have very
    # different authority -- the arm tops out near 0.39 m/s on y but 0.83 m/s on x
    # (see CURVE_VELOCITY_GAIN_REAL) -- so a y-dominated mild bow saturates at a
    # lower path speed than a tall narrow one that spends its time moving in x.
    #
    # The three values span 2x and are anchored at the top by what the arm can
    # actually hold: the fastest one saturates the y axis around the apex, and is
    # therefore the widest sweep the arm can produce in 20 steps. They are NOT
    # spread down to a crawl on purpose -- with the step count fixed at 20, a
    # slower path speed simply means less of the arc gets covered, and the point
    # of these trials is to cover as much of a full-width arc as possible.
    speeds_m_s: tuple[float, ...] = (0.15, 0.30, 0.50)

    @property
    def aspect(self) -> float:
        return self.height_m / max(self.half_width_m, 1e-9)


CURVES = {
    # arc_wide's half-width is already the workspace maximum: 0.36 m is half the
    # 0.74 m y span, less the boundary margin. It cannot be made wider.
    "arc_wide": Curve("arc_wide", 0.36, 0.16,
                      "mild half arc, bottom corner to bottom corner",
                      speeds_m_s=(0.22, 0.32, 0.45)),
    "arc_medium": Curve("arc_medium", 0.33, 0.26,
                        "rounder half arc, starts nearer the bottom centre",
                        speeds_m_s=(0.24, 0.35, 0.50)),
    "arc_tight": Curve("arc_tight", 0.30, 0.36,
                       "tall half arc, starts nearest the bottom centre",
                       speeds_m_s=(0.28, 0.42, 0.60)),
}

DEFAULT_CURVES = "arc_wide,arc_medium,arc_tight"
# Speeds for the curves, in m/s of paddle travel along the arc. Only used when
# --curve-speeds is given explicitly; otherwise each shape uses its own sweep
# (see CURVES above).
DEFAULT_CURVE_SPEEDS = None
# Nominal control rate. Used to convert a path speed into a per-step displacement.
DEFAULT_STEP_HZ = 20.0
# Floor on steps per arc for the legacy open-loop schedule, so a fast tight curve
# stays a curve rather than a two-point corner.
MIN_CURVE_STEPS = 6

# ---------------------------------------------------------------------------
# What an action actually does to the paddle
# ---------------------------------------------------------------------------
# The action is NOT a per-step displacement. Both simulators rebuild the command
# target from the CURRENT pose every step --
#
#     target = pose_now + action * move_lims        (AirHockeyReal.get_transition)
#
# -- and hand it to a tracking controller (servoL on the UR5, a PID in Box2D), so
# the command never integrates. What the action sets is a constant lead distance,
# and a constant lead settles at a constant SPEED:
#
#     v_realised ~= CURVE_VELOCITY_GAIN * (action * move_lims)      [m/s]
#
# The action is therefore a velocity command, and the realised displacement per
# step is only ``gain * dt`` of the commanded one. Measured (see
# notes/scratch/experiments/):
#
#   real UR5   servoL lookahead=0.2 gain=700, dt~0.048 s
#              y: 0.386 m/s at action 1.0 -> G = 0.386/0.12 = 3.22 1/s
#              x: 0.82  m/s at action 1.0 -> G = 0.82 /0.26 = 3.15 1/s
#              => G ~= 3.2, i.e. only ~15% of the commanded step is realised.
#   Box2D      dt = 0.05 s, sysid PID -> ~50% of the commanded step is realised,
#              i.e. G ~= 10 1/s.
#
# This is the bug the closed-loop tracker below exists to fix: the old open-loop
# schedule wrote ``action = displacement / move_lims``, which is only true if the
# paddle reaches its target within one step. In Box2D (G*dt = 0.5) a flat
# ``tracking_gain = 2.0`` papered over it and the arcs looked full width; on the
# robot (G*dt = 0.15) the same schedule traced barely a third of the table.
CURVE_VELOCITY_GAIN_REAL = 3.2
CURVE_VELOCITY_GAIN_SIM = 10.0
# How far ahead of the paddle's own projection onto the arc the carrot sits, in
# multiples of one step of travel (speed * dt). ~2 steps is enough to keep the
# heading smooth without cutting the corner at the apex.
DEFAULT_CURVE_LOOKAHEAD_STEPS = 2.0
# Floor on the carrot distance so the heading stays well-defined when the paddle
# is nearly stopped (start of the sweep, or a speed low enough that speed*dt is
# comparable to the position noise).
MIN_CURVE_LOOKAHEAD_M = 0.02
DEFAULT_CURVE_TRACKING = "closed-loop"
CURVE_TRACKING_MODES = ("closed-loop", "open-loop")
# Legacy open-loop schedule only (--curve-tracking open-loop). Kept so the
# pre-fix behaviour is still reproducible for comparison; see the note above for
# why 2.0 is a Box2D number that does not transfer to the robot.
DEFAULT_CURVE_TRACKING_GAIN = 2.0

# ---------------------------------------------------------------------------
# Closed-loop arc tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveTrack:
    """Everything the closed-loop arc tracker needs, resolved at plan time.

    Carried on the ``Trial`` so both collectors can build an identical tracker
    from the saved plan; the per-step actions themselves are computed online from
    the measured paddle pose, not baked in.
    """

    x_base: float
    y_centre: float
    half_width_m: float
    height_m: float
    speed_m_s: float
    move_lims: tuple[float, float]
    velocity_gain: float
    dt: float
    n_steps: int
    lookahead_steps: float = DEFAULT_CURVE_LOOKAHEAD_STEPS

    @property
    def arc_length_m(self) -> float:
        return semi_ellipse_arc_length(self.half_width_m, self.height_m)

    def tracker(self) -> "ArcTracker":
        return ArcTracker(self)


class ArcTracker:
    """Pure-pursuit follower for one half-ellipse, in ROBOT frame.

    Each step it projects the MEASURED paddle position onto the arc, places a
    carrot a short distance further along, and commands the action whose realised
    velocity (``v = velocity_gain * action * move_lims``, see the note above)
    points at that carrot at the requested path speed::

        action = clip( speed * unit(carrot - p) / (velocity_gain * move_lims), -1, 1 )

    Two properties matter here, and neither holds for the old open-loop schedule:

    * The traced path is the arc itself, whatever the arm's real tracking gain
      turns out to be -- a mis-estimated ``velocity_gain`` changes how FAST the
      sweep goes, not what shape it is, because the carrot is re-derived from the
      measured pose every step.
    * When the requested speed is past what the axis can deliver, the action
      simply saturates at 1.0 and the paddle sweeps as far along the arc as it
      can in the timesteps available. That is the intended behaviour for the wide
      arcs: 20 steps at 20 Hz is not enough time to cross the whole table, so the
      trial covers the first part of the full-width arc rather than covering all
      of a shrunken one.
    """

    # Polyline resolution. 800 samples puts the discretisation error on the
    # projection well under a millimetre for the widest arc.
    SAMPLES = 800

    def __init__(self, track: CurveTrack):
        self.track = track
        a, b = float(track.half_width_m), float(track.height_m)
        thetas = np.linspace(-np.pi / 2.0, np.pi / 2.0, self.SAMPLES)
        self.points = np.stack(
            [track.x_base - b * np.cos(thetas), track.y_centre + a * np.sin(thetas)],
            axis=1,
        )
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.total_s = float(self.s[-1])
        self.move_lims = np.asarray(track.move_lims, dtype=float)
        self.dt = float(track.dt)
        self.speed = float(track.speed_m_s)
        self.velocity_gain = float(track.velocity_gain)
        self.lookahead_m = max(
            float(MIN_CURVE_LOOKAHEAD_M),
            float(track.lookahead_steps) * self.speed * self.dt,
        )
        # Monotone progress: the projection is only ever allowed to advance, so a
        # paddle that swings wide near the apex cannot be dragged back to the
        # matching point on the rising half.
        self._i = 0

    # -- geometry -----------------------------------------------------------
    def _project(self, p: np.ndarray) -> int:
        """Index of the nearest arc sample at or ahead of the last projection."""
        d = np.linalg.norm(self.points[self._i:] - p, axis=1)
        self._i += int(np.argmin(d))
        return self._i

    def _point_at(self, s_target: float) -> np.ndarray:
        s_target = float(np.clip(s_target, 0.0, self.total_s))
        j = int(np.searchsorted(self.s, s_target, side="right")) - 1
        j = int(np.clip(j, 0, len(self.s) - 2))
        span = self.s[j + 1] - self.s[j]
        frac = 0.0 if span <= 1e-12 else (s_target - self.s[j]) / span
        return self.points[j] + frac * (self.points[j + 1] - self.points[j])

    # -- control ------------------------------------------------------------
    def action(self, paddle_robot_xy) -> np.ndarray:
        """Action for this step, from the measured paddle pose (ROBOT frame)."""
        p = np.asarray(paddle_robot_xy, dtype=float)[:2]
        i = self._project(p)
        carrot = self._point_at(self.s[i] + self.lookahead_m)
        d = carrot - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return np.zeros(2, dtype=np.float32)
        # Ease into the final waypoint instead of overshooting it: once the arc
        # has less than one step of travel left, ask only for what is left.
        speed = min(self.speed, dist / self.dt)
        lead = (speed / max(self.velocity_gain, 1e-9)) * (d / dist)
        return np.clip(lead / self.move_lims, -1.0, 1.0).astype(np.float32)

    def progress(self, paddle_robot_xy) -> float:
        """Fraction of the arc covered so far, by the monotone projection."""
        self._project(np.asarray(paddle_robot_xy, dtype=float)[:2])
        return float(self.s[self._i] / self.total_s) if self.total_s > 0 else 0.0


def simulate_arc_tracker(track: CurveTrack):
    """Dry-run preview: run the tracker against the identified first-order arm.

    The arm model is exactly the one the gain was measured from --- one step moves
    the paddle by ``velocity_gain * dt * (action * move_lims)`` --- so this gives
    an honest estimate of the action profile, the realised y span and how much of
    the arc fits in ``track.n_steps`` steps, without touching hardware. It ignores
    the couple of steps the real arm spends ramping up, so it reads slightly
    optimistic on the fastest conditions.

    Returns ``(actions[N, 2], poses[N + 1, 2], arc_fraction)``, where
    ``arc_fraction`` is how much of the arc's LENGTH the sweep gets through --- the
    honest "reach" number, unlike the y span, which understates a tall arc that
    spends much of its length climbing in x.
    """
    tracker = track.tracker()
    move_lims = np.asarray(track.move_lims, dtype=float)
    p = np.array([track.x_base, track.y_centre - track.half_width_m], dtype=float)
    poses = [p.copy()]
    actions = []
    for _ in range(int(track.n_steps)):
        a = np.asarray(tracker.action(p), dtype=float)
        p = p + track.velocity_gain * track.dt * (a * move_lims)
        actions.append(a)
        poses.append(p.copy())
    return (np.asarray(actions, dtype=np.float32), np.asarray(poses, dtype=float),
            tracker.progress(p))


DEFAULT_DIRECTIONS = "+x,-x,+y,-y,diagpos,diagneg"
DEFAULT_DELTAS = [0.33, 0.66, 1.0]
DEFAULT_REPEATS = 3
DEFAULT_ACTION_STEPS = 20
DEFAULT_SETTLE_STEPS = 3
# Inset from the workspace boundary for the max-room start poses. The arm should
# not be parked exactly on a limit it will immediately be clipped against.
DEFAULT_START_MARGIN_M = 0.01
# x used for the off-axis coordinate of the y-axis trials (and the whole fixed-start
# pose): the real config's reset_positions["hitting"] pose.
DEFAULT_BASE_ROBOT_XY = (-0.68, 0.0)


@dataclass(frozen=True)
class Trial:
    """One scripted segment: a constant-action line, or a swept half arc."""

    index: int
    delta: float
    repeat: int
    direction: Direction | None = None
    curve: Curve | None = None
    # Per-axis constant action for line trials, filled by resolve_plan_actions.
    action_vec: tuple[float, float] | None = None
    # Per-step action sequence for curve trials, filled by build_curve_plan. For
    # a closed-loop curve this is the PREVIEW schedule (what the tracker is
    # predicted to command against the identified arm model), used for plan
    # printing and as the fallback if a collector cannot supply a live pose; the
    # actions actually sent come from `make_tracker()`.
    schedule: tuple[tuple[float, float], ...] | None = None
    # Closed-loop arc tracking spec; None for line trials and for curve trials
    # built with --curve-tracking open-loop.
    track: CurveTrack | None = None

    @property
    def is_curve(self) -> bool:
        return self.curve is not None

    @property
    def is_closed_loop(self) -> bool:
        return self.track is not None

    def make_tracker(self) -> ArcTracker | None:
        """Fresh pure-pursuit tracker for this trial, or None if open-loop."""
        return self.track.tracker() if self.track is not None else None

    @property
    def key(self) -> str:
        return self.curve.key if self.is_curve else self.direction.key

    @property
    def description(self) -> str:
        return self.curve.description if self.is_curve else self.direction.description

    @property
    def trial_type(self) -> str:
        return "curve" if self.is_curve else "line"

    @property
    def action(self) -> np.ndarray:
        """Representative action: the constant one, or the curve's peak step."""
        if self.is_curve:
            sched = np.asarray(self.schedule, dtype=np.float32)
            return sched[int(np.argmax(np.linalg.norm(sched, axis=1)))]
        if self.action_vec is not None:
            return np.asarray(self.action_vec, dtype=np.float32)
        return (np.asarray(self.direction.vec, dtype=np.float32) * np.float32(self.delta))

    def step_actions(self, default_steps: int) -> np.ndarray:
        """The [N, 2] action sequence this trial holds, one row per timestep.

        Line trials repeat their constant action ``default_steps`` times; curve
        trials carry their own length, since a faster sweep needs fewer steps to
        cover the same 180 degrees.
        """
        if self.is_curve:
            return np.asarray(self.schedule, dtype=np.float32)
        return np.tile(np.asarray(self.action, dtype=np.float32), (int(default_steps), 1))

    @property
    def axis(self) -> str:
        return "curve" if self.is_curve else self.direction.axis

    @property
    def sign(self) -> int:
        return 0 if self.is_curve else self.direction.sign

    @property
    def group(self) -> str:
        if self.is_curve:
            return f"{self.curve.key}_v{self.delta:.2f}"
        return f"{self.direction.key}_delta{self.delta:.2f}"

    @property
    def name(self) -> str:
        return f"traj_{self.index:03d}_{self.group}_trial{self.repeat}"


def parse_curves(spec: str) -> list[Curve]:
    """Parse ``"arc_wide,arc_tight"`` into Curves. ``none``/empty gives no curves."""
    curves: list[Curve] = []
    for raw in (spec or "").split(","):
        token = raw.strip().lower()
        if not token or token == "none":
            continue
        if token not in CURVES:
            raise SystemExit(f"Bad --curves token {raw!r}; choose from {sorted(CURVES)}")
        if CURVES[token] not in curves:
            curves.append(CURVES[token])
    return curves


def parse_directions(spec: str) -> list[Direction]:
    """Parse e.g. ``"+x,-y,diagpos"`` into ``Direction`` objects, order preserved."""
    directions: list[Direction] = []
    for raw in spec.split(","):
        token = raw.strip().lower().replace(" ", "")
        if not token or token == "none":
            continue
        key = DIRECTION_ALIASES.get(token)
        if key is None:
            raise SystemExit(
                f"Bad --directions token {raw!r}; choose from "
                f"{sorted(set(DIRECTION_ALIASES))}"
            )
        direction = DIRECTIONS[key]
        if direction not in directions:
            directions.append(direction)
    return directions


def build_trial_plan(
    directions: list[Direction],
    deltas: list[float],
    repeats: int,
    order: str,
) -> list[Trial]:
    """Expand directions x deltas x repeats into an ordered trial list.

    ``grouped`` runs the repeats of one condition back-to-back (easiest to eyeball
    for repeatability); ``interleaved`` cycles through every condition once per
    round, which spreads any slow drift in the robot/table across conditions.
    """
    conditions = [(d, delta) for d in directions for delta in deltas]
    if order == "grouped":
        ordered = [(d, delta, r) for d, delta in conditions for r in range(1, repeats + 1)]
    elif order == "interleaved":
        ordered = [(d, delta, r) for r in range(1, repeats + 1) for d, delta in conditions]
    else:
        raise SystemExit(f"Unknown --order {order!r}; choose grouped or interleaved")
    return [
        Trial(index=i, direction=d, delta=float(delta), repeat=r)
        for i, (d, delta, r) in enumerate(ordered)
    ]


def step_displacement(direction: Direction, delta: float, move_lims) -> np.ndarray:
    """Commanded per-step displacement (m) for one condition, before clipping."""
    return np.asarray(direction.vec, dtype=float) * float(delta) * np.asarray(move_lims, dtype=float)


def resolve_action(
    direction: Direction,
    delta: float,
    *,
    mode: str = DEFAULT_DELTA_MODE,
    start_xy=None,
    lims=None,
    edge_lims=None,
    move_lims=None,
    action_steps: int = DEFAULT_ACTION_STEPS,
) -> np.ndarray:
    """Per-axis action for one condition.

    ``action`` mode (the default): ``delta`` IS the action. Every axis the
    direction travels gets ``+-delta`` and the others are zero, so at
    ``delta = 0.33``:

        +x -> ( 0.33,  0.00)      -y      -> ( 0.00, -0.33)
        -x -> (-0.33,  0.00)      diagpos -> (-0.33, +0.33)
        +y -> ( 0.00,  0.33)      diagneg -> (-0.33, -0.33)

    Both diagonals carry a NEGATIVE x component: they travel toward table centre
    (the strike direction, "up" the table), and differ only in the y sign.

    Because ``rmax_x`` (0.26) and ``rmax_y`` (0.12) differ and the workspace is
    0.33 m x 0.73 m, the axes run out at different times: x saturates within a
    couple of steps and the paddle then slides along the far edge for the
    remaining timesteps. That is accepted behaviour in this mode.

    ``workspace`` mode: ``delta`` is instead the fraction of the *available room*
    to traverse over the trial, and the per-axis components are set so both axes
    cover their share in the same ``action_steps``:

        step_i   = delta * room_i / action_steps        (metres per step)
        action_i = step_i / move_lims_i                 (normalised)

    The commanded path is then a straight line to the opposite corner instead of
    an L, and at ``delta <= 1`` nothing clips -- at the cost of decoupling
    ``delta`` from the action actually sent, and of capping speed at
    ``room / (action_steps * rmax)`` per axis.
    """
    vec = np.asarray(direction.vec, dtype=float)
    if mode == "action":
        return vec * float(delta)
    if mode != "workspace":
        raise SystemExit(f"Unknown delta mode {mode!r}; choose from {list(DELTA_MODES)}")

    room = np.maximum(0.0, room_in_direction(start_xy, direction, lims, edge_lims))
    steps = max(1, int(action_steps))
    per_step = np.sign(vec) * float(delta) * room / steps
    action = per_step / np.asarray(move_lims, dtype=float)
    return np.clip(action, -1.0, 1.0)


def resolve_plan_actions(
    trials: list[Trial],
    start_poses: dict,
    *,
    mode: str,
    lims,
    edge_lims,
    move_lims,
    action_steps: int,
) -> list[Trial]:
    """Return the plan witheach trial's per-axis action filled in."""
    resolved = []
    for trial in trials:
        action = resolve_action(
            trial.direction, trial.delta, mode=mode,
            start_xy=start_poses[trial.direction.key], lims=lims, edge_lims=edge_lims,
            move_lims=move_lims, action_steps=action_steps,
        )
        resolved.append(replace(trial, action_vec=(float(action[0]), float(action[1]))))
    return resolved


def steps_for_full_scale(direction: Direction, start_xy, lims, edge_lims, move_lims) -> int:
    """Steps at which delta=1.0 would command the full action range on some axis.

    i.e. the largest ``action_steps`` for which ``workspace`` mode still reaches
    |action| = 1 somewhere. Beyond this, trials get slower as they get longer.
    """
    room = np.maximum(0.0, room_in_direction(start_xy, direction, lims, edge_lims))
    lims_arr = np.asarray(move_lims, dtype=float)
    ratios = [room[i] / lims_arr[i] for i in range(2) if direction.vec[i]]
    return max(1, int(np.floor(min(ratios)))) if ratios else 0


def _infer_gain(trial, move_lims) -> float:
    """Recover the tracking gain a curve trial was built with, for reporting."""
    sched = np.asarray(trial.schedule, dtype=float)
    steps = sched * np.asarray(move_lims, dtype=float)
    span = float(steps[:, 1].sum())
    return span / (2.0 * trial.curve.half_width_m) if trial.curve.half_width_m else 1.0


def print_plan(trials: list[Trial], move_lims, action_steps: int, step_hz: float = 20.0) -> None:
    """One row per condition: action, per-step displacement, speed, travel."""
    lines = [t for t in trials if not t.is_curve]
    curves = [t for t in trials if t.is_curve]
    parts = []
    if lines:
        parts.append(f"{len(lines)} line @ {action_steps} steps")
    if curves:
        lengths = sorted(len(t.schedule) for t in curves)
        parts.append(f"{len(curves)} curve @ {lengths[0]}-{lengths[-1]} steps")
    print(f"\n[plan] {len(trials)} trials ({', '.join(parts)})")
    if lines:
        print(f"  {'condition':<22} {'action (x,y)':>16} {'m/step':>16} {'m/s':>7} {'travel(m)':>10}  reps")
        seen: set[str] = set()
        for trial in lines:
            if trial.group in seen:
                continue
            seen.add(trial.group)
            action = np.asarray(trial.action, dtype=float)
            step = action * np.asarray(move_lims, dtype=float)
            speed = float(np.linalg.norm(step)) * float(step_hz)
            travel = float(np.linalg.norm(step)) * int(action_steps)
            n_repeat = sum(1 for t in lines if t.group == trial.group)
            print(
                f"  {trial.group:<22} ({action[0]:+.3f},{action[1]:+.3f}) "
                f"({step[0]:+.4f},{step[1]:+.4f}) {speed:7.3f} {travel:10.3f}  x{n_repeat}"
            )
    closed = [t for t in curves if t.is_closed_loop]
    open_loop = [t for t in curves if not t.is_closed_loop]
    if closed:
        table_width = 2.0 * max(t.track.half_width_m for t in closed)
        print(f"\n  closed-loop arc tracking (action = velocity command; "
              f"G = {closed[0].track.velocity_gain:g} 1/s)")
        print(f"  {'condition':<22} {'m/s':>6} {'steps':>6} {'arc(m)':>7} {'reach':>6} "
              f"{'y span':>7} {'of table':>9} {'peak|ax|':>8} {'peak|ay|':>8} {'full@':>6}  reps")
        seen = set()
        for trial in closed:
            if trial.group in seen:
                continue
            seen.add(trial.group)
            track = trial.track
            _acts, poses, reach = simulate_arc_tracker(track)
            sched = np.asarray(trial.schedule, dtype=float)
            span = float(poses[:, 1].max() - poses[:, 1].min())
            n_repeat = sum(1 for t in closed if t.group == trial.group)
            # Steps this condition would need to finish the sweep, so the
            # trade at 20 is explicit rather than implied.
            full_steps = int(np.ceil(track.n_steps / max(reach, 1e-9)))
            print(
                f"  {trial.group:<22} {trial.delta:6.2f} {track.n_steps:6d} "
                f"{track.arc_length_m:7.3f} {reach:5.0%} {span:7.3f} "
                f"{span / max(table_width, 1e-9):8.0%} "
                f"{np.max(np.abs(sched[:, 0])):8.3f} {np.max(np.abs(sched[:, 1])):8.3f} "
                f"{full_steps:6d}  x{n_repeat}"
            )
        print("  (y span / reach are predictions against the identified first-order arm. "
              "'reach' < 100% means the\n   sweep runs out of timesteps part-way along a "
              "full-width arc -- expected at 20 steps, since the arm\n   tops out near "
              f"{closed[0].track.velocity_gain * move_lims[1]:.2f} m/s laterally. 'full@' is the "
              "--action-steps that would complete the sweep.)")
    if open_loop:
        print(f"\n  open-loop arc schedule (legacy; commands displacement, not velocity)")
        print(f"  {'condition':<22} {'target':>7} {'cmd m/s':>8} {'gain':>5} {'steps':>6} "
              f"{'deg/step':>9} {'peak|ax|':>8} {'peak|ay|':>8}  reps")
        seen = set()
        for trial in open_loop:
            if trial.group in seen:
                continue
            seen.add(trial.group)
            sched = np.asarray(trial.schedule, dtype=float)
            steps = sched * np.asarray(move_lims, dtype=float)
            per_step = np.linalg.norm(steps, axis=1)
            # Commanded speed carries the tracking gain, so it reads high; the
            # paddle lags by roughly the same factor, landing near `target`.
            peak_speed = float(np.max(per_step)) * float(step_hz)
            gain = float(getattr(trial, "_gain", 0.0)) or _infer_gain(trial, move_lims)
            n_repeat = sum(1 for t in open_loop if t.group == trial.group)
            print(
                f"  {trial.group:<22} {trial.delta:7.2f} {peak_speed:8.3f} {gain:5.2f} "
                f"{len(sched):6d} {180.0 / len(sched):9.1f} "
                f"{np.max(np.abs(sched[:, 0])):8.3f} {np.max(np.abs(sched[:, 1])):8.3f}  x{n_repeat}"
            )


# ---------------------------------------------------------------------------
# Start poses
# ---------------------------------------------------------------------------


def room_in_direction(start_robot_xy, direction: Direction, lims, edge_lims) -> np.ndarray:
    """Per-axis metres of workspace available from ``start_robot_xy``.

    Returns ``[room_x, room_y]``, with 0 on any axis the direction does not travel.
    The x entry uses the chamfered ``effective_x_max`` at the start pose's y.
    """
    x_min_lim, _, y_min, y_max = lims
    x, y = float(start_robot_xy[0]), float(start_robot_xy[1])
    sx, sy = direction.vec
    room_x = 0.0
    if sx > 0:
        room_x = effective_x_max(y, lims, edge_lims) - x
    elif sx < 0:
        room_x = x - x_min_lim
    room_y = 0.0
    if sy > 0:
        room_y = y_max - y
    elif sy < 0:
        room_y = y - y_min
    return np.array([room_x, room_y], dtype=float)


def steps_to_saturation(start_robot_xy, direction: Direction, delta: float,
                        move_lims, lims, edge_lims) -> int:
    """Steps before the first travelled axis runs out of workspace."""
    room = room_in_direction(start_robot_xy, direction, lims, edge_lims)
    step = np.abs(step_displacement(direction, delta, move_lims))
    counts = [room[i] / step[i] for i in range(2) if step[i] > 1e-9]
    if not counts:
        return 0
    return max(1, int(np.ceil(min(counts))))


def start_pose_for(
    direction: Direction,
    lims,
    edge_lims,
    *,
    mode: str = "max-room",
    base_xy=DEFAULT_BASE_ROBOT_XY,
    margin: float = DEFAULT_START_MARGIN_M,
) -> np.ndarray:
    """Initial paddle pose (ROBOT frame) for one direction.

    ``max-room`` parks the paddle at the far end of every axis it is about to
    travel, so the whole run is spent in free space instead of pinned against a
    limit. A ``-y`` trial starts at ``y_max``; a ``+x`` trial at ``x_min``; the
    ``diagpos`` trial, travelling ``(-x, +y)``, starts at the bottom-LEFT corner
    and the ``diagneg`` trial at the bottom-RIGHT. Axes the direction does not
    travel keep their ``base_xy`` coordinate, so every trial on an axis shares one
    starting line.

    ``fixed`` returns ``base_xy`` for every direction (one common start pose,
    matching the real config's reset pose).

    ``margin`` insets the pose from the boundary. y is resolved first because the
    reachable x_max is chamfered near the far corners --- at |y| ~ 0.35 it pulls in
    from -0.42 to about -0.49 --- so a corner start has to be clamped against
    ``effective_x_max`` at its own y or it would sit outside the workspace.
    """
    base_x, base_y = float(base_xy[0]), float(base_xy[1])
    if mode == "fixed":
        return np.array([base_x, base_y], dtype=float)
    if mode != "max-room":
        raise SystemExit(f"Unknown start mode {mode!r}; choose max-room or fixed")

    x_min_lim, _, y_min, y_max = lims
    sx, sy = direction.vec

    if sy > 0:
        y = y_min + margin
    elif sy < 0:
        y = y_max - margin
    else:
        y = base_y

    x_max_here = effective_x_max(y, lims, edge_lims) - margin
    if sx > 0:
        x = x_min_lim + margin
    elif sx < 0:
        x = x_max_here
    else:
        x = base_x
    x = float(np.clip(x, x_min_lim + margin, x_max_here))
    return np.array([x, y], dtype=float)


# ---------------------------------------------------------------------------
# Shared CLI flags
# ---------------------------------------------------------------------------


def curve_geometry(curve: Curve, lims, edge_lims, margin: float = DEFAULT_START_MARGIN_M):
    """Fit a curve to the workspace: (start_xy, half_width, height, y_centre).

    The chord sits at the largest x the arc's two endpoints can both reach. That
    is set by the far-edge chamfer, which pulls x_max in from -0.42 to about -0.49
    at |y| ~ 0.35 -- so a corner-to-corner arc starts lower down the table than a
    narrow one. Height is then clamped to whatever room is left above it.
    """
    x_min_lim, _, y_min, y_max = lims
    y_centre = 0.5 * (y_min + y_max)
    half_width = min(float(curve.half_width_m), 0.5 * (y_max - y_min) - margin)
    y_start, y_end = y_centre - half_width, y_centre + half_width
    x_base = min(
        effective_x_max(y_start, lims, edge_lims),
        effective_x_max(y_end, lims, edge_lims),
    ) - margin
    height = min(float(curve.height_m), x_base - (x_min_lim + margin))
    return np.array([x_base, y_start], dtype=float), half_width, height, y_centre


def semi_ellipse_arc_length(a: float, b: float, samples: int = 2000) -> float:
    """Path length of the half arc, for converting a target speed into steps.

    Trapezoid rule written out rather than via numpy's helper: that helper is
    ``np.trapz`` on numpy 1.x and ``np.trapezoid`` on 2.x, and the robot runs an
    older numpy than the dev venv.
    """
    t = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(samples))
    speed = np.sqrt((b * np.sin(t)) ** 2 + (a * np.cos(t)) ** 2)
    return float(np.sum(0.5 * (speed[:-1] + speed[1:]) * np.diff(t)))


def curve_action_schedule(
    curve: Curve,
    speed_m_s: float,
    lims,
    edge_lims,
    move_lims,
    margin: float = DEFAULT_START_MARGIN_M,
    min_steps: int = MIN_CURVE_STEPS,
    step_hz: float = DEFAULT_STEP_HZ,
    tracking_gain: float = DEFAULT_CURVE_TRACKING_GAIN,
):
    """Per-step actions that sweep one half arc. Returns (actions[N,2], info).

    The path is ``x(t) = x_base - b cos t``, ``y(t) = y_centre + a sin t`` for
    ``t`` running -90deg -> +90deg, so per step::

        dx = b sin(t) dt        dy = a cos(t) dt
        ax = dx / rmax_x        ay = dy / rmax_y

    ``dy >= 0`` throughout, so the paddle tracks left to right without reversing,
    while ``dx`` flips sign at the apex -- up the table, then back down.

    ``speed_m_s`` selects how fast: the step count is ``arc_length / (speed * dt)``,
    so the sweep is spread over however many steps hold that speed. The four
    speeds are therefore four heading rates around the same arc.

    Speed is specified in m/s rather than as an action magnitude on purpose. A
    line trial saturates against a workspace limit within a few steps, so a large
    commanded action is only briefly realised; a curve never saturates, so the
    commanded speed is sustained for the entire sweep and a number that looks
    reasonable in action units can be an unreasonable thing to ask of the arm.

    ``tracking_gain`` scales the whole schedule to compensate for the paddle
    lagging its target. Without it the realised arc is only ~57% as wide as the
    one commanded, which is why arcs look small on the robot. The geometry fields
    in ``info`` describe the COMMANDED ellipse; the realised path is that shape
    scaled by however much of it the servo actually delivers.
    """
    start_xy, a, b, y_centre = curve_geometry(curve, lims, edge_lims, margin)
    rmax_x, rmax_y = float(move_lims[0]), float(move_lims[1])
    dt = 1.0 / max(1e-6, float(step_hz))
    arc_length = semi_ellipse_arc_length(a, b)
    n_steps = max(int(min_steps), int(round(arc_length / max(float(speed_m_s) * dt, 1e-9))))
    d_theta = np.pi / n_steps
    # Midpoint rule, so the summed displacement matches the ellipse rather than
    # drifting by half a step.
    thetas = -np.pi / 2.0 + (np.arange(n_steps) + 0.5) * d_theta
    actions = np.stack(
        [b * np.sin(thetas) * d_theta / rmax_x, a * np.cos(thetas) * d_theta / rmax_y],
        axis=1,
    )
    # Rescale each axis so the SUMMED command hits the ellipse exactly. Quadrature
    # error is a few tenths of a percent at 20 steps but grows as the sweep gets
    # coarse, and a short arc should still be the arc that was asked for. The x
    # column is antisymmetric so scaling leaves its net displacement at zero.
    steps = actions * np.array([rmax_x, rmax_y])
    apex = float(np.abs(np.cumsum(steps[:, 0]).min()))
    if apex > 1e-9:
        actions[:, 0] *= b / apex
    span = float(steps[:, 1].sum())
    if span > 1e-9:
        actions[:, 1] *= (2.0 * a) / span
    # Apply the tracking gain, but never past the point where an action would clip:
    # clipping flattens the velocity profile and the commanded path stops being the
    # ellipse that was asked for. Slow sweeps have headroom and get the full gain;
    # fast ones are already commanding large actions and get whatever is left.
    peak_unit = float(np.max(np.abs(actions)))
    gain_ceiling = 1.0 / peak_unit if peak_unit > 1e-9 else float(tracking_gain)
    gain = float(min(float(tracking_gain), gain_ceiling))
    actions = np.clip(actions * gain, -1.0, 1.0)
    info = {
        "tracking": "open-loop",
        "start_xy": start_xy,
        "tracking_gain_requested": float(tracking_gain),
        "tracking_gain": gain,
        "half_width_m": float(a),
        "height_m": float(b),
        "y_centre": float(y_centre),
        "arc_length_m": float(arc_length),
        "n_steps": int(n_steps),
        "d_theta_rad": float(d_theta),
        "target_speed_m_s": float(speed_m_s),
        "realised_speed_m_s": float(arc_length / (n_steps * dt)),
        "peak_ax": float(np.max(np.abs(actions[:, 0]))),
        "peak_ay": float(np.max(np.abs(actions[:, 1]))),
    }
    return actions, info


def build_curve_track(
    curve: Curve,
    speed_m_s: float,
    lims,
    edge_lims,
    move_lims,
    margin: float = DEFAULT_START_MARGIN_M,
    step_hz: float = DEFAULT_STEP_HZ,
    velocity_gain: float = CURVE_VELOCITY_GAIN_REAL,
    n_steps: int = DEFAULT_ACTION_STEPS,
    lookahead_steps: float = DEFAULT_CURVE_LOOKAHEAD_STEPS,
):
    """Closed-loop spec for one (shape, speed), plus its dry-run prediction."""
    start_xy, a, b, y_centre = curve_geometry(curve, lims, edge_lims, margin)
    track = CurveTrack(
        x_base=float(start_xy[0]),
        y_centre=float(y_centre),
        half_width_m=float(a),
        height_m=float(b),
        speed_m_s=float(speed_m_s),
        move_lims=(float(move_lims[0]), float(move_lims[1])),
        velocity_gain=float(velocity_gain),
        dt=1.0 / max(1e-6, float(step_hz)),
        n_steps=int(n_steps),
        lookahead_steps=float(lookahead_steps),
    )
    actions, poses, arc_fraction = simulate_arc_tracker(track)
    arc_length = track.arc_length_m
    predicted_span = float(poses[:, 1].max() - poses[:, 1].min())
    info = {
        "tracking": "closed-loop",
        "start_xy": np.asarray(start_xy, dtype=float),
        "half_width_m": float(a),
        "height_m": float(b),
        "y_centre": float(y_centre),
        "arc_length_m": float(arc_length),
        "n_steps": int(n_steps),
        "target_speed_m_s": float(speed_m_s),
        "velocity_gain": float(velocity_gain),
        "lookahead_m": float(track.tracker().lookahead_m),
        # Predictions against the identified first-order arm (see
        # simulate_arc_tracker); the robot ramps up over ~3 steps, so these read
        # a little optimistic on the fastest conditions.
        "predicted_y_span_m": predicted_span,
        "predicted_arc_fraction": float(arc_fraction),
        "predicted_peak_ax": float(np.max(np.abs(actions[:, 0]))) if len(actions) else 0.0,
        "predicted_peak_ay": float(np.max(np.abs(actions[:, 1]))) if len(actions) else 0.0,
    }
    return track, actions, info


def build_curve_plan(
    curves: list[Curve],
    speeds: list[float] | None,
    repeats: int,
    order: str,
    lims,
    edge_lims,
    move_lims,
    margin: float = DEFAULT_START_MARGIN_M,
    min_steps: int = MIN_CURVE_STEPS,
    step_hz: float = DEFAULT_STEP_HZ,
    tracking_gain: float = DEFAULT_CURVE_TRACKING_GAIN,
    tracking: str = DEFAULT_CURVE_TRACKING,
    velocity_gain: float = CURVE_VELOCITY_GAIN_REAL,
    n_steps: int = DEFAULT_ACTION_STEPS,
    lookahead_steps: float = DEFAULT_CURVE_LOOKAHEAD_STEPS,
):
    """Curve trials plus their start poses, ordered like the line plan.

    ``speeds`` of None gives each shape its own sweep; a list applies the same
    speeds to every shape.

    ``tracking="closed-loop"`` (the default) attaches a ``CurveTrack`` to every
    trial and fills ``schedule`` with the predicted preview only; the collector
    drives the arc from the measured pose. ``tracking="open-loop"`` restores the
    pre-fix behaviour: a fixed schedule of per-step displacements, whose length
    is derived from arc length / speed rather than fixed at ``n_steps``.
    """
    if tracking not in CURVE_TRACKING_MODES:
        raise SystemExit(
            f"Unknown --curve-tracking {tracking!r}; choose from {list(CURVE_TRACKING_MODES)}"
        )
    conditions = [(c, v) for c in curves for v in (speeds or c.speeds_m_s)]
    if order == "grouped":
        ordered = [(c, v, r) for c, v in conditions for r in range(1, repeats + 1)]
    elif order == "interleaved":
        ordered = [(c, v, r) for r in range(1, repeats + 1) for c, v in conditions]
    else:
        raise SystemExit(f"Unknown --order {order!r}; choose grouped or interleaved")

    trials, start_poses, infos = [], {}, {}
    for c, v, r in ordered:
        if tracking == "closed-loop":
            track, actions, info = build_curve_track(
                c, v, lims, edge_lims, move_lims, margin, step_hz,
                velocity_gain, n_steps, lookahead_steps,
            )
        else:
            track = None
            actions, info = curve_action_schedule(
                c, v, lims, edge_lims, move_lims, margin, min_steps, step_hz, tracking_gain
            )
        start_poses[c.key] = info["start_xy"]
        infos[f"{c.key}_v{float(v):.2f}"] = info
        trials.append(Trial(
            index=0, curve=c, delta=float(v), repeat=r,
            schedule=tuple((float(ax), float(ay)) for ax, ay in actions),
            track=track,
        ))
    return trials, start_poses, infos


def reindex(trials: list[Trial]) -> list[Trial]:
    """Renumber a concatenated plan so trial_name indices stay contiguous."""
    return [replace(t, index=i) for i, t in enumerate(trials)]


def add_plan_arguments(parser) -> None:
    """Register the plan flags shared by the real and sim collectors."""
    parser.add_argument("--directions", type=str, default=DEFAULT_DIRECTIONS,
                        help="Comma-separated travel directions: +x -x +y -y diagpos diagneg "
                             "(diagpos = bottom-left -> top-right, diagneg = bottom-right -> "
                             "top-left; also spellable -x+y / -x-y). A value starting with '-' "
                             "must use the equals form so argparse does not read it as a flag: "
                             "--directions=-x,diagpos")
    parser.add_argument("--delta-mode", type=str, default=DEFAULT_DELTA_MODE, choices=list(DELTA_MODES),
                        help="action (default): --deltas ARE the action -- every travelled axis "
                             "gets +-delta, others zero, so a diagonal at 0.33 is (-0.33, +-0.33). "
                             "x saturates first and the paddle slides along the far edge for the "
                             "remaining steps. workspace: --deltas are fractions of the "
                             "traversable room and the per-axis action is derived so both axes "
                             "finish together (straight path, never clips, but slower).")
    parser.add_argument("--deltas", type=float, nargs="+", default=list(DEFAULT_DELTAS),
                        help="The three magnitudes to sweep. In action mode (default) these are "
                             "the per-axis action magnitudes themselves; in workspace mode they "
                             "are fractions of the available room.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="Trials per (direction, delta) condition.")
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS,
                        help="Constant-action timesteps per trial.")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS,
                        help="Zero-action steps recorded before the constant action, so each trial "
                             "has a measured at-rest baseline. Flagged via the is_settle_step dataset.")
    parser.add_argument("--curves", type=str, default=DEFAULT_CURVES,
                        help="Comma-separated half-arc shapes: arc_wide (mild, bottom corner to "
                             "bottom corner), arc_medium (rounder, starts nearer the centre), "
                             "arc_tight (tall and narrow, starts nearest the centre). "
                             "'none' runs no curves.")
    parser.add_argument("--curve-speeds", type=float, nargs="+", default=DEFAULT_CURVE_SPEEDS,
                        help="Override the per-shape speed sweeps (m/s of paddle travel along the "
                             "arc) with one list applied to every shape. By default each shape "
                             "uses its own three speeds: arc_wide 0.15/0.28/0.45, arc_medium "
                             "0.15/0.30/0.50, arc_tight 0.18/0.35/0.60 -- tapered by how much of "
                             "each shape is y motion, which is the axis with the least authority.")
    parser.add_argument("--curve-tracking", type=str, default=DEFAULT_CURVE_TRACKING,
                        choices=list(CURVE_TRACKING_MODES),
                        help="closed-loop (default): each step's action is computed online from "
                             "the MEASURED paddle pose so the arc is traced at its true "
                             "workspace-wide geometry. open-loop: the legacy fixed schedule of "
                             "per-step displacements, which the arm realises at only ~15%% on the "
                             "robot -- kept for reproducing pre-fix runs.")
    parser.add_argument("--curve-steps", type=int, default=None,
                        help="Timesteps per curve trial (closed-loop tracking only). Defaults to "
                             "--action-steps, so lines and curves are the same length. A "
                             "full-width arc needs roughly 55 steps to complete; at 20 it covers "
                             "the first part of the arc, which is the intended trade.")
    parser.add_argument("--curve-velocity-gain", type=float, default=None,
                        help="Realised paddle speed per metre of commanded lead, 1/s. The action "
                             "is a velocity command (target is rebuilt from the current pose each "
                             "step), so this converts a path speed into an action. Measured: 3.2 "
                             "on the real UR5, ~10 in Box2D. Each collector defaults to its own; "
                             "getting it wrong changes the sweep SPEED, not its shape.")
    parser.add_argument("--curve-lookahead-steps", type=float, default=DEFAULT_CURVE_LOOKAHEAD_STEPS,
                        help="Carrot distance ahead of the paddle's projection onto the arc, in "
                             "steps of travel. Larger cuts the apex; smaller gets twitchy.")
    parser.add_argument("--curve-tracking-gain", type=float, default=DEFAULT_CURVE_TRACKING_GAIN,
                        help="Open-loop tracking only: scale the fixed schedule to compensate for "
                             "the paddle lagging its target. Calibrated in Box2D and does not "
                             "transfer to the robot; ignored under closed-loop tracking.")
    parser.add_argument("--min-curve-steps", type=int, default=MIN_CURVE_STEPS,
                        help="Open-loop tracking only: floor on steps per arc, so a fast tight "
                             "sweep stays a curve.")
    parser.add_argument("--order", type=str, default="grouped", choices=["grouped", "interleaved"],
                        help="grouped: repeats back-to-back. interleaved: one round per repeat.")


def add_start_pose_arguments(parser) -> None:
    parser.add_argument("--start-mode", type=str, default="max-room", choices=["max-room", "fixed"],
                        help="max-room: each direction starts at the far end of every axis it "
                             "travels, so the paddle has the full workspace. "
                             "fixed: every trial starts at --base-robot-xy.")
    parser.add_argument("--base-robot-xy", type=float, nargs=2, default=list(DEFAULT_BASE_ROBOT_XY),
                        help="ROBOT-frame pose used for axes a direction does not travel in "
                             "max-room mode, and as the single start pose in fixed mode.")
    parser.add_argument("--start-margin", type=float, default=DEFAULT_START_MARGIN_M,
                        help="Inset from the workspace boundary for max-room start poses (m).")


# ---------------------------------------------------------------------------
# Offline geometry (for --dry-run, which must not construct an env)
# ---------------------------------------------------------------------------


def real_preview_geometry(sim_params: dict):
    """(lims, edge_lims, move_lims) as AirHockeyReal would build them.

    The workspace limits are hardcoded in the env rather than read from config,
    so they come from the module constant; the corner-cut biases are derived the
    same way ``__init__`` derives them.
    """
    from airhockey.sims.air_hockey_real import REAL_WORKSPACE_LIMS

    lims = tuple(REAL_WORKSPACE_LIMS)
    top_abs = float(sim_params.get("top_abs", 0.8))
    bot_abs = float(sim_params.get("bot_abs", 0.1))
    y_extent = float(sim_params.get("corner_cut_y_extent", CORNER_CUT_Y_EXTENT))
    bias_p, bias_m = corner_cut_biases(lims[1], lims[2], lims[3], slope=top_abs, y_extent=y_extent)
    move_lims = (float(sim_params.get("rmax_x", 0.26)), float(sim_params.get("rmax_y", 0.12)))
    return lims, (top_abs, bot_abs, bias_p, bias_m), move_lims


def box2d_preview_geometry(sim_params: dict):
    """(lims, edge_lims, move_lims) as AirHockeyBox2D would build them."""
    lims = (
        float(sim_params.get("x_min_lim", -0.85)),
        float(sim_params.get("x_max_lim", -0.45)),
        float(sim_params.get("y_min", -0.37)),
        float(sim_params.get("y_max", 0.37)),
    )
    edge_lims = (
        float(sim_params.get("top_abs", 0.8)),
        float(sim_params.get("bot_abs", 0.1)),
        float(sim_params.get("max_bias_p", -0.15)),
        float(sim_params.get("max_bias_m", -0.15)),
    )
    move_lims = (float(sim_params.get("rmax_x", 0.26)), float(sim_params.get("rmax_y", 0.12)))
    return lims, edge_lims, move_lims
