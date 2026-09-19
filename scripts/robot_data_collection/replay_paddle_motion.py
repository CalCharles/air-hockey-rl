"""Replay a collected paddle-motion session through Box2D and overlay the two tracks.

Pipeline, per trial in the source session:

  1. read the recorded **start pose** and **action sequence** out of the HDF5,
  2. reset Box2D to that start pose (puck parked and frozen) and step it through
     exactly those actions,
  3. record the trajectory the sim went through,
  4. render a GIF with BOTH tracks overlaid on the Box2D frame --- the source
     trajectory from the file and the replay trajectory the sim just produced ---
     plus the per-step position error between them.

Real sessions also carry the camera's puck reading (``train_vals`` columns 32-34),
and that gets drawn into the same Box2D frame in cyan: trail plus a ring at the
current sample, hollow while the detector is coasting on a stale position. It is
an overlay, not a body --- the Box2D puck stays parked and frozen, so the replayed
paddle never collides with it and the paddle metrics mean the same thing they did
before.

The GIF also opens with the APPROACH: ``arm_puck_track`` holds every camera poll
between "trial armed" and the puck crossing the trigger line, and those samples
are resampled onto the trial's frame rate and played first, with both paddles
held at the start pose because that is exactly what the robot did --- it does not
move until the trigger fires. So one GIF covers the puck from the moment the
trial started, not from the halfway line. ``--arm-preroll-s`` trims it. For a ``puck_collision`` trial that is the point: you see the real puck's
approach and rebound against the paddle track the sim reproduces, and any offset
between the two is the camera/sim disagreement you are hunting.

The source session can be either kind of collection:

  * a **real** session from ``collect_paddle_motion.py`` --- the overlay is then a
    genuine sim-vs-real comparison, and the error curve is the transfer gap;
  * a **sim** session from ``collect_paddle_motion_sim.py`` --- the overlay should
    land on top of itself, which is how you check the replay pipeline is faithful
    before trusting it on real data.

The script reads which kind it is from the file layout and labels the GIF legend
accordingly. Positions are handled in ROBOT frame throughout (table frame minus
``center_offset_constant``), the one frame both collectors agree on.

Usage::

    python -m scripts.robot_data_collection.replay_paddle_motion \\
        --session data/robot_data_collection/paddle_motion_sim_20260909_1608

    # a subset, and no GIFs
    python -m scripts.robot_data_collection.replay_paddle_motion \\
        --session <dir> --trials traj_018 traj_024 --no-gifs
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from airhockey.renderers.render import AirHockeyRenderer  # noqa: E402
from scripts.robot_data_collection.collect_paddle_motion_sim import (  # noqa: E402
    DEFAULT_CONFIG,
    GIF_FPS,
    GIF_WIDTH,
    load_env,
    paddle_state,
    park_and_freeze_puck,
    save_gif,
    target_position,
    to_gif_frame,
)

# BGR, drawn on the Box2D frame before the RGB conversion.
SOURCE_COLOR = (0, 190, 0)      # green   -- the paddle trajectory recorded in the file
REPLAY_COLOR = (255, 0, 200)    # magenta -- the paddle track Box2D just produced
PUCK_COLOR = (255, 220, 0)      # cyan    -- the puck the camera saw during the real trial
OUTLINE_COLOR = (0, 0, 0)

# train_vals column slices for real-collected sessions
# (airhockey/sims/real/proprioceptive_state.py).
REAL_POSE_XY = slice(5, 7)
REAL_SPEED_XY = slice(11, 13)
REAL_DESIRED_XY = slice(26, 28)
# Camera-derived puck state, logged in TABLE frame (air_hockey_real.py writes
# `protected_puck_pos + center_offset_constant` into these columns). The flag is
# 0 for a fresh detection and non-zero when the detector fell back to the last
# known position.
REAL_PUCK_XY = slice(32, 34)
REAL_PUCK_OCCLUDED = 34
VALS_TIME_COL = 0
# `find_red_hockey_puck*` returns x == -2 (robot frame) when it has never seen
# the puck at all. That is a sentinel, not a position, so arm-phase rows carrying
# it are dropped rather than drawn at the table's far corner.
PUCK_NEVER_SEEN_X_ROBOT = -2.0


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def load_source_trial(path: Path, real_step_shift: bool = True) -> dict:
    """Read start pose + actions + recorded trajectory from one collected HDF5.

    Handles both collector layouts. Everything comes back in ROBOT frame, and on
    the post-step convention (``positions[i]`` is the state ``actions[i]``
    produced) --- see ``real_step_shift`` below.
    """
    with h5py.File(path, "r") as hf:
        attrs = dict(hf.attrs)
        actions = np.asarray(hf["actions"][:], dtype=np.float64)
        is_settle = np.asarray(hf["is_settle_step"][:], dtype=np.int8) if "is_settle_step" in hf else None

        puck = None
        puck_occluded = None
        puck_start = None
        arm_track = None
        if "arm_puck_track" in hf:
            # The collector polls the camera from the moment the trial arms until
            # the puck crosses the trigger line, so this covers the whole approach
            # -- everything that happened BEFORE the first env step. Columns:
            # frame_time_s, puck_x_obs, puck_y_obs, puck_occluded.
            arm_track = np.asarray(hf["arm_puck_track"][:], dtype=np.float64).reshape(-1, 4)
        if "paddle_pos_robot" in hf:
            kind = "sim"
            positions = np.asarray(hf["paddle_pos_robot"][:], dtype=np.float64)
            velocities = np.asarray(hf["paddle_vel"][:], dtype=np.float64)
            targets = np.asarray(hf["target_pos_robot"][:], dtype=np.float64) if "target_pos_robot" in hf else None
            start_xy = np.asarray(attrs["start_pos_robot"], dtype=np.float64)[:2]
        elif "train_vals" in hf:
            kind = "real"
            vals = np.asarray(hf["train_vals"][:], dtype=np.float64)
            positions = vals[:, REAL_POSE_XY]
            velocities = vals[:, REAL_SPEED_XY]
            targets = vals[:, REAL_DESIRED_XY]
            # Camera puck, table frame -> robot frame, so it lives in the same
            # frame as everything else here and re-enters table frame once, in
            # draw_track/draw_puck.
            puck_offset = np.array(
                [float(attrs.get("center_offset_constant", 0.0)), 0.0], dtype=np.float64
            )
            puck = vals[:, REAL_PUCK_XY] - puck_offset
            puck_occluded = vals[:, REAL_PUCK_OCCLUDED]
            # `start_pose` is the settled TCP pose measured just before the trial.
            start_xy = np.asarray(attrs["start_pose"], dtype=np.float64)[:2]
            if real_step_shift:
                # AirHockeyReal.get_transition reads the TCP pose (_resolve_state_pose_speed)
                # and appends the row (get_state_array -> self.vals.append) BEFORE it sends that
                # step's ctrl.servoL -- grep those call sites to confirm the ordering. So
                # train_vals[i] is the state observed *before* action[i] took effect. The
                # sim collector records post-step state. Shift the real trajectory forward
                # one step so positions[i] is the state action[i] produced, and drop the
                # final action whose result is therefore unobserved. Without this every
                # comparison is biased by one 50 ms step.
                positions = positions[1:]
                velocities = velocities[1:]
                targets = targets[:-1]      # desired_pose[i] IS the command sent at step i
                actions = actions[:-1]
                if is_settle is not None:
                    is_settle = is_settle[:-1]
                # The puck is read on the same row as the pose, so it shifts with
                # it. Row 0 -- the state before action 0 -- is dropped from the
                # trajectory but IS the state at the start pose, so keep it for
                # the frame the GIF draws before the first step.
                puck_start = (puck[0].copy(), float(puck_occluded[0]))
                puck = puck[1:]
                puck_occluded = puck_occluded[1:]
        else:
            raise SystemExit(
                f"{path.name}: neither 'paddle_pos_robot' (sim collection) nor "
                "'train_vals' (real collection) present -- not a paddle-motion trial file."
            )

    if actions.shape[0] != positions.shape[0]:
        raise SystemExit(
            f"{path.name}: {actions.shape[0]} actions but {positions.shape[0]} recorded "
            "positions; refusing to replay a misaligned trial."
        )

    # Stitched approach+trial puck track in TABLE frame on a trigger-relative
    # clock, kept raw (stale samples included) so any downstream analysis can
    # apply its own filtering. Columns match `arm_puck_track`: t_s, x_obs,
    # y_obs, occluded.
    puck_track_table = None
    if kind == "real":
        trigger = float(attrs.get("trigger_time", vals[0, VALS_TIME_COL]))
        trial_track = np.column_stack([
            vals[:, VALS_TIME_COL] - trigger,
            vals[:, REAL_PUCK_XY],
            vals[:, REAL_PUCK_OCCLUDED],
        ])
        if arm_track is not None and arm_track.shape[0] > 0:
            arm_rel = arm_track.copy()
            arm_rel[:, 0] -= trigger
            puck_track_table = np.vstack([arm_rel, trial_track])
        else:
            puck_track_table = trial_track

    arm_puck = None
    arm_puck_occluded = None
    arm_puck_time = None
    if arm_track is not None and arm_track.shape[0] > 0:
        arm_offset = np.array(
            [float(attrs.get("center_offset_constant", 0.0)), 0.0], dtype=np.float64
        )
        arm_xy = arm_track[:, 1:3] - arm_offset
        # Drop the leading rows from before the detector had ever seen the puck
        # (operator still holding it / puck off the table). Those carry the
        # never-seen sentinel, and drawing it would invent an approach that
        # never happened.
        real = np.abs(arm_xy[:, 0] - PUCK_NEVER_SEEN_X_ROBOT) > 1e-6
        first = int(np.argmax(real)) if bool(np.any(real)) else len(arm_xy)
        if first < len(arm_xy):
            arm_puck = arm_xy[first:]
            arm_puck_occluded = arm_track[first:, 3]
            # Seconds relative to the trigger, so the HUD can count down to the
            # first env step regardless of when the operator armed.
            arm_puck_time = arm_track[first:, 0] - float(arm_track[-1, 0])

    def _attr(name, default=None):
        value = attrs.get(name, default)
        return value.item() if isinstance(value, np.generic) else value

    return {
        "path": path,
        "kind": kind,
        "name": str(_attr("trial_name", path.stem)),
        "axis": str(_attr("axis", "?")),
        "sign": int(_attr("direction_sign", 0)),
        "delta": float(_attr("action_delta", float("nan"))),
        "repeat": int(_attr("repeat", 0)),
        "settle_steps": int(_attr("settle_steps", 0)),
        "action_steps": int(_attr("action_steps", 0)),
        "actions": actions,
        "is_settle": is_settle,
        "start_xy_robot": start_xy,
        "positions_robot": positions,
        "velocities": velocities,
        "targets_robot": targets,
        "puck_robot": puck,
        "puck_occluded": puck_occluded,
        "puck_start": puck_start,
        "arm_puck_robot": arm_puck,
        "arm_puck_occluded": arm_puck_occluded,
        "arm_puck_time_s": arm_puck_time,
        "puck_track_table": puck_track_table,
    }


def discover_trials(session: Path, selectors: list[str] | None) -> list[Path]:
    files = sorted(p for p in session.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"No .hdf5 trial files in {session}")
    if selectors:
        files = [p for p in files if any(sel in p.name for sel in selectors)]
        if not files:
            raise SystemExit(f"--trials {selectors} matched nothing in {session}")
    return files


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def draw_track(
    frame: np.ndarray,
    renderer: AirHockeyRenderer,
    points_robot: np.ndarray,
    center_offset: float,
    color,
    ring_radius_px: int,
    *,
    line_thickness: int,
    outline: bool,
) -> None:
    """Draw a trail through `points_robot` and a ring at the last point.

    Points are robot frame; the renderer wants table frame, so the offset goes
    back on here. ``world_xy_to_output_pixel`` maps into the FINAL frame buffer
    (post vertical rotate), so this must be called on the frame get_frame()
    returned, not on an intermediate.

    The two tracks are drawn at different widths (source thick, replay thin) so a
    perfect match still reads as a coloured core inside a coloured halo rather
    than one track silently hiding the other.
    """
    if len(points_robot) == 0:
        return
    pixels = [
        renderer.world_xy_to_output_pixel(float(p[0]) + center_offset, float(p[1]))
        for p in points_robot
    ]
    pts = np.array([[int(round(px)), int(round(py))] for px, py in pixels], dtype=np.int32)
    if len(pts) > 1:
        if outline:
            cv2.polylines(frame, [pts], False, OUTLINE_COLOR, line_thickness + 2, cv2.LINE_AA)
        cv2.polylines(frame, [pts], False, color, line_thickness, cv2.LINE_AA)
    center = (int(pts[-1][0]), int(pts[-1][1]))
    if outline:
        cv2.circle(frame, center, ring_radius_px, OUTLINE_COLOR, line_thickness + 2, cv2.LINE_AA)
    cv2.circle(frame, center, ring_radius_px, color, line_thickness, cv2.LINE_AA)
    cv2.circle(frame, center, 2, color, -1, cv2.LINE_AA)


def draw_puck(
    frame: np.ndarray,
    renderer: AirHockeyRenderer,
    points_robot: np.ndarray,
    occluded: np.ndarray,
    center_offset: float,
    puck_radius_px: int,
) -> None:
    """Draw the REAL (camera-observed) puck track onto the Box2D frame.

    This is an overlay only --- the Box2D puck body stays parked and frozen, so
    the replayed paddle never touches it and the paddle metrics are unaffected.
    What you are looking at is "where the camera said the puck was" painted into
    the sim's coordinate frame, which is exactly the comparison a collision trial
    needs: the real puck against the paddle the sim reproduces.

    Occluded samples (the detector fell back to its last known position) are
    dropped from the trail and drawn hollow when current, so a frozen puck that
    is really just a stale reading can't be mistaken for a measurement.
    """
    if points_robot is None or len(points_robot) == 0:
        return
    occluded = (
        np.zeros(len(points_robot), dtype=bool)
        if occluded is None
        else np.asarray(occluded, dtype=float) != 0.0
    )
    pixels = np.array(
        [
            renderer.world_xy_to_output_pixel(float(p[0]) + center_offset, float(p[1]))
            for p in points_robot
        ],
        dtype=float,
    )
    pixels = np.round(pixels).astype(np.int32)

    # Trail: polyline over each run of consecutive fresh samples, so the line
    # never bridges an occlusion gap with an invented straight segment.
    run: list[np.ndarray] = []
    for pixel, is_occluded in zip(pixels, occluded):
        if is_occluded:
            if len(run) > 1:
                cv2.polylines(frame, [np.array(run)], False, PUCK_COLOR, 2, cv2.LINE_AA)
            run = []
            continue
        run.append(pixel)
    if len(run) > 1:
        cv2.polylines(frame, [np.array(run)], False, PUCK_COLOR, 2, cv2.LINE_AA)

    center = (int(pixels[-1][0]), int(pixels[-1][1]))
    cv2.circle(frame, center, puck_radius_px + 2, OUTLINE_COLOR, 2, cv2.LINE_AA)
    if occluded[-1]:
        cv2.circle(frame, center, puck_radius_px, PUCK_COLOR, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, center, puck_radius_px, PUCK_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 2, PUCK_COLOR, -1, cv2.LINE_AA)


def resample_arm_indices(times_s: np.ndarray, dt: float, max_preroll_s: float | None) -> list[int]:
    """One arm-phase sample per `dt` of wall clock, latest-sample-wins.

    The camera polls at ~30 Hz while the GIF advances one frame per env step
    (20 Hz), so the approach has to be put on the trial's time base or it would
    play back at the wrong speed. ``times_s`` is seconds relative to the trigger
    (<= 0). Returns indices INTO the raw track; each returned index is the newest
    sample at or before that frame's timestamp, and the trail is still drawn from
    every raw sample up to it, so nothing is thrown away visually.
    """
    if len(times_s) == 0:
        return []
    start = float(times_s[0])
    if max_preroll_s is not None:
        start = max(start, -abs(float(max_preroll_s)))
    if start >= 0.0:
        return []
    n_frames = int(np.floor(-start / dt))
    indices = []
    for k in range(n_frames):
        target = start + k * dt
        candidates = np.nonzero(times_s <= target)[0]
        if len(candidates) == 0:
            continue
        indices.append(int(candidates[-1]))
    return indices


def compose_frame(
    renderer: AirHockeyRenderer,
    source_hist: np.ndarray,
    replay_hist: np.ndarray,
    center_offset: float,
    paddle_radius_px: int,
    gif_width: int,
    puck_hist: np.ndarray | None = None,
    puck_occluded_hist: np.ndarray | None = None,
    puck_radius_px: int = 0,
) -> np.ndarray:
    """Box2D frame + both paddle tracks + the real puck overlay, resized for the GIF."""
    frame = renderer.get_frame().copy()
    # Source underneath and wider; replay on top and narrower.
    draw_track(frame, renderer, source_hist, center_offset, SOURCE_COLOR, paddle_radius_px,
               line_thickness=5, outline=True)
    draw_track(frame, renderer, replay_hist, center_offset, REPLAY_COLOR,
               max(2, paddle_radius_px - 3), line_thickness=2, outline=False)
    if puck_hist is not None and puck_radius_px > 0:
        draw_puck(frame, renderer, puck_hist, puck_occluded_hist, center_offset, puck_radius_px)
    return to_gif_frame(frame, gif_width)


def annotate(
    frame_rgb: np.ndarray,
    trial: dict,
    step_i: int,
    n_steps: int,
    error_m: float,
    source_label: str,
    show_puck: bool = False,
    phase_line: str | None = None,
) -> np.ndarray:
    """Legend + step + current error, drawn at native GIF resolution.

    ``phase_line`` replaces the step/error row for approach-phase frames, where
    no env step has run yet and there is no paddle error to report.
    """
    frame = np.ascontiguousarray(frame_rgb)
    sign = "+" if trial["sign"] > 0 else "-"
    header = [
        f"{trial['axis']}{sign} d{trial['delta']:.2f} t{trial['repeat']}",
        phase_line if phase_line is not None
        else f"step {step_i}/{n_steps}  err {error_m * 1000:5.1f}mm",
    ]
    for row, text in enumerate(header):
        origin = (3, 11 + row * 11)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    # Legend, bottom-left. Colors are BGR constants; frame is RGB here.
    legend = [(source_label, SOURCE_COLOR[::-1]), ("replay", REPLAY_COLOR[::-1])]
    if show_puck:
        legend.append(("real puck", PUCK_COLOR[::-1]))
    base_y = frame.shape[0] - 6 - (len(legend) - 1) * 11
    for row, (text, color) in enumerate(legend):
        y = base_y + row * 11
        cv2.line(frame, (4, y - 3), (14, y - 3), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(frame, (4, y - 3), (14, y - 3), tuple(int(c) for c in color), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_trial(
    env,
    trial: dict,
    park_xy_table: np.ndarray,
    renderer: AirHockeyRenderer | None,
    gif_width: int,
    gif_fps: int = 20,
    arm_preroll_s: float | None = None,
) -> dict:
    """Step Box2D through the recorded actions from the recorded start pose."""
    sim = env.simulator
    center_offset = float(sim.center_offset_constant)
    start_table = np.asarray(trial["start_xy_robot"], dtype=np.float64) + np.array([center_offset, 0.0])
    start_state = np.concatenate([start_table, [0.0, 0.0], park_xy_table, [0.0, 0.0]])
    env.reset_from_state(start_state)
    env.current_timestep = 0
    env.episode_return = 0.0
    env.episode_length = 0
    env.success_in_ep = False
    park_and_freeze_puck(env, park_xy_table)

    offset = np.array([center_offset, 0.0])
    source_positions = np.asarray(trial["positions_robot"], dtype=np.float64)
    n_steps = source_positions.shape[0]

    replay_positions: list[np.ndarray] = []
    replay_velocities: list[np.ndarray] = []
    replay_targets: list[np.ndarray] = []
    frames: list[np.ndarray] = []

    start_pos_table, _ = paddle_state(env)
    start_pos_robot = start_pos_table - offset
    paddle_radius_px = max(3, int(round(float(sim.paddle_radius) * float(env.ppm))))

    # Real puck track, if this source session recorded one. Frame k of the GIF
    # shows samples [0, k], so the puck advances in lockstep with both paddles.
    source_puck = trial.get("puck_robot")
    source_puck_occl = trial.get("puck_occluded")
    puck_start = trial.get("puck_start")
    has_puck = source_puck is not None and len(source_puck) > 0
    puck_radius_px = max(2, int(round(float(sim.puck_radius) * float(env.ppm)))) if has_puck else 0

    # Approach phase: everything the camera saw between "trial armed" and the
    # puck crossing the trigger line, i.e. before the robot was allowed to move.
    # It is prepended to the GIF as pre-roll and stays on screen as trail during
    # the trial, so one GIF covers the puck start-to-end rather than starting
    # mid-flight at the halfway line.
    arm_puck = trial.get("arm_puck_robot") if has_puck else None
    arm_puck_occl = trial.get("arm_puck_occluded")
    arm_times = trial.get("arm_puck_time_s")
    has_arm = arm_puck is not None and len(arm_puck) > 0 and arm_times is not None
    arm_frame_indices = (
        resample_arm_indices(arm_times, 1.0 / max(int(gif_fps), 1), arm_preroll_s)
        if has_arm else []
    )

    def puck_hist_through(step_i: int):
        """Puck samples visible at GIF frame `step_i` (0 = the start pose).

        Always includes the whole approach, so the trail the viewer sees during
        the trial reaches back to where the puck entered the table.
        """
        if not has_puck:
            return None, None
        points_parts = []
        occl_parts = []
        if has_arm:
            points_parts.append(arm_puck)
            occl_parts.append(arm_puck_occl)
        if puck_start is not None:
            points_parts.append(np.asarray(puck_start[0]).reshape(1, 2))
            occl_parts.append(np.array([puck_start[1]]))
        if step_i > 0:
            points_parts.append(source_puck[:step_i])
            occl_parts.append(source_puck_occl[:step_i])
        if not points_parts:
            return None, None
        return np.vstack(points_parts), np.concatenate(occl_parts)

    def arm_puck_hist_through(raw_index: int):
        """Every raw approach sample up to (and including) `raw_index`."""
        return arm_puck[: raw_index + 1], arm_puck_occl[: raw_index + 1]

    if renderer is not None and arm_frame_indices:
        # Paddle is parked at the start pose for the whole approach: the robot
        # does not move until the trigger fires, which is the protocol this
        # visualisation must not misrepresent.
        for raw_index in arm_frame_indices:
            puck_hist, puck_occl_hist = arm_puck_hist_through(raw_index)
            frames.append(annotate(
                compose_frame(renderer, np.array([trial["start_xy_robot"]]),
                              np.array([start_pos_robot]), center_offset,
                              paddle_radius_px, gif_width,
                              puck_hist=puck_hist, puck_occluded_hist=puck_occl_hist,
                              puck_radius_px=puck_radius_px),
                trial, 0, n_steps, 0.0, f"source ({trial['kind']})", show_puck=True,
                phase_line=f"approach  t{float(arm_times[raw_index]):+5.1f}s",
            ))

    if renderer is not None:
        puck_hist, puck_occl_hist = puck_hist_through(0)
        frames.append(annotate(
            compose_frame(renderer, np.array([trial["start_xy_robot"]]), np.array([start_pos_robot]),
                          center_offset, paddle_radius_px, gif_width,
                          puck_hist=puck_hist, puck_occluded_hist=puck_occl_hist,
                          puck_radius_px=puck_radius_px),
            trial, 0, n_steps, float(np.linalg.norm(start_pos_robot - trial["start_xy_robot"])),
            f"source ({trial['kind']})", show_puck=has_puck,
        ))

    for step_i in range(n_steps):
        env.step(np.asarray(trial["actions"][step_i], dtype=np.float32))
        park_and_freeze_puck(env, park_xy_table)
        pos_table, vel = paddle_state(env)
        replay_positions.append(pos_table - offset)
        replay_velocities.append(vel)
        replay_targets.append(target_position(env) - offset)
        if renderer is not None:
            source_hist = np.vstack([trial["start_xy_robot"], source_positions[: step_i + 1]])
            replay_hist = np.vstack([start_pos_robot, np.asarray(replay_positions)])
            error = float(np.linalg.norm(replay_positions[-1] - source_positions[step_i]))
            puck_hist, puck_occl_hist = puck_hist_through(step_i + 1)
            frames.append(annotate(
                compose_frame(renderer, source_hist, replay_hist, center_offset,
                              paddle_radius_px, gif_width,
                              puck_hist=puck_hist, puck_occluded_hist=puck_occl_hist,
                              puck_radius_px=puck_radius_px),
                trial, step_i + 1, n_steps, error, f"source ({trial['kind']})",
                show_puck=has_puck,
            ))

    replay_positions_arr = np.stack(replay_positions, axis=0)
    metrics = compute_metrics(source_positions, replay_positions_arr)
    return {
        "start_pos_robot": start_pos_robot,
        "replay_pos_robot": replay_positions_arr,
        "replay_vel": np.stack(replay_velocities, axis=0),
        "replay_target_robot": np.stack(replay_targets, axis=0),
        "position_error": metrics["_pos_err"],
        "delta_error": metrics["_delta_err"],
        "metrics": metrics,
        "frames": frames,
    }


def compute_metrics(source_pos: np.ndarray, replay_pos: np.ndarray) -> dict:
    """Position- and delta-level agreement between the two tracks.

    ``pos_err(t)  = ||p_src(t) - p_rep(t)||``            (absolute position gap)
    ``delta_err(t)= ||dp_src(t) - dp_rep(t)||``, where ``dp(t) = p(t) - p(t-1)``
                                                        (per-step motion gap, i.e.
                                                         a velocity-level residual)

    Position error accumulates -- once the two tracks separate they stay apart --
    so the delta metrics are the ones that localise *where* the dynamics differ.
    """
    pos_err = np.linalg.norm(source_pos - replay_pos, axis=1)
    source_delta = np.diff(source_pos, axis=0)
    replay_delta = np.diff(replay_pos, axis=0)
    delta_err = np.linalg.norm(source_delta - replay_delta, axis=1)
    return {
        "n_steps": int(pos_err.shape[0]),
        "sum_pos_err_mm": float(np.sum(pos_err) * 1000.0),
        "sum_delta_err_mm": float(np.sum(delta_err) * 1000.0),
        "max_pos_err_mm": float(np.max(pos_err) * 1000.0),
        "max_delta_err_mm": float(np.max(delta_err) * 1000.0),
        "mean_pos_err_mm": float(np.mean(pos_err) * 1000.0),
        "mean_delta_err_mm": float(np.mean(delta_err) * 1000.0),
        "_pos_err": pos_err,
        "_delta_err": delta_err,
    }


METRIC_KEYS = ["sum_pos_err_mm", "sum_delta_err_mm", "max_pos_err_mm", "max_delta_err_mm"]
METRIC_HEADERS = {
    "sum_pos_err_mm": "sum |dpos|",
    "sum_delta_err_mm": "sum |ddelta|",
    "max_pos_err_mm": "max |dpos|",
    "max_delta_err_mm": "max |ddelta|",
}


def aggregate(entries: list[dict]) -> dict:
    """Roll a set of per-trial metrics up into one row.

    Sums add (they are per-trial totals); maxima take the worst case. Means are
    per-trial averages so the aggregate row stays comparable to an individual one.
    """
    return {
        "n_trials": len(entries),
        "n_steps": int(sum(e["n_steps"] for e in entries)),
        "sum_pos_err_mm": float(sum(e["sum_pos_err_mm"] for e in entries)),
        "sum_delta_err_mm": float(sum(e["sum_delta_err_mm"] for e in entries)),
        "max_pos_err_mm": float(max(e["max_pos_err_mm"] for e in entries)),
        "max_delta_err_mm": float(max(e["max_delta_err_mm"] for e in entries)),
        "mean_sum_pos_err_mm": float(np.mean([e["sum_pos_err_mm"] for e in entries])),
        "mean_sum_delta_err_mm": float(np.mean([e["sum_delta_err_mm"] for e in entries])),
    }


def render_table(entries: list[dict]) -> str:
    """Per-trial rows, per-condition subtotals, and one overall aggregate."""
    lines = []
    name_w = max([len("condition / trial")] + [len(e["trial_name"]) for e in entries]) + 2
    header = f"{'condition / trial':<{name_w}}{'steps':>7}" + "".join(
        f"{METRIC_HEADERS[k]:>14}" for k in METRIC_KEYS
    )
    lines.append(header)
    lines.append("-" * len(header))

    conditions: dict[str, list[dict]] = {}
    for entry in entries:
        sign = "+" if entry["direction_sign"] > 0 else "-"
        conditions.setdefault(f"{entry['axis']}{sign} d{entry['action_delta']:.2f}", []).append(entry)

    for cond, rows in conditions.items():
        for entry in rows:
            lines.append(
                f"{'  ' + entry['trial_name']:<{name_w}}{entry['n_steps']:>7}"
                + "".join(f"{entry[k]:>14.1f}" for k in METRIC_KEYS)
            )
        agg = aggregate(rows)
        lines.append(
            f"{cond + ' (subtotal)':<{name_w}}{agg['n_steps']:>7}"
            + "".join(f"{agg[k]:>14.1f}" for k in METRIC_KEYS)
        )
        lines.append("")

    total = aggregate(entries)
    lines.append("=" * len(header))
    lines.append(
        f"{'TOTAL (' + str(total['n_trials']) + ' trials)':<{name_w}}{total['n_steps']:>7}"
        + "".join(f"{total[k]:>14.1f}" for k in METRIC_KEYS)
    )
    lines.append(
        f"{'  per-trial mean':<{name_w}}{'':>7}"
        f"{total['mean_sum_pos_err_mm']:>14.1f}{total['mean_sum_delta_err_mm']:>14.1f}"
        f"{'':>14}{'':>14}"
    )
    lines.append(
        "\nSums are per-trial totals over all timesteps; maxima are worst-case over all "
        "timesteps.\n|dpos| = ||p_source - p_replay||; |ddelta| = per-step motion difference "
        "||dp_source - dp_replay||. All values in mm."
    )
    return "\n".join(lines)


def write_metrics_files(out_dir: Path, entries: list[dict]) -> None:
    import csv

    fields = ["trial_name", "axis", "direction_sign", "action_delta", "repeat", "n_steps",
              *METRIC_KEYS, "mean_pos_err_mm", "mean_delta_err_mm"]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(render_table(entries) + "\n")


def write_replay_hdf5(path: Path, trial: dict, record: dict, session_meta: dict) -> dict:
    with h5py.File(path, "w") as hf:
        hf.create_dataset("actions", data=trial["actions"])
        hf.create_dataset("source_pos_robot", data=trial["positions_robot"])
        hf.create_dataset("source_vel", data=trial["velocities"])
        if trial["targets_robot"] is not None:
            hf.create_dataset("source_target_robot", data=trial["targets_robot"])
        if trial["is_settle"] is not None:
            hf.create_dataset("is_settle_step", data=trial["is_settle"])
        hf.create_dataset("replay_pos_robot", data=record["replay_pos_robot"])
        hf.create_dataset("replay_vel", data=record["replay_vel"])
        hf.create_dataset("replay_target_robot", data=record["replay_target_robot"])
        hf.create_dataset("position_error", data=record["position_error"])
        hf.create_dataset("delta_error", data=record["delta_error"])
        if trial.get("puck_track_table") is not None:
            # Approach + trial, one clock, table frame: what the camera saw for
            # the whole trial. Kept here so a physics fit can run off the replay
            # output without going back to the collection file.
            hf.create_dataset("source_puck_track", data=trial["puck_track_table"])
            hf.attrs["source_puck_track_columns"] = [
                "t_s_rel_trigger", "puck_x_obs", "puck_y_obs", "puck_occluded",
            ]
        hf.attrs["trial_name"] = trial["name"]
        hf.attrs["source_kind"] = trial["kind"]
        hf.attrs["source_file"] = str(trial["path"])
        hf.attrs["axis"] = trial["axis"]
        hf.attrs["direction_sign"] = trial["sign"]
        hf.attrs["action_delta"] = trial["delta"]
        hf.attrs["repeat"] = trial["repeat"]
        hf.attrs["settle_steps"] = trial["settle_steps"]
        hf.attrs["action_steps"] = trial["action_steps"]
        hf.attrs["start_xy_robot_source"] = trial["start_xy_robot"]
        hf.attrs["start_xy_robot_replay"] = record["start_pos_robot"]
        for key in ("replay_config_path", "workspace_lims", "hist_len",
                    "center_offset_constant", "session_start_iso"):
            hf.attrs[key] = session_meta[key]
    error = record["position_error"]
    metrics = {k: v for k, v in record["metrics"].items() if not k.startswith("_")}
    return {
        **metrics,
        "trial_name": trial["name"],
        "file": path.name,
        "source_file": trial["path"].name,
        "source_kind": trial["kind"],
        "axis": trial["axis"],
        "direction_sign": trial["sign"],
        "action_delta": trial["delta"],
        "repeat": trial["repeat"],
        "start_xy_robot_source": [float(v) for v in trial["start_xy_robot"]],
        "final_position_error_mm": float(error[-1] * 1000.0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a collected paddle-motion session through Box2D and overlay both tracks.",
    )
    parser.add_argument("--session", type=str, required=True,
                        help="Directory produced by either collector (contains *.hdf5 + manifest.json).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Default: <session>/replay.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Box2D env config used for the replay.")
    parser.add_argument("--trials", type=str, nargs="+", default=None,
                        help="Only replay trials whose filename contains one of these substrings.")
    parser.add_argument("--no-real-step-shift", action="store_true",
                        help="Do NOT shift real train_vals forward one step. The shift is on by "
                             "default because the real env logs each row before sending that "
                             "step's servoL; disable only to inspect the raw alignment.")
    parser.add_argument("--no-gifs", action="store_true", help="Skip GIF rendering.")
    parser.add_argument("--arm-preroll-s", type=float, default=None,
                        help="Seconds of the pre-trigger approach to show before the trial. "
                             "Default: the whole thing, from the camera's first sight of the "
                             "puck to the trigger. Use this to trim a long operator idle; "
                             "0 skips the approach entirely.")
    parser.add_argument("--gif-width", type=int, default=GIF_WIDTH)
    parser.add_argument("--gif-fps", type=int, default=GIF_FPS)
    args = parser.parse_args()

    session = Path(args.session)
    if not session.is_dir():
        raise SystemExit(f"--session is not a directory: {session}")
    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    trial_paths = discover_trials(session, args.trials)
    trials = [load_source_trial(p, real_step_shift=not args.no_real_step_shift)
              for p in trial_paths]

    env, _cfg = load_env(config_path)
    sim = env.simulator
    out_dir = Path(args.out_dir) if args.out_dir else session / "replay"
    gif_dir = out_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    park_xy_table = np.array([-(float(sim.length) / 2.0 - 0.01), 0.0], dtype=np.float64)
    session_meta = {
        "replay_config_path": str(config_path),
        "workspace_lims": np.asarray(sim.lims, dtype=float),
        "hist_len": int(sim.hist_len),
        "center_offset_constant": float(sim.center_offset_constant),
        "session_start_iso": datetime.now().astimezone().isoformat(),
    }

    renderer = None
    if not args.no_gifs:
        renderer = AirHockeyRenderer(
            env, orientation="vertical", show_target_position=True, show_acceleration_arrow=False,
        )

    kinds = sorted({t["kind"] for t in trials})
    print(f"[replay] source session={session} ({len(trials)} trials, kind={'/'.join(kinds)})")
    print(f"[replay] replay config={config_path}")
    print(f"[replay] out_dir={out_dir.resolve()}")

    entries: list[dict] = []
    direction_frames: dict[str, list[np.ndarray]] = {}
    for i, trial in enumerate(trials):
        record = replay_trial(env, trial, park_xy_table, renderer, int(args.gif_width),
                              gif_fps=int(args.gif_fps), arm_preroll_s=args.arm_preroll_s)
        entry = write_replay_hdf5(out_dir / f"{trial['name']}_replay.hdf5", trial, record, session_meta)
        entries.append(entry)
        if record["frames"]:
            save_gif(gif_dir / f"{trial['name']}_overlay.gif", record["frames"], int(args.gif_fps))
            key = f"{trial['axis']}{'pos' if trial['sign'] > 0 else 'neg'}"
            direction_frames.setdefault(key, []).extend(record["frames"])
        arm_frames = len(record["frames"]) - (entry["n_steps"] + 1) if record["frames"] else 0
        if arm_frames > 0:
            print(f"    approach: {arm_frames} pre-roll frames "
                  f"({arm_frames / max(int(args.gif_fps), 1):.1f}s) before the trigger")
        print(
            f"[{i + 1}/{len(trials)}] {trial['name']}: {entry['n_steps']} steps, "
            f"mean |dpos| {entry['mean_pos_err_mm']:.2f} mm, max {entry['max_pos_err_mm']:.2f} mm, "
            f"max |ddelta| {entry['max_delta_err_mm']:.2f} mm"
        )

    summary_gifs = []
    for key, frames in direction_frames.items():
        save_gif(gif_dir / f"summary_{key}_overlay.gif", frames, int(args.gif_fps))
        summary_gifs.append(f"summary_{key}_overlay.gif")

    total = aggregate(entries) if entries else {}
    manifest = {
        "session_start_iso": session_meta["session_start_iso"],
        "session_end_iso": datetime.now().astimezone().isoformat(),
        "source_session": str(session.resolve()),
        "source_kinds": kinds,
        "replay_config_path": str(config_path),
        "out_dir": str(out_dir.resolve()),
        "num_trials": len(entries),
        "aggregate": total,
        "gif_fps": int(args.gif_fps),
        "gif_width": int(args.gif_width),
        "summary_gifs": summary_gifs,
        "trials": entries,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if entries:
        write_metrics_files(out_dir, entries)
        print()
        print(render_table(entries))
    print(f"\n[replay] {len(entries)} trials replayed -> {out_dir.resolve()}")
    print(f"[replay] metrics written to metrics.csv / metrics.txt")
    if summary_gifs:
        print(f"[replay] GIFs in {gif_dir.resolve()} ({len(entries)} overlay + {len(summary_gifs)} summary)")


if __name__ == "__main__":
    main()
