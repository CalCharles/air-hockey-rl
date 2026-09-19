"""Open-loop paddle-motion data collection on the real UR5.

Runs a scripted battery of *controlled* paddle trajectories: from a fixed
initial pose, hold one constant action for N timesteps, save the trajectory,
reset, repeat. No policy, no puck interaction, no timestep smoothing.

Default plan (54 trajectories, one invocation)::

    directions: +x  -x            vertical   (along table length; -x = strike)
                +y  -y            horizontal (lateral)
                diagpos           bottom-left  -> top-right
                diagneg           bottom-right -> top-left
    delta     : 0.33, 0.66, 1.00  (the action itself: every travelled axis gets
                                   +-delta, others zero, so a diagonal at 0.33
                                   is (-0.33, +-0.33))
    repeats   : 3
    steps     : 20 constant-action steps per trial (plus a short zero-action settle)

Each direction starts at the far end of every axis it travels (see
``trial_plan.start_pose_for``), which is what makes the diagonals run corner to
corner.

Each trial is written to its own HDF5 file under ``--out-dir`` and indexed in
``manifest.json``.

Why "no timestep smoothing" needs the bundled config
----------------------------------------------------
``AirHockeyReal.get_transition`` passes every commanded target through
``filter_update``, which averages the last ``hist_len`` (target - pose) deltas.
``configs/robot_data_collection/paddle_motion_config.yaml`` sets ``hist_len: 1``,
where that average degenerates to the raw target, and zeroes the three
``transition_hold_steps_*`` knobs so no recovery hold can silently overwrite a
commanded action. Pass a different ``--config`` and you lose those guarantees;
the script prints a warning if it detects them missing.

What ``delta`` means (``--delta-mode action``, the default)
-----------------------------------------------------------
``delta`` IS the action. Every axis a direction travels gets ``+-delta``; axes it
does not travel are zero. At ``delta = 0.33``::

    +x -> ( 0.33,  0.00)      -y      -> ( 0.00, -0.33)
    -x -> (-0.33,  0.00)      diagpos -> (-0.33, +0.33)
    +y -> ( 0.00,  0.33)      diagneg -> (-0.33, -0.33)

Both diagonals carry a negative x component: they travel toward table centre (the
strike direction) and differ only in the y sign.

``rmax_x`` (0.26 m/step at |action|=1) and ``rmax_y`` (0.12) differ, and so does
the room on each axis, so the axes run out at different times: x saturates within
a couple of steps and the paddle then slides along the far edge for the remaining
timesteps. Accepted here --- the interesting part is the transient and the
steady-state velocity plateau before that. The script prints the predicted
saturation step per direction up front.

``--delta-mode workspace`` is the alternative: ``delta`` becomes the fraction of
the available room to cover over the trial and the per-axis action is derived so
both axes finish together (straight corner-to-corner path, never clips), at the
cost of speed and of decoupling ``delta`` from the action actually sent.

Usage::

    python -m scripts.robot_data_collection.collect_paddle_motion \\
        --out-dir data/robot_data_collection/paddle_motion_$(date +%Y%m%d_%H%M)

    # inspect the plan without touching the robot
    python -m scripts.robot_data_collection.collect_paddle_motion --dry-run

Second experiment: puck-paddle collisions (``--puck-collision``)
-----------------------------------------------------------------
Everything above describes the default battery, which never touches a puck.
``--puck-collision`` runs a different, interactive experiment out of the same
file and leaves the default path completely alone.

One trial: you type four things -- the puck release height (a free-text LABEL
-- "top", "3/4", "1/2" -- recorded with the trial and used for nothing else),
the delta, the y offset, and the trigger line. The last ENTER sends the arm to
the bottom of the table, shifted sideways by that offset; the script then watches
the overhead
camera with the env's own puck detector and, the instant a fresh detection
crosses the trigger line moving toward the robot, commands the constant action
``(-delta, 0)`` straight up the table into the incoming puck. A zero-action tail
records the rebound. Sweep delta over 0.33 / 0.66 / 1.0, the release height over
top / three-quarters / half, and the y offset over whatever set of impact
geometries you want -- 0 for a centred hit, +-0.02 / +-0.05 for progressively
more glancing ones.

Delta 0 is allowed and is the useful control condition: the action is (0, 0), so
the paddle drives to its offset start pose and then holds it while the puck
arrives, giving the restitution of the paddle face with no paddle velocity mixed
in. Everything records exactly as it does for a real strike.

The y offset
------------
Signed metres off the session start pose, applied through the RESET POSE, so the
arm travels there as part of the same moveL that parks it rather than as extra
commanded steps that would land in the saved trajectory. By the time the script
arms, the paddle is already where the trial wants it, and it does not move again
until the strike. Offsets that would leave the y workspace are clipped, loudly.

The paddle never moves sideways during a trial: the whole lateral geometry is
the one number you typed, which is what makes a battery of offsets a clean sweep
of impact points rather than a set of whatever-the-tracker-did outcomes.

The trigger line
----------------
Asked per trial, in observation x, and it is the black line NEAREST THE ROBOT --
not the centre line. That marking is physical: where it lands in observation
coordinates depends on the table, the camera and the homography, none of which
this repo knows, so there is no built-in default. ``--trigger-x`` only seeds the
first prompt; after that a bare ENTER reuses the previous trial's answer, and you
type a number on the trials where you want to move the line. For reference the
centre line is 0.0, the robot's end of the table +0.97 and the far end -0.97, so
the near line is a positive number; ``--show-arm-view`` draws your chosen line on
the rectified camera frame so you can put it on the black one.

Asking it last makes that final ENTER a "ready?" gate -- nothing moves until it
is answered::

    python -m scripts.robot_data_collection.collect_paddle_motion --puck-collision \\
        --out-dir data/robot_data_collection/puck_collision_$(date +%Y%m%d_%H%M)

    # same, with the first trigger prompt pre-filled
    python -m scripts.robot_data_collection.collect_paddle_motion --puck-collision \\
        --trigger-x 0.6 --out-dir <dir>

    # put the line on the black one, once
    python -m scripts.robot_data_collection.collect_paddle_motion --puck-collision \\
        --show-arm-view --trigger-x 0.6

    # geometry only, no robot, no camera
    python -m scripts.robot_data_collection.collect_paddle_motion --puck-collision --dry-run

Restarting into a directory that already holds trials continues the numbering
(highest ``collision_<n>_...`` on disk, plus one) instead of overwriting from
zero, and the previous ``manifest.json`` is carried forward under
``previous_sessions``.

Each saved trial reports the trigger line it used (``trigger_x_obs``), the
measured detection-to-motion latency (``trigger_to_action_s``) and the full
camera-rate approach track (``arm_puck_track``), which is what you use to decide
whether to type an earlier line on the next trial.
Each trial's file records the offset it used (``y_offset``), the pose it parked
at (``start_robot_xy_commanded``) and the unshifted session pose
(``session_start_robot_xy``), and the offset is in the filename too.

Third experiment: reversal jerk (``--jerk``)
---------------------------------------------
Paddle only --- no puck, no camera. How violently does the arm shake when a
trajectory reverses, and how much of that goes away if you ramp into the
turnaround instead of reversing at full speed?

One trial holds a constant action, ramps it linearly to zero over ``t`` steps
and immediately reverses, in ``--jerk-steps`` (20) commanded steps::

    phase   steps            action on the travel axis
    ---------------------------------------------------------------
    out     n_out            +delta                    (constant)
    slow    t                +delta * (t - k)/t, k=1..t  -> ends at 0
    back    N - n_out - t    -delta                    (constant)

``n_out`` and ``t`` are both typed per trial: "10 steps out, t slowing, 20 - 10 - t
back" is the split, and it is used exactly as typed. The ramp is straight-line
interpolation from delta to zero -- consecutive scales differ by exactly 1/t and
the t-th step commands precisely 0, so the paddle is at rest at the turnaround
before ``-delta`` is sent.

Four conditions, the two halves of the experiment::

    vertical   (x, along the table length)   up_down     down_up
    horizontal (y, lateral)                  right_left  left_right

Each starts at the far end of the axis it is about to travel --- ``up_down``
parks at the bottom of the table, ``right_left`` at the left edge --- so the
whole out-and-back runs in free space rather than against a limit.

``t`` is the experiment. At ``t = 0`` the commanded action jumps from ``+delta``
to ``-delta`` between two consecutive timesteps: the hardest reversal the action
space can express. Raising ``t`` replaces that jump with a linear ramp through
zero, so the paddle arrives at the turnaround already at rest. Sweep ``t`` at
fixed delta and read ``acc_x/y/z`` out of ``train_vals``.

Like the collision battery, it is interactive and asks per trial --- condition,
delta, ``n_out``, then ``t``, each taking a bare ENTER to reuse the previous
answer and ``q`` to finish --- and that last ENTER is the "ready?" gate, since
the arm moves next. ``--jerk-condition`` / ``--jerk-delta`` /
``--jerk-out-steps`` / ``--jerk-slowdown-steps`` seed the first prompts, and
``t`` is capped at the steps ``n_out`` leaves behind so the three phases always
sum to ``N``.

Watch the room, because nothing clips ``n_out`` for you. The action is a
velocity command of ``rmax_axis * delta`` metres per step, so ``n_out = 10`` at
delta 1.0 asks for 2.6 m on x where the workspace is 0.41 m: the paddle pins
against the far edge after two steps and stalls there, and the ramp and the
reversal then happen from a standstill. The plan report prints the predicted pin
step (and the largest ``n_out`` that stays inside) for every condition x delta
before anything moves, and each trial prints it again as it starts. y is the
roomier axis, 0.74 m at ``rmax_y = 0.12``. Trials that overrun still run and
still record --- ``saturates_at_step`` is in the file.

::

    python -m scripts.robot_data_collection.collect_paddle_motion --jerk \\
        --out-dir data/robot_data_collection/reversal_jerk_$(date +%Y%m%d_%H%M)

    # first prompts pre-filled: vertical up-then-down, delta 0.66, 10 out + 4 ramp
    python -m scripts.robot_data_collection.collect_paddle_motion --jerk \\
        --jerk-condition up_down --jerk-delta 0.66 \\
        --jerk-out-steps 10 --jerk-slowdown-steps 4

    # the out/slow/back split for every condition x delta, no robot
    python -m scripts.robot_data_collection.collect_paddle_motion --jerk --dry-run

Restarting into a directory that already holds ``jerk_<n>_...`` trials continues
the numbering and carries the old manifest forward under ``previous_sessions``,
exactly as the collision battery does.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Force the repo root ahead of any sourced-ROS `scripts` package.
# See scripts/real/README.md ("Sourced ROS env: avoiding the scripts package collision").
REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from datetime import datetime  # noqa: E402

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from airhockey import AirHockeyEnv  # noqa: E402
# save_collect / the overlay constants are the puck-detection path the real env
# itself runs (control_parameters.camera_callback and AirHockeyReal.poll_puck_detection);
# the collision experiment re-runs it directly so it can watch the puck between steps.
from airhockey.sims.real.control_parameters import (  # noqa: E402
    offset_constants,
    save_collect,
    visual_downscale_constant,
)
from airhockey.sims.real.overlay_utils import robot_to_display_pixel_int  # noqa: E402
from airhockey.sims.real.robot_control import apply_negative_z_force  # noqa: E402
from airhockey.sims.real.trajectory_merging import merge_trajectory  # noqa: E402
from scripts.robot_data_collection.trial_plan import (  # noqa: E402
    DIRECTIONS,
    Direction,
    Trial,
    add_plan_arguments,
    add_start_pose_arguments,
    CURVE_VELOCITY_GAIN_REAL,
    build_curve_plan,
    build_trial_plan,
    parse_curves,
    parse_directions,
    reindex,
    print_plan,
    room_in_direction,
    real_preview_geometry,
    resolve_plan_actions,
    curve_geometry,
    start_pose_for,
    step_displacement,
    steps_for_full_scale,
    steps_to_saturation,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_data_collection" / "paddle_motion_config.yaml"

# train_vals column layout, mirrored from airhockey/sims/real/proprioceptive_state.py
# so the saved files are self-describing. `pose_*` is the raw TCP pose in ROBOT
# frame (x ~ -0.68 at the reset pose); add center_offset_constant (1.2) to compare
# against the sim's table-frame paddle position.
VALS_COLUMN_NAMES = [
    "cur_time", "tidx", "i", "estop", "safety",
    *[f"pose_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")],
    *[f"speed_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")],
    *[f"force_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")],
    *[f"acc_{k}" for k in ("x", "y", "z")],
    *[f"desired_pose_{k}" for k in ("x", "y", "z", "rx", "ry", "rz")],
    "puck_x", "puck_y", "puck_occluded",
]


def default_out_dir(experiment: str) -> str:
    """Timestamped session directory, named after the experiment that fills it."""
    return f"data/robot_data_collection/{experiment}_{datetime.now():%Y%m%d_%H%M%S}"


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


def load_env(config_path: Path) -> tuple[AirHockeyEnv, dict]:
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ah = dict(cfg["air_hockey"])
    if "seed" not in ah:
        seed_cfg = cfg.get("seed", 0)
        if isinstance(seed_cfg, (list, tuple)):
            seed_cfg = seed_cfg[0] if len(seed_cfg) > 0 else 0
        ah["seed"] = int(seed_cfg)
    if "n_training_steps" not in ah:
        ah["n_training_steps"] = int(cfg.get("n_training_steps", 1))
    ah.setdefault("return_goal_obs", False)
    sim_params = ah.get("simulator_params", {})
    if isinstance(sim_params, dict):
        # Per-reset space prompt would stall a 36-trial batch; the session gate
        # in main() is the single place a human is asked to confirm.
        sim_params["wait_for_space_to_start"] = False
    return AirHockeyEnv(ah), cfg


def check_no_smoothing(cfg: dict) -> list[str]:
    """Return human-readable warnings if the config reintroduces smoothing."""
    sim_params = cfg.get("air_hockey", {}).get("simulator_params", {}) or {}
    warnings = []
    hist_len = int(sim_params.get("hist_len", 2))
    if hist_len != 1:
        warnings.append(
            f"hist_len={hist_len} (expected 1): filter_update will average the last "
            f"{hist_len} target deltas, i.e. commanded actions ARE smoothed."
        )
    for key in (
        "transition_hold_steps_on_estop_enter",
        "transition_hold_steps_on_estop_clear",
        "transition_hold_steps_on_safety_rearm",
    ):
        steps = int(sim_params.get(key, 0))
        if steps > 0:
            warnings.append(
                f"{key}={steps} (expected 0): a hold can replace commanded actions "
                "with hold-in-place targets mid-trial."
            )
    return warnings


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def settle_after_reset(sim, seconds: float) -> np.ndarray:
    """Hold still for `seconds` after a reset, keeping the paddle clamped.

    Two things happen in this window. The obvious one: the arm finishes settling
    at the reset pose, so every trial starts from the same at-rest state instead
    of from whatever residual motion the reset moveL left behind.

    The non-obvious one: UR's forceMode has a ~2 s controller-side timeout, and
    nothing refreshes it while we idle here -- reset()'s last clamp call is
    already several sleeps old by the time we return. Left alone, a 3 s pause
    drops compliance and the paddle lifts off the table before the trial starts.
    So the wait is sliced and re-clamped, the same way get_transition refreshes
    it every step. See notes/docs/environments/real-world/paddle-clamping-coverage-gap.md.

    Returns the settled TCP pose (falls back to the simulator's cached pose).
    """
    deadline = time.time() + max(0.0, float(seconds))
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        if not sim.control_off and not sim.above_table:
            try:
                if bool(sim.robot_command_readiness()["command_ready"]):
                    apply_negative_z_force(sim.ctrl, sim.rcv)
            except Exception as exc:
                print(f"  [settle] clamp refresh skipped: {exc}")
        time.sleep(min(0.25, remaining))
    try:
        return np.array(sim.rcv.getActualTCPPose(), dtype=float)
    except Exception:
        return np.array(getattr(sim, "pose", np.zeros(6)), dtype=float)


def paddle_robot_xy(sim) -> np.ndarray:
    """Current paddle position in ROBOT frame, as the env itself sees it.

    ``get_transition`` anchors every command on ``getTargetTCPPose()``
    (``target = pose_now + action * move_lims``), so the closed-loop arc tracker
    has to close its loop on exactly that quantity -- not on the actual TCP pose,
    which trails it -- or the commanded lead would be computed against a
    different reference than the one the env adds it to.
    """
    pose, _speed = sim._safe_target_pose_speed()
    return np.asarray(pose, dtype=float)[:2]


def run_trial(env: AirHockeyEnv, trial: Trial, action_steps: int, settle_steps: int,
              reset_settle_s: float, start_robot_xy: np.ndarray) -> dict:
    """Reset to the initial pose, then run the trial's action schedule.

    Line trials hold one constant action for N steps. Curve trials with
    closed-loop tracking recompute their action every step from the measured
    paddle pose (see ``trial_plan.ArcTracker``), which is what makes the realised
    arc the full-width one instead of ~15% of it.

    Returns a record with the per-step arrays plus the simulator's own
    proprioceptive ``vals`` / image buffers, harvested before the next reset
    clears them.
    """
    sim = env.simulator
    # Retarget the reset pose for this direction. reset_pose[0] is the mutable
    # [x, y, z, rx, ry, rz] list the env's own random/preset reset paths write to,
    # so reset() moveL's to whatever we leave here.
    sim.reset_pose[0][0] = float(start_robot_xy[0])
    sim.reset_pose[0][1] = float(start_robot_xy[1])
    obs, _info = env.reset()
    start_pose = settle_after_reset(sim, reset_settle_s)
    step_actions = trial.step_actions(action_steps)
    tracker = trial.make_tracker()
    zero_action = np.zeros(2, dtype=np.float32)

    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    is_settle: list[int] = []
    step_start_times: list[float] = []
    step_end_times: list[float] = []
    block_reasons: list[str] = []
    protective_stop = False
    aborted_at: int | None = None

    schedule = [(zero_action, 1)] * settle_steps + [(a, 0) for a in step_actions]
    for step_i, (step_action, settle_flag) in enumerate(schedule):
        if tracker is not None and not settle_flag:
            # Closed loop: the schedule entry is only the dry-run preview; the
            # action actually sent chases a carrot on the arc from where the
            # paddle really is.
            step_action = tracker.action(paddle_robot_xy(sim))
        t0 = time.time()
        obs, _reward, _terminated, _truncated, info = env.step(step_action)
        t1 = time.time()
        actions.append(np.asarray(step_action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        is_settle.append(int(settle_flag))
        step_start_times.append(t0)
        step_end_times.append(t1)
        block_reasons.append(str(info.get("command_block_reason", "none")))
        if bool(info.get("protective_stop", False)) or not bool(info.get("robot_step_ready", True)):
            protective_stop = True
            aborted_at = step_i
            print(
                f"  !! {trial.name}: robot not stepping at step {step_i} "
                f"(protective_stop={info.get('protective_stop')}, "
                f"reason={info.get('command_block_reason')}); aborting trial"
            )
            break

    return {
        "start_pose": start_pose,
        "start_robot_xy_commanded": np.asarray(start_robot_xy, dtype=float),
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 2), dtype=np.float32),
        "observations": np.stack(observations, axis=0),
        "is_settle": np.asarray(is_settle, dtype=np.int8),
        "step_start_times": np.asarray(step_start_times, dtype=np.float64),
        "step_end_times": np.asarray(step_end_times, dtype=np.float64),
        "command_block_reasons": block_reasons,
        "protective_stop": protective_stop,
        "aborted_at": aborted_at,
        # Harvested now: the next env.reset() clears sim.vals / sim.images.
        "sim_vals": list(sim.vals),
        "sim_images": list(sim.images),
    }


def write_trial_hdf5(
    path: Path,
    trial: Trial,
    record: dict,
    session_meta: dict,
    save_images: bool,
) -> dict:
    """Write one trial to HDF5 and return its manifest entry."""
    sim_vals = record["sim_vals"]
    sim_images = record["sim_images"]

    imgs = None
    if save_images and len(sim_images) > 0:
        imgs, vals = merge_trajectory(session_meta["image_path"], sim_images, sim_vals)
        if vals is None:  # image/value counts disagreed; keep the proprioception
            print(f"  !! {trial.name}: image/value misalignment, saving values only")
            imgs = None
            vals = np.stack(sim_vals, axis=0) if len(sim_vals) else np.zeros((0, 35))
    else:
        vals = np.stack(sim_vals, axis=0) if len(sim_vals) else np.zeros((0, 35))

    with h5py.File(path, "w") as hf:
        if imgs is not None:
            hf.create_dataset("train_img", shape=imgs.shape, compression="gzip", compression_opts=9, data=imgs)
        hf.create_dataset("train_vals", shape=vals.shape, compression="gzip", compression_opts=9, data=vals)
        hf.create_dataset("actions", data=record["actions"])
        hf.create_dataset("observations", data=record["observations"])
        hf.create_dataset("is_settle_step", data=record["is_settle"])
        hf.create_dataset("step_start_time", data=record["step_start_times"])
        hf.create_dataset("step_end_time", data=record["step_end_times"])
        hf.create_dataset(
            "command_block_reason",
            data=np.array(record["command_block_reasons"], dtype=h5py.string_dtype()),
        )
        hf.attrs["vals_column_names"] = VALS_COLUMN_NAMES
        hf.attrs["trial_name"] = trial.name
        hf.attrs["trial_index"] = trial.index
        hf.attrs["trial_type"] = trial.trial_type
        hf.attrs["direction_key"] = trial.key
        hf.attrs["direction_description"] = trial.description
        if trial.is_curve:
            # A curve holds a different action every step, so `action` below is the
            # peak one; the full sequence is the `actions` dataset.
            hf.attrs["direction_vec"] = np.zeros(2, dtype=np.int8)
            hf.attrs["curve_half_width_m"] = trial.curve.half_width_m
            hf.attrs["curve_height_m"] = trial.curve.height_m
            hf.attrs["curve_aspect"] = trial.curve.aspect
            hf.attrs["curve_target_speed_m_s"] = trial.delta
            hf.attrs["curve_heading_rate_deg_per_step"] = 180.0 / max(1, len(trial.schedule))
            hf.attrs["curve_tracking"] = "closed-loop" if trial.is_closed_loop else "open-loop"
            if trial.is_closed_loop:
                track = trial.track
                hf.attrs["curve_velocity_gain"] = track.velocity_gain
                hf.attrs["curve_lookahead_steps"] = track.lookahead_steps
                hf.attrs["curve_arc_length_m"] = track.arc_length_m
                hf.attrs["curve_x_base"] = track.x_base
                hf.attrs["curve_y_centre"] = track.y_centre
                # The preview the plan was printed from, so a saved trial can be
                # compared against what the tracker was predicted to command.
                hf.create_dataset("curve_preview_actions",
                                  data=np.asarray(trial.schedule, dtype=np.float32))
        else:
            hf.attrs["direction_vec"] = np.asarray(trial.direction.vec, dtype=np.int8)
        # Convenience fields: axis is "x"/"y" for single-axis trials, "xy" for a
        # diagonal and "curve" for an arc; direction_vec carries the truth.
        hf.attrs["axis"] = trial.axis
        hf.attrs["direction_sign"] = trial.sign
        hf.attrs["action_delta"] = trial.delta
        hf.attrs["action"] = trial.action
        hf.attrs["repeat"] = trial.repeat
        hf.attrs["settle_steps"] = session_meta["settle_steps"]
        hf.attrs["action_steps"] = session_meta["action_steps"]
        hf.attrs["start_pose"] = record["start_pose"]
        hf.attrs["start_robot_xy_commanded"] = record["start_robot_xy_commanded"]
        hf.attrs["protective_stop"] = record["protective_stop"]
        hf.attrs["aborted_at_step"] = -1 if record["aborted_at"] is None else record["aborted_at"]
        for key in (
            "config_path", "reset_pose", "move_lims", "workspace_lims", "edge_lims",
            "hist_len", "control_type", "control_mode", "block_time", "session_start_iso",
            "delta_mode", "curve_tracking", "curve_velocity_gain", "curve_steps",
        ):
            hf.attrs[key] = session_meta[key]

    return {
        "trial_name": trial.name,
        "file": path.name,
        "index": trial.index,
        "trial_type": trial.trial_type,
        "direction_key": trial.key,
        "direction_vec": ([0, 0] if trial.is_curve else [int(v) for v in trial.direction.vec]),
        "curve_aspect": (float(trial.curve.aspect) if trial.is_curve else None),
        "curve_tracking": (("closed-loop" if trial.is_closed_loop else "open-loop")
                           if trial.is_curve else None),
        "n_action_steps": int(len(trial.schedule)) if trial.is_curve else None,
        "axis": trial.axis,
        "direction_sign": trial.sign,
        "action_delta": trial.delta,
        "action": [float(v) for v in trial.action],
        "repeat": trial.repeat,
        "num_steps": int(vals.shape[0]),
        "settle_steps": session_meta["settle_steps"],
        "action_steps": session_meta["action_steps"],
        "start_pose_xy": [float(record["start_pose"][0]), float(record["start_pose"][1])],
        "start_robot_xy_commanded": [float(v) for v in record["start_robot_xy_commanded"]],
        "has_images": imgs is not None,
        "protective_stop": bool(record["protective_stop"]),
        "aborted_at_step": record["aborted_at"],
    }


# ---------------------------------------------------------------------------
# Experiment 2: puck-paddle collision (--puck-collision)
# ---------------------------------------------------------------------------
#
# Protocol, one trial at a time:
#
#   1. the operator types the puck release height (a free-text LABEL, recorded
#      but never used to command anything) and the delta;
#   2. ENTER -> the arm resets to the bottom of the table (same start pose the
#      `-x` vertical trials use) and settles;
#   3. the script ARMS: it polls the overhead camera and runs the same puck
#      detector the env uses, printing where it sees the puck;
#   4. the operator releases the puck; the moment a fresh detection crosses the
#      table's centre line moving toward the robot, the constant action
#      ``(-delta, 0)`` is commanded for --collision-action-steps steps, i.e.
#      the paddle drives straight up the table into the incoming puck;
#   5. a zero-action tail records the puck's post-collision flight.
#
# Frames. The detector returns ROBOT-frame metres; the env (and therefore this
# script, and the saved `puck_x` column) adds ``center_offset_constant`` (1.2)
# to get OBSERVATION/TABLE frame, whose origin is the centre of the table:
#
#     x_obs = +length/2 = +0.9652   bottom rail, behind the robot
#     x_obs =  0.0                  the bold black centre line  <- the trigger
#     x_obs = -length/2 = -0.9652   top rail, far end
#
# so the puck rolling down toward the robot has an INCREASING x_obs, the paddle
# lives at x_obs in [0.37, 0.78], and "up the table" is -x in both frames.

COLLISION_DIRECTION_KEY = "xneg"  # vertical, toward table centre -- the strike direction

# Table centre line in OBSERVATION frame. table_x_top/-bot are -+length/2
# (airhockey/airhockey_base.py), so the midpoint is exactly zero.
#
# It is NOT the trigger: this experiment fires on the black line NEAREST THE
# ROBOT, which is a physical marking on the real table and has no coordinate
# anywhere in this repo -- so it is asked per trial (--trigger-x only seeds the
# first prompt). The centre line is kept here as the one reference point the
# frame definition does pin down.
TABLE_MIDLINE_OBS_X = 0.0
# Half the table length (airhockey/airhockey_base.py: length 1.9304), i.e. the
# rails, used only to reject an obviously-off-the-table trigger value.
TABLE_HALF_LENGTH_OBS_X = 0.9652
# The paddle's parked x. A trigger line beyond it means the puck arrives before
# the strike is commanded, which is worth a note but not an error.
PADDLE_START_OBS_X = 0.770


@dataclass
class CollisionTrialSpec:
    """One operator-specified collision trial."""

    index: int
    height_label: str          # free text, documentation only ("top", "3/4", "1/2")
    delta: float               # |action| on x; the action sent is (-delta, 0)
    y_offset: float = 0.0      # metres off the start pose, signed; +y is left of the robot
    trigger_x: float = TABLE_MIDLINE_OBS_X   # obs-frame x of the line that fires the strike

    @property
    def y_slug(self) -> str:
        # Signed and zero-padded so a directory of trials sorts and reads
        # sensibly: y+0.050, y-0.050, y+0.000.
        return f"y{self.y_offset:+.3f}"

    @property
    def start_robot_xy(self) -> np.ndarray:
        """Where this trial parks the paddle: the session start pose, y-shifted."""
        return np.array([0.0, float(self.y_offset)], dtype=float)

    @property
    def height_slug(self) -> str:
        slug = "".join(ch if ch.isalnum() else "-" for ch in self.height_label.strip().lower())
        slug = "-".join(part for part in slug.split("-") if part)
        return slug or "unlabelled"

    @property
    def action(self) -> np.ndarray:
        # delta 0 gives the zero action: the paddle holds its offset start pose
        # for the whole "strike", which is the stationary-paddle control trial.
        return np.array([-float(self.delta), 0.0], dtype=np.float32)

    @property
    def name(self) -> str:
        return (f"collision_{self.index:03d}_h{self.height_slug}"
                f"_delta{self.delta:.2f}_{self.y_slug}")


def refresh_paddle_clamp(sim) -> None:
    """Re-apply the paddle's downward force so UR's forceMode does not time out.

    The arm phase can idle for many seconds waiting on the operator, which is far
    longer than forceMode's ~2 s controller-side timeout: without this the paddle
    goes compliant-free and lifts off the table before the puck ever arrives.
    Same reasoning (and same guards) as ``settle_after_reset``. See
    notes/docs/environments/real-world/paddle-clamping-coverage-gap.md.
    """
    if sim.control_off or sim.above_table:
        return
    try:
        if bool(sim.robot_command_readiness()["command_ready"]):
            apply_negative_z_force(sim.ctrl, sim.rcv)
    except Exception as exc:
        print(f"  [clamp] refresh skipped: {exc}")


def poll_puck_once(sim, keep_frame: bool = False):
    """One camera frame -> one puck detection, in the env's own frame.

    This is ``AirHockeyReal.poll_puck_detection`` (the detection-only tick the
    rollout startup gate uses: it advances no timestep and commands no motion)
    with two deliberate differences:

      * the rectified frame is handed back so the arm view can draw on it;
      * nothing is appended to ``sim.images``. ``merge_trajectory`` pairs
        ``sim.images`` with ``sim.vals`` one-to-one and only ``env.step``
        appends to ``vals``, so an arm-phase frame left in ``images`` would
        shift every image of the saved trial by one and trip the misalignment
        guard in ``write_trial_hdf5``.

    ``sim.puck_history`` IS appended to, which is the point: the detector gates
    candidates on the predicted next position, so it needs the same warm history
    during the arm phase that it has mid-rollout, and the trial's first
    observation then carries a real history instead of the reset sentinels.

    Returns ``(puck, frame, frame_received_s)`` with ``puck = (x, y, occluded)``,
    x in OBSERVATION frame and ``occluded == 0`` meaning a fresh detection.
    """
    image, _save_img, frame_received_s = save_collect(
        sim.cap, None, None, None,
        show=False, lims=None, edge_lims=None, region_x_offset=sim.x_offset,
    )
    puck = np.array(
        sim.puck_detector(image, sim.puck_history, rotate=False, **sim.puck_detector_kwargs),
        dtype=float,
    )
    # Detector hit (occluded == 0) is detector-frame x and needs the centre
    # offset; the occlusion fallback already comes back in state frame.
    if int(puck[2]) == 0:
        puck[0] += sim.center_offset_constant
    sim.puck_history.append(puck)
    sim.puck = puck[:2]

    # Keep the paddle history warm from the same telemetry the env uses.
    tcp_target_pose, tcp_target_speed = sim._safe_target_pose_speed()
    state_pose, state_speed, _ = sim._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
    sim.pose = np.array(state_pose, dtype=float)
    sim.speed = np.array(state_speed, dtype=float)
    paddle_xy = sim._paddle_observation_xy_from_pose(sim.pose[:2])
    sim.paddle_history.append([float(paddle_xy[0]), float(paddle_xy[1]), 0])

    return puck, (image if keep_frame else None), frame_received_s


def draw_arm_view(frame, sim, puck, trigger_x: float, waited_s: float, rate_hz: float) -> None:
    """Show the rectified camera frame with the trigger line and the puck on it.

    Purely an operator aid for --show-arm-view: it is how you confirm that
    ``--trigger-x`` really lands on the bold black centre line of YOUR table
    before trusting the trigger.
    """
    view = frame.copy()
    trigger_px = robot_to_display_pixel_int(
        obs_x_to_robot_x(sim, trigger_x), 0.0,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )[0]
    cv2.line(view, (trigger_px, 0), (trigger_px, view.shape[0]), (0, 255, 255), 2)

    fresh = int(puck[2]) == 0
    if float(puck[0]) > -1.5:  # -2 + offset is the never-seen sentinel
        center = robot_to_display_pixel_int(
            obs_x_to_robot_x(sim, puck[0]), float(puck[1]),
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
        )
        cv2.drawMarker(view, center, (0, 255, 0) if fresh else (0, 165, 255),
                       cv2.MARKER_CROSS, 24, 2)
    lines = [
        f"ARMED  trigger x_obs={trigger_x:+.3f}  waited {waited_s:5.1f}s  {rate_hz:4.1f} Hz",
        (f"puck x_obs={puck[0]:+.3f} y={puck[1]:+.3f} " + ("DETECTED" if fresh else "stale")),
        "release the puck; Ctrl-C aborts the session",
    ]
    for i, line in enumerate(lines):
        org = (10, 24 + 22 * i)
        cv2.putText(view, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("arm", view)
    cv2.waitKey(1)


def obs_x_to_robot_x(sim, x_obs: float) -> float:
    """OBSERVATION-frame x -> ROBOT-frame x (the inverse of the env's offset)."""
    return float(x_obs) - float(sim.center_offset_constant)


def trial_start_robot_xy(sim, session_start_xy, y_offset: float,
                         margin: float = 0.02) -> np.ndarray:
    """Session start pose shifted by `y_offset`, clipped into the y workspace.

    Clipping is loud rather than silent: an offset the arm cannot reach would
    otherwise be recorded as if it had been applied, and every trial at that
    setting would quietly be a different experiment from the one you typed.
    """
    start = np.asarray(session_start_xy, dtype=float).copy()
    wanted = float(start[1]) + float(y_offset)
    y_lo = float(sim.lims[2]) + float(margin)
    y_hi = float(sim.lims[3]) - float(margin)
    clipped = float(np.clip(wanted, y_lo, y_hi))
    if abs(clipped - wanted) > 1e-9:
        print(f"  !! y offset {y_offset:+.3f} would put the paddle at y={wanted:+.3f}, "
              f"outside the workspace [{y_lo:+.3f}, {y_hi:+.3f}]; clipped to {clipped:+.3f}")
    start[1] = clipped
    return start


def wait_for_puck_trigger(sim, trigger_x: float, timeout_s: float, *,
                          min_far_frames: int, clamp_every_s: float,
                          print_every: int, show_view: bool) -> dict:
    """Poll the camera until the puck crosses ``trigger_x`` toward the robot.

    The trigger is a genuine CROSSING, not a half-plane test: at least
    ``min_far_frames`` fresh detections must first land on the far side of the
    line (x_obs < trigger_x) before a fresh detection at or past it fires. That
    rules out arming on a puck that is already sitting on the robot's half, and
    on a single spurious detection.

    Stale frames (the detector's last-known fallback, ``occluded != 0``) are
    recorded but never trigger --- a hand reaching in to release the puck is
    exactly when the detector goes stale, and firing on a held-over position
    would launch the paddle at nothing.

    Returns a dict with the trigger outcome and the full arm-phase puck track.
    """
    track: list[tuple[float, float, float, float]] = []
    t_start = time.time()
    last_clamp = 0.0
    far_frames = 0
    fresh_frames = 0
    polls = 0
    prev_fresh: tuple[float, float] | None = None  # (t, x_obs)
    trigger_puck = None
    trigger_time = None
    approach_speed = float("nan")
    reason = "timeout"

    print(f"  [arm] ARMED -- release the puck. Trigger: fresh detection crossing "
          f"x_obs={trigger_x:+.3f} (robot x={obs_x_to_robot_x(sim, trigger_x):+.3f}) "
          f"toward the robot, after {min_far_frames} frame(s) beyond it. "
          f"Timeout {timeout_s:.0f}s.")

    while True:
        now = time.time()
        waited = now - t_start
        if waited > timeout_s:
            break
        if now - last_clamp >= clamp_every_s:
            refresh_paddle_clamp(sim)
            last_clamp = now

        puck, frame, frame_t = poll_puck_once(sim, keep_frame=show_view)
        polls += 1
        track.append((float(frame_t), float(puck[0]), float(puck[1]), float(puck[2])))
        rate_hz = polls / max(waited, 1e-6)

        fresh = int(puck[2]) == 0
        if fresh:
            fresh_frames += 1
            x_obs = float(puck[0])
            if x_obs < trigger_x:
                far_frames += 1
            elif far_frames >= min_far_frames:
                if prev_fresh is not None:
                    dt = float(frame_t) - prev_fresh[0]
                    if dt > 1e-6:
                        approach_speed = (x_obs - prev_fresh[1]) / dt
                trigger_puck = np.asarray(puck, dtype=float)
                trigger_time = float(frame_t)
                reason = "crossed"
                break
            prev_fresh = (float(frame_t), x_obs)

        if show_view and frame is not None:
            draw_arm_view(frame, sim, puck, trigger_x, waited, rate_hz)
        if print_every > 0 and polls % print_every == 0:
            print(f"  [arm] {waited:5.1f}s  {rate_hz:4.1f} Hz  "
                  f"{'det  ' if fresh else 'stale'} x_obs={puck[0]:+.3f} y={puck[1]:+.3f}  "
                  f"far_frames={far_frames}")

    waited_s = time.time() - t_start
    fresh_x = [row[1] for row in track if int(row[3]) == 0]
    if reason == "crossed":
        print(f"  [arm] TRIGGER after {waited_s:.2f}s at x_obs={trigger_puck[0]:+.3f} "
              f"y={trigger_puck[1]:+.3f} (approach {approach_speed:+.2f} m/s) -- striking.")
    else:
        detail = (f"{fresh_frames}/{polls} fresh frames, "
                  f"fresh x_obs range [{min(fresh_x):+.3f}, {max(fresh_x):+.3f}]"
                  if fresh_x else f"NO fresh detections in {polls} frames")
        print(f"  [arm] timed out after {waited_s:.1f}s without a crossing ({detail}). "
              f"Trial discarded.")

    return {
        "triggered": reason == "crossed",
        "reason": reason,
        "track": np.asarray(track, dtype=np.float64).reshape(-1, 4),
        "polls": polls,
        "fresh_frames": fresh_frames,
        "far_frames": far_frames,
        "waited_s": float(waited_s),
        "rate_hz": float(polls / max(waited_s, 1e-6)),
        "trigger_puck": trigger_puck,
        "trigger_time": trigger_time,
        "approach_speed_obs_m_s": float(approach_speed),
    }


def run_collision_trial(env: AirHockeyEnv, spec: CollisionTrialSpec, session: dict,
                        start_robot_xy: np.ndarray) -> dict:
    """Reset to the bottom (y-offset by the trial), wait for the crossing, strike.

    Structurally the same as ``run_trial`` -- same reset retarget, same settle,
    same per-step bookkeeping and protective-stop abort -- with the arm phase
    spliced in between the settle and the constant action, and a zero-action
    tail after it so the puck's post-collision flight is recorded too.

    ``spec.y_offset`` shifts the paddle sideways off the session start pose, and
    it is applied HERE, through the reset pose, so the arm travels there as part
    of the same moveL that parks it -- not as extra commanded steps that would
    land in the saved trajectory. By the time the script arms, the paddle is
    already sitting where the trial wants it and it does not move again until
    the strike.

    Returns ``{"triggered": False, ...}`` and nothing else if the puck never
    crossed; the caller does not write a file for those.
    """
    sim = env.simulator
    trial_xy = trial_start_robot_xy(sim, start_robot_xy, spec.y_offset)
    sim.reset_pose[0][0] = float(trial_xy[0])
    sim.reset_pose[0][1] = float(trial_xy[1])
    obs, _info = env.reset()
    start_pose = settle_after_reset(sim, session["reset_settle_s"])

    zero_action = np.zeros(2, dtype=np.float32)
    action = spec.action

    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    phases: list[str] = []
    is_settle: list[int] = []
    step_start_times: list[float] = []
    step_end_times: list[float] = []
    block_reasons: list[str] = []
    terminated_flags: list[int] = []
    truncated_flags: list[int] = []
    protective_stop = False
    aborted_at: int | None = None

    def run_steps(schedule) -> bool:
        """Step through `schedule`, recording as we go. False = abort the trial."""
        nonlocal obs, protective_stop, aborted_at
        for step_action, phase in schedule:
            t0 = time.time()
            obs, _reward, terminated, truncated, info = env.step(step_action)
            t1 = time.time()
            actions.append(np.asarray(step_action, dtype=np.float32))
            observations.append(np.asarray(obs, dtype=np.float32))
            phases.append(phase)
            is_settle.append(int(phase == "settle"))
            step_start_times.append(t0)
            step_end_times.append(t1)
            block_reasons.append(str(info.get("command_block_reason", "none")))
            # The juggle task terminates the moment the puck reaches the bottom
            # rail, which for this experiment is a normal outcome rather than a
            # reason to stop: the flags are recorded and the schedule runs on.
            terminated_flags.append(int(bool(terminated)))
            truncated_flags.append(int(bool(truncated)))
            if bool(info.get("protective_stop", False)) or not bool(info.get("robot_step_ready", True)):
                protective_stop = True
                aborted_at = len(actions) - 1
                print(
                    f"  !! {spec.name}: robot not stepping at step {aborted_at} "
                    f"(protective_stop={info.get('protective_stop')}, "
                    f"reason={info.get('command_block_reason')}); aborting trial"
                )
                return False
        return True

    # Pre-trigger zero-action baseline. Off by default: it would sit in `vals`
    # separated from the strike by however long the operator took to release the
    # puck, which reads as a gap in the trajectory rather than a baseline.
    if not run_steps([(zero_action, "settle")] * int(session["collision_settle_steps"])):
        return {"triggered": False, "reason": "protective_stop_before_arm"}

    arm = wait_for_puck_trigger(
        sim, float(spec.trigger_x), float(session["arm_timeout_s"]),
        min_far_frames=int(session["min_far_frames"]),
        clamp_every_s=float(session["clamp_every_s"]),
        print_every=int(session["arm_print_every"]),
        show_view=bool(session["show_arm_view"]),
    )
    if not arm["triggered"]:
        return {"triggered": False, "reason": arm["reason"], "arm": arm}

    strike_t0 = time.time()
    run_steps(
        [(action, "action")] * int(session["collision_action_steps"])
        + [(zero_action, "post")] * int(session["collision_post_steps"])
    )

    return {
        "triggered": True,
        "reason": arm["reason"],
        "arm": arm,
        # Detection-to-first-command latency: how far past the line the puck had
        # actually travelled by the time the paddle was told to move.
        "trigger_to_action_s": strike_t0 - float(arm["trigger_time"]),
        "start_pose": start_pose,
        "start_robot_xy_commanded": np.asarray(trial_xy, dtype=float),
        "session_start_robot_xy": np.asarray(start_robot_xy, dtype=float),
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 2), dtype=np.float32),
        "observations": np.stack(observations, axis=0),
        "step_phase": phases,
        "is_settle": np.asarray(is_settle, dtype=np.int8),
        "step_start_times": np.asarray(step_start_times, dtype=np.float64),
        "step_end_times": np.asarray(step_end_times, dtype=np.float64),
        "command_block_reasons": block_reasons,
        "terminated": np.asarray(terminated_flags, dtype=np.int8),
        "truncated": np.asarray(truncated_flags, dtype=np.int8),
        "protective_stop": protective_stop,
        "aborted_at": aborted_at,
        # Harvested now: the next env.reset() clears sim.vals / sim.images.
        "sim_vals": list(sim.vals),
        "sim_images": list(sim.images),
    }


def write_collision_trial_hdf5(path: Path, spec: CollisionTrialSpec, record: dict,
                               session_meta: dict, save_images: bool) -> dict:
    """Write one collision trial to HDF5 and return its manifest entry."""
    sim_vals = record["sim_vals"]
    sim_images = record["sim_images"]

    imgs = None
    if save_images and len(sim_images) > 0:
        imgs, vals = merge_trajectory(session_meta["image_path"], sim_images, sim_vals)
        if vals is None:  # image/value counts disagreed; keep the proprioception
            print(f"  !! {spec.name}: image/value misalignment, saving values only")
            imgs = None
            vals = np.stack(sim_vals, axis=0) if len(sim_vals) else np.zeros((0, 35))
    else:
        vals = np.stack(sim_vals, axis=0) if len(sim_vals) else np.zeros((0, 35))

    arm = record["arm"]
    with h5py.File(path, "w") as hf:
        if imgs is not None:
            hf.create_dataset("train_img", shape=imgs.shape, compression="gzip",
                              compression_opts=9, data=imgs)
        hf.create_dataset("train_vals", shape=vals.shape, compression="gzip",
                          compression_opts=9, data=vals)
        hf.create_dataset("actions", data=record["actions"])
        hf.create_dataset("observations", data=record["observations"])
        hf.create_dataset("is_settle_step", data=record["is_settle"])
        hf.create_dataset("step_phase",
                          data=np.array(record["step_phase"], dtype=h5py.string_dtype()))
        hf.create_dataset("step_start_time", data=record["step_start_times"])
        hf.create_dataset("step_end_time", data=record["step_end_times"])
        hf.create_dataset("terminated", data=record["terminated"])
        hf.create_dataset("truncated", data=record["truncated"])
        hf.create_dataset(
            "command_block_reason",
            data=np.array(record["command_block_reasons"], dtype=h5py.string_dtype()),
        )
        # Arm phase: the puck's approach before the strike, at camera rate rather
        # than at env-step rate. Columns: frame_time, x_obs, y_obs, occluded.
        hf.create_dataset("arm_puck_track", data=arm["track"])
        hf.attrs["arm_puck_track_columns"] = ["frame_time_s", "puck_x_obs", "puck_y_obs", "puck_occluded"]

        hf.attrs["vals_column_names"] = VALS_COLUMN_NAMES
        hf.attrs["trial_name"] = spec.name
        hf.attrs["trial_index"] = spec.index
        hf.attrs["trial_type"] = "puck_collision"
        hf.attrs["experiment"] = "puck_collision"
        # Operator-entered, documentation only: nothing in the run depends on it.
        hf.attrs["puck_height_label"] = spec.height_label
        # Operator-entered lateral offset off the session start pose, applied via
        # the reset pose. `start_robot_xy_commanded` is the offset pose the trial
        # actually parked at (after workspace clipping); `session_start_robot_xy`
        # is the unshifted one every trial shares.
        hf.attrs["y_offset"] = float(spec.y_offset)
        hf.attrs["session_start_robot_xy"] = record["session_start_robot_xy"]
        hf.attrs["direction_key"] = COLLISION_DIRECTION_KEY
        hf.attrs["direction_description"] = "vertical, up the table -- the strike direction"
        hf.attrs["direction_vec"] = np.array([-1, 0], dtype=np.int8)
        hf.attrs["axis"] = "x"
        hf.attrs["direction_sign"] = -1
        hf.attrs["action_delta"] = float(spec.delta)
        hf.attrs["action"] = spec.action
        hf.attrs["trigger_x_obs"] = float(spec.trigger_x)
        hf.attrs["trigger_puck_obs"] = np.asarray(arm["trigger_puck"], dtype=float)
        hf.attrs["trigger_time"] = float(arm["trigger_time"])
        hf.attrs["trigger_to_action_s"] = float(record["trigger_to_action_s"])
        hf.attrs["trigger_approach_speed_obs_m_s"] = float(arm["approach_speed_obs_m_s"])
        hf.attrs["arm_wait_s"] = float(arm["waited_s"])
        hf.attrs["arm_poll_rate_hz"] = float(arm["rate_hz"])
        hf.attrs["arm_polls"] = int(arm["polls"])
        hf.attrs["arm_fresh_frames"] = int(arm["fresh_frames"])
        hf.attrs["settle_steps"] = int(session_meta["collision_settle_steps"])
        hf.attrs["action_steps"] = int(session_meta["collision_action_steps"])
        hf.attrs["post_steps"] = int(session_meta["collision_post_steps"])
        hf.attrs["start_pose"] = record["start_pose"]
        hf.attrs["start_robot_xy_commanded"] = record["start_robot_xy_commanded"]
        hf.attrs["protective_stop"] = record["protective_stop"]
        hf.attrs["aborted_at_step"] = -1 if record["aborted_at"] is None else record["aborted_at"]
        for key in (
            "config_path", "reset_pose", "move_lims", "workspace_lims", "edge_lims",
            "hist_len", "control_type", "control_mode", "block_time", "session_start_iso",
            "center_offset_constant",
        ):
            hf.attrs[key] = session_meta[key]

    return {
        "trial_name": spec.name,
        "file": path.name,
        "index": spec.index,
        "trial_type": "puck_collision",
        "puck_height_label": spec.height_label,
        "action_delta": float(spec.delta),
        "y_offset": float(spec.y_offset),
        "action": [float(v) for v in spec.action],
        "direction_key": COLLISION_DIRECTION_KEY,
        "num_steps": int(vals.shape[0]),
        "settle_steps": int(session_meta["collision_settle_steps"]),
        "action_steps": int(session_meta["collision_action_steps"]),
        "post_steps": int(session_meta["collision_post_steps"]),
        "trigger_x_obs": float(spec.trigger_x),
        "trigger_puck_obs": [float(v) for v in arm["trigger_puck"]],
        "trigger_to_action_s": float(record["trigger_to_action_s"]),
        "trigger_approach_speed_obs_m_s": float(arm["approach_speed_obs_m_s"]),
        "arm_wait_s": float(arm["waited_s"]),
        "arm_poll_rate_hz": float(arm["rate_hz"]),
        "start_pose_xy": [float(record["start_pose"][0]), float(record["start_pose"][1])],
        "start_robot_xy_commanded": [float(v) for v in record["start_robot_xy_commanded"]],
        "has_images": imgs is not None,
        "protective_stop": bool(record["protective_stop"]),
        "aborted_at_step": record["aborted_at"],
        "terminated_any": bool(np.any(record["terminated"])),
    }


COLLISION_INDEX_RE = re.compile(r"^collision_(\d+)_")


def next_trial_index(out_dir: Path, prefix: str, pattern: re.Pattern) -> int:
    """First unused trial index in `out_dir`: highest one already there, plus one.

    Trials are keyed by the index in their filename, so restarting a session
    into a directory that already holds trials has to continue the numbering
    rather than restart it -- otherwise trial 0 of the new run silently
    overwrites trial 0 of the old one, and the manifest of the second run
    describes files that are no longer what it says they are.

    Anything in the directory that does not parse as `<prefix>_<n>_...` is
    ignored, so a hand-renamed or half-written file cannot push the counter to a
    nonsense value.
    """
    highest = -1
    for path in sorted(out_dir.glob(f"{prefix}_*.hdf5")):
        match = pattern.match(path.name)
        if match is not None:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def load_previous_sessions(manifest_path: Path) -> list[dict]:
    """Earlier sessions recorded in `manifest_path`, oldest first, flattened.

    A restarted session writes its own manifest over the old one. The trial
    HDF5s survive that (they are never overwritten now that the index
    continues), but the session-level record -- which args produced which
    trials -- would be lost, so the previous manifest is carried forward here
    instead. The chain is flattened, so ten sessions in one directory give ten
    entries rather than ten levels of nesting.
    """
    if not manifest_path.is_file():
        return []
    try:
        with open(manifest_path) as f:
            prior = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[collision] existing manifest.json could not be read ({exc}); "
              f"it will be replaced and its session record lost.")
        return []
    chain = list(prior.pop("previous_sessions", []))
    chain.append(prior)
    return chain


_QUIT = object()


def _ask(prompt: str, default, parse):
    """One prompt with its own retry loop. Returns the value, or ``_QUIT``.

    Bad input re-asks THIS question only -- getting the offset wrong should not
    make you retype the height and the delta. A bare ENTER takes `default`
    (``None`` means there is no default yet, so an answer is required), and
    q/quit/exit at any prompt abandons the trial.

    `parse` raises ValueError with the operator-facing message.
    """
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("q", "quit", "exit"):
            return _QUIT
        if not raw:
            if default is None:
                print("  an answer is needed here (or q to finish).")
                continue
            return default
        try:
            return parse(raw)
        except ValueError as exc:
            print(f"  {exc}")


def _parse_height(raw: str) -> str:
    return raw


def _parse_delta(raw: str) -> float:
    """Action magnitude, 0 to 1 inclusive. Shared by the collision and jerk prompts.

    Zero is allowed and means exactly what it says: the action is (0, 0) and the
    paddle holds its start pose for the whole trial, recording the same way every
    other trial does. For the collision battery that is the no-strike control --
    the restitution of the paddle face with no paddle velocity in it. For the jerk
    battery it is the at-rest noise floor the reversals are read against.
    """
    try:
        delta = float(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not a number.")
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta must be in [0, 1]; got {delta:g}.")
    return delta


def _parse_trigger_x(raw: str) -> float:
    try:
        trigger_x = float(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not a number.")
    if abs(trigger_x) > TABLE_HALF_LENGTH_OBS_X:
        raise ValueError(
            f"{trigger_x:g} is off the table -- observation x runs "
            f"[{-TABLE_HALF_LENGTH_OBS_X:+.3f}, {TABLE_HALF_LENGTH_OBS_X:+.3f}], "
            f"centre line 0.0, robot's end positive."
        )
    return trigger_x


def _parse_y_offset(raw: str) -> float:
    try:
        offset = float(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not a number.")
    if abs(offset) > 1.0:
        raise ValueError(
            f"{offset:g} m is not a plausible offset -- this is METRES, not centimetres. "
            "The paddle's whole y workspace is about 0.74 m wide."
        )
    return offset


def prompt_collision_trial(index: int, prev_height: str | None,
                           prev_delta: float | None,
                           prev_y_offset: float | None,
                           prev_trigger_x: float | None) -> CollisionTrialSpec | None:
    """Ask for this trial's height, delta, y offset and trigger line. None = quit.

    Every prompt accepts a bare ENTER to reuse the previous trial's value, so a
    block of repeats at one condition is four ENTERs. The trigger line is asked
    last and is normally an ENTER, which makes that final keystroke a "ready?"
    gate: the arm moves next, straight to the offset start pose.

    ``prev_trigger_x`` seeds the first trial from ``--trigger-x`` when it was
    given. With neither, there is no default and an answer is required -- that
    line is a marking on your physical table and nothing here knows where it is.
    """
    height_suffix = f" [{prev_height}]" if prev_height else ""
    height = _ask(
        f"\n[collision] trial {index}: puck release height "
        f"(e.g. top, 3/4, 1/2 -- label only){height_suffix}, or q to finish: ",
        prev_height, _parse_height,
    )
    if height is _QUIT:
        return None

    delta_suffix = f" [{prev_delta:g}]" if prev_delta is not None else ""
    delta = _ask(
        f"[collision] trial {index}: delta -- vertical action magnitude, "
        f"0 <= d <= 1 (0.33 / 0.66 / 1.0; 0 = paddle holds still){delta_suffix}: ",
        prev_delta, _parse_delta,
    )
    if delta is _QUIT:
        return None

    default_y = 0.0 if prev_y_offset is None else float(prev_y_offset)
    y_offset = _ask(
        f"[collision] trial {index}: y offset in METRES off the start pose, "
        f"signed (+ = +y, - = -y) [{default_y:+.3f}]: ",
        default_y, _parse_y_offset,
    )
    if y_offset is _QUIT:
        return None

    trigger_suffix = f" [{prev_trigger_x:+.3f}]" if prev_trigger_x is not None else ""
    trigger_x = _ask(
        f"[collision] trial {index}: trigger x_obs -- the black line nearest the robot, "
        f"NOT the centre line (centre 0.0, robot's end +{TABLE_HALF_LENGTH_OBS_X:.3f})"
        f"{trigger_suffix}: ",
        prev_trigger_x, _parse_trigger_x,
    )
    if trigger_x is _QUIT:
        return None
    if float(trigger_x) > PADDLE_START_OBS_X:
        print(f"  note: {float(trigger_x):+.3f} is past the paddle's start pose "
              f"({PADDLE_START_OBS_X:+.3f}) -- the puck reaches the paddle before the "
              f"strike is commanded. Fine if that is what you want.")

    return CollisionTrialSpec(index=index, height_label=str(height),
                              delta=float(delta), y_offset=float(y_offset),
                              trigger_x=float(trigger_x))


def report_collision_plan(sim_or_none, start_xy, lims, edge_lims, move_lims, args) -> None:
    """Print the fixed geometry of the collision battery before it runs."""
    center_offset = (float(sim_or_none.center_offset_constant)
                     if sim_or_none is not None else 1.2)
    print(f"[collision] start pose (bottom of the table, ROBOT frame) = "
          f"({start_xy[0]:+.3f}, {start_xy[1]:+.3f})  "
          f"-> x_obs {start_xy[0] + center_offset:+.3f}")
    if args.trigger_x is None:
        print("[collision] trigger line: asked per trial (no --trigger-x seed given). "
              "It is the black line NEAREST THE ROBOT, in observation x, where the centre "
              f"line is 0.0 and the robot's end is +{TABLE_HALF_LENGTH_OBS_X:.3f}; the puck "
              "crosses it moving toward the robot, i.e. with x_obs increasing.")
    else:
        trigger_x = float(args.trigger_x)
        print(f"[collision] trigger line: asked per trial, first prompt seeded with "
              f"x_obs={trigger_x:+.3f} (robot x={trigger_x - center_offset:+.3f}); the puck "
              f"crosses it moving toward the robot, i.e. with x_obs increasing.")
    print(f"[collision] action = (-delta, 0): straight up the table. "
          f"rmax_x={move_lims[0]:g} m/step, so at |action|=1 the paddle covers "
          f"{move_lims[0]:g} m per step.")
    for delta in (0.33, 0.66, 1.0):
        n = steps_to_saturation(start_xy, DIRECTIONS[COLLISION_DIRECTION_KEY], delta,
                                move_lims, lims, edge_lims)
        print(f"[collision]   delta={delta:<5g} step {delta * move_lims[0]:.3f} m  "
              f"-> reaches the far edge of the workspace after ~{n} step(s)")
    print(f"[collision] {args.collision_settle_steps} settle + "
          f"{args.collision_action_steps} action + {args.collision_post_steps} post steps "
          f"per trial; arm timeout {args.arm_timeout_s:g}s.")


def run_puck_collision_session(args, config_path: Path) -> None:
    """Interactive puck-paddle collision battery (the --puck-collision path)."""
    args.out_dir = args.out_dir or default_out_dir("puck_collision")
    if args.dry_run:
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sim_params = cfg["air_hockey"]["simulator_params"]
        lims, edge_lims, move_lims = real_preview_geometry(sim_params)
        start_xy = start_pose_for(
            DIRECTIONS[COLLISION_DIRECTION_KEY], lims, edge_lims,
            mode=args.start_mode, base_xy=args.base_robot_xy, margin=float(args.start_margin),
        )
        report_collision_plan(None, start_xy, lims, edge_lims, move_lims, args)
        for warning in check_no_smoothing(cfg):
            print(f"[warn] {warning}")
        print(f"\n[dry-run] would write collision trials to {args.out_dir}")
        return

    env, cfg = load_env(config_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = env.simulator
    move_lims = tuple(float(v) for v in sim.move_lims)

    if sim.cap is None or sim.puck_detector is None:
        raise SystemExit(
            "The collision experiment needs the camera and a puck detector, but the env "
            f"has cap={sim.cap} detector={sim.puck_detector}. Check control_mode (must not "
            "be 'mouse'/'mimic'/'observe') and puck_detector in the config."
        )
    if int(args.camera_buffersize) > 0:
        # The trigger is latency-critical: a driver-side backlog means the puck is
        # already past the line by the time we see it cross. The env leaves this
        # unset; test_camera_puck_detection.py sets it to 1 for the same reason.
        ok = bool(sim.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(args.camera_buffersize)))
        print(f"[collision] camera buffersize -> {args.camera_buffersize} "
              f"({'accepted' if ok else 'REJECTED by the backend; expect extra latency'})")

    session_meta = {
        "config_path": str(config_path),
        "image_path": sim.image_path,
        "reset_settle_s": float(args.reset_settle_s),
        "collision_settle_steps": int(args.collision_settle_steps),
        "collision_action_steps": int(args.collision_action_steps),
        "collision_post_steps": int(args.collision_post_steps),
        "trigger_x_seed": None if args.trigger_x is None else float(args.trigger_x),
        "arm_timeout_s": float(args.arm_timeout_s),
        "min_far_frames": int(args.trigger_min_far_frames),
        "clamp_every_s": float(args.arm_clamp_every_s),
        "arm_print_every": int(args.arm_print_every),
        "show_arm_view": bool(args.show_arm_view),
        "reset_pose": np.asarray(sim.reset_pose[0], dtype=float),
        "move_lims": np.asarray(move_lims, dtype=float),
        "workspace_lims": np.asarray(sim.lims, dtype=float),
        "edge_lims": np.asarray(sim.edge_lims, dtype=float),
        "hist_len": int(sim.hist_len),
        "control_type": str(sim.control_type),
        "control_mode": str(sim.control_mode),
        "block_time": float(sim.block_time),
        "center_offset_constant": float(sim.center_offset_constant),
        "session_start_iso": datetime.now().astimezone().isoformat(),
    }

    print(f"[collision] config={config_path}")
    print(f"[collision] out_dir={out_dir.resolve()}")
    print(f"[collision] workspace x{tuple(sim.lims[:2])} y{tuple(sim.lims[2:])}  "
          f"move_lims (m/step at |action|=1) = {move_lims}")
    for warning in check_no_smoothing(cfg):
        print(f"[warn] {warning}")

    start_xy = start_pose_for(
        DIRECTIONS[COLLISION_DIRECTION_KEY], sim.lims, sim.edge_lims,
        mode=args.start_mode, base_xy=args.base_robot_xy, margin=float(args.start_margin),
    )
    session_meta["start_robot_xy"] = [float(v) for v in start_xy]
    report_collision_plan(sim, start_xy, sim.lims, sim.edge_lims, move_lims, args)

    if not args.no_wait:
        input("\n[collision] Clear the table and workspace, then press ENTER to begin the "
              "session (Ctrl-C to abort)... ")

    manifest_entries: list[dict] = []
    skipped: list[dict] = []
    interrupted = False
    index = next_trial_index(out_dir, "collision", COLLISION_INDEX_RE)
    first_index = index
    if index > 0:
        print(f"[collision] {out_dir} already holds trials up to index {index - 1}; "
              f"this session starts at {index}.")
    prev_height: str | None = None
    prev_delta: float | None = None
    prev_y_offset: float | None = None
    prev_trigger_x: float | None = (None if args.trigger_x is None else float(args.trigger_x))
    try:
        while True:
            spec = prompt_collision_trial(index, prev_height, prev_delta, prev_y_offset,
                                          prev_trigger_x)
            if spec is None:
                break
            prev_height, prev_delta = spec.height_label, spec.delta
            prev_y_offset, prev_trigger_x = spec.y_offset, spec.trigger_x
            trial_xy = trial_start_robot_xy(sim, start_xy, spec.y_offset)
            print(f"\n[{index}] {spec.name}  action={spec.action}  "
                  f"(height label {spec.height_label!r})")
            print(f"  start pose y {start_xy[1]:+.3f} {spec.y_offset:+.3f} "
                  f"-> {trial_xy[1]:+.3f} (robot frame);  "
                  f"trigger x_obs={spec.trigger_x:+.3f}")
            if spec.delta == 0.0:
                print("  delta 0: the paddle will NOT move at the crossing -- it holds this "
                      "pose and the puck arrives at a stationary paddle.")
            print("  moving to the bottom of the table -- keep clear of the arm.")

            record = run_collision_trial(env, spec, session_meta, start_xy)
            if not record["triggered"]:
                skipped.append({"trial_name": spec.name, "index": index,
                                "puck_height_label": spec.height_label,
                                "action_delta": float(spec.delta),
                                "y_offset": float(spec.y_offset),
                                "trigger_x_obs": float(spec.trigger_x),
                                "reason": record["reason"]})
                print(f"  not saved ({record['reason']}); re-enter the trial to retry.")
                continue

            entry = write_collision_trial_hdf5(
                out_dir / f"{spec.name}.hdf5", spec, record, session_meta,
                save_images=not args.no_save_images,
            )
            manifest_entries.append(entry)
            index += 1
            print(f"  saved {entry['file']} ({entry['num_steps']} steps, "
                  f"images={entry['has_images']}, "
                  f"trigger->action {entry['trigger_to_action_s'] * 1000:.0f} ms)")
            if record["protective_stop"]:
                print("[collision] Stopping the session: the robot reported a protective "
                      "stop. Clear it and re-run.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[collision] interrupted; writing manifest for completed trials.")
    finally:
        manifest = {
            "experiment": "puck_collision",
            "session_start_iso": session_meta["session_start_iso"],
            "session_end_iso": datetime.now().astimezone().isoformat(),
            "config_path": str(config_path),
            "out_dir": str(out_dir.resolve()),
            "interrupted": interrupted,
            "completed_trials": len(manifest_entries),
            "skipped_trials": skipped,
            "direction": COLLISION_DIRECTION_KEY,
            "trigger_x_obs_seed": None if args.trigger_x is None else float(args.trigger_x),
            "trigger_min_far_frames": int(args.trigger_min_far_frames),
            "arm_timeout_s": float(args.arm_timeout_s),
            "camera_buffersize": int(args.camera_buffersize),
            "collision_settle_steps": int(args.collision_settle_steps),
            "collision_action_steps": int(args.collision_action_steps),
            "collision_post_steps": int(args.collision_post_steps),
            "reset_settle_s": float(args.reset_settle_s),
            "start_mode": args.start_mode,
            "start_robot_xy": session_meta.get("start_robot_xy"),
            "config_reset_pose": [float(v) for v in session_meta["reset_pose"]],
            "move_lims": [float(v) for v in move_lims],
            "workspace_lims": [float(v) for v in sim.lims],
            "center_offset_constant": session_meta["center_offset_constant"],
            "hist_len": session_meta["hist_len"],
            "vals_column_names": VALS_COLUMN_NAMES,
            "trials": manifest_entries,
            "first_trial_index": first_index,
            # Sessions that wrote into this same directory before this one,
            # oldest first. Their trials are still on disk under their own
            # indices; this keeps the record of what produced them.
            "previous_sessions": load_previous_sessions(out_dir / "manifest.json"),
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        prior = len(manifest["previous_sessions"])
        print(f"\n[collision] wrote {len(manifest_entries)} trajectories + manifest.json "
              f"to {out_dir.resolve()}"
              + (f" ({prior} earlier session(s) carried forward in the manifest)"
                 if prior else ""))
        if args.show_arm_view:
            try:
                cv2.destroyWindow("arm")
            except Exception:
                pass
        if not interrupted:
            try:
                env.reset()
            except Exception as exc:
                print(f"[collision] final reset skipped: {exc}")
        close_fn = getattr(env.simulator, "close", None) or getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def add_collision_arguments(parser) -> None:
    """Flags for the --puck-collision experiment. None of them affect the default path."""
    group = parser.add_argument_group("puck-paddle collision experiment (--puck-collision)")
    group.add_argument("--puck-collision", action="store_true",
                       help="Run the interactive puck-paddle collision battery INSTEAD of the "
                            "scripted paddle-motion plan: per trial you type the puck release "
                            "height and the delta, the arm resets to the bottom of the table, "
                            "and the strike fires when the camera sees the puck cross the "
                            "table's centre line.")
    group.add_argument("--trigger-x", type=float, default=None,
                       help="Seed for the per-trial trigger-line prompt. Each trial asks for the "
                            "OBSERVATION-frame x of the trigger line -- the black line NEAREST "
                            "THE ROBOT, not the centre line (which is 0.0; the robot half is "
                            "positive, the far half negative) -- and a bare ENTER reuses the "
                            "previous trial's answer. Pass this to have a value ready at the "
                            "first prompt; omit it and the first trial must type one, since that "
                            "line is a marking on your table and nothing in this repo knows where "
                            "it is. Measure it once with --show-arm-view.")
    group.add_argument("--trigger-min-far-frames", type=int, default=2,
                       help="Fresh detections required on the far side of the line before a "
                            "crossing can fire, so the trigger needs a real approach and cannot "
                            "arm on a puck already sitting on the robot's half.")
    group.add_argument("--arm-timeout-s", type=float, default=25.0,
                       help="Seconds to wait for the crossing before discarding the trial and "
                            "returning to the prompt.")
    group.add_argument("--arm-print-every", type=int, default=15,
                       help="Print the detected puck position every N camera frames while armed "
                            "(0 = never).")
    group.add_argument("--arm-clamp-every-s", type=float, default=0.5,
                       help="Re-apply the paddle's downward force this often while armed. UR's "
                            "forceMode times out after ~2 s, so an unclamped wait lifts the "
                            "paddle off the table before the puck arrives.")
    group.add_argument("--show-arm-view", action="store_true",
                       help="Show the rectified camera frame with the trigger line and the "
                            "detected puck while armed. Use it once to confirm --trigger-x lands "
                            "on your table's centre line.")
    group.add_argument("--camera-buffersize", type=int, default=1,
                       help="cv2.CAP_PROP_BUFFERSIZE applied to the env's capture for this "
                            "session (0 = leave the backend default). 1 keeps the trigger "
                            "reading the newest frame instead of a queued one.")
    group.add_argument("--collision-action-steps", type=int, default=20,
                       help="Constant-action steps commanded after the trigger.")
    group.add_argument("--collision-post-steps", type=int, default=20,
                       help="Zero-action steps recorded after the strike, so the puck's "
                            "post-collision flight is in the same file.")
    group.add_argument("--collision-settle-steps", type=int, default=0,
                       help="Zero-action steps recorded BEFORE arming. Off by default: they "
                            "would be separated from the strike by however long the operator "
                            "takes to release the puck, so they read as a gap rather than a "
                            "baseline.")



# ---------------------------------------------------------------------------
# Experiment 3: reversal jerk (--jerk)
# ---------------------------------------------------------------------------
#
# Paddle only -- no puck, no camera trigger. One trial holds a constant action,
# ramps it linearly to zero over t steps, and immediately reverses:
#
#     phase   steps            action along the travel axis
#     ----------------------------------------------------------------
#     out     n_out            +delta                       (constant)
#     slow    t                +delta * (t - k)/t, k = 1..t  -> ends at 0
#     back    N - n_out - t    -delta                       (constant)
#
# summing to N = --jerk-steps (20) commanded steps. n_out and t are both typed
# per trial: "10 steps out, t slowing, 20 - 10 - t back" is the split, and the
# workspace geometry does not resize it. The paddle starts at the far end of the
# axis it is about to travel -- ``start_pose_for`` on the OUTBOUND direction, the
# same "opposite end of the table" rule the straight-line battery uses -- so the
# outbound leg has as much room as the table can give it.
#
# The ramp is a straight line from delta to zero. Taking the last outbound step
# as k = 0 and the reversal as k = t + 1, the scale factor (t - k)/t is exactly
# linear interpolation between them: it leaves delta, passes through the
# intermediate values evenly, and the t-th step commands precisely 0, so the
# paddle is at rest at the turnaround before -delta is sent.
#
# What t buys you. At t = 0 the commanded action steps from +delta to -delta
# between two consecutive timesteps: the hardest reversal the action space can
# express, and the one that shakes the arm. Raising t spreads that same reversal
# over a linear ramp. Sweeping t at fixed delta and fixed n_out is the
# experiment; ``acc_x/y/z`` in ``train_vals`` is where the jerk shows up.
#
# Watch the room, because nothing here clips for you. The action is a velocity
# command of rmax_axis * delta metres per step, so n_out = 10 at delta 1.0 asks
# for 2.6 m on x, where the workspace is 0.41 m: the paddle pins against the far
# edge after two steps and spends the rest of the outbound leg stalled there,
# which makes the reversal start from a standstill and measures nothing. The
# plan report prints the predicted pin step for every condition x delta before
# the arm moves, and each trial prints it again as it starts. y is the roomier
# axis (0.74 m at rmax_y = 0.12, six steps at delta 1.0). Trials that overrun
# still run and still record -- ``saturates_at_step`` is in the file -- but a
# clean reversal wants n_out * step_m to stay inside the room.

# The four conditions, in prompt order (a bare "1".."4" at the prompt picks one).
JERK_CONDITION_ORDER = ("up_down", "down_up", "right_left", "left_right")


@dataclass(frozen=True)
class JerkCondition:
    """An out-and-back pair: which direction the paddle travels, then reverses into."""

    key: str
    out_key: str    # DIRECTIONS key for the outbound leg
    back_key: str   # DIRECTIONS key for the return leg (the opposite sign)
    description: str

    @property
    def out(self) -> Direction:
        return DIRECTIONS[self.out_key]

    @property
    def back(self) -> Direction:
        return DIRECTIONS[self.back_key]

    @property
    def axis(self) -> str:
        """"x" (vertical, along the table length) or "y" (horizontal, lateral)."""
        return self.out.axis

    @property
    def axis_index(self) -> int:
        return 0 if self.axis == "x" else 1


# "Vertical" is x (along the table length, -x = away from the robot, up the
# table); "horizontal" is y. Each condition starts at the far end of its
# outbound direction, so `up_down` starts at the bottom and `down_up` at the top.
JERK_CONDITIONS = {
    "up_down": JerkCondition(
        "up_down", "xneg", "xpos",
        "vertical: up the table (away from the robot) first, then back down"),
    "down_up": JerkCondition(
        "down_up", "xpos", "xneg",
        "vertical: down the table (toward the robot) first, then back up"),
    "right_left": JerkCondition(
        "right_left", "ypos", "yneg",
        "horizontal: toward +y (right) first, then back left"),
    "left_right": JerkCondition(
        "left_right", "yneg", "ypos",
        "horizontal: toward -y (left) first, then back right"),
}

JERK_CONDITION_ALIASES = {
    "up_down": "up_down", "up-down": "up_down", "updown": "up_down",
    "up": "up_down", "ud": "up_down",
    "down_up": "down_up", "down-up": "down_up", "downup": "down_up",
    "down": "down_up", "du": "down_up",
    "right_left": "right_left", "right-left": "right_left", "rightleft": "right_left",
    "right": "right_left", "rl": "right_left",
    "left_right": "left_right", "left-right": "left_right", "leftright": "left_right",
    "left": "left_right", "lr": "left_right",
}

JERK_INDEX_RE = re.compile(r"^jerk_(\d+)_")


@dataclass
class JerkTrialSpec:
    """One operator-specified reversal-jerk trial."""

    index: int
    condition: JerkCondition
    delta: float           # |action| on the travel axis during the cruise phases
    out_steps: int         # n_out: constant-action steps before the ramp
    slowdown_steps: int    # t: ramp steps between the outbound cruise and the reversal
    total_steps: int       # N: commanded steps in the whole trial; back = N - n_out - t
    # Which repeat of an identical condition this is. Always 1 for a typed trial;
    # the scripted batch counts 1..repeats. Not in the filename -- the trial index
    # already makes that unique -- but recorded, so repeats can be grouped.
    repeat: int = 1

    @property
    def name(self) -> str:
        return (f"jerk_{self.index:03d}_{self.condition.key}"
                f"_delta{self.delta:.2f}_out{self.out_steps:02d}"
                f"_slow{self.slowdown_steps:02d}")

    @property
    def back_steps(self) -> int:
        """N - n_out - t. Never negative: the prompts bound n_out + t by N."""
        return max(0, int(self.total_steps) - int(self.out_steps) - int(self.slowdown_steps))

    @property
    def action(self) -> np.ndarray:
        """Representative action: the outbound cruise one."""
        return (np.asarray(self.condition.out.vec, dtype=np.float32)
                * np.float32(self.delta))


def slowdown_scales(slowdown_steps: int) -> np.ndarray:
    """Linear ramp from delta down to zero over t steps: (t - k)/t for k = 1..t.

    Straight-line interpolation, with the last outbound step as k = 0 (scale 1,
    i.e. delta) and the reversal as k = t + 1: consecutive scales differ by
    exactly 1/t, and the t-th entry is exactly 0, so the paddle is commanded to
    rest at the turnaround rather than reversing out of a moving state.

    t = 1 therefore degenerates to a single zero step -- a one-timestep pause --
    which is the smallest non-trivial thing "slow down first" can mean, and t = 0
    gives no ramp at all: the bang-bang reversal this experiment compares against.
    """
    t = int(slowdown_steps)
    if t <= 0:
        return np.zeros(0, dtype=float)
    k = np.arange(1, t + 1, dtype=float)
    return (t - k) / t


def max_out_steps(room_m: float, step_m: float, ramp_m: float) -> int:
    """Largest n_out whose cruise + ramp still fits in `room_m`.

    The ramp is charged what it actually coasts through (``step_m * (t-1)/2``),
    not a full step per ramp step -- charging it full steps understates the
    answer badly at large t, which is the whole range this experiment lives in.
    """
    if step_m <= 1e-9:
        return 0
    return max(0, int(np.floor((room_m - ramp_m) / step_m)))


AUTO_OUT_STEPS = -1   # --jerk-out-steps auto: size n_out per trial from the room

# Seconds per commanded step (AirHockeyReal's block_time default). Only used for
# the offline travel estimate below; the trial itself is paced by the env.
JERK_STEP_DT_S = 0.049


def arm_step_travel_m(delta: float, rmax: float) -> float:
    """Metres the PADDLE actually covers in one step at `delta` on an axis.

    NOT ``delta * rmax``. That is how fast the servoL TARGET runs: every step
    ``take_action`` sets the target to ``pose + action * rmax`` and the arm chases
    it, lagging behind. The target saturates against the workspace within a step
    or two at the large deltas while the paddle is still accelerating well short
    of the edge, which is why sizing the outbound leg off the target made the
    trials far too short on the real robot.

    The arm's measured sustained speed is ``CURVE_VELOCITY_GAIN_REAL * action *
    rmax`` (3.2 1/s, fitted to the straight-line battery in
    data/robot_data_collection/paddle_motion_20260909_1808: 0.386 m/s on y and
    0.82 m/s on x at action 1.0), so one step covers that times the step period --
    about a sixth of what the target moves.

    This is the steady-state speed, so it still overestimates the first few steps
    while the arm is spinning up; it is an upper bound on travel, which is the
    safe direction for a room check.
    """
    return float(CURVE_VELOCITY_GAIN_REAL) * float(delta) * float(rmax) * JERK_STEP_DT_S


def resolve_out_steps(spec: JerkTrialSpec, lims, edge_lims, move_lims, args) -> JerkTrialSpec:
    """Fill in an AUTO n_out: the most cruise steps that still leave room to ramp.

    Sized per trial, because the room that matters depends on the axis, the delta
    and t all at once. Always keeps at least one cruise step and one return step,
    so a trial still has a reversal to measure even where the table is too short
    for the full ramp (x at the large deltas) -- those stay flagged by
    ``saturates_at_step`` rather than being silently dropped.
    """
    if spec.out_steps != AUTO_OUT_STEPS:
        return spec
    probe = replace(spec, out_steps=0)
    plan = jerk_phase_plan(probe, jerk_start_pose(spec.condition, lims, edge_lims, args),
                           lims, edge_lims, move_lims)
    # Half the non-ramp budget, so the return leg can retrace the trip out: both
    # legs run at the same speed, so an even split is what brings the paddle back
    # to where it started. The room only reduces this where the axis is genuinely
    # too short (x at delta 1.0), which is rare now that travel is estimated from
    # the arm's real speed rather than the target's.
    even = (int(spec.total_steps) - int(spec.slowdown_steps)) // 2
    room_limited = max_out_steps(plan["room_m"], plan["step_m"], plan["ramp_m"])
    n = max(1, min(even, room_limited))
    return replace(spec, out_steps=n)


def jerk_phase_plan(spec: JerkTrialSpec, start_robot_xy, lims, edge_lims, move_lims) -> dict:
    """The trial's out / slow / back split, plus how it sits against the room.

    The split is exactly what was typed -- ``n_out`` constant-action steps, ``t``
    ramp steps, and the rest of the budget on the return. Nothing here resizes
    it. The geometry is measured only so the caller can SAY, before the arm
    moves, whether the outbound leg has somewhere to go: the action is a velocity
    command, so ``n_out`` steps at ``delta`` ask for ``n_out * rmax * delta``
    metres and the paddle simply pins against the far edge once that exceeds the
    room, taking the reversal with it.
    """
    axis_i = spec.condition.axis_index
    rmax = float(move_lims[axis_i])
    # What the paddle covers, not what the target covers -- see arm_step_travel_m.
    step_m = arm_step_travel_m(spec.delta, rmax)
    room = float(room_in_direction(start_robot_xy, spec.condition.out, lims, edge_lims)[axis_i])
    scales = slowdown_scales(spec.slowdown_steps)

    n_out = int(spec.out_steps)
    n_slow = int(spec.slowdown_steps)
    n_back = spec.back_steps

    out_m = step_m * n_out
    ramp_m = step_m * float(scales.sum())
    travel_m = out_m + ramp_m
    back_m = step_m * n_back
    per_step = np.concatenate([np.full(n_out, step_m), step_m * scales])
    over = np.nonzero(np.cumsum(per_step) > room + 1e-9)[0]
    return {
        "n_out": n_out,
        "n_slow": n_slow,
        "n_back": n_back,
        "step_m": step_m,
        "room_m": room,
        "out_m": out_m,
        "ramp_m": ramp_m,
        "travel_m": travel_m,
        "back_m": back_m,
        # 1-indexed commanded step at which the paddle first has more travel
        # commanded than workspace left; -1 if the outbound leg fits.
        "saturates_at_step": int(over[0]) + 1 if over.size else -1,
        "overruns": bool(over.size),
        # Largest n_out that would have stayed inside the workspace at this
        # delta and t -- the number to retype when a trial overruns.
        "max_out_steps": max_out_steps(room, step_m, ramp_m),
        # The return is shorter than the trip out, so the paddle does not end up
        # back where it started. Fine for a reversal measurement, but it means
        # consecutive trials do not start from the same place unless each one
        # resets -- which they do.
        "returns_short": bool(back_m < travel_m - 1e-9),
    }


def jerk_step_actions(spec: JerkTrialSpec, plan: dict) -> list[tuple[np.ndarray, str]]:
    """The trial's (action, phase) schedule: n_out cruise, t ramp, n_back return."""
    out_vec = np.asarray(spec.condition.out.vec, dtype=np.float32)
    back_vec = np.asarray(spec.condition.back.vec, dtype=np.float32)
    delta = np.float32(spec.delta)
    schedule = [(out_vec * delta, "out")] * plan["n_out"]
    schedule += [(out_vec * delta * np.float32(s), "slow")
                 for s in slowdown_scales(spec.slowdown_steps)]
    schedule += [(back_vec * delta, "back")] * plan["n_back"]
    return schedule


def run_jerk_trial(env: AirHockeyEnv, spec: JerkTrialSpec, session: dict,
                   start_robot_xy: np.ndarray, plan: dict) -> dict:
    """Reset to the start of the outbound leg, then run out / slow / back.

    Structurally ``run_collision_trial`` without the arm phase: same reset
    retarget, same settle, same per-step bookkeeping and protective-stop abort.
    The phase of every step is recorded alongside it, which is what lets the
    analysis line the acceleration trace up against the commanded reversal.
    """
    sim = env.simulator
    sim.reset_pose[0][0] = float(start_robot_xy[0])
    sim.reset_pose[0][1] = float(start_robot_xy[1])
    obs, _info = env.reset()
    start_pose = settle_after_reset(sim, session["reset_settle_s"])

    zero_action = np.zeros(2, dtype=np.float32)
    schedule = ([(zero_action, "settle")] * int(session["jerk_settle_steps"])
                + jerk_step_actions(spec, plan))

    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    phases: list[str] = []
    is_settle: list[int] = []
    step_start_times: list[float] = []
    step_end_times: list[float] = []
    block_reasons: list[str] = []
    protective_stop = False
    aborted_at: int | None = None

    for step_i, (step_action, phase) in enumerate(schedule):
        t0 = time.time()
        obs, _reward, _terminated, _truncated, info = env.step(step_action)
        t1 = time.time()
        actions.append(np.asarray(step_action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        phases.append(phase)
        is_settle.append(int(phase == "settle"))
        step_start_times.append(t0)
        step_end_times.append(t1)
        block_reasons.append(str(info.get("command_block_reason", "none")))
        if bool(info.get("protective_stop", False)) or not bool(info.get("robot_step_ready", True)):
            protective_stop = True
            aborted_at = step_i
            print(
                f"  !! {spec.name}: robot not stepping at step {step_i} "
                f"(protective_stop={info.get('protective_stop')}, "
                f"reason={info.get('command_block_reason')}); aborting trial"
            )
            break

    return {
        "start_pose": start_pose,
        "start_robot_xy_commanded": np.asarray(start_robot_xy, dtype=float),
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 2), dtype=np.float32),
        "observations": np.stack(observations, axis=0),
        "step_phase": phases,
        "is_settle": np.asarray(is_settle, dtype=np.int8),
        "step_start_times": np.asarray(step_start_times, dtype=np.float64),
        "step_end_times": np.asarray(step_end_times, dtype=np.float64),
        "command_block_reasons": block_reasons,
        "protective_stop": protective_stop,
        "aborted_at": aborted_at,
        # Harvested now: the next env.reset() clears sim.vals / sim.images.
        "sim_vals": list(sim.vals),
        "sim_images": list(sim.images),
    }


def write_jerk_trial_hdf5(path: Path, spec: JerkTrialSpec, record: dict, plan: dict,
                          session_meta: dict, save_images: bool) -> dict:
    """Write one reversal-jerk trial to HDF5 and return its manifest entry."""
    sim_vals = record["sim_vals"]
    sim_images = record["sim_images"]

    imgs = None
    if save_images and len(sim_images) > 0:
        imgs, vals = merge_trajectory(session_meta["image_path"], sim_images, sim_vals)
        if vals is None:  # image/value counts disagreed; keep the proprioception
            print(f"  !! {spec.name}: image/value misalignment, saving values only")
            imgs = None
            vals = np.stack(sim_vals, axis=0) if len(sim_vals) else np.zeros((0, 35))
    else:
        vals = np.stack(sim_vals, axis=0) if len(sim_vals) else np.zeros((0, 35))

    cond = spec.condition
    with h5py.File(path, "w") as hf:
        if imgs is not None:
            hf.create_dataset("train_img", shape=imgs.shape, compression="gzip",
                              compression_opts=9, data=imgs)
        hf.create_dataset("train_vals", shape=vals.shape, compression="gzip",
                          compression_opts=9, data=vals)
        hf.create_dataset("actions", data=record["actions"])
        hf.create_dataset("observations", data=record["observations"])
        hf.create_dataset("is_settle_step", data=record["is_settle"])
        # "settle" / "out" / "slow" / "back", one per commanded step: the label
        # the acceleration trace is segmented by.
        hf.create_dataset("step_phase",
                          data=np.array(record["step_phase"], dtype=h5py.string_dtype()))
        hf.create_dataset("step_start_time", data=record["step_start_times"])
        hf.create_dataset("step_end_time", data=record["step_end_times"])
        hf.create_dataset(
            "command_block_reason",
            data=np.array(record["command_block_reasons"], dtype=h5py.string_dtype()),
        )
        # The ramp multipliers this trial used, so the schedule is reconstructable
        # from the file without re-deriving it from t.
        hf.create_dataset("slowdown_scales", data=slowdown_scales(spec.slowdown_steps))

        hf.attrs["vals_column_names"] = VALS_COLUMN_NAMES
        hf.attrs["trial_name"] = spec.name
        hf.attrs["trial_index"] = spec.index
        hf.attrs["trial_type"] = "reversal_jerk"
        hf.attrs["experiment"] = "reversal_jerk"
        hf.attrs["condition_key"] = cond.key
        hf.attrs["condition_description"] = cond.description
        hf.attrs["axis"] = cond.axis
        hf.attrs["motion"] = "vertical" if cond.axis == "x" else "horizontal"
        hf.attrs["out_direction_key"] = cond.out_key
        hf.attrs["back_direction_key"] = cond.back_key
        hf.attrs["out_direction_vec"] = np.asarray(cond.out.vec, dtype=np.int8)
        hf.attrs["back_direction_vec"] = np.asarray(cond.back.vec, dtype=np.int8)
        hf.attrs["direction_sign"] = cond.out.sign
        hf.attrs["action_delta"] = float(spec.delta)
        hf.attrs["action"] = spec.action
        hf.attrs["slowdown_steps"] = int(spec.slowdown_steps)
        hf.attrs["total_steps"] = int(spec.total_steps)
        hf.attrs["repeat"] = int(spec.repeat)
        hf.attrs["out_steps"] = int(plan["n_out"])
        hf.attrs["back_steps"] = int(plan["n_back"])
        hf.attrs["settle_steps"] = int(session_meta["jerk_settle_steps"])
        # Geometry the split was fitted to, in metres on the travel axis.
        hf.attrs["step_m"] = float(plan["step_m"])
        hf.attrs["room_m"] = float(plan["room_m"])
        hf.attrs["out_m"] = float(plan["out_m"])
        hf.attrs["ramp_m"] = float(plan["ramp_m"])
        hf.attrs["travel_m"] = float(plan["travel_m"])
        hf.attrs["back_m"] = float(plan["back_m"])
        # -1 = the outbound leg fits; otherwise the 1-indexed commanded step at
        # which the paddle starts pinning against the far edge.
        hf.attrs["saturates_at_step"] = int(plan["saturates_at_step"])
        hf.attrs["max_out_steps_fitting"] = int(plan["max_out_steps"])
        hf.attrs["start_pose"] = record["start_pose"]
        hf.attrs["start_robot_xy_commanded"] = record["start_robot_xy_commanded"]
        hf.attrs["protective_stop"] = record["protective_stop"]
        hf.attrs["aborted_at_step"] = -1 if record["aborted_at"] is None else record["aborted_at"]
        for key in (
            "config_path", "reset_pose", "move_lims", "workspace_lims", "edge_lims",
            "hist_len", "control_type", "control_mode", "block_time", "session_start_iso",
            "center_offset_constant",
        ):
            hf.attrs[key] = session_meta[key]

    return {
        "trial_name": spec.name,
        "file": path.name,
        "index": spec.index,
        "trial_type": "reversal_jerk",
        "condition_key": cond.key,
        "axis": cond.axis,
        "motion": "vertical" if cond.axis == "x" else "horizontal",
        "out_direction_key": cond.out_key,
        "back_direction_key": cond.back_key,
        "action_delta": float(spec.delta),
        "action": [float(v) for v in spec.action],
        "slowdown_steps": int(spec.slowdown_steps),
        "total_steps": int(spec.total_steps),
        "repeat": int(spec.repeat),
        "out_steps": int(plan["n_out"]),
        "back_steps": int(plan["n_back"]),
        "settle_steps": int(session_meta["jerk_settle_steps"]),
        "num_steps": int(vals.shape[0]),
        "room_m": float(plan["room_m"]),
        "travel_m": float(plan["travel_m"]),
        "back_m": float(plan["back_m"]),
        "saturates_at_step": int(plan["saturates_at_step"]),
        "max_out_steps_fitting": int(plan["max_out_steps"]),
        "start_pose_xy": [float(record["start_pose"][0]), float(record["start_pose"][1])],
        "start_robot_xy_commanded": [float(v) for v in record["start_robot_xy_commanded"]],
        "has_images": imgs is not None,
        "protective_stop": bool(record["protective_stop"]),
        "aborted_at_step": record["aborted_at"],
    }


# The battery this experiment was built to collect, as `condition:deltas:ts:repeats`
# blocks run in the order written. Trials inside a block go delta-major, then t, then
# repeat -- so a block sweeps t at one delta before moving to the next delta, which is
# the comparison the experiment is for.
#
#   up_down     0.66 and 1.00, t = 3..0, once each,  20 steps  ->   8
#   right_left  0.33/0.66/1.00, t = 3..0, x3,         44 steps  ->  36
#                                                                  ----
#                                                                    44
#
# Both blocks stop at t = 3. On up_down that is forced: the vertical axis is only
# 0.40 m, so a longer ramp coasts further than the table has left and t = 4 or 5
# there would record the far edge rather than the commanded deceleration. y has
# the room for the full 5..0 sweep but is trimmed to match, so the same t values
# are compared across both axes.
#
# right_left runs 44-step trials rather than 20. The paddle's speed is
# gain * delta * rmax and rmax_y (0.12) is less than half rmax_x (0.26), so at
# 20 steps a lateral trial crawls -- 17 cm of a 73 cm axis at delta 1.0, against
# 37 cm of 40 cm for the vertical one. 44 = 20 * (0.26 / 0.12) restores roughly
# the same travel per trial, which is what makes the two axes comparable.
#
# down_up and left_right are the mirrored halves and are deliberately not in the
# default plan; add them back as extra blocks if you want the symmetric set.
DEFAULT_JERK_BATCH = (
    "up_down:0.66,1.0:3,2,1,0:1;"
    "right_left:0.33,0.66,1.0:3,2,1,0:3::44"
)


def parse_jerk_batch(spec: str, total_steps: int, out_steps: int) -> list[JerkTrialSpec]:
    """``cond:deltas:ts:repeats;...`` -> the trial list, in run order.

    Indices are placeholders: ``run_jerk_session`` stamps the real one as each
    trial starts, so a batch dropped into a directory that already holds trials
    continues its numbering like any other session.

    Everything is validated here, before the env is built and the arm is powered:
    a typo in a 174-trial plan should not surface as an exception forty minutes
    into a session.
    """
    trials: list[JerkTrialSpec] = []
    for block in (b.strip() for b in str(spec).split(";")):
        if not block:
            continue
        parts = block.split(":")
        if len(parts) not in (4, 5, 6):
            raise SystemExit(
                f"--jerk-batch block {block!r} needs 4 to 6 colon-separated fields "
                f"(condition:deltas:ts:repeats[:n_out[:total_steps]]), got {len(parts)}."
            )
        cond_raw, deltas_raw, ts_raw, repeats_raw = parts[:4]
        try:
            condition = _parse_condition(cond_raw)
        except ValueError as exc:
            raise SystemExit(f"--jerk-batch block {block!r}: {exc}")
        try:
            deltas = [float(v) for v in deltas_raw.split(",") if v.strip()]
            ts = [int(v) for v in ts_raw.split(",") if v.strip()]
            repeats = int(repeats_raw)
            # Optional 5th field: this block's n_out. The travel axis and the delta
            # both change how far n_out steps actually reach, so one number for the
            # whole batch cannot suit both x (0.41 m) and y (0.74 m) -- this is how
            # you give the short axis a shorter cruise.
            block_out_steps = (int(parts[4]) if len(parts) >= 5 and parts[4].strip()
                               else int(out_steps))
            # Optional 6th field: this block's trial length. The paddle's speed is
            # gain * delta * rmax, and rmax_y (0.12) is less than half rmax_x (0.26),
            # so the same step count covers less than half the distance on y. This
            # is how a lateral block buys back the travel with a longer trial.
            block_total_steps = (int(parts[5]) if len(parts) == 6 and parts[5].strip()
                                 else int(total_steps))
        except ValueError as exc:
            raise SystemExit(f"--jerk-batch block {block!r}: {exc}")
        if block_out_steps < 0 and block_out_steps != AUTO_OUT_STEPS:
            raise SystemExit(f"--jerk-batch block {block!r}: n_out cannot be negative.")
        if block_total_steps < 1:
            raise SystemExit(f"--jerk-batch block {block!r}: total_steps must be >= 1.")
        if not deltas or not ts:
            raise SystemExit(f"--jerk-batch block {block!r}: empty delta or t list.")
        if repeats < 1:
            raise SystemExit(f"--jerk-batch block {block!r}: repeats must be >= 1.")
        for delta in deltas:
            if not (0.0 <= delta <= 1.0):
                raise SystemExit(
                    f"--jerk-batch block {block!r}: delta {delta:g} is outside [0, 1].")
            for t in ts:
                if t < 0:
                    raise SystemExit(
                        f"--jerk-batch block {block!r}: t {t} cannot be negative.")
                if block_out_steps != AUTO_OUT_STEPS and block_out_steps + t > block_total_steps:
                    raise SystemExit(
                        f"--jerk-batch block {block!r}: n_out {block_out_steps} + t {t} "
                        f"exceeds the block's {block_total_steps} steps; lower n_out or t."
                    )
                for repeat in range(1, repeats + 1):
                    trials.append(JerkTrialSpec(
                        index=0, condition=condition, delta=float(delta),
                        out_steps=block_out_steps, slowdown_steps=int(t),
                        total_steps=block_total_steps, repeat=repeat,
                    ))
    if not trials:
        raise SystemExit("--jerk-batch produced no trials; check the spec string.")
    return trials


def report_jerk_batch(trials: list[JerkTrialSpec], lims, edge_lims, move_lims, args) -> None:
    """Print the scripted plan grouped by condition x delta, with the room check.

    The per-delta line is where an unrunnable combination shows up: n_out is one
    number for the whole batch, so a delta that overruns overruns on every trial
    of that block, and it is worth seeing that before committing the session.
    """
    trials = [resolve_out_steps(s, lims, edge_lims, move_lims, args) for s in trials]
    n_outs = sorted({s.out_steps for s in trials})
    n_out_desc = (f"n_out={n_outs[0]}" if len(n_outs) == 1
                  else f"n_out {n_outs[0]}-{n_outs[-1]} by block")
    print(f"[jerk] scripted batch: {len(trials)} trials, "
          f"{n_out_desc} of {args.jerk_steps} steps each, no prompts.")
    seen: list[tuple[str, float, int]] = []
    for spec in trials:
        key = (spec.condition.key, spec.delta, spec.out_steps)
        if key in seen:
            continue
        seen.append(key)
        group = [s for s in trials if (s.condition.key, s.delta, s.out_steps) == key]
        ts = sorted({s.slowdown_steps for s in group}, reverse=True)
        reps = max(s.repeat for s in group)
        start_xy = jerk_start_pose(spec.condition, lims, edge_lims, args)
        plan = jerk_phase_plan(spec, start_xy, lims, edge_lims, move_lims)
        # A row can cover several t values that share an n_out, and a longer ramp
        # coasts further -- so name the t values that actually overrun rather than
        # tarring the whole row with the first one's verdict.
        bad = sorted({s.slowdown_steps for s in group
                      if jerk_phase_plan(s, start_xy, lims, edge_lims,
                                         move_lims)["overruns"]}, reverse=True)
        note = ""
        if bad:
            worst = jerk_phase_plan(
                replace(spec, slowdown_steps=bad[0]), start_xy, lims, edge_lims, move_lims)
            note = (f"   !! t={','.join(str(t) for t in bad)} overrun: "
                    f"{worst['travel_m']:.3f} m of {worst['room_m']:.3f} m, "
                    f"pins at step {worst['saturates_at_step']}")
        print(f"[jerk]   {spec.condition.key:<11} delta {spec.delta:<5g} "
              f"t {','.join(str(t) for t in ts)}  x{reps}  n_out {spec.out_steps:<2d}"
              f" -> {len(group):>3} trials   step {plan['step_m']:.3f} m{note}")
    # Deterministic part only: the reset moveL between trials is not timed here
    # and adds a few seconds each, so this is a floor, not an estimate.
    overrunning = sum(
        1 for s in trials
        if jerk_phase_plan(s, jerk_start_pose(s.condition, lims, edge_lims, args),
                           lims, edge_lims, move_lims)["overruns"]
    )
    if overrunning:
        print(f"[jerk] {overrunning} of {len(trials)} trials have an outbound leg longer than "
              f"the table: the paddle stalls against the far edge before the ramp starts, so "
              f"the reversal happens from a standstill. Lower n_out (globally with "
              f"--jerk-out-steps, or per block with the 5th spec field) if that is not what "
              f"you want.")
    per_trial_s = int(args.jerk_steps) * 0.05 + float(args.reset_settle_s)
    total_min = len(trials) * per_trial_s / 60.0
    print(f"[jerk] at least {total_min:.0f} min of stepping + settling "
          f"({per_trial_s:.1f} s/trial), plus the reset move before each trial.")


def _parse_condition(raw: str) -> JerkCondition:
    """"up"/"up_down"/"1" -> the condition. Bare 1..4 index JERK_CONDITION_ORDER."""
    token = raw.strip().lower().replace(" ", "_")
    if token.isdigit():
        i = int(token)
        if not (1 <= i <= len(JERK_CONDITION_ORDER)):
            raise ValueError(f"pick 1-{len(JERK_CONDITION_ORDER)}; got {token}.")
        return JERK_CONDITIONS[JERK_CONDITION_ORDER[i - 1]]
    key = JERK_CONDITION_ALIASES.get(token)
    if key is None:
        raise ValueError(
            f"{raw!r} is not a condition. Use 1-4, or one of: "
            + ", ".join(JERK_CONDITION_ORDER) + " (up / down / right / left also work)."
        )
    return JERK_CONDITIONS[key]


def _step_count_parser(upper: int, label: str):
    """Parser for a whole number of timesteps in [0, `upper`].

    The bound is what keeps the three phases summing to the trial budget: n_out
    is capped at N, and t is then capped at whatever N - n_out is left, so
    `back` can never come out negative and the operator is told at the prompt
    rather than discovering a silently clipped trial in the saved file.
    """
    def parse(raw: str) -> int:
        try:
            steps = int(raw)
        except ValueError:
            raise ValueError(f"{raw!r} is not a whole number of timesteps.")
        if steps < 0:
            raise ValueError(f"{label} cannot be negative; got {steps}.")
        if steps > upper:
            raise ValueError(f"{label} must be at most {upper} here; got {steps}.")
        return steps
    return parse


def prompt_jerk_trial(index: int, prev_condition: JerkCondition | None,
                      prev_delta: float | None, prev_out_steps: int | None,
                      prev_slowdown: int | None, total_steps: int) -> JerkTrialSpec | None:
    """Ask for this trial's condition, delta, n_out and t. None = quit.

    Same shape as ``prompt_collision_trial``: every prompt takes a bare ENTER to
    reuse the previous trial's value and q to finish, and the last one doubles as
    the "ready?" gate -- the arm moves as soon as it is answered.

    n_out is asked before t so t can be bounded by the steps n_out leaves behind,
    which is what guarantees out + t + back == total_steps.
    """
    listing = ", ".join(f"{i + 1}={key}" for i, key in enumerate(JERK_CONDITION_ORDER))
    cond_suffix = f" [{prev_condition.key}]" if prev_condition is not None else ""
    condition = _ask(
        f"\n[jerk] trial {index}: condition ({listing}){cond_suffix}, or q to finish: ",
        prev_condition, _parse_condition,
    )
    if condition is _QUIT:
        return None

    delta_suffix = f" [{prev_delta:g}]" if prev_delta is not None else ""
    delta = _ask(
        f"[jerk] trial {index}: delta -- action magnitude on the travel axis, "
        f"0 <= d <= 1 (0.33 / 0.66 / 1.0){delta_suffix}: ",
        prev_delta, _parse_delta,
    )
    if delta is _QUIT:
        return None

    out_suffix = f" [{prev_out_steps}]" if prev_out_steps is not None else ""
    out_steps = _ask(
        f"[jerk] trial {index}: n_out -- timesteps held at full delta in the initial "
        f"direction, out of {total_steps}{out_suffix}: ",
        prev_out_steps, _step_count_parser(total_steps, "n_out"),
    )
    if out_steps is _QUIT:
        return None

    remaining = int(total_steps) - int(out_steps)
    slow_default = (None if prev_slowdown is None else min(int(prev_slowdown), remaining))
    slow_suffix = f" [{slow_default}]" if slow_default is not None else ""
    slowdown = _ask(
        f"[jerk] trial {index}: t -- timesteps ramping linearly to a stop before the "
        f"reversal, 0 to {remaining} (0 = reverse at full delta){slow_suffix}: ",
        slow_default, _step_count_parser(remaining, "t"),
    )
    if slowdown is _QUIT:
        return None

    spec = JerkTrialSpec(index=index, condition=condition, delta=float(delta),
                         out_steps=int(out_steps), slowdown_steps=int(slowdown),
                         total_steps=int(total_steps))
    if spec.back_steps == 0:
        print(f"  note: n_out={int(out_steps)} + t={int(slowdown)} uses the whole "
              f"{total_steps}-step trial, so there is no return leg and nothing reverses.")
    return spec


def jerk_start_pose(condition: JerkCondition, lims, edge_lims, args) -> np.ndarray:
    """Where the trial parks: the far end of the axis it is about to travel.

    ``start_pose_for`` on the OUTBOUND direction, which is exactly the "start at
    the opposite end of the table from where the arm ends up" rule -- `up_down`
    starts at the bottom, `right_left` at the left edge, and so on.
    """
    return start_pose_for(
        condition.out, lims, edge_lims,
        mode=args.start_mode, base_xy=args.base_robot_xy, margin=float(args.start_margin),
    )


def report_jerk_plan(lims, edge_lims, move_lims, args) -> None:
    """Preview the seeded split against every condition x delta before the arm moves.

    The split itself is typed per trial; what this cannot be read off the prompt
    is whether the outbound leg fits, so that is the point of the printout.
    """
    t = int(args.jerk_slowdown_steps)
    n_out = (0 if args.jerk_out_steps == AUTO_OUT_STEPS else int(args.jerk_out_steps))
    n_back = max(0, int(args.jerk_steps) - n_out - t)
    print(f"[jerk] {args.jerk_steps} commanded steps per trial "
          f"({args.jerk_settle_steps} zero-action settle step(s) first), split "
          f"out + t + back; seeded at {n_out} + {t} + {n_back}. "
          f"rmax = {move_lims[0]:g} m/step on x (vertical), "
          f"{move_lims[1]:g} m/step on y (horizontal) at |action|=1.")
    print("[jerk] n_out and t are typed per trial -- the numbers below are what the "
          "seeded split would do on each axis, so you can pick an n_out the room can "
          "actually take.")
    for key in JERK_CONDITION_ORDER:
        cond = JERK_CONDITIONS[key]
        start_xy = jerk_start_pose(cond, lims, edge_lims, args)
        room = float(room_in_direction(start_xy, cond.out, lims, edge_lims)[cond.axis_index])
        print(f"[jerk] {key:<11} start robot xy=({start_xy[0]:+.3f},{start_xy[1]:+.3f}), "
              f"room {room:.3f} m on {cond.axis}.  [{cond.description}]")
        for delta in args.jerk_deltas:
            spec = JerkTrialSpec(index=0, condition=cond, delta=float(delta),
                                 out_steps=n_out, slowdown_steps=t,
                                 total_steps=int(args.jerk_steps))
            plan = jerk_phase_plan(spec, start_xy, lims, edge_lims, move_lims)
            note = ""
            if plan["overruns"]:
                fits = max_out_steps(plan["room_m"], plan["step_m"], plan["ramp_m"])
                note = (f"  !! asks {plan['travel_m']:.3f} m of {plan['room_m']:.3f} m -- "
                        f"pins against the far edge at step {plan['saturates_at_step']}; "
                        f"n_out <= {fits} would stay inside at this delta and t")
            print(f"[jerk]   delta={delta:<5g} step {plan['step_m']:.3f} m "
                  f"-> out {plan['out_m']:.3f} m + ramp {plan['ramp_m']:.3f} m "
                  f"= {plan['travel_m']:.3f} m{note}")


def run_jerk_session(args, config_path: Path) -> None:
    """Interactive reversal-jerk battery (the --jerk path)."""
    args.out_dir = args.out_dir or default_out_dir("reversal_jerk")
    # Validated here rather than at the first prompt: a typo in either should
    # fail before the env is constructed and the arm is powered, not after the
    # operator has already cleared the table and pressed ENTER.
    if int(args.jerk_steps) < 1:
        raise SystemExit(f"--jerk-steps must be at least 1; got {args.jerk_steps}.")
    if int(args.jerk_slowdown_steps) < 0:
        raise SystemExit(
            f"--jerk-slowdown-steps cannot be negative; got {args.jerk_slowdown_steps}.")
    if str(args.jerk_out_steps).strip().lower() == "auto":
        args.jerk_out_steps = AUTO_OUT_STEPS
    else:
        try:
            args.jerk_out_steps = int(args.jerk_out_steps)
        except ValueError:
            raise SystemExit(
                f"--jerk-out-steps takes a whole number or 'auto'; got {args.jerk_out_steps!r}.")
        if args.jerk_out_steps < 0:
            raise SystemExit(f"--jerk-out-steps cannot be negative; got {args.jerk_out_steps}.")
        if args.jerk_out_steps + int(args.jerk_slowdown_steps) > int(args.jerk_steps):
            raise SystemExit(
                f"--jerk-out-steps {args.jerk_out_steps} + --jerk-slowdown-steps "
                f"{args.jerk_slowdown_steps} exceeds --jerk-steps {args.jerk_steps}; the seeded "
                f"split has to leave room for the return leg.")
    try:
        condition_seed = (None if args.jerk_condition is None
                          else _parse_condition(args.jerk_condition))
    except ValueError as exc:
        raise SystemExit(f"--jerk-condition: {exc}")
    # Built (and fully validated) before the env exists, so a typo in the spec
    # fails now rather than partway through a long unattended session.
    batch_trials = (parse_jerk_batch(args.jerk_batch, int(args.jerk_steps),
                                     int(args.jerk_out_steps))
                    if args.jerk_batch else None)
    if args.dry_run:
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sim_params = cfg["air_hockey"]["simulator_params"]
        lims, edge_lims, move_lims = real_preview_geometry(sim_params)
        if batch_trials is not None:
            report_jerk_batch(batch_trials, lims, edge_lims, move_lims, args)
        else:
            report_jerk_plan(lims, edge_lims, move_lims, args)
        for warning in check_no_smoothing(cfg):
            print(f"[warn] {warning}")
        print(f"\n[dry-run] would write jerk trials to {args.out_dir}")
        return

    env, cfg = load_env(config_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = env.simulator
    move_lims = tuple(float(v) for v in sim.move_lims)

    session_meta = {
        "config_path": str(config_path),
        "image_path": sim.image_path,
        "reset_settle_s": float(args.reset_settle_s),
        "jerk_settle_steps": int(args.jerk_settle_steps),
        "jerk_steps": int(args.jerk_steps),
        "reset_pose": np.asarray(sim.reset_pose[0], dtype=float),
        "move_lims": np.asarray(move_lims, dtype=float),
        "workspace_lims": np.asarray(sim.lims, dtype=float),
        "edge_lims": np.asarray(sim.edge_lims, dtype=float),
        "hist_len": int(sim.hist_len),
        "control_type": str(sim.control_type),
        "control_mode": str(sim.control_mode),
        "block_time": float(sim.block_time),
        "center_offset_constant": float(sim.center_offset_constant),
        "session_start_iso": datetime.now().astimezone().isoformat(),
    }

    print(f"[jerk] config={config_path}")
    print(f"[jerk] out_dir={out_dir.resolve()}")
    print(f"[jerk] workspace x{tuple(sim.lims[:2])} y{tuple(sim.lims[2:])}  "
          f"move_lims (m/step at |action|=1) = {move_lims}")
    for warning in check_no_smoothing(cfg):
        print(f"[warn] {warning}")
    if batch_trials is not None:
        report_jerk_batch(batch_trials, sim.lims, sim.edge_lims, move_lims, args)
    else:
        report_jerk_plan(sim.lims, sim.edge_lims, move_lims, args)

    if not args.no_wait:
        input("\n[jerk] Clear the table and workspace, then press ENTER to begin the "
              "session (Ctrl-C to abort)... ")

    manifest_entries: list[dict] = []
    interrupted = False
    index = next_trial_index(out_dir, "jerk", JERK_INDEX_RE)
    first_index = index
    if index > 0:
        print(f"[jerk] {out_dir} already holds trials up to index {index - 1}; "
              f"this session starts at {index}.")
    prev_condition: JerkCondition | None = condition_seed
    prev_delta: float | None = (None if args.jerk_delta is None else float(args.jerk_delta))
    prev_out_steps: int | None = (None if args.jerk_out_steps == AUTO_OUT_STEPS
                                  else int(args.jerk_out_steps))
    prev_slowdown: int | None = int(args.jerk_slowdown_steps)
    batch_i = 0
    try:
        while True:
            if batch_trials is not None:
                if batch_i >= len(batch_trials):
                    break
                # The placeholder index from parse_jerk_batch is replaced here, so a
                # batch continues the directory's numbering like any other session.
                spec = resolve_out_steps(
                    replace(batch_trials[batch_i], index=index),
                    sim.lims, sim.edge_lims, move_lims, args)
                batch_i += 1
                print(f"\n[jerk] batch {batch_i}/{len(batch_trials)}", end="")
            else:
                spec = prompt_jerk_trial(index, prev_condition, prev_delta, prev_out_steps,
                                         prev_slowdown, int(args.jerk_steps))
                if spec is None:
                    break
                prev_condition, prev_delta = spec.condition, spec.delta
                prev_out_steps, prev_slowdown = spec.out_steps, spec.slowdown_steps

            start_xy = jerk_start_pose(spec.condition, sim.lims, sim.edge_lims, args)
            plan = jerk_phase_plan(spec, start_xy, sim.lims, sim.edge_lims, move_lims)
            print(f"\n[{index}] {spec.name}  [{spec.condition.description}]")
            print(f"  {plan['n_out']} out + {plan['n_slow']} slow + {plan['n_back']} back "
                  f"= {plan['n_out'] + plan['n_slow'] + plan['n_back']} steps at "
                  f"|action|={spec.delta:g} ({plan['step_m']:.3f} m/step); "
                  f"travel {plan['travel_m']:.3f} m of {plan['room_m']:.3f} m room.")
            if plan["overruns"]:
                fits = max_out_steps(plan["room_m"], plan["step_m"], plan["ramp_m"])
                print(f"  !! the outbound leg does not fit: it asks {plan['travel_m']:.3f} m "
                      f"of {plan['room_m']:.3f} m, so the paddle pins against the far edge at "
                      f"step {plan['saturates_at_step']} and stalls there -- the ramp and the "
                      f"reversal then happen from a standstill. Recorded either way; "
                      f"n_out <= {fits} would stay inside at this delta and t.")
            elif plan["returns_short"]:
                print(f"  note: the return covers {plan['back_m']:.3f} m of the "
                      f"{plan['travel_m']:.3f} m gone out, so the paddle finishes short of "
                      f"where it started. Each trial resets, so this only matters if you "
                      f"wanted a closed round trip.")
            if plan["n_back"] == 0:
                print("  !! no return leg -- nothing reverses in this trial.")
            print(f"  start robot xy=({start_xy[0]:+.3f},{start_xy[1]:+.3f}) -- "
                  f"moving there now, keep clear of the arm.")

            record = run_jerk_trial(env, spec, session_meta, start_xy, plan)
            entry = write_jerk_trial_hdf5(
                out_dir / f"{spec.name}.hdf5", spec, record, plan, session_meta,
                save_images=not args.no_save_images,
            )
            manifest_entries.append(entry)
            index += 1
            print(f"  saved {entry['file']} ({entry['num_steps']} steps, "
                  f"images={entry['has_images']})")
            if record["protective_stop"]:
                print("[jerk] Stopping the session: the robot reported a protective stop. "
                      "Clear it and re-run.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[jerk] interrupted; writing manifest for completed trials.")
    finally:
        manifest = {
            "experiment": "reversal_jerk",
            "session_start_iso": session_meta["session_start_iso"],
            "session_end_iso": datetime.now().astimezone().isoformat(),
            "config_path": str(config_path),
            "out_dir": str(out_dir.resolve()),
            "interrupted": interrupted,
            "completed_trials": len(manifest_entries),
            "conditions": list(JERK_CONDITION_ORDER),
            "jerk_steps": int(args.jerk_steps),
            "jerk_settle_steps": int(args.jerk_settle_steps),
            "jerk_deltas": [float(d) for d in args.jerk_deltas],
            "jerk_batch": (args.jerk_batch or None),
            "planned_trials": (None if batch_trials is None else len(batch_trials)),
            "out_steps_seed": int(args.jerk_out_steps),
            "slowdown_steps_seed": int(args.jerk_slowdown_steps),
            "delta_seed": (None if args.jerk_delta is None else float(args.jerk_delta)),
            "condition_seed": args.jerk_condition,
            "reset_settle_s": float(args.reset_settle_s),
            "start_mode": args.start_mode,
            "config_reset_pose": [float(v) for v in session_meta["reset_pose"]],
            "move_lims": [float(v) for v in move_lims],
            "workspace_lims": [float(v) for v in sim.lims],
            "center_offset_constant": session_meta["center_offset_constant"],
            "hist_len": session_meta["hist_len"],
            "vals_column_names": VALS_COLUMN_NAMES,
            "trials": manifest_entries,
            "first_trial_index": first_index,
            # Sessions that wrote into this same directory before this one,
            # oldest first; their trials are still on disk under their own indices.
            "previous_sessions": load_previous_sessions(out_dir / "manifest.json"),
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        prior = len(manifest["previous_sessions"])
        print(f"\n[jerk] wrote {len(manifest_entries)} trajectories + manifest.json "
              f"to {out_dir.resolve()}"
              + (f" ({prior} earlier session(s) carried forward in the manifest)"
                 if prior else ""))
        if not interrupted:
            try:
                env.reset()
            except Exception as exc:
                print(f"[jerk] final reset skipped: {exc}")
        close_fn = getattr(env.simulator, "close", None) or getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def add_jerk_arguments(parser) -> None:
    """Flags for the --jerk experiment. None of them affect the other two paths."""
    group = parser.add_argument_group("reversal-jerk experiment (--jerk)")
    group.add_argument("--jerk", action="store_true",
                       help="Run the interactive reversal-jerk battery INSTEAD of the scripted "
                            "paddle-motion plan: per trial you type the condition (vertical "
                            "up-then-down / down-then-up, horizontal right-then-left / "
                            "left-then-right), the delta and t, and the paddle drives from one "
                            "end of the axis to the other, ramps to a stop over t steps, and "
                            "drives straight back. No puck.")
    group.add_argument("--jerk-steps", type=int, default=20,
                       help="Commanded steps per trial, split out + t + back.")
    group.add_argument("--jerk-batch", type=str, nargs="?", const=DEFAULT_JERK_BATCH,
                       default=None, metavar="SPEC",
                       help="Run a scripted battery with NO per-trial prompts. Bare "
                            "--jerk-batch runs the default plan (up_down at 0.66 and 1.0 x "
                            "t=3..0 once each, then right_left at 0.33/0.66/1.0 x t=3..0 "
                            "three times each; 44 trials). Pass a "
                            "SPEC to run something else: semicolon-separated "
                            "'condition:deltas:ts:repeats[:n_out[:total_steps]]' blocks, run in "
                            "the order written, e.g. "
                            "'up_down:0.66,1.0:3,2,1,0:1;right_left:0.33:3,0:3::44'. The optional "
                            "5th field sets that block's n_out (otherwise --jerk-out-steps) and "
                            "the 6th its trial length (otherwise --jerk-steps); leave a field "
                            "empty to keep the default. The 6th is how a lateral block buys back "
                            "travel, since rmax_y is less than half rmax_x. The one ENTER safety "
                            "gate still applies unless you also pass --no-wait.")
    group.add_argument("--jerk-out-steps", type=str, default="auto", metavar="N|auto",
                       help="Timesteps held at full delta before the ramp starts (seeds the "
                            "prompt in interactive mode; sets every trial in --jerk-batch). "
                            "Default 'auto' sizes it per trial to the largest cruise that still "
                            "leaves room for the ramp on that axis at that delta and t, so the "
                            "paddle is moving when the ramp starts and the ramp finishes before "
                            "the edge. Give a number to fix it by hand.")
    group.add_argument("--jerk-slowdown-steps", type=int, default=0,
                       help="Seed for the per-trial t prompt: timesteps spent ramping the action "
                            "linearly to zero before the reversal. 0 is the bang-bang reversal "
                            "(+delta straight to -delta), which is the baseline this experiment "
                            "compares the ramps against.")
    group.add_argument("--jerk-delta", type=float, default=None,
                       help="Seed for the per-trial delta prompt (a bare ENTER then reuses the "
                            "previous trial's answer). Omit it and the first trial must type "
                            "one. Ignored under --jerk-batch, which carries its own deltas.")
    group.add_argument("--jerk-condition", type=str, default=None,
                       help="Seed for the per-trial condition prompt: up_down, down_up, "
                            "right_left or left_right (up / down / right / left also work).")
    group.add_argument("--jerk-deltas", type=float, nargs="+", default=[0.33, 0.66, 1.0],
                       help="Deltas the plan report previews per condition. It does not run "
                            "them -- the trials are the ones you type -- it just shows how far "
                            "the seeded n_out would travel at each, so you can see which ones "
                            "run out of table before you start.")
    group.add_argument("--jerk-settle-steps", type=int, default=0,
                       help="Zero-action steps recorded before the outbound leg, on top of "
                            "--reset-settle-s. Useful as an at-rest acceleration baseline in "
                            "the same file as the reversal.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def curve_velocity_gain(args) -> float:
    """Action-to-velocity gain for the arc tracker: CLI override, else the UR5's.

    3.2 1/s was measured off the straight-line battery in
    data/robot_data_collection/paddle_motion_20260909_1808 (0.386 m/s sustained
    at action 1.0 on y with rmax_y = 0.12; 0.82 m/s on x with rmax_x = 0.26).
    """
    if args.curve_velocity_gain is not None:
        return float(args.curve_velocity_gain)
    return float(CURVE_VELOCITY_GAIN_REAL)


def curve_steps(args) -> int:
    """Timesteps per curve trial: --curve-steps, else the same as the lines."""
    return int(args.curve_steps) if args.curve_steps else int(args.action_steps)


def _resolve_start_and_actions(trials, directions, args, lims, edge_lims, move_lims):
    """Build the whole plan: line trials, then curve trials, with their start poses.

    Line trials get a start pose per direction and a constant per-axis action;
    curve trials carry their own start pose and a per-step action sequence whose
    length depends on the sweep speed. The two lists are concatenated and
    renumbered so trial indices stay contiguous.
    """
    start_poses = {
        d.key: start_pose_for(
            d, lims, edge_lims,
            mode=args.start_mode, base_xy=args.base_robot_xy, margin=float(args.start_margin),
        )
        for d in directions
    }
    trials = resolve_plan_actions(
        trials, start_poses, mode=args.delta_mode, lims=lims, edge_lims=edge_lims,
        move_lims=move_lims, action_steps=int(args.action_steps),
    )
    curves = parse_curves(args.curves)
    if curves:
        curve_trials, curve_starts, _infos = build_curve_plan(
            curves, (list(args.curve_speeds) if args.curve_speeds else None),
            int(args.repeats), args.order,
            lims, edge_lims, move_lims,
            margin=float(args.start_margin), min_steps=int(args.min_curve_steps),
            tracking_gain=float(args.curve_tracking_gain),
            tracking=str(args.curve_tracking),
            velocity_gain=curve_velocity_gain(args),
            n_steps=curve_steps(args),
            lookahead_steps=float(args.curve_lookahead_steps),
        )
        start_poses.update(curve_starts)
        trials = reindex(list(trials) + list(curve_trials))
    if not trials:
        raise SystemExit("--directions and --curves are both empty; nothing to run")
    return start_poses, trials


def report_start_poses(directions, start_poses, args, lims, edge_lims, move_lims):
    """Print each direction's start pose, room, and what limits its top speed."""
    curves = parse_curves(args.curves)
    if curves and args.curve_tracking == "closed-loop":
        gain = curve_velocity_gain(args)
        print(f"[curve] closed-loop tracking, action->velocity gain {gain:g} 1/s "
              f"(action 1.0 sustains {gain * move_lims[0]:.2f} m/s in x, "
              f"{gain * move_lims[1]:.2f} m/s in y), {curve_steps(args)} steps per trial.")
    for c in curves:
        start_xy, a, b, y_centre = curve_geometry(c, lims, edge_lims, float(args.start_margin))
        print(
            f"[curve] {c.key:<11} start robot xy=({start_xy[0]:+.3f},{start_xy[1]:+.3f}) "
            f"-> apex x={start_xy[0] - b:+.3f}, end y={y_centre + a:+.3f}; "
            f"half-width {a:.3f} m, height {b:.3f} m, aspect {b / a:.2f}.  [{c.description}]"
        )
    for d in directions:
        start_xy = start_poses[d.key]
        room = room_in_direction(start_xy, d, lims, edge_lims)
        travelled = " ".join(f"{ax}:{room[i]:.3f}m" for i, ax in enumerate("xy") if d.vec[i])
        if args.delta_mode == "workspace":
            full = steps_for_full_scale(d, start_xy, lims, edge_lims, move_lims)
            tail = (f"delta=1.0 traverses it in {int(args.action_steps)} steps; "
                    f"|action| would hit 1.0 at ~{full} step(s)")
        else:
            n = steps_to_saturation(start_xy, d, max(args.deltas), move_lims, lims, edge_lims)
            tail = f"at delta={max(args.deltas):g} the first axis leaves the workspace after ~{n} step(s)"
        print(
            f"[start] {d.key:<8} start robot xy=({start_xy[0]:+.3f},{start_xy[1]:+.3f}) "
            f"-> room {travelled}; {tail}.  [{d.description}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect open-loop constant-action paddle trajectories on the real robot.",
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Real-robot env config YAML (default: the bundled no-smoothing config).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for the per-trial HDF5 files and manifest.json. "
                             "Defaults to data/robot_data_collection/<experiment>_<timestamp>.")
    add_plan_arguments(parser)
    add_start_pose_arguments(parser)
    parser.add_argument("--reset-settle-s", type=float, default=3.0,
                        help="Seconds to hold at the reset pose before each trial starts, so the "
                             "arm is fully at rest. The paddle is re-clamped during the wait "
                             "(forceMode times out after ~2 s).")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Skip the train_img dataset (frames are still captured for puck detection).")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip the 'press ENTER to start' safety gate.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the trial plan and exit without constructing the env.")
    add_collision_arguments(parser)
    add_jerk_arguments(parser)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    if args.puck_collision and args.jerk:
        raise SystemExit("--puck-collision and --jerk are separate experiments; pick one.")

    if args.puck_collision:
        # Separate experiment, separate entry point: nothing below this runs, so
        # the scripted paddle-motion plan is untouched by the collision flags.
        run_puck_collision_session(args, config_path)
        return

    if args.jerk:
        # Likewise: the reversal-jerk battery is its own entry point and shares
        # only the env loading, the start-pose geometry and the prompt helpers.
        run_jerk_session(args, config_path)
        return

    args.out_dir = args.out_dir or default_out_dir("paddle_motion")
    directions = parse_directions(args.directions)
    trials = build_trial_plan(directions, list(args.deltas), int(args.repeats), args.order)

    if args.dry_run:
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sim_params = cfg["air_hockey"]["simulator_params"]
        lims, edge_lims, move_lims = real_preview_geometry(sim_params)
        start_poses, trials = _resolve_start_and_actions(trials, directions, args, lims, edge_lims, move_lims)
        report_start_poses(directions, start_poses, args, lims, edge_lims, move_lims)
        print_plan(trials, move_lims, int(args.action_steps))
        for warning in check_no_smoothing(cfg):
            print(f"[warn] {warning}")
        print(f"\n[dry-run] would write {len(trials)} files to {args.out_dir}")
        return

    # Connect first: a failed RTDE/camera connect should not leave an empty
    # session directory behind.
    env, cfg = load_env(config_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = env.simulator
    move_lims = tuple(float(v) for v in sim.move_lims)

    session_meta = {
        "config_path": str(config_path),
        "image_path": sim.image_path,
        "settle_steps": int(args.settle_steps),
        "action_steps": int(args.action_steps),
        "delta_mode": str(args.delta_mode),
        "curve_tracking": str(args.curve_tracking),
        "curve_velocity_gain": curve_velocity_gain(args),
        "curve_steps": curve_steps(args),
        "reset_pose": np.asarray(sim.reset_pose[0], dtype=float),
        "move_lims": np.asarray(move_lims, dtype=float),
        "workspace_lims": np.asarray(sim.lims, dtype=float),
        "edge_lims": np.asarray(sim.edge_lims, dtype=float),
        "hist_len": int(sim.hist_len),
        "control_type": str(sim.control_type),
        "control_mode": str(sim.control_mode),
        "block_time": float(sim.block_time),
        "session_start_iso": datetime.now().astimezone().isoformat(),
    }

    print(f"[collect] config={config_path}")
    print(f"[collect] out_dir={out_dir.resolve()}")
    print(f"[collect] start mode={args.start_mode} (config reset pose was {session_meta['reset_pose'][:2]})")
    print(f"[collect] workspace x{tuple(sim.lims[:2])} y{tuple(sim.lims[2:])}  "
          f"move_lims (m/step at |action|=1) = {move_lims}")

    for warning in check_no_smoothing(cfg):
        print(f"[warn] {warning}")

    start_poses, trials = _resolve_start_and_actions(
        trials, directions, args, sim.lims, sim.edge_lims, move_lims
    )
    session_meta["start_poses_robot"] = {
        key: [float(v) for v in xy] for key, xy in start_poses.items()
    }
    report_start_poses(directions, start_poses, args, sim.lims, sim.edge_lims, move_lims)
    print_plan(trials, move_lims, int(args.action_steps))

    if not args.no_wait:
        input("\n[collect] Clear the table and workspace, then press ENTER to start (Ctrl-C to abort)... ")

    manifest_entries: list[dict] = []
    interrupted = False
    try:
        for trial in trials:
            print(f"\n[{trial.index + 1}/{len(trials)}] {trial.name}  action={trial.action}")
            start_robot_xy = start_poses[trial.key]
            record = run_trial(env, trial, int(args.action_steps), int(args.settle_steps),
                               float(args.reset_settle_s), start_robot_xy)
            entry = write_trial_hdf5(
                out_dir / f"{trial.name}.hdf5", trial, record, session_meta,
                save_images=not args.no_save_images,
            )
            manifest_entries.append(entry)
            print(f"  saved {entry['file']} ({entry['num_steps']} steps, images={entry['has_images']})")
            if record["protective_stop"]:
                print("[collect] Stopping the session: the robot reported a protective stop. "
                      "Clear it and re-run the remaining conditions with --directions/--deltas.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[collect] interrupted; writing manifest for completed trials.")
    finally:
        manifest = {
            "session_start_iso": session_meta["session_start_iso"],
            "session_end_iso": datetime.now().astimezone().isoformat(),
            "config_path": str(config_path),
            "out_dir": str(out_dir.resolve()),
            "interrupted": interrupted,
            "planned_trials": len(trials),
            "completed_trials": len(manifest_entries),
            "directions": [d.key for d in directions],
            "direction_vecs": {d.key: [int(v) for v in d.vec] for d in directions},
            "deltas": [float(d) for d in args.deltas],
            "repeats": int(args.repeats),
            "action_steps": int(args.action_steps),
            "settle_steps": int(args.settle_steps),
            "delta_mode": args.delta_mode,
            "curves": [c.key for c in parse_curves(args.curves)],
            "curve_tracking": str(args.curve_tracking),
            "curve_velocity_gain": curve_velocity_gain(args),
            "curve_steps": curve_steps(args),
            "curve_lookahead_steps": float(args.curve_lookahead_steps),
            "curve_speeds": ([float(v) for v in args.curve_speeds]
                             if args.curve_speeds else None),
            "reset_settle_s": float(args.reset_settle_s),
            "order": args.order,
            "start_mode": args.start_mode,
            "start_poses_robot": session_meta["start_poses_robot"],
            "config_reset_pose": [float(v) for v in session_meta["reset_pose"]],
            "move_lims": [float(v) for v in move_lims],
            "workspace_lims": [float(v) for v in sim.lims],
            "hist_len": session_meta["hist_len"],
            "vals_column_names": VALS_COLUMN_NAMES,
            "trials": manifest_entries,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[collect] wrote {len(manifest_entries)} trajectories + manifest.json to {out_dir.resolve()}")
        # Park the paddle back at the initial pose: the last trial ends wherever it
        # ran out of workspace. Skipped after Ctrl-C, where the operator asked for
        # the robot to stop moving.
        if not interrupted:
            try:
                env.reset()
            except Exception as exc:
                print(f"[collect] final reset skipped: {exc}")
        # AirHockeyBaseEnv inherits gymnasium's no-op close(); the simulator's own
        # close() is what releases the camera and the async render process.
        close_fn = getattr(env.simulator, "close", None) or getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
