"""Open-loop paddle-motion data collection in Box2D --- sim counterpart of
``collect_paddle_motion.py``.

Runs the exact same scripted battery as the real-robot collector (same
``trial_plan`` module, so the same flags give the same conditions in the same
order): from a per-direction initial paddle pose, hold one constant action for N
timesteps, save the trajectory, reset, repeat. Six directions --- four axis-aligned
plus the two diagonals (bottom-left -> top-right, bottom-right -> top-left) ---
each holding a constant action where ``delta`` is the action itself (a diagonal
at 0.33 is ``(-0.33, +-0.33)``). See ``trial_plan.resolve_action``. The puck is parked at the far end
of the table and frozen, so nothing perturbs the paddle --- matching a real
session collected on a cleared table.

``--curves`` adds the same closed-loop half-arc family as the real collector.
The tracker is identical and the requested path speeds are the same; only the
action-to-velocity gain differs (Box2D's paddle PID is ~3x stiffer than the
UR5's servo), so the same arc is commanded with smaller actions here.

Each trial gets its own HDF5 file plus a GIF; every direction also gets a summary
GIF spanning its whole delta sweep.

Comparing against the real battery later
----------------------------------------
The directly comparable quantities are, per timestep:

    sim  ``paddle_pos_robot``   <->  real ``train_vals[:, 5:7]``   (pose_x, pose_y)
    sim  ``target_pos_robot``   <->  real ``train_vals[:, 26:28]`` (desired_pose_x/y)
    sim  ``paddle_vel``         <->  real ``train_vals[:, 11:13]`` (speed_x, speed_y)

Both sides are in ROBOT frame (table frame minus ``center_offset_constant``), both
start from the same pose, and both ran the same action schedule with smoothing
disabled. ``scripts/visualization/replay_real_in_sim.py`` is the existing
side-by-side renderer if you want frames rather than curves.

Usage::

    python -m scripts.robot_data_collection.collect_paddle_motion_sim \\
        --out-dir data/robot_data_collection/paddle_motion_sim_$(date +%Y%m%d_%H%M)

    python -m scripts.robot_data_collection.collect_paddle_motion_sim --dry-run
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
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from airhockey import AirHockeyEnv  # noqa: E402
from airhockey.renderers.render import AirHockeyRenderer  # noqa: E402
from scripts.robot_data_collection.trial_plan import (  # noqa: E402
    Trial,
    add_plan_arguments,
    add_start_pose_arguments,
    CURVE_VELOCITY_GAIN_SIM,
    build_curve_plan,
    build_trial_plan,
    parse_curves,
    parse_directions,
    reindex,
    print_plan,
    room_in_direction,
    box2d_preview_geometry,
    resolve_plan_actions,
    curve_geometry,
    start_pose_for,
    step_displacement,
    steps_for_full_scale,
    steps_to_saturation,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "robot_data_collection" / "paddle_motion_sim_config.yaml"

GIF_FPS = 20  # matches the env's 20 Hz step rate, so GIFs play at real time
GIF_WIDTH = 160  # repo convention for Box2D qualitative GIFs


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


def load_env(config_path: Path) -> tuple[AirHockeyEnv, dict]:
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ah = dict(cfg["air_hockey"])
    if "seed" not in ah:
        ah["seed"] = int(cfg.get("seed", 0))
    if "n_training_steps" not in ah:
        ah["n_training_steps"] = int(cfg.get("n_training_steps", 1))
    ah.setdefault("return_goal_obs", False)
    return AirHockeyEnv(ah), cfg


def check_no_smoothing(cfg: dict) -> list[str]:
    sim_params = cfg.get("air_hockey", {}).get("simulator_params", {}) or {}
    hist_len = int(sim_params.get("hist_len", 2))
    if hist_len != 1:
        return [
            f"hist_len={hist_len} (expected 1): _filter_update will average the last "
            f"{hist_len} target deltas, i.e. commanded actions ARE smoothed."
        ]
    return []


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def park_and_freeze_puck(env: AirHockeyEnv, park_xy_table) -> None:
    """Pin every puck body at the far end with zero velocity.

    Re-applied every step: the puck is a dynamic body under gravity, so a one-off
    placement would drift back down the table and eventually reach the paddle.
    """
    sim = env.simulator
    park_box2d = sim.base_coord_to_box2d(park_xy_table)
    for puck_body in sim.pucks.values():
        puck_body.position = park_box2d
        puck_body.linearVelocity = (0.0, 0.0)
        puck_body.angularVelocity = 0.0


def paddle_state(env: AirHockeyEnv) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth paddle (position, velocity) in table frame."""
    paddle = env.current_state["paddles"]["paddle_ego"]
    return (
        np.asarray(paddle["position"][:2], dtype=np.float64),
        np.asarray(paddle["velocity"][:2], dtype=np.float64),
    )


def target_position(env: AirHockeyEnv) -> np.ndarray:
    """Last PID target in table frame (NaN before the first step)."""
    target = getattr(env.simulator, "last_target_position", None)
    if target is None:
        return np.full(2, np.nan, dtype=np.float64)
    return np.asarray(target[:2], dtype=np.float64)


def to_gif_frame(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    """BGR full-size frame -> RGB frame resized to `width`, aspect preserved."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height = max(1, int(round(rgb.shape[0] * width / rgb.shape[1])))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def annotate(frame: np.ndarray, trial: Trial, step_i: int, n_steps: int,
             paddle_robot_xy: np.ndarray, phase: str) -> np.ndarray:
    """Draw trial identity / step / paddle pose on an ALREADY-RESIZED RGB frame.

    Annotating after the resize keeps the text at native output resolution; a label
    drawn on the 360 px render first would be downscaled into mush at the
    conventional 160 px GIF width.
    """
    frame = np.ascontiguousarray(frame)
    lines = [
        f"{trial.key} d{trial.delta:.2f} t{trial.repeat}",
        f"{phase[:3]} {step_i}/{n_steps}",
        f"{paddle_robot_xy[0]:+.3f} {paddle_robot_xy[1]:+.3f}",
    ]
    for row, text in enumerate(lines):
        origin = (3, 11 + row * 11)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_PLAIN, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_trial(
    env: AirHockeyEnv,
    trial: Trial,
    action_steps: int,
    settle_steps: int,
    start_robot_xy: np.ndarray,
    park_xy_table: np.ndarray,
    renderer: AirHockeyRenderer | None,
    gif_width: int,
) -> dict:
    """Reset to the initial pose, then hold one constant action for N steps."""
    center_offset_pre = float(env.simulator.center_offset_constant)
    start_table_xy = np.asarray(start_robot_xy, dtype=np.float64) + np.array([center_offset_pre, 0.0])
    start_state = np.concatenate([start_table_xy, [0.0, 0.0], park_xy_table, [0.0, 0.0]])
    obs, _info = env.reset_from_state(start_state)
    # reset_from_state (unlike reset) leaves the episode counters alone, so a long
    # multi-trial session would eventually trip max_timesteps truncation.
    env.current_timestep = 0
    env.episode_return = 0.0
    env.episode_length = 0
    env.success_in_ep = False
    park_and_freeze_puck(env, park_xy_table)

    center_offset = float(env.simulator.center_offset_constant)
    start_pos_table, _ = paddle_state(env)
    step_actions = trial.step_actions(action_steps)
    tracker = trial.make_tracker()
    zero_action = np.zeros(2, dtype=np.float32)
    schedule = [(zero_action, 1)] * settle_steps + [(a, 0) for a in step_actions]

    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    is_settle: list[int] = []
    frames: list[np.ndarray] = []

    if renderer is not None:
        frames.append(annotate(
            to_gif_frame(renderer.get_frame(), gif_width), trial, 0, len(schedule),
            start_pos_table - np.array([center_offset, 0.0]), "start",
        ))

    for step_i, (step_action, settle_flag) in enumerate(schedule, start=1):
        if tracker is not None and not settle_flag:
            # Closed loop: the schedule entry is only the dry-run preview; the
            # action actually sent chases a carrot on the arc from where the
            # paddle really is. Same feedback signal as the real collector, in
            # the same ROBOT frame.
            pos_table_now, _ = paddle_state(env)
            step_action = tracker.action(pos_table_now - np.array([center_offset, 0.0]))
        obs, _reward, _terminated, _truncated, _info = env.step(step_action)
        park_and_freeze_puck(env, park_xy_table)
        pos_table, vel = paddle_state(env)
        actions.append(np.asarray(step_action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        positions.append(pos_table)
        velocities.append(vel)
        targets.append(target_position(env))
        is_settle.append(int(settle_flag))
        if renderer is not None:
            frames.append(annotate(
                to_gif_frame(renderer.get_frame(), gif_width), trial, step_i, len(schedule),
                pos_table - np.array([center_offset, 0.0]),
                "settle" if settle_flag else "action",
            ))

    positions_table = np.stack(positions, axis=0)
    targets_table = np.stack(targets, axis=0)
    offset = np.array([center_offset, 0.0])
    return {
        "start_robot_xy_commanded": np.asarray(start_robot_xy, dtype=np.float64),
        "start_pos_table": start_pos_table,
        "start_pos_robot": start_pos_table - offset,
        "actions": np.stack(actions, axis=0),
        "observations": np.stack(observations, axis=0),
        "paddle_pos_table": positions_table,
        "paddle_pos_robot": positions_table - offset,
        "paddle_vel": np.stack(velocities, axis=0),
        "target_pos_table": targets_table,
        "target_pos_robot": targets_table - offset,
        "is_settle": np.asarray(is_settle, dtype=np.int8),
        "frames": frames,
    }


def write_trial_hdf5(path: Path, trial: Trial, record: dict, session_meta: dict) -> dict:
    with h5py.File(path, "w") as hf:
        for key in (
            "actions", "observations", "paddle_pos_table", "paddle_pos_robot",
            "paddle_vel", "target_pos_table", "target_pos_robot",
        ):
            hf.create_dataset(key, data=record[key])
        hf.create_dataset("is_settle_step", data=record["is_settle"])
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
                hf.attrs["curve_velocity_gain"] = trial.track.velocity_gain
                hf.attrs["curve_lookahead_steps"] = trial.track.lookahead_steps
                hf.attrs["curve_arc_length_m"] = trial.track.arc_length_m
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
        hf.attrs["start_pos_table"] = record["start_pos_table"]
        hf.attrs["start_pos_robot"] = record["start_pos_robot"]
        hf.attrs["start_robot_xy_commanded"] = record["start_robot_xy_commanded"]
        for key in (
            "config_path", "move_lims", "workspace_lims", "edge_lims", "hist_len",
            "center_offset_constant", "time_per_step", "session_start_iso",
            "park_puck_xy_table", "start_mode", "delta_mode",
            "curve_tracking", "curve_velocity_gain", "curve_steps",
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
        "n_action_steps": int(len(trial.schedule)) if trial.is_curve else None,
        "axis": trial.axis,
        "direction_sign": trial.sign,
        "action_delta": trial.delta,
        "action": [float(v) for v in trial.action],
        "repeat": trial.repeat,
        "num_steps": int(record["actions"].shape[0]),
        "settle_steps": session_meta["settle_steps"],
        "action_steps": session_meta["action_steps"],
        "start_pos_robot": [float(v) for v in record["start_pos_robot"]],
        "start_robot_xy_commanded": [float(v) for v in record["start_robot_xy_commanded"]],
        "final_pos_robot": [float(v) for v in record["paddle_pos_robot"][-1]],
        "displacement_robot": [
            float(record["paddle_pos_robot"][-1][0] - record["start_pos_robot"][0]),
            float(record["paddle_pos_robot"][-1][1] - record["start_pos_robot"][1]),
        ],
    }


def save_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, format="GIF", loop=0, duration=int(1000 / fps))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def curve_velocity_gain(args) -> float:
    """Action-to-velocity gain for the arc tracker: CLI override, else Box2D's.

    Box2D's paddle PID closes ~50% of the commanded lead per 0.05 s step, i.e.
    G ~= 10 1/s -- about 3x stiffer than the UR5's 3.2 (see
    trial_plan.CURVE_VELOCITY_GAIN_REAL). That difference is exactly why the old
    open-loop schedule looked right here and came out a third of the table wide
    on the robot.
    """
    if args.curve_velocity_gain is not None:
        return float(args.curve_velocity_gain)
    return float(CURVE_VELOCITY_GAIN_SIM)


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
        description="Collect open-loop constant-action paddle trajectories in Box2D.",
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Box2D env config YAML (default: the bundled no-smoothing sim config).")
    parser.add_argument("--out-dir", type=str,
                        default=f"data/robot_data_collection/paddle_motion_sim_{datetime.now():%Y%m%d_%H%M%S}",
                        help="Directory for the per-trial HDF5 files, GIFs and manifest.json.")
    add_plan_arguments(parser)
    add_start_pose_arguments(parser)
    parser.add_argument("--no-gifs", action="store_true", help="Skip GIF rendering.")
    parser.add_argument("--gif-width", type=int, default=GIF_WIDTH,
                        help="GIF width in px, aspect preserved (repo convention: 160).")
    parser.add_argument("--gif-fps", type=int, default=GIF_FPS,
                        help="GIF frame rate (20 = real time at the env's 20 Hz step rate).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the trial plan and exit without constructing the env.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    directions = parse_directions(args.directions)
    trials = build_trial_plan(directions, list(args.deltas), int(args.repeats), args.order)

    if args.dry_run:
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        sim_params = cfg["air_hockey"]["simulator_params"]
        lims, edge_lims, move_lims = box2d_preview_geometry(sim_params)
        start_poses, trials = _resolve_start_and_actions(trials, directions, args, lims, edge_lims, move_lims)
        report_start_poses(directions, start_poses, args, lims, edge_lims, move_lims)
        print_plan(trials, move_lims, int(args.action_steps))
        for warning in check_no_smoothing(cfg):
            print(f"[warn] {warning}")
        print(f"\n[dry-run] would write {len(trials)} files to {args.out_dir}")
        return

    env, cfg = load_env(config_path)
    sim = env.simulator
    out_dir = Path(args.out_dir)
    gif_dir = out_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    center_offset = float(sim.center_offset_constant)
    # Far end of the table, out of the paddle's reachable workspace.
    park_xy_table = np.array([-(float(sim.length) / 2.0 - 0.01), 0.0], dtype=np.float64)
    move_lims = tuple(float(v) for v in np.asarray(sim.move_lims).reshape(-1)[:2])
    start_poses, trials = _resolve_start_and_actions(
        trials, directions, args, sim.lims, sim.edge_lims, move_lims
    )

    session_meta = {
        "config_path": str(config_path),
        "settle_steps": int(args.settle_steps),
        "action_steps": int(args.action_steps),
        "delta_mode": str(args.delta_mode),
        "curve_tracking": str(args.curve_tracking),
        "curve_velocity_gain": curve_velocity_gain(args),
        "curve_steps": curve_steps(args),
        "move_lims": np.asarray(move_lims, dtype=float),
        "workspace_lims": np.asarray(sim.lims, dtype=float),
        "edge_lims": np.asarray(sim.edge_lims, dtype=float),
        "hist_len": int(sim.hist_len),
        "center_offset_constant": center_offset,
        "time_per_step": float(sim.time_per_step),
        "park_puck_xy_table": park_xy_table,
        "start_mode": str(args.start_mode),
        "session_start_iso": datetime.now().astimezone().isoformat(),
    }

    renderer = None
    if not args.no_gifs:
        renderer = AirHockeyRenderer(
            env, orientation="vertical", show_target_position=True, show_acceleration_arrow=False,
        )

    print(f"[collect-sim] config={config_path}")
    print(f"[collect-sim] out_dir={out_dir.resolve()}")
    print(f"[collect-sim] start mode={args.start_mode}")
    print(f"[collect-sim] workspace x{tuple(sim.lims[:2])} y{tuple(sim.lims[2:])}  move_lims={move_lims}")
    print(f"[collect-sim] puck parked+frozen at table {park_xy_table}")
    for warning in check_no_smoothing(cfg):
        print(f"[warn] {warning}")

    report_start_poses(directions, start_poses, args, sim.lims, sim.edge_lims, move_lims)
    print_plan(trials, move_lims, int(args.action_steps))

    manifest_entries: list[dict] = []
    direction_frames: dict[str, list[np.ndarray]] = {}
    for trial in trials:
        record = run_trial(
            env, trial, int(args.action_steps), int(args.settle_steps),
            start_poses[trial.key], park_xy_table, renderer, int(args.gif_width),
        )
        entry = write_trial_hdf5(out_dir / f"{trial.name}.hdf5", trial, record, session_meta)
        manifest_entries.append(entry)
        if record["frames"]:
            save_gif(gif_dir / f"{trial.name}.gif", record["frames"], int(args.gif_fps))
            key = trial.key
            direction_frames.setdefault(key, []).extend(record["frames"])
        dx, dy = entry["displacement_robot"]
        print(
            f"[{trial.index + 1}/{len(trials)}] {trial.name}: "
            f"{entry['num_steps']} steps, net displacement dx={dx:+.4f} dy={dy:+.4f} m"
        )

    summary_gifs = []
    for key, frames in direction_frames.items():
        summary_path = gif_dir / f"summary_{key}.gif"
        save_gif(summary_path, frames, int(args.gif_fps))
        summary_gifs.append(summary_path.name)

    manifest = {
        "session_start_iso": session_meta["session_start_iso"],
        "session_end_iso": datetime.now().astimezone().isoformat(),
        "source": "box2d",
        "config_path": str(config_path),
        "out_dir": str(out_dir.resolve()),
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
        "order": args.order,
        "start_mode": args.start_mode,
        "start_poses_robot": {
            key: [float(v) for v in xy] for key, xy in start_poses.items()
        },
        "park_puck_xy_table": [float(v) for v in park_xy_table],
        "move_lims": [float(v) for v in move_lims],
        "workspace_lims": [float(v) for v in sim.lims],
        "hist_len": session_meta["hist_len"],
        "gif_fps": int(args.gif_fps),
        "gif_width": int(args.gif_width),
        "summary_gifs": summary_gifs,
        "trials": manifest_entries,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[collect-sim] wrote {len(manifest_entries)} trajectories + manifest.json to {out_dir.resolve()}")
    if summary_gifs:
        print(f"[collect-sim] GIFs in {gif_dir.resolve()} ({len(manifest_entries)} per-trial + "
              f"{len(summary_gifs)} summary: {', '.join(summary_gifs)})")


if __name__ == "__main__":
    main()
