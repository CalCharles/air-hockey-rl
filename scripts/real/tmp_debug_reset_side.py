"""TEMPORARY: run the puck reset FSM on one side only, with per-step logging.

Debug harness for the reset policy's right-side (+y) behaviour. It does not edit
``rollout_reset_policy_real.py`` or anything the eval / online-finetune
entrypoints run: the side is forced by setting
``rollout_reset_policy_real._DEBUG_FORCE_END_SIDE`` in this process only.

Why not ``rollout_reset_policy_real.py --force-end-side right``: that flag works
through the rng, but with ``_DEBUG_FORCE_END_SIDE = "alternate"`` the FSM takes
its side from the alternation and never asks the rng, so the flag is ignored.
That script also never calls ``set_reset_workspace``, so it runs without the
reset-only widening (``reset_y_max_extension_m``,
``reset_right_corner_x_extension_m``, ``reset_hist_len``) that the eval /
finetune reset runner applies. This script runs each reset the way
``real_reset_runner.run_reset_fsm`` does: widen the workspace, build the FSM,
step it until done, restore the workspace.

Each reset waits for Enter (or starts right away with ``--continuous``). Per
reset it prints a summary and writes a CSV with one row per step (phase, action,
paddle / puck in the TCP frame the FSM plans in, occlusion, path index).

Usage (real-robot machine, repo root):

    python -m scripts.real.tmp_debug_reset_side                      # right side, hist4 config
    python -m scripts.real.tmp_debug_reset_side --side left          # compare with the left side
    python -m scripts.real.tmp_debug_reset_side --continuous --num-resets 10
    python -m scripts.real.tmp_debug_reset_side --reset-y-max-extension-m 0.03 \\
        --reset-right-corner-x-extension-m 0.04                        # try other right-side reach

Left-ending / right-ending side-wall fixes (2026-10-12) now live in ResetPolicyFSM
(``left_end_wall_press_m``, ``right_end_hold_y`` ...) and AirHockeyReal.set_reset_end_side
(``reset_left_corner_x_extension_m`` for both end sides), so eval / online finetune get them
too; the flags below only override them for this run.

Keys while running: x = exit after the current step, h = abort the current reset
and hard-reset the robot.
"""
import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

import scripts.real.rollout_reset_policy_real as rrp
from airhockey import AirHockeyEnv
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from scripts.td3.helper.real_td3_runtime import _latest_camera_frame, _prepare_air_hockey_config

# Simulator params a CLI flag can override for this run only (None = keep config / default).
_SIM_OVERRIDES = (
    "reset_x_max_extension_m",
    "reset_y_max_extension_m",
    "reset_side_wall_clearance_m",
    "reset_right_corner_x_extension_m",
    "reset_left_corner_x_extension_m",
    "reset_hist_len",
)

_CSV_FIELDS = (
    "step", "wall_t", "phase", "action_x", "action_y",
    "paddle_tcp_x", "paddle_tcp_y", "paddle_actual_x", "paddle_actual_y", "puck_tcp_x", "puck_tcp_y", "puck_occluded",
    "path_idx", "n_waypoints", "stage2_cycles", "done_reason",
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config-path", default="configs/real_configs/rollout_td3_config_hist4.yaml")
    p.add_argument("--side", choices=("right", "left"), default="right",
                   help="Corner every reset ENDS at (burst / strike fires there); right = +y.")
    p.add_argument("--num-resets", type=int, default=0, help="Stop after this many resets (0 = until 'x').")
    p.add_argument("--continuous", action="store_true",
                   help="Start the next reset right after the previous one instead of waiting for Enter.")
    p.add_argument("--max-steps", type=int, default=600,
                   help="Abort a reset that has not finished after this many steps.")
    p.add_argument("--success-line", type=float, default=0.5,
                   help="shared_success_threshold_proportion_from_bottom for the FSM.")
    p.add_argument("--ignore-wall", action="store_true",
                   help="success_ignore_wall=True: hand over above the success line even with the puck on a side wall.")
    p.add_argument("--handoff-wall-gap-m", type=float, default=None,
                   help="ResetPolicyFSM handoff_wall_gap_m (min puck-edge-to-side-wall gap at hand-off).")
    p.add_argument("--no-widen", action="store_true",
                   help="Skip set_reset_workspace (run on the normal policy workspace).")
    p.add_argument("--show-path", action="store_true", help="Show the planned sweep on the camera frame.")
    p.add_argument("--debug-control", action="store_true", help="Keep the simulator's [control_debug] prints.")
    p.add_argument("--right-end-wall-press-m", type=float, default=None,
                   help="ResetPolicyFSM right_end_wall_press_m (right-ending sweeps; 0 = old sweep).")
    p.add_argument("--right-end-press-until-y-m", type=float, default=None,
                   help="ResetPolicyFSM right_end_press_until_y_m.")
    p.add_argument("--left-end-wall-press-m", type=float, default=0.03,
                   help="Left-ending sweeps: aim x this far past the path (into the bottom limit) while the "
                        "waypoint y is below --left-end-press-below-y-m. 0 = old sweep.")
    p.add_argument("--left-end-press-below-y-m", type=float, default=0.0,
                   help="Left-ending sweeps: press only where the waypoint y is below this (TCP frame; "
                        "the left corner is -y), i.e. on the way into the left corner.")
    p.add_argument("--no-right-end-hold-y", action="store_true",
                   help="Right-ending resets: old behaviour (hits made from wherever y ended up, 1-3 cm short "
                        "of the right limit).")
    p.add_argument("--right-end-wall-y-offset-m", type=float, default=0.0,
                   help="Right-ending resets: hold y this far inside the right limit (0 = at the limit, paddle "
                        "edge ~6 mm off the wall).")
    p.add_argument("--right-end-arrive-y-m", type=float, default=0.005,
                   help="Right-ending resets: sweep ends only once the paddle is within this of the end y.")
    p.add_argument("--no-timed-strike", action="store_true",
                   help="right_end_timed_strike=False: old fixed-distance second-hit trigger.")
    for name in ("timed_strike_contact_steps", "timed_strike_camera_latency_steps", "timed_strike_paddle_travel_m"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=None, help=f"ResetPolicyFSM {name}.")
    p.add_argument("--no-strike-pullback", action="store_true",
                   help="right_end_strike_pullback_margin_m=None: wait for the puck where the paddle is (no run-up).")
    p.add_argument("--no-full-force-strike", action="store_true",
                   help="right_end_full_force_strike=False: the old ramped strike (-0.3, -0.6, -0.8, then full).")
    p.add_argument("--no-video", action="store_true", help="Skip the per-reset camera .mp4.")
    p.add_argument("--video-fps", type=int, default=20, help="Playback fps of the camera .mp4 (control rate is 20 Hz).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="real_runs/tmp_debug_reset_side")
    for name in _SIM_OVERRIDES:
        p.add_argument("--" + name.replace("_", "-"), type=int if name == "reset_hist_len" else float, default=None)
    return p.parse_args()


def _fmt_lims(sim) -> str:
    x_min, x_max, y_min, y_max = (float(v) for v in sim.lims)
    top_abs, bot_abs, bias_p, bias_m = (float(v) for v in sim.edge_lims)
    return (f"x=[{x_min:+.4f}, {x_max:+.4f}] y=[{y_min:+.4f}, {y_max:+.4f}] "
            f"edge(top_abs={top_abs:.3f} bot_abs={bot_abs:.3f} bias_p(+y)={bias_p:+.4f} bias_m(-y)={bias_m:+.4f})")


class _CameraVideo:
    """Per-reset camera .mp4 (same frames the finetune stack saves), opened on the first frame."""

    def __init__(self, path: Path, fps: int):
        self.path, self.fps = path, int(fps)
        self.writer, self.size, self.n_frames, self.n_missing = None, None, 0, 0

    def write(self, env, text_rows):
        frame = _latest_camera_frame(env)
        if frame is None:
            self.n_missing += 1
            return
        frame = np.ascontiguousarray(frame[:, :, :3] if frame.shape[2] >= 3 else np.repeat(frame, 3, axis=2),
                                     dtype=np.uint8)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                self.path = self.path.with_suffix(".avi")
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"XVID"), self.fps, self.size)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        for i, row in enumerate(text_rows):
            cv2.putText(frame, row, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        self.writer.write(frame)
        self.n_frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.n_frames:
            print(f"[debug_reset]   camera video: {self.path} ({self.n_frames} frames, {self.n_missing} missing)")
        else:
            print(f"[debug_reset]   camera video: no frames available ({self.n_missing} missing)")


def _run_one_reset(env, rng, args, reset_idx: int, out_dir: Path, nbc) -> dict:
    sim = env.simulator
    widen = None if args.no_widen else getattr(sim, "set_reset_workspace", None)
    if widen is not None:
        widen(True)
    print(f"\n[debug_reset] reset {reset_idx}: workspace {_fmt_lims(sim)}"
          f"{'' if widen is not None else ' (not widened)'}")

    fsm = rrp.ResetPolicyFSM(
        env,
        rng,
        left_end_wall_press_m=args.left_end_wall_press_m,
        left_end_press_below_y_m=args.left_end_press_below_y_m,
        right_end_hold_y=not args.no_right_end_hold_y,
        right_end_wall_y_offset_m=args.right_end_wall_y_offset_m,
        right_end_arrive_y_m=args.right_end_arrive_y_m,
        shared_success_threshold_proportion_from_bottom=float(args.success_line),
        success_ignore_wall=args.ignore_wall,
        capture_second_hit_frame=False,
        **{k: getattr(args, k) for k in ("right_end_wall_press_m", "right_end_press_until_y_m", "handoff_wall_gap_m",
                                              "timed_strike_contact_steps", "timed_strike_camera_latency_steps",
                                              "timed_strike_paddle_travel_m")
           if getattr(args, k) is not None},
        **({"right_end_timed_strike": False} if args.no_timed_strike else {}),
        **({"right_end_strike_pullback_margin_m": None} if args.no_strike_pullback else {}),
        **({"right_end_full_force_strike": False} if args.no_full_force_strike else {}),
    )
    wp = fsm.path_waypoints
    print(f"[debug_reset] start_side={fsm.start_side} end_side={fsm._forced_end_side} "
          f"sweep ({wp[0, 0]:+.4f},{wp[0, 1]:+.4f}) -> ({wp[-1, 0]:+.4f},{wp[-1, 1]:+.4f}) "
          f"strike_mag={-fsm._strike_actions[-1]:.3f} puck_proximity_m={fsm.puck_proximity_m:.3f}")
    if args.show_path:
        rrp.show_reset_path_on_camera(fsm, sim)

    csv_path = out_dir / f"reset_{reset_idx:03d}_{fsm._forced_end_side}.csv"
    video = None if args.no_video else _CameraVideo(csv_path.with_suffix(".mp4"), args.video_fps)
    phase_steps = Counter()
    phase_order = []
    puck_y_max = paddle_y_max = -np.inf
    aborted = None
    t0 = time.time()
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            while not fsm.done:
                state = sim.get_current_state()
                phase_before = fsm.phase
                action = np.asarray(fsm.step(state), dtype=np.float32)
                if not phase_order or phase_order[-1] != phase_before:
                    phase_order.append(phase_before)
                phase_steps[phase_before] += 1

                paddle = fsm._get_tcp_position(state)  # commanded target pose
                try:
                    paddle_actual = np.asarray(sim.rcv.getActualTCPPose(), dtype=float)[:2]
                except Exception:
                    paddle_actual = np.full(2, np.nan)
                puck = fsm._get_puck_pos(state)
                occluded = bool(fsm._puck_is_occluded(state))
                paddle_y_max = max(paddle_y_max, float(paddle[1]))
                if not occluded:
                    puck_y_max = max(puck_y_max, float(puck[1]))
                writer.writerow({
                    "step": fsm.total_steps, "wall_t": f"{time.time() - t0:.3f}", "phase": phase_before,
                    "action_x": f"{action[0]:+.4f}", "action_y": f"{action[1]:+.4f}",
                    "paddle_tcp_x": f"{paddle[0]:+.4f}", "paddle_tcp_y": f"{paddle[1]:+.4f}",
                    "paddle_actual_x": f"{paddle_actual[0]:+.4f}", "paddle_actual_y": f"{paddle_actual[1]:+.4f}",
                    "puck_tcp_x": f"{puck[0]:+.4f}", "puck_tcp_y": f"{puck[1]:+.4f}",
                    "puck_occluded": int(occluded), "path_idx": fsm.path_idx, "n_waypoints": len(wp),
                    "stage2_cycles": fsm.stage2_cycle_count, "done_reason": fsm.done_reason,
                })
                env.step(action)
                if video is not None:
                    video.write(env, [f"reset {reset_idx} {fsm._forced_end_side}  step {fsm.total_steps}",
                                      f"{phase_before} -> {fsm.phase}"])
                # No env.reset() between debug resets, so step()'s per-frame image list would grow unbounded.
                if isinstance(getattr(sim, "images", None), list) and len(sim.images) > 1:
                    del sim.images[:-1]

                key = nbc.get_data()
                if key in ("x", "h"):
                    aborted = "operator_exit" if key == "x" else "operator_hard_reset"
                    break
                if fsm.total_steps >= args.max_steps:
                    aborted = "max_steps"
                    break
    finally:
        fsm.close()
        if video is not None:
            video.close()
        if widen is not None:
            widen(False)

    reason = aborted or fsm.done_reason
    state = sim.get_current_state()
    puck_x = float(state["pucks"][0]["position"][0])
    paddle_x = float(state["paddles"]["paddle_ego"]["position"][0])
    summary = {
        "reset": reset_idx, "side": fsm._forced_end_side, "reason": reason,
        "success_stage": fsm.last_success_stage if reason == "success" else "-",
        "steps": fsm.total_steps, "wall_s": round(time.time() - t0, 2),
        "stage2_cycles": fsm.stage2_cycle_count,
        "max_paddle_tcp_y": round(paddle_y_max, 4), "max_puck_tcp_y": round(puck_y_max, 4),
        "puck_above_paddle_at_end": puck_x < paddle_x,
    }
    print(f"[debug_reset] reset {reset_idx} DONE reason={reason} steps={fsm.total_steps} "
          f"stage2_cycles={fsm.stage2_cycle_count} puck_above_paddle={summary['puck_above_paddle_at_end']}")
    print(f"[debug_reset]   phases: " + " -> ".join(phase_order))
    print(f"[debug_reset]   steps/phase: " + ", ".join(f"{k}={v}" for k, v in phase_steps.items()))
    print(f"[debug_reset]   max paddle tcp y={paddle_y_max:+.4f}  max visible puck tcp y={puck_y_max:+.4f}  "
          f"(sweep end y={wp[-1, 1]:+.4f})")
    print(f"[debug_reset]   per-step log: {csv_path}")

    if reason in ("hard_reset_required", "operator_hard_reset", "max_steps"):
        rrp.hard_reset_with_pause(env, reason=f"debug_reset_{reason}", pause_s=1.0)
    return summary


def main():
    args = _parse_args()
    # Process-local: only this script's FSMs see it. The source file keeps "alternate".
    rrp._DEBUG_FORCE_END_SIDE = args.side

    # Same loader as the eval / finetune stack (the rollout_td3 configs have no
    # top-level n_training_steps / algorithm, which rrp.load_air_hockey_params needs).
    with open(args.config_path, "r") as f:
        params = _prepare_air_hockey_config(yaml.safe_load(f), seed=args.seed)
    params["simulator_params"] = dict(params.get("simulator_params", {}))
    sim_params = params["simulator_params"]
    sim_params["wait_for_space_to_start"] = False
    if not args.debug_control:
        sim_params["debug_control"] = False
    for name in _SIM_OVERRIDES:
        value = getattr(args, name)
        if value is not None:
            sim_params[name] = value
            print(f"[debug_reset] override {name}={value}")
    params["max_timesteps"] = max(100_000, int(params.get("max_timesteps", 0)))

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = AirHockeyEnv(params)
    sim = env.simulator
    print(f"[debug_reset] side={args.side} config={args.config_path} out={out_dir}")
    print(f"[debug_reset] normal workspace {_fmt_lims(sim)}")
    for name in _SIM_OVERRIDES:
        print(f"[debug_reset]   {name}={getattr(sim, name, 'n/a')}")
    print(f"[debug_reset] left-ending: corner_x_extension={getattr(sim, 'reset_left_corner_x_extension_m', 'n/a')} "
          f"wall_press={args.left_end_wall_press_m:.3f} below_y={args.left_end_press_below_y_m:+.3f}")
    print(f"[debug_reset] right-ending fix: hold_y={not args.no_right_end_hold_y} "
          f"wall_y_offset={args.right_end_wall_y_offset_m:.3f} arrive_y={args.right_end_arrive_y_m:.3f}")

    rng = np.random.default_rng(args.seed)
    summaries = []
    try:
        with NonBlockingConsole() as nbc:
            reset_idx = 0
            while args.num_resets <= 0 or reset_idx < args.num_resets:
                if not args.continuous:
                    print("\n[debug_reset] put the puck where you want it, then press Enter (x to quit)")
                    key = False
                    while key not in ("\n", "\r", "x"):
                        time.sleep(0.05)
                        key = nbc.get_data()
                    if key == "x":
                        break
                summary = _run_one_reset(env, rng, args, reset_idx, out_dir, nbc)
                summaries.append(summary)
                reset_idx += 1
                if summary["reason"] == "operator_exit":
                    break
                if args.continuous:
                    time.sleep(1.0)
    finally:
        if summaries:
            with open(out_dir / "summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                writer.writeheader()
                writer.writerows(summaries)
            n_ok = sum(s["reason"] == "success" for s in summaries)
            print(f"\n[debug_reset] {n_ok}/{len(summaries)} resets succeeded on side={args.side}; "
                  f"summary: {out_dir / 'summary.csv'}")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
