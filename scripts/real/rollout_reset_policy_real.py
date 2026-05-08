import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from airhockey.sims.real.coordinate_transform import get_clip_limits
from airhockey.sims.real.overlay_utils import robot_to_display_pixel_int
from scripts.real.agent import Agent as LegacyMLPAgent


def load_air_hockey_params(config_path: str, save_path_override: str = None) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    air_hockey_params = dict(cfg["air_hockey"])
    air_hockey_params["n_training_steps"] = cfg["n_training_steps"]
    seed_cfg = cfg.get("seed", 0)
    if isinstance(seed_cfg, (list, tuple)):
        seed_cfg = seed_cfg[0] if len(seed_cfg) > 0 else 0
    air_hockey_params["seed"] = int(seed_cfg)
    if cfg["algorithm"] == "sac" and "goal" in cfg["air_hockey"]["task"]:
        air_hockey_params["return_goal_obs"] = True
    else:
        air_hockey_params["return_goal_obs"] = False

    if save_path_override is not None:
        air_hockey_params["simulator_params"]["save_path"] = save_path_override
    return air_hockey_params


def infer_policy_dims_from_state_dict(state_dict):
    actor_input_dim = int(state_dict["actor.0.weight"].shape[1])
    action_dim = int(state_dict["actor_mean_head.weight"].shape[0])
    return actor_input_dim, action_dim


class ResetPolicyFSM:
    """Five-phase FSM for resetting the puck from the bottom of the table.

    Phases:
      1. goto_start    -- move paddle to the starting corner of the edge loop
      2. edge_loop     -- sweep along the bottom boundary from one side to the other
      3. upward_burst  -- flick paddle upward (negative x) for burst_steps
      4. wait_for_puck -- hold position until puck falls within puck_proximity_m
      5. strike        -- ramping upward strike [-0.3, -0.6, -1.0, -1.0, -1.0]
    """

    def __init__(
        self,
        env: AirHockeyEnv,
        rng: np.random.Generator,
        loop_max_delta_m: float = 0.1,
        burst_action_m: float = 0.2,
        burst_steps: int = 5,
        goto_start_arrive_m: float = 0.05,
        waypoint_advance_m: float = 0.05,
        puck_proximity_m: float = 0.5,
        post_upward_check_steps: int = 20,
        post_window_debug_log: bool = False,
        shared_success_threshold_proportion_from_bottom: float = 0.5,
        off_wall_abs_y_m: float = 0.35,
        min_off_wall_window_steps: int = 5,
        max_stage2_cycles: int = 5,
        capture_second_hit_frame: bool = True,
        async_second_hit_write: bool = False,
        show_second_hit_window: bool = False,
    ):
        self.env = env
        self.rng = rng
        self.loop_max_delta_m = float(loop_max_delta_m)
        self.burst_action_m = float(burst_action_m)
        self.burst_steps = int(burst_steps)
        self.goto_start_arrive_m = float(goto_start_arrive_m)
        self.waypoint_advance_m = float(waypoint_advance_m)
        self.post_upward_check_steps = max(1, int(post_upward_check_steps))
        self.post_window_debug_log = bool(post_window_debug_log)
        self.shared_success_threshold_proportion_from_bottom = float(
            shared_success_threshold_proportion_from_bottom
        )
        self.off_wall_abs_y_m = float(off_wall_abs_y_m)
        self.min_height_window_steps = 1
        self.min_off_wall_window_steps = max(1, int(min_off_wall_window_steps))
        self.max_stage2_cycles = max(1, int(max_stage2_cycles))
        strike_mag = float(self.rng.uniform(0.8, 1.0))
        self._strike_actions = [
            -0.3 * strike_mag,
            -0.6 * strike_mag,
            -0.8 * strike_mag,
            -strike_mag,
            -strike_mag,
            -strike_mag,
            -strike_mag,
            -strike_mag,
            -strike_mag,
        ]
        self.puck_proximity_m = float(self.rng.uniform(0.5, 0.6)) # less delay than before

        simulator = self.env.simulator
        self._lims = np.array(
            getattr(simulator, "lims", (-0.79, -0.375, -0.36, 0.36)),
            dtype=np.float32,
        )
        self._edge_lims = np.array(
            getattr(simulator, "edge_lims", (0.8, 0.1, -0.15, -0.15)),
            dtype=np.float32,
        )
        self._move_lims = np.array(
            getattr(simulator, "move_lims", (0.26, 0.12)),
            dtype=np.float32,
        )
        self._paddle_x_off = float(getattr(simulator, "paddle_additional_x_offset", 0.0))
        self._paddle_y_off = float(getattr(simulator, "paddle_additional_y_offset", 0.0))
        self._center_offset = float(getattr(simulator, "center_offset_constant", 0.0))
        self._x_offset = float(getattr(simulator, "x_offset", 0.0))

        self.phase = "goto_start"
        self.phase_steps = 0
        self.total_steps = 0
        self.done = False
        self.done_reason = "in_progress"
        self.last_success_stage = "unknown"
        self.last_success_motion = "unknown"
        self.stage2_cycle_count = 0
        self._window_steps_left = 0
        self._window_off_wall_count = 0
        self._window_height_count = 0
        self._window_shared_gate_count = 0
        self._window_kind = "none"
        self._window_target_offset_x_m = 0.075
        self._window_start_tcp = None
        self._window_target_tcp = None
        self._pending_window_finalize = None
        self._wait_hold_y_tcp = None
        self.path_idx = 0
        self.path_waypoints = np.zeros((1, 2), dtype=np.float32)
        self.force_log_interval_steps = 25
        self._captured_second_hit_frame = False
        self.capture_second_hit_frame = bool(capture_second_hit_frame)
        self.async_second_hit_write = bool(async_second_hit_write)
        self.show_second_hit_window = bool(show_second_hit_window)
        self._second_hit_frame_dir = Path("real_runs/async_td3/second_hit_frames")
        self._second_hit_writer = (
            ThreadPoolExecutor(max_workers=1)
            if self.capture_second_hit_frame and self.async_second_hit_write
            else None
        )
        self._build_edge_loop_path()

    def close(self) -> None:
        if self._second_hit_writer is not None:
            self._second_hit_writer.shutdown(wait=False, cancel_futures=False)
            self._second_hit_writer = None

    def _get_tcp_position(self, state_info: dict) -> np.ndarray:
        """Raw TCP position used for motion commands (path waypoints live in this frame)."""
        simulator = self.env.simulator
        if hasattr(simulator, "rcv"):
            try:
                tcp_pose = simulator.rcv.getTargetTCPPose()
                return np.array(tcp_pose[:2], dtype=np.float32)
            except Exception:
                pass
        if hasattr(simulator, "pose"):
            try:
                return np.array(simulator.pose[:2], dtype=np.float32)
            except Exception:
                pass
        paddle_obs = np.array(state_info["paddles"]["paddle_ego"]["position"], dtype=np.float32)
        return np.array(
            [
                paddle_obs[0] - self._x_offset - self._paddle_x_off,
                paddle_obs[1] - self._paddle_y_off,
            ],
            dtype=np.float32,
        )

    def _get_puck_pos(self, state_info: dict) -> np.ndarray:
        """Puck position in TCP-aligned frame (center_offset removed)."""
        puck_obs = np.array(state_info["pucks"][0]["position"], dtype=np.float32)
        return np.array([puck_obs[0] - self._center_offset, puck_obs[1]], dtype=np.float32)

    def _read_robot_wrench(self) -> np.ndarray:
        """Read the current TCP wrench from the real robot receiver when available."""
        simulator = self.env.simulator
        rcv = getattr(simulator, "rcv", None)
        if rcv is None:
            return None
        try:
            wrench = np.asarray(rcv.getActualTCPForce(), dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if wrench.size < 6:
            return None
        return wrench[:6].astype(np.float32, copy=False)

    def _maybe_log_robot_normal_force(self) -> None:
        interval = int(self.force_log_interval_steps)
        if interval <= 0 or (self.total_steps % interval) != 0:
            return
        wrench = self._read_robot_wrench()
        if wrench is None:
            print(
                f"[reset_fsm_force] step={self.total_steps} phase={self.phase} "
                "robot_wrench_unavailable=1"
            )
            return
        raw_fz = float(wrench[2])
        normal_force_n = abs(raw_fz)
        print(
            f"[reset_fsm_force] step={self.total_steps} phase={self.phase} "
            f"normal_force_n={normal_force_n:.3f} raw_fz={raw_fz:+.3f} "
            f"wrench=({float(wrench[0]):+.3f},{float(wrench[1]):+.3f},{float(wrench[2]):+.3f},"
            f"{float(wrench[3]):+.3f},{float(wrench[4]):+.3f},{float(wrench[5]):+.3f})"
        )

    def _meters_delta_to_action(self, delta_xy_m: np.ndarray) -> np.ndarray:
        move_lims = np.maximum(self._move_lims, 1e-6)
        action = delta_xy_m / move_lims
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _project_displacement_to_action_box(self, target_delta_xy_m: np.ndarray) -> np.ndarray:
        """Project desired displacement to the action box, matching TD3 primitive semantics."""
        target_delta = np.array(target_delta_xy_m, dtype=np.float32).reshape(2)
        scale = np.maximum(self._move_lims.astype(np.float32), 1e-6)
        normalized = target_delta / scale
        max_abs = float(np.max(np.abs(normalized)))
        projection_div = max(1.0, max_abs)
        return (normalized / projection_div).astype(np.float32)

    def _build_edge_loop_path(self) -> None:
        """Build waypoints tracing the bottom edge of the trapezoidal workspace."""
        _, _, y_min_lim, y_max_lim = self._lims
        self.start_side = "left" if self.rng.random() < 0.5 else "right"

        y_margin = 0.00
        y_min = float(y_min_lim + y_margin)
        y_max = float(y_max_lim - y_margin)
        y_start = y_min if self.start_side == "left" else y_max
        y_end = y_max if self.start_side == "left" else y_min

        n_points = 44
        ys = np.linspace(y_start, y_end, n_points, dtype=np.float32)
        x_pts = np.zeros_like(ys)
        loop_inset = 0.01
        loop_bulge = 0.00
        for i, y_val in enumerate(ys):
            _, x_max, _, _ = get_clip_limits(
                0.0,
                float(y_val),
                tuple(self._lims.tolist()),
                tuple(self._edge_lims.tolist()),
            )
            t = i / max(1, n_points - 1)
            inset = loop_inset + loop_bulge * np.sin(np.pi * t)
            x_pts[i] = float(x_max) - inset
        self.path_waypoints = np.stack([x_pts, ys], axis=1).astype(np.float32)
        self.path_idx = 0

    def _lookahead_target_on_path(self, paddle_pos: np.ndarray, lookahead_m: float) -> np.ndarray:
        while self.path_idx < len(self.path_waypoints) - 1:
            current_wp = self.path_waypoints[self.path_idx]
            if float(np.linalg.norm(current_wp - paddle_pos)) < self.waypoint_advance_m:
                self.path_idx += 1
            else:
                break

        if self.path_idx >= len(self.path_waypoints) - 1:
            return self.path_waypoints[-1]

        remaining = float(lookahead_m)
        j = self.path_idx
        while j < len(self.path_waypoints) - 1:
            p0 = self.path_waypoints[j]
            p1 = self.path_waypoints[j + 1]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1e-8:
                j += 1
                continue
            if remaining <= seg_len:
                alpha = remaining / seg_len
                return (p0 + alpha * seg).astype(np.float32)
            remaining -= seg_len
            j += 1
        return self.path_waypoints[-1]

    def _toward_target(self, paddle_pos: np.ndarray, target: np.ndarray, max_delta_m: float) -> np.ndarray:
        delta = target - paddle_pos
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float32)
        scaled_delta = delta * min(1.0, max_delta_m / norm)
        return self._meters_delta_to_action(scaled_delta)

    def _upward_burst_action(self) -> np.ndarray:
        """Upward on the table = negative x in TCP frame."""
        burst_delta_m = np.array([-self.burst_action_m, 0.0], dtype=np.float32)
        return self._meters_delta_to_action(burst_delta_m)

    def draw_path_overlay(self, frame, offset_constants=None, visual_downscale_constant=2.0):
        """Draw the planned edge-loop path and burst arrow on a camera frame."""
        if frame is None or len(self.path_waypoints) < 2:
            return frame

        sim = self.env.simulator
        if offset_constants is None:
            offset_constants = getattr(sim, "offset_constants", np.array((2250, 500)))
        vds = float(visual_downscale_constant)

        def _to_px(x_m, y_m):
            return robot_to_display_pixel_int(
                x_m, y_m,
                offset_constants=offset_constants,
                visual_downscale_constant=vds,
            )

        pts = np.array(
            [_to_px(float(wp[0]), float(wp[1])) for wp in self.path_waypoints],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        start_px = _to_px(float(self.path_waypoints[0, 0]), float(self.path_waypoints[0, 1]))
        end_px = _to_px(float(self.path_waypoints[-1, 0]), float(self.path_waypoints[-1, 1]))
        cv2.circle(frame, start_px, 8, (255, 100, 0), -1)
        cv2.circle(frame, end_px, 8, (0, 0, 255), -1)

        burst_len_m = 0.10
        end_xy = self.path_waypoints[-1]
        arrow_end_xy = (float(end_xy[0]) - burst_len_m, float(end_xy[1]))
        arrow_end_px = _to_px(*arrow_end_xy)
        cv2.arrowedLine(frame, end_px, arrow_end_px, (0, 0, 255), 2, tipLength=0.35)

        mid = len(self.path_waypoints) // 2
        wp0, wpm, wpn = self.path_waypoints[0], self.path_waypoints[mid], self.path_waypoints[-1]
        print(
            f"[reset_path] {len(self.path_waypoints)} waypoints, side={self.start_side}\n"
            f"  start=({wp0[0]:.4f}, {wp0[1]:.4f})  mid=({wpm[0]:.4f}, {wpm[1]:.4f})  "
            f"end=({wpn[0]:.4f}, {wpn[1]:.4f})  burst_dir=(-x, 0)"
        )
        return frame

    def _puck_is_occluded(self, state_info: dict) -> bool:
        return int(np.asarray(state_info["pucks"][0].get("occluded", 0)).reshape(-1)[0]) > 0

    def _quarter_line_tcp_x(self) -> float:
        """Quarter line between bottom and midline in TCP-aligned frame."""
        table_midline = (float(self.env.table_x_top) + float(self.env.table_x_bot)) / 2.0
        quarter_from_bottom = (float(self.env.table_x_bot) + table_midline) / 2.0
        return quarter_from_bottom - self._center_offset

    def _shared_success_height_tcp_x(self) -> float:
        """Shared upward threshold for all stages in TCP-aligned frame."""
        return self._line_tcp_x_from_bottom_proportion(
            self.shared_success_threshold_proportion_from_bottom
        )

    def _line_tcp_x_from_bottom_proportion(self, proportion_from_bottom: float) -> float:
        """Line position in TCP frame for a given [bottom->top] table proportion."""
        bottom = float(self.env.table_x_bot)
        top = float(self.env.table_x_top)
        line_world_x = bottom + float(proportion_from_bottom) * (top - bottom)
        return line_world_x - self._center_offset

    def _get_puck_motion_from_history(self, state_info: dict) -> tuple[bool, bool]:
        """Return (puck_falling, puck_rising) from history in table-x convention."""
        puck_history = state_info["pucks"][0].get("history", [])
        if len(puck_history) < 3:
            return False, False
        older_x = float(puck_history[-3][0])
        curr_x = float(puck_history[-1][0])
        puck_falling = (curr_x - older_x) > 0
        puck_rising = (curr_x - older_x) < 0
        return puck_falling, puck_rising

    def _log_phase_check(
        self,
        state_info: dict,
        phase_name: str,
        check_name: str,
        passed: bool,
        window_stats: dict = None,
    ) -> None:
        """Single-line log for phase completion checks."""
        puck = state_info["pucks"][0]
        puck_world_x = float(puck["position"][0])
        puck_pos = self._get_puck_pos(state_info)
        puck_tcp_x = float(puck_pos[0])
        puck_y = float(puck_pos[1])
        threshold_lookup = {
            "quarter": self._quarter_line_tcp_x(),
            "shared_success_gate": self._shared_success_height_tcp_x(),
        }
        threshold = threshold_lookup.get(check_name, self._quarter_line_tcp_x())
        result = "pass" if passed else "fail"
        off_wall = abs(puck_y) <= self.off_wall_abs_y_m
        shared_gate_steps = -1
        off_wall_steps = -1
        height_steps = -1
        required_height_steps = 1
        required_steps = self.min_off_wall_window_steps
        if isinstance(window_stats, dict):
            shared_gate_steps = int(window_stats.get("shared_gate_steps", -1))
            off_wall_steps = int(window_stats.get("off_wall_steps", -1))
            height_steps = int(window_stats.get("height_steps", -1))
            required_height_steps = int(window_stats.get("required_height_steps", 1))
            required_steps = int(window_stats.get("required_steps", self.min_off_wall_window_steps))
        print(
            f"[reset_fsm] {phase_name}_done check={check_name} result={result} "
            f"puck_world_x={puck_world_x:+.4f} puck_tcp_x={puck_tcp_x:+.4f} "
            f"threshold_tcp_x={threshold:+.4f} puck_y={puck_y:+.4f} "
            f"off_wall={int(off_wall)} y_abs_limit={self.off_wall_abs_y_m:.3f} "
            f"shared_gate_steps={shared_gate_steps} off_wall_steps={off_wall_steps} "
            f"height_steps={height_steps} required_height_steps={required_height_steps} "
            f"required_off_wall_steps={required_steps} "
            f"stage2_cycles={self.stage2_cycle_count}/{self.max_stage2_cycles} "
            f"total_steps={self.total_steps}"
        )

    def _log_new_round_start(self, reason: str) -> None:
        print(f"[reset_fsm] new_round_start reason={reason} total_steps={self.total_steps}")

    def _motion_estimate_label(self, state_info: dict) -> str:
        puck_falling, puck_rising = self._get_puck_motion_from_history(state_info)
        if puck_rising:
            return "up"
        if puck_falling:
            return "down"
        return "unknown"

    def _mark_success(self, state_info: dict, stage: str) -> None:
        motion = self._motion_estimate_label(state_info)
        self.last_success_stage = stage
        self.last_success_motion = motion
        self.done_reason = "success"
        self.done = True
        print(f"[reset_fsm] success stage={stage} motion_estimate={motion} total_steps={self.total_steps}")

    def _restart_round(self, reason: str) -> None:
        self.done_reason = "restart_round"
        self.stage2_cycle_count = 0
        self._captured_second_hit_frame = False
        self._wait_hold_y_tcp = None
        self._log_new_round_start(reason=reason)
        self._build_edge_loop_path()
        self.phase = "goto_start"
        self.phase_steps = 0

    def _set_terminal_reason(self, reason: str) -> None:
        self.done_reason = str(reason)
        self.done = True

    def _start_post_upward_window(
        self,
        kind: str,
        state_info: dict,
        cached_tcp: np.ndarray  = None,
    ) -> None:
        self._window_kind = str(kind)
        self._window_steps_left = int(self.post_upward_check_steps)
        self._window_off_wall_count = 0
        self._window_height_count = 0
        self._window_shared_gate_count = 0
        window_start_tcp = (
            np.array(cached_tcp, dtype=np.float32)
            if cached_tcp is not None
            else self._get_tcp_position(state_info)
        )
        target_tcp = np.array(
            [window_start_tcp[0] + self._window_target_offset_x_m, window_start_tcp[1]],
            dtype=np.float32,
        )
        self._window_start_tcp = window_start_tcp.astype(np.float32)
        self._window_target_tcp = target_tcp
        if self.post_window_debug_log:
            print(
                f"[reset_fsm] post_window_start kind={self._window_kind} "
                f"start_tcp=({self._window_start_tcp[0]:+.4f},{self._window_start_tcp[1]:+.4f}) "
                f"target_tcp=({self._window_target_tcp[0]:+.4f},{self._window_target_tcp[1]:+.4f}) "
                f"steps={self._window_steps_left}"
            )

    def _clear_post_upward_window_target(self) -> None:
        self._window_start_tcp = None
        self._window_target_tcp = None

    def _enter_wait_for_puck(self, cached_tcp: np.ndarray = None) -> None:
        self.phase = "wait_for_puck"
        tcp = (
            np.array(cached_tcp, dtype=np.float32)
            if cached_tcp is not None
            else None
        )
        if tcp is not None and tcp.shape[0] >= 2:
            self._wait_hold_y_tcp = float(tcp[1])
        else:
            self._wait_hold_y_tcp = None

    def _wait_for_puck_hold_action(
        self,
        state_info: dict,
        cached_tcp: np.ndarray = None,
    ) -> np.ndarray:
        current_tcp = (
            np.array(cached_tcp, dtype=np.float32)
            if cached_tcp is not None
            else self._get_tcp_position(state_info)
        )
        if self._wait_hold_y_tcp is None:
            self._wait_hold_y_tcp = float(current_tcp[1])
        y_delta_m = float(self._wait_hold_y_tcp) - float(current_tcp[1])
        hold_delta_m = np.array([0.0, y_delta_m], dtype=np.float32)
        action = self._project_displacement_to_action_box(hold_delta_m)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action[0] = 0.0
        return action

    def _current_window_downward_action(
        self,
        state_info: dict,
        cached_tcp: np.ndarray  = None,
    ) -> np.ndarray:
        if self._window_target_tcp is None:
            return np.zeros(2, dtype=np.float32)
        current_tcp = (
            np.array(cached_tcp, dtype=np.float32)
            if cached_tcp is not None
            else self._get_tcp_position(state_info)
        )
        puck_world = np.array(state_info["pucks"][0]["position"], dtype=np.float32)
        remaining_delta_m = self._window_target_tcp - current_tcp
        action = self._project_displacement_to_action_box(remaining_delta_m)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.post_window_debug_log:
            print(
                f"[reset_fsm] post_window_track kind={self._window_kind} steps_left={self._window_steps_left} "
                f"curr_tcp=({current_tcp[0]:+.4f},{current_tcp[1]:+.4f}) "
                f"puck_world=({puck_world[0]:+.4f},{puck_world[1]:+.4f}) "
                f"target_tcp=({self._window_target_tcp[0]:+.4f},{self._window_target_tcp[1]:+.4f}) "
                f"remaining=({remaining_delta_m[0]:+.4f},{remaining_delta_m[1]:+.4f}) "
                f"action=({action[0]:+.3f},{action[1]:+.3f})"
            )
        return action

    def _step_post_upward_window(
        self,
        state_info: dict,
        cached_tcp: np.ndarray  = None,
    ) -> np.ndarray:
        if self.phase == "post_first_upward_check":
            check_name = "shared_success_gate"
            phase_name = "stage1_upward"
        elif self.phase == "post_second_upward_check":
            check_name = "shared_success_gate"
            phase_name = "stage2_upward"
        else:
            return np.zeros(2, dtype=np.float32)

        off_wall_now = False
        above_height_now = False
        shared_gate_now = False
        if not self._puck_is_occluded(state_info):
            puck_pos = self._get_puck_pos(state_info)
            puck_tcp_x = float(puck_pos[0])
            puck_y = float(puck_pos[1])
            above_height_now = puck_tcp_x <= self._shared_success_height_tcp_x()
            off_wall_now = abs(puck_y) <= self.off_wall_abs_y_m
            shared_gate_now = bool(above_height_now and off_wall_now)
        if off_wall_now:
            self._window_off_wall_count += 1
        if above_height_now:
            self._window_height_count += 1
        if shared_gate_now:
            self._window_shared_gate_count += 1

        action = self._current_window_downward_action(state_info, cached_tcp=cached_tcp)
        passed_now = bool(
            (self._window_height_count >= self.min_height_window_steps)
            and (self._window_off_wall_count >= self.min_off_wall_window_steps)
        )
        if passed_now:
            self._pending_window_finalize = {
                "kind": self._window_kind,
                "phase_name": phase_name,
                "check_name": check_name,
                "passed": True,
                "shared_gate_steps": int(self._window_shared_gate_count),
                "off_wall_steps": int(self._window_off_wall_count),
                "height_steps": int(self._window_height_count),
                "required_height_steps": int(self.min_height_window_steps),
                "required_steps": int(self.min_off_wall_window_steps),
                "early_exit": True,
            }
            return self._finalize_post_upward_window(state_info)
        self._window_steps_left -= 1
        if self._window_steps_left <= 0:
            passed = bool(
                (self._window_height_count >= self.min_height_window_steps)
                and (self._window_off_wall_count >= self.min_off_wall_window_steps)
            )
            self._pending_window_finalize = {
                "kind": self._window_kind,
                "phase_name": phase_name,
                "check_name": check_name,
                "passed": passed,
                "shared_gate_steps": int(self._window_shared_gate_count),
                "off_wall_steps": int(self._window_off_wall_count),
                "height_steps": int(self._window_height_count),
                "required_height_steps": int(self.min_height_window_steps),
                "required_steps": int(self.min_off_wall_window_steps),
                "early_exit": False,
            }
        return action

    def _finalize_post_upward_window(self, state_info: dict) -> np.ndarray:
        if self._pending_window_finalize is None:
            return np.zeros(2, dtype=np.float32)
        finalize = dict(self._pending_window_finalize)
        self._pending_window_finalize = None
        self._clear_post_upward_window_target()
        self._log_phase_check(
            state_info,
            phase_name=str(finalize["phase_name"]),
            check_name=str(finalize["check_name"]),
            passed=bool(finalize["passed"]),
            window_stats=finalize,
        )
        if finalize["kind"] == "first":
            if bool(finalize["passed"]):
                self._mark_success(state_info, stage="stage1_upward")
                return np.zeros(2, dtype=np.float32)
            self.stage2_cycle_count = 0
            self._captured_second_hit_frame = False
            self._enter_wait_for_puck()
            return np.zeros(2, dtype=np.float32)
        if bool(finalize["passed"]):
            self._mark_success(state_info, stage="stage2_upward")
            return np.zeros(2, dtype=np.float32)
        self.stage2_cycle_count += 1
        if self.stage2_cycle_count >= self.max_stage2_cycles:
            print(
                f"[reset_fsm] stage2_max_cycles_reached "
                f"count={self.stage2_cycle_count} limit={self.max_stage2_cycles} -> hard reset required"
            )
            self._set_terminal_reason("hard_reset_required")
            return np.zeros(2, dtype=np.float32)
        self._captured_second_hit_frame = False
        print(
            f"[reset_fsm] stage2_retry cycle={self.stage2_cycle_count}/{self.max_stage2_cycles} "
            f"reason=shared_success_gate_not_met"
        )
        self._enter_wait_for_puck()
        return np.zeros(2, dtype=np.float32)

    @staticmethod
    def _write_second_hit_frames(frame, out_path: Path, latest_path: Path) -> tuple[bool, bool]:
        wrote_timestamped = cv2.imwrite(str(out_path), frame)
        wrote_latest = cv2.imwrite(str(latest_path), frame)
        return bool(wrote_timestamped), bool(wrote_latest)

    def _capture_second_hit_trigger_frame(self, dist_m: float, puck_falling: bool) -> None:
        """Capture transformed camera frame on the same control step before strike is sent."""
        if not self.capture_second_hit_frame:
            return
        simulator = self.env.simulator
        cap = getattr(simulator, "cap", None)
        if cap is None:
            print("[second_hit_capture] No camera available; skipping capture.")
            return
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            print("[second_hit_capture] Failed to read camera frame.")
            return
        from airhockey.sims.real.control_parameters import (
            homography_transform,
        )
        frame, _ = homography_transform(raw_frame, get_save=True, rotate=False)
        if frame is None:
            print("[second_hit_capture] Homography transform failed; skipping capture.")
            return

        text_rows = [
            f"phase={self.phase}",
            f"total_steps={self.total_steps}",
            f"dist={dist_m:.3f}m",
            f"puck_falling={int(bool(puck_falling))}",
            f"shared_success_threshold_tcp_x={self._shared_success_height_tcp_x():+.3f}",
            f"quarter_tcp_x={self._quarter_line_tcp_x():+.3f}",
        ]
        for i, row in enumerate(text_rows):
            y = 28 + i * 26
            cv2.putText(frame, row, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        self._second_hit_frame_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self._second_hit_frame_dir / f"second_hit_trigger_{timestamp}_step{self.total_steps}.png"
        latest_path = self._second_hit_frame_dir / "latest_second_hit_trigger.png"
        if self._second_hit_writer is None:
            wrote_timestamped, wrote_latest = self._write_second_hit_frames(frame, out_path, latest_path)
        else:
            self._second_hit_writer.submit(
                self._write_second_hit_frames,
                frame.copy(),
                out_path,
                latest_path,
            )
            wrote_timestamped, wrote_latest = True, True
            print(f"[second_hit_capture] queued async write {out_path}")
        if wrote_timestamped:
            print(f"[second_hit_capture] saved {out_path}")
            if wrote_latest:
                print(f"[second_hit_capture] updated {latest_path}")
        else:
            print(f"[second_hit_capture] Failed to save {out_path}")
        if self.show_second_hit_window:
            # Show immediately for real-time verification at trigger step.
            cv2.imshow("second_hit_trigger", frame)
            cv2.waitKey(1)

    def step(self, state_info: dict) -> np.ndarray:
        if self.done:
            return np.zeros(2, dtype=np.float32)

        self.total_steps += 1
        self._maybe_log_robot_normal_force()
        paddle_tcp = self._get_tcp_position(state_info)

        if self._pending_window_finalize is not None:
            return self._finalize_post_upward_window(state_info)

        if self.phase in ("post_first_upward_check", "post_second_upward_check"):
            return self._step_post_upward_window(state_info, cached_tcp=paddle_tcp)

        if self.phase == "goto_start":
            start_pos = self.path_waypoints[0]
            dist_to_start = float(np.linalg.norm(start_pos - paddle_tcp))
            if dist_to_start < self.goto_start_arrive_m:
                self.phase = "edge_loop"
                self.path_idx = 0
                self.phase_steps = 0
            return self._toward_target(paddle_tcp, start_pos, self.loop_max_delta_m)

        if self.phase == "edge_loop":
            self.phase_steps += 1
            target = self._lookahead_target_on_path(paddle_tcp, self.loop_max_delta_m)
            end_dist = float(np.linalg.norm(self.path_waypoints[-1] - paddle_tcp))
            at_end = self.path_idx >= len(self.path_waypoints) - 2 and end_dist < 0.05
            if at_end:
                self.phase = "upward_burst"
                self.phase_steps = self.burst_steps
                return self._upward_burst_action()
            return self._toward_target(paddle_tcp, target, self.loop_max_delta_m)

        if self.phase == "upward_burst":
            self.phase_steps -= 1
            if self.phase_steps <= 0:
                self.phase = "post_first_upward_check"
                self._start_post_upward_window(
                    kind="first",
                    state_info=state_info,
                    cached_tcp=paddle_tcp,
                )
                return self._step_post_upward_window(state_info, cached_tcp=paddle_tcp)
            return self._upward_burst_action()

        if self.phase == "wait_for_puck":
            hold_action = self._wait_for_puck_hold_action(state_info, cached_tcp=paddle_tcp)
            puck_x = float(state_info["pucks"][0]["position"][0])
            paddle_x = float(state_info["paddles"]["paddle_ego"]["position"][0])
            if puck_x >= paddle_x:
                self._restart_round(reason="wait_for_puck_puck_below_paddle")
                return np.zeros(2, dtype=np.float32)
            if self._puck_is_occluded(state_info):
                return hold_action
            puck_pos = self._get_puck_pos(state_info)
            dist = float(np.linalg.norm(paddle_tcp - puck_pos))
            puck_falling, _ = self._get_puck_motion_from_history(state_info)
            if dist <= self.puck_proximity_m and puck_falling:
                if not self._captured_second_hit_frame:
                    # Capture now (same loop iteration) before issuing strike action.
                    self._capture_second_hit_trigger_frame(dist_m=dist, puck_falling=puck_falling)
                    self._captured_second_hit_frame = True
                self.phase = "strike"
                self._wait_hold_y_tcp = None
                self.phase_steps = len(self._strike_actions)
                action_x = self._strike_actions[0]
                self.phase_steps -= 1
                return np.array([action_x, 0.0], dtype=np.float32)
            return hold_action

        if self.phase == "strike":
            if self.phase_steps <= 0:
                self.phase = "post_second_upward_check"
                self._start_post_upward_window(
                    kind="second",
                    state_info=state_info,
                    cached_tcp=paddle_tcp,
                )
                return self._step_post_upward_window(state_info, cached_tcp=paddle_tcp)
            idx = len(self._strike_actions) - self.phase_steps
            action_x = self._strike_actions[idx]
            self.phase_steps -= 1
            return np.array([action_x, 0.0], dtype=np.float32)

        self._set_terminal_reason("unknown_phase")
        return np.zeros(2, dtype=np.float32)


def build_model_if_requested(args, eval_env):
    if args.model is None:
        return None, False, None
    device = torch.device(args.device)
    state_dict = torch.load(args.model, map_location=device)
    model_obs_dim, model_action_dim = infer_policy_dims_from_state_dict(state_dict)
    model = LegacyMLPAgent(
        eval_env,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_size=args.agent_hidden_size,
    )
    model.load_state_dict(state_dict)
    model = model.to(device=device)
    model.eval()
    use_last_action = model_obs_dim > eval_env.single_observation_space.shape[0]
    last_action = torch.zeros((1, model_action_dim), dtype=torch.float32, device=device)
    return model, use_last_action, last_action


def show_reset_path_on_camera(fsm, simulator):
    """Grab a camera frame and display the planned reset path overlay."""
    from airhockey.sims.real.control_parameters import (
        visual_downscale_constant,
        offset_constants,
        homography_transform,
    )
    cap = getattr(simulator, "cap", None)
    if cap is None:
        print("[reset_path] No camera available; skipping path visualization.")
        return
    ret, raw_frame = cap.read()
    if not ret or raw_frame is None:
        print("[reset_path] Failed to read camera frame.")
        return
    frame, _ = homography_transform(raw_frame, get_save=True, rotate=False)
    fsm.draw_path_overlay(frame, offset_constants=offset_constants,
                          visual_downscale_constant=visual_downscale_constant)
    cv2.imshow("reset_path", frame)
    cv2.waitKey(1)


def build_obs_from_state(state: dict, obs_type: str):
    return get_observation_by_type(
        state,
        obs_type=obs_type,
        puck_history=state["pucks"][0]["history"],
        paddle_history=state["paddles"]["paddle_ego"]["history"],
    )


def get_env_state(eval_env: AirHockeyEnv) -> dict:
    state = getattr(eval_env, "current_state", None)
    if isinstance(state, dict) and "pucks" in state:
        return state
    return eval_env.simulator.get_current_state()


def refresh_state_obs(eval_env: AirHockeyEnv, obs_type: str) -> tuple[dict, np.ndarray]:
    state = get_env_state(eval_env)
    obs = build_obs_from_state(state, obs_type=obs_type)
    return state, obs


def compute_failure(
    state: dict,
    table_x_bot: float,
    bottom_margin: float,
    bottom_fail_count: int,
    occluded_fail_count: int,
    counters: dict,
) -> bool:
    puck = state["pucks"][0]
    puck_x = float(puck["position"][0])
    puck_occ = int(np.asarray(puck.get("occluded", 0)).reshape(-1)[0]) > 0
    if puck_x >= (table_x_bot - bottom_margin):
        counters["bottom"] += 1
    else:
        counters["bottom"] = 0
    if puck_occ:
        counters["occ"] += 1
    else:
        counters["occ"] = 0
    return counters["bottom"] >= bottom_fail_count or counters["occ"] >= occluded_fail_count


def hard_reset_with_pause(eval_env: AirHockeyEnv, reason: str, pause_s: float = 3.0) -> None:
    """Force robot to initial pose and wait before normal control resumes."""
    print(f"[fallback_reset] reason={reason} -> hard env reset")
    simulator = getattr(eval_env, "simulator", None)
    if simulator is not None:
        if hasattr(simulator, "wait_for_space_to_start"):
            try:
                simulator.wait_for_space_to_start = False
            except Exception:
                pass
        real_env = getattr(simulator, "air_hockey_env", None)
        if real_env is not None and hasattr(real_env, "wait_for_space_to_start"):
            try:
                real_env.wait_for_space_to_start = False
            except Exception:
                pass
    try:
        eval_env.reset(seed=None, write_traj=False)
    except TypeError:
        eval_env.reset(seed=None)
    print(f"[fallback_reset] pausing for {pause_s:.1f}s before resuming normal control")
    time.sleep(float(pause_s))


def enter_reset_mode(
    eval_env: AirHockeyEnv,
    rng: np.random.Generator,
    *,
    post_window_debug_log: bool,
    shared_success_threshold_proportion_from_bottom: float,
    reason: str,
    show_reset_path_overlay: bool,
    capture_second_hit_frame: bool,
    async_second_hit_write: bool,
    show_second_hit_window: bool,
    fsm_cls=ResetPolicyFSM,
) -> tuple[str, ResetPolicyFSM]:
    reset_fsm = fsm_cls(
        eval_env,
        rng,
        post_window_debug_log=post_window_debug_log,
        shared_success_threshold_proportion_from_bottom=(
            shared_success_threshold_proportion_from_bottom
        ),
        capture_second_hit_frame=capture_second_hit_frame,
        async_second_hit_write=async_second_hit_write,
        show_second_hit_window=show_second_hit_window,
    )
    reset_fsm._log_new_round_start(reason=reason)
    if show_reset_path_overlay:
        show_reset_path_on_camera(reset_fsm, eval_env.simulator)
    return "reset", reset_fsm


def restore_normal_runtime_state(
    use_last_action: bool,
    last_action_for_policy,
    reset_fsm: ResetPolicyFSM ,
    *,
    cooldown: int,
    startup_counter: int,
    clear_last_action: bool,
) -> tuple[str, None, dict, int, int]:
    if reset_fsm is not None:
        reset_fsm.close()
    if clear_last_action and use_last_action and last_action_for_policy is not None:
        last_action_for_policy.zero_()
    return "normal", None, {"bottom": 0, "occ": 0}, int(startup_counter), int(cooldown)


def resolve_timing_optimization_flags(args) -> dict:
    disable_reset_path_overlay = bool(args.disable_reset_path_overlay)
    disable_second_hit_capture = bool(args.disable_second_hit_capture)
    disable_second_hit_preview_window = bool(args.disable_second_hit_preview_window)
    async_second_hit_write = bool(args.async_second_hit_write)

    if args.timing_optimized_mode:
        disable_reset_path_overlay = True
        disable_second_hit_capture = True
        disable_second_hit_preview_window = True
        async_second_hit_write = True

    return {
        "show_reset_path_overlay": not disable_reset_path_overlay,
        "capture_second_hit_frame": not disable_second_hit_capture,
        "show_second_hit_window": not disable_second_hit_preview_window,
        "async_second_hit_write": async_second_hit_write,
    }


def handle_normal_mode_step(
    obs: np.ndarray,
    model,
    use_last_action: bool,
    last_action_for_policy,
    normal_action: np.ndarray,
    startup_counter: int,
    args,
) -> tuple[np.ndarray, int, torch.Tensor ]:
    if model is None:
        action = np.copy(normal_action)
    else:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        policy_obs = obs_t
        if use_last_action:
            policy_obs = torch.cat([policy_obs, last_action_for_policy], dim=-1)
        with torch.inference_mode():
            action = model(policy_obs).detach().cpu().numpy().squeeze()
        action = action / float(model.action_scale.item())
        action = np.clip(action, -1.0, 1.0)
        if use_last_action:
            last_action_for_policy = torch.as_tensor(
                action,
                dtype=torch.float32,
                device=args.device,
            ).unsqueeze(0)

    if args.policy_stand_still:
        action = np.zeros(2, dtype=np.float32)
        if use_last_action and last_action_for_policy is not None:
            last_action_for_policy.zero_()

    if startup_counter < int(args.startup_hold_steps):
        action = np.zeros(2, dtype=np.float32)
        startup_counter += 1

    return action, startup_counter, last_action_for_policy


def handle_reset_mode_step(
    state: dict,
    reset_fsm: ResetPolicyFSM,
    eval_env: AirHockeyEnv,
    use_last_action: bool,
    last_action_for_policy,
    fail_counters: dict,
    startup_counter: int,
    reset_cooldown: int,
) -> tuple[np.ndarray, str, ResetPolicyFSM , dict, int, int, bool]:
    action = reset_fsm.step(state)
    if not reset_fsm.done:
        return (
            action,
            "reset",
            reset_fsm,
            fail_counters,
            startup_counter,
            reset_cooldown,
            False,
        )

    done_reason = getattr(reset_fsm, "done_reason", "unknown")
    if done_reason == "hard_reset_required":
        print("[reset] hard reset requested by FSM after max stage-2 retries.")
        hard_reset_with_pause(
            eval_env=eval_env,
            reason="reset_fsm_stage2_max_retries",
            pause_s=3.0,
        )
        mode, reset_fsm, fail_counters, startup_counter, reset_cooldown = (
            restore_normal_runtime_state(
                use_last_action=use_last_action,
                last_action_for_policy=last_action_for_policy,
                reset_fsm=reset_fsm,
                cooldown=40,
                startup_counter=0,
                clear_last_action=True,
            )
        )
        return (
            action,
            mode,
            reset_fsm,
            fail_counters,
            startup_counter,
            reset_cooldown,
            True,
        )

    puck_x = float(state["pucks"][0]["position"][0])
    paddle_x = float(state["paddles"]["paddle_ego"]["position"][0])
    success_stage = getattr(reset_fsm, "last_success_stage", "unknown")
    success_motion = getattr(reset_fsm, "last_success_motion", "unknown")
    if puck_x < paddle_x:
        print(
            f"[reset] SUCCESS: stage={success_stage} motion_estimate={success_motion} "
            f"puck_x={puck_x:.3f} < paddle_x={paddle_x:.3f}"
        )
    else:
        print(
            f"[reset] cycle done stage={success_stage} motion_estimate={success_motion} "
            f"but puck still below paddle (puck_x={puck_x:.3f} >= paddle_x={paddle_x:.3f})"
        )
    mode, reset_fsm, fail_counters, startup_counter, reset_cooldown = (
        restore_normal_runtime_state(
            use_last_action=use_last_action,
            last_action_for_policy=last_action_for_policy,
            reset_fsm=reset_fsm,
            cooldown=40,
            startup_counter=startup_counter,
            clear_last_action=False,
        )
    )
    return (
        action,
        mode,
        reset_fsm,
        fail_counters,
        startup_counter,
        reset_cooldown,
        False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-robot rollout with failure-triggered reset policy.")
    parser.add_argument("--config-path", type=str, default="configs/real_configs/rollout_config.yaml")
    parser.add_argument("--save-path", type=str, default=None, help="Override trajectory save path.")
    parser.add_argument("--model", type=str, default=None, help="Optional juggling model path for normal mode.")
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--agent-hidden-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normal-action-x", type=float, default=0.0, help="Fallback normal-mode action x.")
    parser.add_argument("--normal-action-y", type=float, default=0.0, help="Fallback normal-mode action y.")
    parser.add_argument("--bottom-margin", type=float, default=0.25, help="Fail when puck_x > table_x_bot - margin.")
    parser.add_argument("--bottom-fail-count", type=int, default=2)
    parser.add_argument("--occluded-fail-count", type=int, default=6)
    parser.add_argument("--startup-hold-steps", type=int, default=10, help="Hold normal action at zero for startup.")
    parser.add_argument(
        "--policy-stand-still",
        action="store_true",
        help="Force zero action during normal/policy mode to visualize reset-to-policy takeover.",
    )
    parser.add_argument(
        "--post-window-debug-log",
        action="store_true",
        help="Enable detailed post-window target-tracking logs during reset.",
    )
    parser.add_argument(
        "--shared-success-threshold-proportion-from-bottom",
        type=float,
        default=0.5,
        help="Bottom->top table proportion for shared reset success threshold (default: 0.4).",
    )
    parser.add_argument(
        "--disable-reset-path-overlay",
        action="store_true",
        help="Disable reset-path camera overlay display (timing optimization).",
    )
    parser.add_argument(
        "--disable-second-hit-capture",
        action="store_true",
        help="Disable second-hit trigger frame capture (timing optimization).",
    )
    parser.add_argument(
        "--disable-second-hit-preview-window",
        action="store_true",
        help="Disable live OpenCV preview window for second-hit capture.",
    )
    parser.add_argument(
        "--async-second-hit-write",
        action="store_true",
        help="Write second-hit capture files asynchronously to reduce control-step blocking.",
    )
    parser.add_argument(
        "--timing-optimized-mode",
        action="store_true",
        help="Enable aggressive timing optimizations (disables capture/overlays, async write).",
    )
    parser.add_argument(
        "--timing-log-every",
        type=int,
        default=0,
        help="Print average control-loop time every N steps (0 disables).",
    )
    parser.add_argument(
        "--use-hybrid-fsm",
        action="store_true",
        help="Use ResetPolicyHybridFSM (programmatic edge-loop + first burst, then frozen juggle policy for the second hit) instead of the legacy ResetPolicyFSM. For isolated testing of the hybrid reset path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal-output mode for watching the FSM: silences simulator [control_debug] prints and the FSM normal-force log, suppresses per-step verbose loop output, and prints one line per (mode, phase) transition instead. Overrides --verbose.",
    )
    parser.add_argument(
        "--force-end-side",
        choices=("left", "right", "random"),
        default="random",
        help="Force the edge-loop FSM such that the paddle ENDS at this side (sweeping from the opposite side toward this one). Useful for reproducing side-specific bugs. Default 'random' uses the FSM's normal coin-flip.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous reset testing: immediately start a new reset cycle after each FSM completion (success or hard_reset_required), instead of returning to normal mode and risking the episode_done hard-reset loop. Pauses 1s between cycles.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.use_hybrid_fsm:
        from scripts.real.rollout_reset_policy_hybrid import ResetPolicyHybridFSM
        reset_fsm_cls = ResetPolicyHybridFSM
    else:
        reset_fsm_cls = ResetPolicyFSM
    if not (0.0 <= float(args.shared_success_threshold_proportion_from_bottom) <= 1.0):
        parser.error("--shared-success-threshold-proportion-from-bottom must be in [0.0, 1.0].")
    if int(args.timing_log_every) < 0:
        parser.error("--timing-log-every must be >= 0.")

    timing_flags = resolve_timing_optimization_flags(args)

    params = load_air_hockey_params(args.config_path, save_path_override=args.save_path)
    sim_params = params.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
        if args.quiet:
            # Silence the simulator's per-step [control_debug] firehose so
            # the FSM phase transitions are visible. Overrides whatever the
            # YAML config sets.
            sim_params["debug_control"] = False
    if args.quiet and args.verbose:
        print("[quiet_mode] --quiet supersedes --verbose; suppressing per-step output.")
        args.verbose = False
    params["max_timesteps"] = max(400, int(params.get("max_timesteps", 300)))
    eval_env = AirHockeyEnv(params)
    model, use_last_action, last_action_for_policy = build_model_if_requested(args, eval_env)
    normal_action = np.array([args.normal_action_x, args.normal_action_y], dtype=np.float32)

    state = get_env_state(eval_env)
    obs_type = params.get("obs_type", "history")
    obs = build_obs_from_state(state, obs_type=obs_type)
    rng = np.random.default_rng(int(params.get("seed", 0)))
    if args.force_end_side != "random":
        # Wrap the rng so .random() always forces the desired end-side in
        # ResetPolicyFSM._build_edge_loop_path. The FSM's `start_side`
        # names where the paddle BEGINS the sweep; the paddle ENDS at the
        # opposite side. So end='right' wants start_side='left' (rng.random()
        # < 0.5 → return 0.0); end='left' wants start_side='right' (return
        # 0.999). All non-random rng methods pass through unchanged so
        # rng.uniform(...) for strike_mag / puck_proximity_m keep stochasticity.
        class _ForcedSideRng:
            def __init__(self, base, forced_value):
                object.__setattr__(self, "_base", base)
                object.__setattr__(self, "_forced_random", forced_value)
            def random(self):
                return self._forced_random
            def __getattr__(self, name):
                return getattr(self._base, name)
        forced_value = 0.0 if args.force_end_side == "right" else 0.999
        forced_start = "left" if args.force_end_side == "right" else "right"
        rng = _ForcedSideRng(rng, forced_value)
        print(
            f"[force_end_side] every reset will end at side='{args.force_end_side}' "
            f"(FSM start_side='{forced_start}')."
        )
    reset_fsm = None
    mode = "normal"
    fail_counters = {"bottom": 0, "occ": 0}
    startup_counter = 0
    reset_cooldown = 0

    print("Running real rollout with reset policy. Keys: y=save+reset, q=reset, r=force-reset, x=exit")
    if args.verbose:
        print(
            "[timing_opts] "
            f"show_reset_path_overlay={int(timing_flags['show_reset_path_overlay'])} "
            f"capture_second_hit_frame={int(timing_flags['capture_second_hit_frame'])} "
            f"show_second_hit_window={int(timing_flags['show_second_hit_window'])} "
            f"async_second_hit_write={int(timing_flags['async_second_hit_write'])}"
        )
    step_counter = 0
    timing_accum_s = 0.0
    timing_count = 0
    prev_quiet_state = ("normal", "-") if args.quiet else None
    if args.quiet:
        print("[quiet_mode] tracking FSM (mode, phase) transitions; one line per change.")
    with NonBlockingConsole() as nbc:
        while True:
            loop_t0 = time.perf_counter()
            fail_now = compute_failure(
                state=state,
                table_x_bot=float(eval_env.table_x_bot),
                bottom_margin=float(args.bottom_margin),
                bottom_fail_count=int(args.bottom_fail_count),
                occluded_fail_count=int(args.occluded_fail_count),
                counters=fail_counters,
            )
            if args.verbose and step_counter % 60 == 0:
                puck_diag = state["pucks"][0]
                paddle_diag = state["paddles"]["paddle_ego"]
                puck_x_diag = float(puck_diag["position"][0])
                paddle_x_diag = float(paddle_diag["position"][0])
                puck_occ_diag = int(np.asarray(puck_diag.get("occluded", 0)).reshape(-1)[0])
                center_offset_diag = float(
                    getattr(
                        getattr(eval_env, "simulator", None),
                        "center_offset_constant",
                        getattr(eval_env, "center_offset_constant", 0.0),
                    )
                )
                bottom_threshold_diag = float(eval_env.table_x_bot) - float(args.bottom_margin)
                puck_tcp_x_diag = puck_x_diag - center_offset_diag
                print(
                    f"[diag] step={step_counter} mode={mode} "
                    f"[env/state frame] puck_x={puck_x_diag:+.4f} paddle_x={paddle_x_diag:+.4f} "
                    f"bottom_threshold_x={bottom_threshold_diag:+.4f} "
                    f"table_x_top={float(eval_env.table_x_top):+.4f} table_x_bot={float(eval_env.table_x_bot):+.4f} "
                    f"[tcp-aligned puck frame] puck_x={puck_tcp_x_diag:+.4f} "
                    f"(env_puck_x - center_offset={center_offset_diag:+.4f}) "
                    f"puck_occ={puck_occ_diag} "
                    f"bottom_count={fail_counters['bottom']} occ_count={fail_counters['occ']} "
                    f"trigger_now={int(puck_x_diag >= bottom_threshold_diag)}"
                )
            step_counter += 1
            if reset_cooldown > 0:
                reset_cooldown -= 1
            if mode == "normal" and fail_now and reset_cooldown <= 0:
                mode, reset_fsm = enter_reset_mode(
                    eval_env,
                    rng,
                    post_window_debug_log=args.post_window_debug_log,
                    shared_success_threshold_proportion_from_bottom=(
                        args.shared_success_threshold_proportion_from_bottom
                    ),
                    reason="reset_trigger_failure_condition",
                    show_reset_path_overlay=timing_flags["show_reset_path_overlay"],
                    capture_second_hit_frame=timing_flags["capture_second_hit_frame"],
                    async_second_hit_write=timing_flags["async_second_hit_write"],
                    show_second_hit_window=timing_flags["show_second_hit_window"],
                    fsm_cls=reset_fsm_cls,
                )
                if args.quiet and reset_fsm is not None:
                    reset_fsm.force_log_interval_steps = 0

            mode_before_handle = mode
            if mode == "reset":
                (
                    action,
                    mode,
                    reset_fsm,
                    fail_counters,
                    startup_counter,
                    reset_cooldown,
                    skip_env_step,
                ) = handle_reset_mode_step(
                    state=state,
                    reset_fsm=reset_fsm,
                    eval_env=eval_env,
                    use_last_action=use_last_action,
                    last_action_for_policy=last_action_for_policy,
                    fail_counters=fail_counters,
                    startup_counter=startup_counter,
                    reset_cooldown=reset_cooldown,
                )
                if skip_env_step:
                    state, obs = refresh_state_obs(eval_env, obs_type=obs_type)
                    continue
            else:
                action, startup_counter, last_action_for_policy = handle_normal_mode_step(
                    obs=obs,
                    model=model,
                    use_last_action=use_last_action,
                    last_action_for_policy=last_action_for_policy,
                    normal_action=normal_action,
                    startup_counter=startup_counter,
                    args=args,
                )

            obs, reward, terminated, truncated, info = eval_env.step(action)
            state = get_env_state(eval_env)
            if args.verbose:
                puck = state["pucks"][0]
                print(
                    f"mode={mode} phase={(reset_fsm.phase if reset_fsm is not None else '-'):<18} "
                    f"action=({float(action[0]):+.3f},{float(action[1]):+.3f}) "
                    f"puck=({float(puck['position'][0]):+.3f},{float(puck['position'][1]):+.3f}) "
                    f"occ={int(np.asarray(puck.get('occluded', 0)).reshape(-1)[0])} "
                    f"rew={float(reward):+.3f} done={terminated or truncated}"
                )
            if args.quiet:
                # One-line print on (mode, phase) change: shows the FSM
                # state machine moving through phases without per-step noise.
                curr_phase = reset_fsm.phase if reset_fsm is not None else "-"
                curr_state = (mode, curr_phase)
                if curr_state != prev_quiet_state:
                    puck = state["pucks"][0]
                    paddle = state["paddles"]["paddle_ego"]
                    prev_label = f"{prev_quiet_state[0]}/{prev_quiet_state[1]}"
                    curr_label = f"{curr_state[0]}/{curr_state[1]}"
                    print(
                        f"[phase] step={step_counter:5d} {prev_label:>32s} -> {curr_label:<32s} "
                        f"puck=({float(puck['position'][0]):+.3f},{float(puck['position'][1]):+.3f}) "
                        f"paddle_x={float(paddle['position'][0]):+.3f} "
                        f"occ={int(np.asarray(puck.get('occluded', 0)).reshape(-1)[0])}"
                    )
                    prev_quiet_state = curr_state

            fsm_just_completed = (mode_before_handle == "reset" and mode == "normal")
            if args.continuous and fsm_just_completed:
                # FSM just finished (success or hard_reset_required). Skip the
                # episode_done hard-reset path entirely and immediately stage
                # another reset cycle. This is what the test harness exists for.
                print("[continuous] FSM cycle done; pausing 1s then starting next cycle.")
                time.sleep(1.0)
                state, obs = refresh_state_obs(eval_env, obs_type=obs_type)
                mode, reset_fsm = enter_reset_mode(
                    eval_env,
                    rng,
                    post_window_debug_log=args.post_window_debug_log,
                    shared_success_threshold_proportion_from_bottom=(
                        args.shared_success_threshold_proportion_from_bottom
                    ),
                    reason="continuous_test_cycle",
                    show_reset_path_overlay=timing_flags["show_reset_path_overlay"],
                    capture_second_hit_frame=timing_flags["capture_second_hit_frame"],
                    async_second_hit_write=timing_flags["async_second_hit_write"],
                    show_second_hit_window=timing_flags["show_second_hit_window"],
                    fsm_cls=reset_fsm_cls,
                )
                if args.quiet and reset_fsm is not None:
                    reset_fsm.force_log_interval_steps = 0
                fail_counters = {"bottom": 0, "occ": 0}
                continue

            if mode == "normal" and bool(terminated or truncated):
                hard_reset_with_pause(
                    eval_env=eval_env,
                    reason="episode_done_without_reset_activation",
                    pause_s=3.0,
                )
                mode, reset_fsm, fail_counters, startup_counter, reset_cooldown = (
                    restore_normal_runtime_state(
                        use_last_action=use_last_action,
                        last_action_for_policy=last_action_for_policy,
                        reset_fsm=reset_fsm,
                        cooldown=0,
                        startup_counter=0,
                        clear_last_action=True,
                    )
                )
                state, obs = refresh_state_obs(eval_env, obs_type=obs_type)
                continue

            key = nbc.get_data()
            if key == "y":
                print("Saving trajectory and resetting...")
                eval_env.reset(seed=None, write_traj=True)
                mode, reset_fsm, fail_counters, startup_counter, reset_cooldown = (
                    restore_normal_runtime_state(
                        use_last_action=use_last_action,
                        last_action_for_policy=last_action_for_policy,
                        reset_fsm=reset_fsm,
                        cooldown=reset_cooldown,
                        startup_counter=0,
                        clear_last_action=True,
                    )
                )
                state, obs = refresh_state_obs(eval_env, obs_type=obs_type)
            elif key == "q":
                print("Resetting without saving...")
                eval_env.reset(seed=None, write_traj=False)
                mode, reset_fsm, fail_counters, startup_counter, reset_cooldown = (
                    restore_normal_runtime_state(
                        use_last_action=use_last_action,
                        last_action_for_policy=last_action_for_policy,
                        reset_fsm=reset_fsm,
                        cooldown=reset_cooldown,
                        startup_counter=0,
                        clear_last_action=True,
                    )
                )
                state, obs = refresh_state_obs(eval_env, obs_type=obs_type)
            elif key == "r":
                print("Force-triggering reset mode...")
                if reset_fsm is not None:
                    reset_fsm.close()
                mode, reset_fsm = enter_reset_mode(
                    eval_env,
                    rng,
                    post_window_debug_log=args.post_window_debug_log,
                    shared_success_threshold_proportion_from_bottom=(
                        args.shared_success_threshold_proportion_from_bottom
                    ),
                    reason="manual_force_reset",
                    show_reset_path_overlay=timing_flags["show_reset_path_overlay"],
                    capture_second_hit_frame=timing_flags["capture_second_hit_frame"],
                    async_second_hit_write=timing_flags["async_second_hit_write"],
                    show_second_hit_window=timing_flags["show_second_hit_window"],
                    fsm_cls=reset_fsm_cls,
                )
                if args.quiet and reset_fsm is not None:
                    reset_fsm.force_log_interval_steps = 0
                fail_counters = {"bottom": 0, "occ": 0}
            elif key == "x":
                print("Exiting...")
                if reset_fsm is not None:
                    reset_fsm.close()
                break

            if int(args.timing_log_every) > 0:
                timing_accum_s += time.perf_counter() - loop_t0
                timing_count += 1
                if timing_count >= int(args.timing_log_every):
                    avg_ms = 1000.0 * timing_accum_s / max(1, timing_count)
                    print(
                        f"[timing] avg_loop_ms={avg_ms:.3f} "
                        f"mode={mode} steps={timing_count} "
                        f"opt={int(args.timing_optimized_mode)}"
                    )
                    timing_accum_s = 0.0
                    timing_count = 0
