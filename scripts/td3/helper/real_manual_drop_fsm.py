"""Manual puck reset: park the paddle, let the operator start the puck by hand.

Two modes:

* ``drop``  -- the operator places the puck at the top of the table and drops it.
* ``flick`` -- the operator flicks the puck up from the bottom of the table; the
  policy starts once the puck crosses ``flick_line_from_robot`` going up.

An alternative to the automatic puck reset (``ResetPolicyFSM``) for evals where
a person restarts every episode by hand. Before each episode the paddle drives
back to the robot's start pose and holds still; the operator places the puck
at the top of the table and drops it, and the policy takes over.

Duck-types the surface ``run_reset_fsm`` consumes (same as
``PaddleRepositionFSM``): constructor ``(env, rng, **kwargs)``,
``step(state_info) -> action``, ``done`` / ``done_reason`` / ``phase`` /
``total_steps`` / ``start_side`` and ``close()``.

Phases:

  1. ``goto_start``       -- drive toward the start pose, at most ``max_step_m``
                             per step.
  2. ``settle``           -- hold still until the paddle has been within
                             ``arrive_m`` of the start pose for ``settle_steps``
                             steps.
  3. ``wait_for_puck_placement`` -- hold still (terminal prompt) until the puck is seen,
                             not occluded, above the top line for
                             ``detect_steps`` consecutive steps.
  4. ``wait_for_release`` -- hold still until the puck has moved ``release_m``
                             toward the robot from where it was held (it was
                             dropped), then hand over. Skipped with
                             ``start_on_detect=True``.

In ``flick`` mode phases 3-4 are replaced by

  3. ``wait_for_flick``  -- hold still (terminal prompt) until the puck, having
                             been seen below the flick line, is seen above it
                             while moving up the table; then hand over.

Handing over on release, not on detection, keeps a puck that is held still
from tripping ``terminate_on_puck_stop`` (20 still steps) as soon as the
policy starts.

Frames: everything is in the env (table) frame, like ``state_info``: x points
from the far wall (``env.table_x_top``) toward the robot (``env.table_x_bot``).
The start pose is the simulator's ``reset_pose`` (robot TCP), converted with
the simulator's own TCP -> observation mapping.
"""
from __future__ import annotations

import numpy as np


class ManualPuckDropFSM:
    """Drive the paddle to the start pose, wait for the operator to drop the puck."""

    def __init__(
        self,
        env,
        rng: np.random.Generator,
        *,
        mode: str = "drop",
        flick_line_from_robot: float = 0.34,
        top_line_from_robot: float = 0.75,
        detect_steps: int = 5,
        release_m: float = 0.02,
        start_on_detect: bool = False,
        max_step_m: float = 0.05,
        arrive_m: float = 0.02,
        settle_steps: int = 5,
    ) -> None:
        if mode not in ("drop", "flick"):
            raise ValueError(f"mode must be 'drop' or 'flick', got {mode!r}")
        self.mode = mode
        self.env = env
        self.rng = rng
        self.detect_steps = max(1, int(detect_steps))
        self.release_m = float(release_m)
        self.start_on_detect = bool(start_on_detect)
        self.max_step_m = float(max_step_m)
        self.arrive_m = float(arrive_m)
        self.settle_steps = max(1, int(settle_steps))

        simulator = getattr(env, "simulator", None)
        self._move_lims = np.asarray(
            getattr(simulator, "move_lims", (0.26, 0.12)), dtype=np.float32
        ).reshape(-1)[:2]

        # "Top of the table": table x at or beyond this line (toward the far wall).
        table_x_bot = float(getattr(env, "table_x_bot", 0.9652))
        table_x_top = float(getattr(env, "table_x_top", -0.9652))
        self.top_line_x = table_x_bot - float(top_line_from_robot) * (table_x_bot - table_x_top)
        # Flick mode: the policy starts once the puck crosses this line going up.
        self.flick_line_x = table_x_bot - float(flick_line_from_robot) * (table_x_bot - table_x_top)

        self.target_xy = self._start_pose_xy()

        self.phase = "goto_start"
        self.total_steps = 0
        self.done = False
        self.done_reason = "in_progress"
        self._settle_count = 0
        self._detect_count = 0
        self._held_xy = []
        self._held_x = None
        self._seen_below_flick_line = False
        self._last_visible_x = None
        line = self.top_line_x if mode == "drop" else self.flick_line_x
        self.start_side = (
            f"manual_{mode} start=({self.target_xy[0]:+.3f},{self.target_xy[1]:+.3f}) "
            f"line_x={line:+.3f}"
        )

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _start_pose_xy(self) -> np.ndarray:
        """Robot start pose (``simulator.reset_pose``) in the observation frame."""
        simulator = self.env.simulator
        reset_pose = getattr(simulator, "reset_pose", None)
        if reset_pose is not None:
            tcp_xy = np.asarray(reset_pose[0][:2], dtype=np.float64)
            to_obs = getattr(simulator, "_paddle_observation_xy_from_pose", None)
            if callable(to_obs):
                return np.asarray(to_obs(tcp_xy), dtype=np.float32).reshape(-1)[:2]
            offset = np.array([
                float(getattr(simulator, "center_offset_constant", 0.0))
                + float(getattr(simulator, "paddle_additional_x_offset", 0.0)),
                float(getattr(simulator, "center_offset_constant_y", 0.0))
                + float(getattr(simulator, "paddle_additional_y_offset", 0.0)),
            ])
            return (tcp_xy + offset).astype(np.float32)
        state = simulator.get_current_state()
        return np.asarray(state["paddles"]["paddle_ego"]["position"], dtype=np.float32).reshape(-1)[:2]

    @staticmethod
    def _paddle_xy(state_info: dict) -> np.ndarray:
        return np.asarray(
            state_info["paddles"]["paddle_ego"]["position"], dtype=np.float32
        ).reshape(-1)[:2]

    @staticmethod
    def _puck(state_info: dict):
        """(xy, visible) for the first puck."""
        puck = state_info["pucks"][0]
        xy = np.asarray(puck["position"], dtype=np.float64).reshape(-1)[:2]
        occluded = int(np.asarray(puck.get("occluded", 0)).reshape(-1)[0]) > 0
        return xy, (not occluded) and bool(np.all(np.isfinite(xy)))

    def _toward_target(self, paddle_xy: np.ndarray) -> np.ndarray:
        delta = self.target_xy - paddle_xy
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float32)
        delta = delta * min(1.0, self.max_step_m / norm)
        action = delta / np.maximum(self._move_lims, 1e-6)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _hand_over(self, reason: str) -> np.ndarray:
        self.done = True
        self.done_reason = "success"
        print(f"[manual_{self.mode}] {reason}; starting the policy (total_steps={self.total_steps})")
        return np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Step.
    # ------------------------------------------------------------------

    def step(self, state_info: dict) -> np.ndarray:
        if self.done:
            return np.zeros(2, dtype=np.float32)
        self.total_steps += 1
        paddle_xy = self._paddle_xy(state_info)

        if self.phase == "goto_start":
            if float(np.linalg.norm(self.target_xy - paddle_xy)) <= self.arrive_m:
                self.phase = "settle"
                self._settle_count = 0
                return np.zeros(2, dtype=np.float32)
            return self._toward_target(paddle_xy)

        if self.phase == "settle":
            if float(np.linalg.norm(self.target_xy - paddle_xy)) > self.arrive_m:
                self.phase = "goto_start"
                return self._toward_target(paddle_xy)
            self._settle_count += 1
            if self._settle_count >= self.settle_steps and self.mode == "flick":
                self.phase = "wait_for_flick"
                self._seen_below_flick_line = False
                self._last_visible_x = None
                print(
                    "\n[manual_flick] Paddle at the start pose. Flick the puck up the table; the "
                    f"policy starts when it crosses table x = {self.flick_line_x:+.3f} m going up."
                )
            elif self._settle_count >= self.settle_steps:
                self.phase = "wait_for_puck_placement"
                self._detect_count = 0
                self._held_xy = []
                print(
                    "\n[manual_drop] Paddle at the start pose. Place the puck at the top of the "
                    f"table (table x <= {self.top_line_x:+.3f} m, i.e. the far part of the table)."
                )
            return np.zeros(2, dtype=np.float32)

        puck_xy, visible = self._puck(state_info)

        if self.phase == "wait_for_flick":
            if visible:
                x = float(puck_xy[0])
                moving_up = self._last_visible_x is not None and x < self._last_visible_x
                if x > self.flick_line_x:
                    self._seen_below_flick_line = True
                elif self._seen_below_flick_line and moving_up:
                    return self._hand_over(f"puck crossed the line going up (x={x:+.3f})")
                self._last_visible_x = x
            return np.zeros(2, dtype=np.float32)

        if self.phase == "wait_for_puck_placement":
            if visible and puck_xy[0] <= self.top_line_x:
                self._detect_count += 1
                self._held_xy.append(puck_xy.copy())
            else:
                self._detect_count = 0
                self._held_xy = []
            if self._detect_count >= self.detect_steps:
                if self.start_on_detect:
                    return self._hand_over("puck detected at the top")
                self._held_x = float(np.median([p[0] for p in self._held_xy]))
                self.phase = "wait_for_release"
                print("[manual_drop] Puck detected. Drop it when ready; the policy starts as it moves.")
            return np.zeros(2, dtype=np.float32)

        if self.phase == "wait_for_release":
            if visible:
                if puck_xy[0] - self._held_x >= self.release_m:
                    return self._hand_over(
                        f"puck released (moved {100 * (puck_xy[0] - self._held_x):.1f} cm toward the robot)"
                    )
                # Still held: measure the drop from the highest point it was held at.
                self._held_x = float(min(self._held_x, puck_xy[0]))
            return np.zeros(2, dtype=np.float32)

        self.done = True
        self.done_reason = "unknown_phase"
        return np.zeros(2, dtype=np.float32)

    def close(self) -> None:
        pass
