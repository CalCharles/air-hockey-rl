"""Paddle-only reset controller for tasks without a puck.

The juggle-family eval uses ``ResetPolicyFSM`` (``scripts/real/
rollout_reset_policy_real.py``) to sweep the puck off the bottom edge and
strike it back up the table. Paddle-only tasks (``paddle_reach_position``,
``paddle_reach_position_velocity``) have no puck to recover; all a reset has
to do is move the paddle to a fresh start pose and let it come to rest so
the policy's first observation is a stationary, non-occluded paddle — the
same start state the Box2D task produces via ``random_paddle_spawn``.

``PaddleRepositionFSM`` duck-types the surface ``run_reset_fsm`` consumes
from ``ResetPolicyFSM``:

  * constructor ``(env, rng)`` (extra kwargs optional),
  * ``step(state_info) -> action`` (normalised [-1, 1] paddle displacement),
  * ``done`` / ``done_reason`` / ``phase`` / ``total_steps`` / ``start_side``,
  * ``close()``.

Phases:

  1. ``goto_start`` — move toward the sampled start pose, at most
     ``max_step_m`` per step (mirrors ``ResetPolicyFSM._toward_target``).
  2. ``settle``     — hold zero action until the paddle has been within
     ``arrive_m`` of the target for ``settle_steps`` consecutive steps and
     its reported speed is below ``settle_speed_mps``.

``done_reason`` is ``"success"`` on a clean settle and
``"hard_reset_required"`` when ``max_total_steps`` elapse first (the
paddle could not reach / hold the target — the runner then falls back to
the physical ``env.reset()`` path, same as the puck FSM's give-up case).

Frames: the target and the paddle position are both taken in the env
(observation) frame — ``state_info["paddles"]["paddle_ego"]["position"]``
and ``env.get_paddle_workspace_bounds`` agree on it — so no TCP-offset
bookkeeping is needed here.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class PaddleRepositionFSM:
    """Drive the paddle to a start pose and let it settle."""

    def __init__(
        self,
        env,
        rng: np.random.Generator,
        *,
        target_xy: Optional[Tuple[float, float]] = None,
        spawn_margin_m: float = 0.02,
        max_step_m: float = 0.10,
        arrive_m: float = 0.02,
        settle_steps: int = 5,
        settle_speed_mps: float = 0.05,
        max_total_steps: int = 200,
    ) -> None:
        self.env = env
        self.rng = rng
        self.spawn_margin_m = float(spawn_margin_m)
        self.max_step_m = float(max_step_m)
        self.arrive_m = float(arrive_m)
        self.settle_steps = max(1, int(settle_steps))
        self.settle_speed_mps = float(settle_speed_mps)
        self.max_total_steps = max(1, int(max_total_steps))

        simulator = getattr(env, "simulator", None)
        self._move_lims = np.asarray(
            getattr(simulator, "move_lims", (0.26, 0.12)), dtype=np.float32
        ).reshape(-1)[:2]

        self.target_xy = (
            np.asarray(target_xy, dtype=np.float32).reshape(2)
            if target_xy is not None
            else self._sample_target()
        )

        self.phase = "goto_start"
        self.phase_steps = 0
        self.total_steps = 0
        self.done = False
        self.done_reason = "in_progress"
        self._settle_count = 0
        # ``run_reset_fsm`` logs ``fsm.start_side`` for the puck FSM; expose a
        # descriptive stand-in so the same log line stays informative.
        self.start_side = f"target=({self.target_xy[0]:+.3f},{self.target_xy[1]:+.3f})"

    # ------------------------------------------------------------------
    # Target sampling.
    # ------------------------------------------------------------------

    def _sample_target(self) -> np.ndarray:
        """Uniform start pose in the reachable workspace when the task uses
        ``random_paddle_spawn``; otherwise the task's fixed default spawn."""
        env = self.env
        if bool(getattr(env, "random_paddle_spawn", False)) and hasattr(
            env, "get_paddle_workspace_bounds"
        ):
            # y first: the corner wedge makes x_max depend on y (same order
            # as ``AirHockeyBaseEnv.sample_paddle_spawn_in_workspace``), but
            # drawn from the reset RNG so the start pose stream is decoupled
            # from the env RNG that samples the goal.
            _, _, y_lo, y_hi = env.get_paddle_workspace_bounds(margin=self.spawn_margin_m)
            y = float(self.rng.uniform(y_lo, y_hi)) if y_hi > y_lo else float(y_lo)
            x_lo, x_hi, _, _ = env.get_paddle_workspace_bounds(margin=self.spawn_margin_m, y=y)
            x = float(self.rng.uniform(x_lo, x_hi)) if x_hi > x_lo else float(x_lo)
            return np.array([x, y], dtype=np.float32)
        if hasattr(env, "get_paddle_configuration"):
            pos, _ = env.get_paddle_configuration("paddle_ego")
            return np.asarray(pos, dtype=np.float32).reshape(2)
        state = env.simulator.get_current_state()
        return np.asarray(
            state["paddles"]["paddle_ego"]["position"], dtype=np.float32
        ).reshape(-1)[:2]

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _paddle_xy(state_info: dict) -> np.ndarray:
        return np.asarray(
            state_info["paddles"]["paddle_ego"]["position"], dtype=np.float32
        ).reshape(-1)[:2]

    @staticmethod
    def _paddle_speed(state_info: dict) -> float:
        vel = state_info["paddles"]["paddle_ego"].get("velocity", (0.0, 0.0))
        vel = np.asarray(vel, dtype=np.float32).reshape(-1)[:2]
        if not np.all(np.isfinite(vel)):
            return 0.0
        return float(np.linalg.norm(vel))

    def _toward_target(self, paddle_xy: np.ndarray) -> np.ndarray:
        delta = self.target_xy - paddle_xy
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float32)
        scaled = delta * min(1.0, self.max_step_m / norm)
        action = scaled / np.maximum(self._move_lims, 1e-6)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _finish(self, reason: str) -> None:
        self.done = True
        self.done_reason = str(reason)

    def close(self) -> None:  # parity with ResetPolicyFSM
        return None

    # ------------------------------------------------------------------
    # Step.
    # ------------------------------------------------------------------

    def step(self, state_info: dict) -> np.ndarray:
        if self.done:
            return np.zeros(2, dtype=np.float32)
        self.total_steps += 1
        paddle_xy = self._paddle_xy(state_info)
        dist = float(np.linalg.norm(self.target_xy - paddle_xy))

        if self.total_steps > self.max_total_steps:
            print(
                "[paddle_reposition] timeout after "
                f"{self.total_steps - 1} steps (phase={self.phase} dist={dist:.3f} m); "
                "requesting hard reset"
            )
            self._finish("hard_reset_required")
            return np.zeros(2, dtype=np.float32)

        if self.phase == "goto_start":
            self.phase_steps += 1
            if dist < self.arrive_m:
                self.phase = "settle"
                self.phase_steps = 0
                self._settle_count = 0
                return np.zeros(2, dtype=np.float32)
            return self._toward_target(paddle_xy)

        if self.phase == "settle":
            self.phase_steps += 1
            if dist > 2.0 * self.arrive_m:
                # Drifted off (PID overshoot / operator nudge): go back.
                self.phase = "goto_start"
                self.phase_steps = 0
                self._settle_count = 0
                return self._toward_target(paddle_xy)
            self._settle_count += 1
            if (
                self._settle_count >= self.settle_steps
                and self._paddle_speed(state_info) <= self.settle_speed_mps
            ):
                self._finish("success")
            return np.zeros(2, dtype=np.float32)

        raise RuntimeError(f"PaddleRepositionFSM: unknown phase {self.phase!r}")
