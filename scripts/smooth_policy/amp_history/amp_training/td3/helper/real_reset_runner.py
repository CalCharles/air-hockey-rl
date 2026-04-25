"""Reset orchestration for the real-world TD3 collector.

Collapses the four reset-FSM call-sites in ``async_td3_real.collector_process``
(startup L1373, soft post-episode L2172, hard-with-FSM L2230, hard-skip-FSM
L2255) into a single ``ResetRunner.run(kind=...)`` method.

Also lifts ``run_reset_fsm`` (L661–737), ``_hard_reset_with_pause``
(L1191–1222), and ``_should_run_reset_policy_at_episode_start`` (L1158–1188)
out of the orchestrator file so the entire reset code path lives here.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import torch

from airhockey import AirHockeyEnv

from .real_collector_reset import (
    merge_reset_fsm_artifact_into_pending,
    soft_reset_prime_paddle_and_extract_previous_puck,
)
from .real_stop_state import _classify_stop_event


# ---------------------------------------------------------------------------
# Reset FSM execution + supporting helpers (lifted from async_td3_real.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingResetArtifact:
    episode_id: int
    partition: str
    done_reason: str
    step_count: int
    rows: list
    images: list
    camera_null_frames: int


@dataclass(frozen=True)
class ResetFSMRunResult:
    total_steps: int
    done_reason: str
    artifact: PendingResetArtifact | None = None


def _reset_stage_id_from_phase(phase: str) -> int:
    phase_name = str(phase)
    if phase_name in ("goto_start", "edge_loop", "upward_burst", "post_first_upward_check"):
        return 0
    if phase_name in ("wait_for_puck", "strike", "post_second_upward_check"):
        return 1
    return -1


def _reset_artifact_partition(done_reason: str) -> str:
    return "success" if str(done_reason) == "success" else "failure"


def _hard_reset_with_pause(
    env: AirHockeyEnv, reason: str, pause_s: float = 3.0
) -> tuple[np.ndarray, dict]:
    """Force physical env reset, then wait before returning to policy collection."""
    print(f"[collector_fallback_reset] reason={reason} -> hard env reset")
    simulator = getattr(env, "simulator", None)
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
    supports_write_traj = False
    try:
        reset_signature = inspect.signature(env.reset)
        supports_write_traj = "write_traj" in reset_signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in reset_signature.parameters.values()
        )
    except (TypeError, ValueError):
        supports_write_traj = False
    if supports_write_traj:
        obs, info = env.reset(seed=None, write_traj=False)
    else:
        obs, info = env.reset(seed=None)
    print(f"[collector_fallback_reset] sleeping {pause_s:.1f}s before resume")
    time.sleep(float(pause_s))
    return obs, info


def _prime_paddle_history_stand_still_non_occluded(env: AirHockeyEnv) -> np.ndarray:
    """Fill paddle history with stationary non-occluded entries and rebuild observation."""
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        obs, _ = env.get_current_state()
        return np.asarray(obs, dtype=np.float32)
    state_info = simulator.get_current_state()
    try:
        paddle_position = np.asarray(
            state_info["paddles"]["paddle_ego"]["position"],
            dtype=np.float64,
        ).reshape(-1)
        paddle_x = float(paddle_position[0])
        paddle_y = float(paddle_position[1])
    except Exception:
        return env.get_observation(
            state_info,
            obs_type=env.obs_type,
            puck_history=simulator.puck_history,
            paddle_history=simulator.paddle_history,
        )
    history_len = int(getattr(simulator, "paddle_history_len", 5))
    simulator.paddle_history = [(paddle_x, paddle_y, 0) for _ in range(max(1, history_len))]
    return env.get_observation(
        state_info,
        obs_type=env.obs_type,
        puck_history=simulator.puck_history,
        paddle_history=simulator.paddle_history,
    )


def _should_run_reset_policy_at_episode_start(
    state_info: dict | None,
    table_x_bot: float | None,
    bottom_margin: float,
    bottom_fail_count: int,
    occluded_fail_count: int,
    counters: dict,
) -> bool:
    """Decide whether to enter reset-policy mode at episode start."""
    if not isinstance(state_info, dict) or table_x_bot is None or not np.isfinite(float(table_x_bot)):
        return False
    try:
        puck = state_info["pucks"][0]
        puck_x = float(puck["position"][0])
        puck_occ = int(np.asarray(puck.get("occluded", 0)).reshape(-1)[0]) > 0
    except Exception:
        return False

    if puck_x >= (float(table_x_bot) - float(bottom_margin)):
        counters["bottom"] = int(counters.get("bottom", 0)) + 1
    else:
        counters["bottom"] = 0

    if puck_occ:
        counters["occ"] = int(counters.get("occ", 0)) + 1
    else:
        counters["occ"] = 0

    return bool(
        counters["bottom"] >= int(bottom_fail_count)
        or counters["occ"] >= int(occluded_fail_count)
    )


def run_reset_fsm(
    env: AirHockeyEnv,
    rng: np.random.Generator,
    artifact_episode_id: int,
    *,
    reset_policy_fsm_cls: Callable,
    build_split_episode_row: Callable,
    latest_camera_frame: Callable,
) -> ResetFSMRunResult:
    """Run the ResetPolicyFSM until done; collect an artifact for the orchestrator.

    These steps are NOT recorded in the policy replay buffer.
    """
    wait_logged = False
    stop_state = _classify_stop_event(env)
    while stop_state.active:
        if not wait_logged:
            print(
                "[reset_fsm] "
                f"stop active; waiting for clear (reason={stop_state.reason})..."
            )
            wait_logged = True
        time.sleep(0.25)
        stop_state = _classify_stop_event(env)
    if wait_logged:
        print("[reset_fsm] stop cleared; resuming reset FSM.")

    fsm = reset_policy_fsm_cls(env, rng)
    reset_rows: list = []
    reset_images: list = []
    reset_camera_null_frames = 0
    print(f"[reset_fsm] starting (side={fsm.start_side})")
    try:
        while not fsm.done:
            state = env.simulator.get_current_state()
            action = fsm.step(state)
            reset_stage_id = _reset_stage_id_from_phase(getattr(fsm, "phase", "unknown"))
            _, _, _, _, step_info = env.step(action)
            stop_state = _classify_stop_event(env, step_info=step_info)
            camera_frame = latest_camera_frame(env)
            if camera_frame is not None:
                reset_images.append(camera_frame)
            else:
                reset_camera_null_frames += 1
            reset_rows.append(
                build_split_episode_row(
                    env=env,
                    action_xy=action,
                    episode_id=artifact_episode_id,
                    episode_step_idx=len(reset_rows),
                    protective_stop_active=stop_state.protective_stop,
                    controller_disconnected=stop_state.controller_disconnected,
                    reset_stage_id=reset_stage_id,
                )
            )
    finally:
        fsm.close()
    done_reason = getattr(fsm, "done_reason", "unknown")
    artifact = None
    if reset_rows:
        artifact = PendingResetArtifact(
            episode_id=int(artifact_episode_id),
            partition=_reset_artifact_partition(done_reason),
            done_reason=str(done_reason),
            step_count=len(reset_rows),
            rows=reset_rows,
            images=reset_images,
            camera_null_frames=int(reset_camera_null_frames),
        )
    print(
        f"[reset_fsm] done after {fsm.total_steps} steps "
        f"(final phase={fsm.phase}, reason={done_reason})"
    )
    if done_reason == "hard_reset_required":
        _hard_reset_with_pause(env, reason="reset_fsm_stage2_max_retries", pause_s=0.0)
    return ResetFSMRunResult(
        total_steps=int(fsm.total_steps),
        done_reason=str(done_reason),
        artifact=artifact,
    )


# ---------------------------------------------------------------------------
# ResetRunner public surface.
# ---------------------------------------------------------------------------


class ResetKind(Enum):
    """All four reset cases the orchestrator can request.

    See plan §3.2 — the full table maps each value to its source line range,
    its ``transition_reason`` string, and the
    ``startup_buffered_message`` flag passed to
    ``merge_reset_fsm_artifact_into_pending``.
    """

    STARTUP = "startup"
    SOFT = "soft"
    HARD_WITH_FSM = "hard_with_fsm"
    HARD_SKIP_FSM = "hard_skip_fsm"


_TRANSITION_REASON_BY_KIND: dict[ResetKind, str] = {
    ResetKind.STARTUP: "startup_reset_to_policy",
    ResetKind.SOFT: "reset_fsm_to_policy",
    ResetKind.HARD_WITH_FSM: "hard_reset_reset_fsm_to_policy",
    ResetKind.HARD_SKIP_FSM: "hard_reset_to_policy",
}


@dataclass
class StopFlags:
    """Snapshot of the orchestrator's per-episode stop flags."""

    had_stop: bool = False
    had_protective_stop: bool = False
    had_controller_disconnect: bool = False


@dataclass
class ResetResult:
    obs: np.ndarray
    artifact: PendingResetArtifact | None
    total_fsm_steps: int
    transition_reason: str
    startup_buffered_message: bool
    kind_actual: ResetKind
    attempts: int
    next_reset_file_id: int
    pending_reset_artifact: PendingResetArtifact | None


@dataclass
class _ResetRunnerCounters:
    """Mutable counters that persist across `ResetRunner.run` calls."""

    bottom: int = 0
    occ: int = 0


class ResetRunner:
    """Reset orchestrator. Blocks until success on each call.

    Owns: the FSM factory, env reference, ``reset_rng``, the soft/hard helpers,
    and the per-attempt counters used by
    ``_should_run_reset_policy_at_episode_start``.

    Does NOT own: actor, replay, learner, transition_hold (returns the
    reason string for the orchestrator to call ``transition_hold.begin``).
    """

    MIN_RESET_DELAY_S: float = 3.0

    def __init__(
        self,
        env: AirHockeyEnv,
        *,
        device: torch.device,
        reset_rng: np.random.Generator,
        reset_policy_fsm_cls: Callable,
        build_split_episode_row: Callable,
        latest_camera_frame: Callable,
        extract_primitive_state_tensors: Callable,
        episode_start_reset_bottom_margin: float = 0.25,
        episode_start_reset_bottom_fail_count: int = 2,
        episode_start_reset_occluded_fail_count: int = 6,
    ) -> None:
        self._env = env
        self._device = device
        self._reset_rng = reset_rng
        self._reset_policy_fsm_cls = reset_policy_fsm_cls
        self._build_split_episode_row = build_split_episode_row
        self._latest_camera_frame = latest_camera_frame
        self._extract_primitive_state_tensors = extract_primitive_state_tensors
        self._bottom_margin = float(episode_start_reset_bottom_margin)
        self._bottom_fail_count = int(episode_start_reset_bottom_fail_count)
        self._occluded_fail_count = int(episode_start_reset_occluded_fail_count)
        self._counters: dict = {"bottom": 0, "occ": 0}

    # ------------------------------------------------------------------
    # Internal helpers (each maps 1:1 to a call-site in collector_process).
    # ------------------------------------------------------------------

    def _run_fsm_once(self, artifact_episode_id: int) -> ResetFSMRunResult:
        return run_reset_fsm(
            self._env,
            self._reset_rng,
            artifact_episode_id=artifact_episode_id,
            reset_policy_fsm_cls=self._reset_policy_fsm_cls,
            build_split_episode_row=self._build_split_episode_row,
            latest_camera_frame=self._latest_camera_frame,
        )

    def _soft_prime(self) -> tuple[np.ndarray, torch.Tensor]:
        return soft_reset_prime_paddle_and_extract_previous_puck(
            self._env,
            device=self._device,
            prime_paddle_history_stand_still_non_occluded=
                _prime_paddle_history_stand_still_non_occluded,
            extract_primitive_state_tensors=self._extract_primitive_state_tensors,
        )

    # ------------------------------------------------------------------
    # Public entrypoint.
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        kind: ResetKind,
        artifact_episode_id: int,
        episode_had_stop_flags: StopFlags,
        episode_end_wall_time: float,
        pending_reset_artifact: PendingResetArtifact | None,
        next_reset_file_id: int,
    ) -> ResetResult:
        """Block until the reset succeeds; return the seeded obs + artifact.

        ``episode_end_wall_time`` is a ``time.time()`` snapshot the orchestrator
        captures the moment the previous policy episode ended. ResetRunner
        applies the soft-path ``artificial_delay_s`` formula
        (``max(0, MIN_RESET_DELAY_S - processing_elapsed_s)``) only on
        ``ResetKind.SOFT``; hard paths use ``_hard_reset_with_pause``'s own
        pause and ignore this field.

        Threads ``pending_reset_artifact`` and ``next_reset_file_id`` through
        ``merge_reset_fsm_artifact_into_pending`` to preserve the existing
        artifact-id rule.
        """
        attempts = 1
        kind_actual = kind
        startup_buffered_message = kind == ResetKind.STARTUP
        total_fsm_steps = 0
        artifact_for_log = None

        if kind == ResetKind.STARTUP:
            # Source: L1373–L1390 (startup before main loop).
            fsm_result = self._run_fsm_once(artifact_episode_id)
            total_fsm_steps += fsm_result.total_steps
            pending_reset_artifact, next_reset_file_id = merge_reset_fsm_artifact_into_pending(
                fsm_result.artifact,
                pending_reset_artifact,
                next_reset_file_id,
                startup_buffered_message=True,
            )
            artifact_for_log = fsm_result.artifact
            obs, _ = self._soft_prime()

        elif kind == ResetKind.SOFT:
            # Source: L2156–L2194 (no stop, not periodic-3).
            processing_elapsed_s = time.time() - float(episode_end_wall_time)
            artificial_delay_s = max(0.0, self.MIN_RESET_DELAY_S - processing_elapsed_s)
            if artificial_delay_s > 0.0:
                time.sleep(artificial_delay_s)
            print(
                "[collector] "
                f"episode_id={int(artifact_episode_id) - 1} "
                f"post_episode_processing_s={processing_elapsed_s:.3f} "
                f"artificial_delay_s={artificial_delay_s:.3f} "
                f"min_reset_delay_s={self.MIN_RESET_DELAY_S:.3f}"
            )
            fsm_result = self._run_fsm_once(artifact_episode_id)
            total_fsm_steps += fsm_result.total_steps
            pending_reset_artifact, next_reset_file_id = merge_reset_fsm_artifact_into_pending(
                fsm_result.artifact,
                pending_reset_artifact,
                next_reset_file_id,
                startup_buffered_message=False,
            )
            artifact_for_log = fsm_result.artifact
            obs, _ = self._soft_prime()

        elif kind == ResetKind.HARD_WITH_FSM or kind == ResetKind.HARD_SKIP_FSM:
            # Source: L2195–L2264 (periodic-3 OR stop-driven).
            if episode_had_stop_flags.had_protective_stop:
                hard_reset_reason = "collector_estop_next_step"
            elif episode_had_stop_flags.had_controller_disconnect:
                hard_reset_reason = "collector_controller_disconnected_next_step"
            else:
                hard_reset_reason = "periodic_every_3_episodes"
            print(
                "[collector] "
                f"episode_id={int(artifact_episode_id) - 1} "
                f"using hard reset path reason={hard_reset_reason}"
            )
            obs, _ = _hard_reset_with_pause(
                env=self._env,
                reason=hard_reset_reason,
                pause_s=self.MIN_RESET_DELAY_S,
            )
            hard_reset_state = self._env.simulator.get_current_state()
            run_reset_policy = _should_run_reset_policy_at_episode_start(
                state_info=hard_reset_state,
                table_x_bot=getattr(self._env, "table_x_bot", None),
                bottom_margin=self._bottom_margin,
                bottom_fail_count=self._bottom_fail_count,
                occluded_fail_count=self._occluded_fail_count,
                counters=self._counters,
            )
            decision = "reset_policy" if run_reset_policy else "policy"
            print(
                "[collector] "
                f"episode_id={int(artifact_episode_id) - 1} "
                f"hard_reset_start_decision={decision} "
                f"bottom_counter={self._counters['bottom']} "
                f"occ_counter={self._counters['occ']}"
            )
            if run_reset_policy:
                kind_actual = ResetKind.HARD_WITH_FSM
                fsm_result = self._run_fsm_once(artifact_episode_id)
                total_fsm_steps += fsm_result.total_steps
                pending_reset_artifact, next_reset_file_id = merge_reset_fsm_artifact_into_pending(
                    fsm_result.artifact,
                    pending_reset_artifact,
                    next_reset_file_id,
                    startup_buffered_message=False,
                )
                artifact_for_log = fsm_result.artifact
                obs, _ = self._soft_prime()
                self._counters["bottom"] = 0
                self._counters["occ"] = 0
            else:
                kind_actual = ResetKind.HARD_SKIP_FSM
                # No FSM, no soft prime — keep obs from _hard_reset_with_pause.
                # primitive state extract is owned by the caller's
                # transition_hold.begin(), so we do nothing further here.

        else:
            raise ValueError(f"Unknown ResetKind: {kind!r}")

        return ResetResult(
            obs=obs,
            artifact=artifact_for_log,
            total_fsm_steps=int(total_fsm_steps),
            transition_reason=_TRANSITION_REASON_BY_KIND[kind_actual],
            startup_buffered_message=startup_buffered_message,
            kind_actual=kind_actual,
            attempts=attempts,
            next_reset_file_id=int(next_reset_file_id),
            pending_reset_artifact=pending_reset_artifact,
        )


def pick_reset_kind(total_episodes: int, stop_flags: StopFlags) -> ResetKind:
    """Pick the reset kind for a normal post-episode boundary.

    Source mapping (L2156–L2200):
      - ``periodic_every_3 OR stop`` → hard reset path
        (orchestrator picks ``HARD_WITH_FSM``; ResetRunner may downgrade
        to ``HARD_SKIP_FSM`` based on env state)
      - else → ``SOFT``
    """
    periodic_hard_reset = (int(total_episodes) % 3) == 0
    if periodic_hard_reset or stop_flags.had_stop:
        return ResetKind.HARD_WITH_FSM
    return ResetKind.SOFT
