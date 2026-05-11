"""PolicyRunner — runs one TD3 policy episode against the real (or sim) env.

Lifts the per-step body and episode-end finalization out of the old
monolithic ``collector_process``. Returns a ``PolicyEpisodeResult`` that
the orchestrator (``collector_process_modular`` in
``extras/async_td3_real.py``) pushes to replay, runs the learner against,
and saves to disk — none of those concerns leak into the runner.

The runner ticks ``TransitionHoldState`` internally, truncates the trajectory
on readiness-fail e-stops, and exposes all delta counters the orchestrator
accumulates between episodes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from airhockey import AirHockeyEnv

from .real_episode_buffers import truncate_collector_episode_for_readiness_fail
from .real_motion_rewards import (
    _compute_motion_reward_components,
    _extract_motion_magnitudes_from_step_info,
    _extract_motion_positions_from_state_info,
    _reset_motion_reward_state,
)
from .real_stop_state import _classify_stop_event
from .real_transition_hold import RolloutContext, TransitionHoldState
from .td3_episode_collection import EpisodeTrajectory


MOTION_METRIC_NAMES = (
    "temporal_valid_fraction",
    "stand_still_reward_raw",
    "temporal_alignment_reward_raw",
    "axis_alignment_reward_raw",
    "velocity_reward_raw",
    "jerk_reward_raw",
    "stand_still_reward_weighted",
    "temporal_alignment_reward_weighted",
    "axis_alignment_reward_weighted",
    "velocity_reward_weighted",
    "jerk_reward_weighted",
)


@dataclass
class StopFlagsSnapshot:
    """Per-episode stop-event accumulators (all default False at episode start)."""

    had_stop: bool = False
    had_protective_stop: bool = False
    had_controller_disconnect: bool = False
    had_readiness_fail_estop: bool = False
    had_human_interrupt: bool = False


@dataclass
class TerminalInfo:
    """Everything the orchestrator needs to compose end-of-episode logs."""

    dones: bool
    truncated: bool
    success: bool
    protective_stop: bool
    controller_disconnect: bool
    readiness_fail_estop: bool
    first_readiness_fail_step_idx: int | None
    first_readiness_fail_reason: str | None
    episode_success: bool
    episode_end_type: str | None
    episode_end_reasons: list
    episode_end_reason: str | None
    readiness_fail_dropped_steps: int
    stop_state_reason: str
    stop_state_artifact_label: str | None
    stop_state_episode_end_type: str | None
    stop_state_episode_end_reason: str | None
    stop_now: bool
    stop_flags: StopFlagsSnapshot


@dataclass
class EpisodeMetrics:
    """Numeric metrics + delta counters the orchestrator accumulates."""

    episode_return: float
    episode_length: float
    episode_task_reward: float
    episode_motion_reward: float
    episode_estop_flag: float
    motion_metric_means: dict
    motion_metric_count: int
    # Latency arrays (already trimmed for readiness-fail).
    puck_detection_latency_ms: list
    model_inference_latency_ms: list
    block_sleep_latency_ms: list
    other_latency_ms: list
    # Camera metadata.
    camera_null_frames: int
    # Delta counters orchestrator accumulates into its own totals.
    delta_total_steps: int
    delta_protective_stop_steps: int
    delta_controller_disconnect_steps: int
    delta_readiness_fail_steps: int
    delta_readiness_fail_estop_dropped_steps: int
    delta_transition_hold_steps: int
    delta_interval_primitive_env_steps: int
    delta_interval_primitive_horizontal_env_steps: int
    delta_interval_target_position_directional_env_steps: int
    delta_human_interrupt_steps: int
    had_protective_stop: bool
    had_controller_disconnect: bool
    had_human_interrupt: bool


@dataclass
class PolicyEpisodeResult:
    trajectory: EpisodeTrajectory
    rows: list
    images: list
    total_env_steps: int
    terminal: TerminalInfo
    metrics: EpisodeMetrics


class PolicyRunner:
    """Runs one policy episode at a time against the env.

    Construction parameters mirror the originals in
    ``collector_process``. The runner reads ``transition_hold.active()``
    in its hot loop and calls ``transition_hold.tick()`` once per step;
    it never calls ``begin`` (that's the orchestrator's responsibility).

    See plan §3.1 for the full per-episode reset list cleared by
    ``seed_after_reset``.
    """

    def __init__(
        self,
        env: AirHockeyEnv,
        actor,
        *,
        device: torch.device,
        args,
        train_args,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        primitive_selector,
        transition_hold: TransitionHoldState,
        ctx: RolloutContext,
        extract_primitive_state_tensors: Callable,
        reset_primitive_rollout_state: Callable,
        deterministic_actor_action: Callable,
        augment_policy_observation: Callable,
        primitive_exploration_chance_for_step: Callable,
        latest_camera_frame: Callable,
        env_timing_info: Callable,
        safe_nonnegative_ms: Callable,
        build_split_episode_row: Callable,
        init_motion_reward_state: Callable,
        readiness_fn: Callable,
    ) -> None:
        self._env = env
        self._actor = actor
        self._device = device
        self._args = args
        self._train_args = train_args
        self._action_low = action_low
        self._action_high = action_high
        self._primitive_selector = primitive_selector
        self._transition_hold = transition_hold
        self._ctx = ctx
        self._extract_primitive_state_tensors = extract_primitive_state_tensors
        self._reset_primitive_rollout_state = reset_primitive_rollout_state
        self._deterministic_actor_action = deterministic_actor_action
        self._augment_policy_observation = augment_policy_observation
        self._primitive_exploration_chance_for_step = primitive_exploration_chance_for_step
        self._latest_camera_frame = latest_camera_frame
        self._env_timing_info = env_timing_info
        self._safe_nonnegative_ms = safe_nonnegative_ms
        self._build_split_episode_row = build_split_episode_row
        self._init_motion_reward_state = init_motion_reward_state
        self._readiness_fn = readiness_fn

        # Initial seed for obs / motion-reward state. Set by orchestrator
        # via `seed_after_initial_reset` before the main loop starts.
        self._obs: np.ndarray | None = None
        self._motion_reward_state = None

        # Per-episode buffers. Cleared inside `seed_after_reset`.
        self._episode_trajectory = EpisodeTrajectory.empty()
        self._episode_rows: list = []
        self._episode_images: list = []
        self._episode_puck_detection_latency_ms: list = []
        self._episode_model_inference_latency_ms: list = []
        self._episode_block_sleep_latency_ms: list = []
        self._episode_other_latency_ms: list = []
        self._episode_camera_null_frames: int = 0

        # Stop / readiness flags (per-episode).
        self._stop_flags = StopFlagsSnapshot()
        self._episode_readiness_first_fail_step_idx: int | None = None
        self._episode_readiness_first_fail_reason: str | None = None

        # Readiness streak (per-step state, reset at episode start).
        self._readiness_fail_streak: int = 0
        self._readiness_fail_first_episode_step_idx: int | None = None
        self._readiness_fail_first_total_step: int | None = None
        self._readiness_fail_prev: bool = False
        self._readiness_fail_prev_reason: str = "none"
        self._readiness_fail_window: int = 5

        # Motion-metric accumulators (per-episode).
        self._episode_motion_metric_sums = {name: 0.0 for name in MOTION_METRIC_NAMES}
        self._episode_motion_metric_count = 0

        # Cumulative counter the orchestrator reads via metrics deltas.
        # We track it internally for the orchestrator's primitive chance
        # anneal lookup (`primitive_exploration_chance_for_step`).
        self._total_steps: int = 0
        # Cumulative readiness-fail step count (NOT reset per-episode); the
        # orchestrator gets a per-episode delta via metrics.
        self._readiness_fail_steps_total: int = 0

    # ------------------------------------------------------------------
    # Initialization hooks called by the orchestrator.
    # ------------------------------------------------------------------

    def set_total_steps(self, total_steps: int) -> None:
        """Seed the runner's running step counter from a checkpoint resume."""
        self._total_steps = int(total_steps)

    def rollback_invalid_episode_steps(self, n: int) -> None:
        """Roll back the running step counter for an episode whose trajectory
        was rejected by validation. The physical steps happened in the world,
        but they don't represent valid policy data — replay/learner/metrics
        already skip them, and the cap/checkpoint cadence should too."""
        if n <= 0:
            return
        self._total_steps = max(0, self._total_steps - int(n))

    @property
    def total_steps(self) -> int:
        return int(self._total_steps)

    def seed_initial(self, obs: np.ndarray, *, motion_reward_horizon: int) -> None:
        """Initial seeding from startup reset (before main loop).

        Sets ``self._obs`` and constructs the initial ``motion_reward_state``
        from current env paddle/puck (matches L1496–1509).
        """
        self._obs = obs
        env = self._env
        initial_state_info = getattr(env, "current_state", None)
        if not isinstance(initial_state_info, dict):
            simulator = getattr(env, "simulator", None)
            if simulator is not None and hasattr(simulator, "get_current_state"):
                try:
                    initial_state_info = simulator.get_current_state()
                except Exception:
                    initial_state_info = None
        initial_paddle_xy, initial_puck_xy = _extract_motion_positions_from_state_info(
            initial_state_info
        )
        self._motion_reward_state = self._init_motion_reward_state(
            int(motion_reward_horizon),
            anchor_paddle_xy=initial_paddle_xy,
            anchor_puck_xy=initial_puck_xy,
        )

    def seed_after_reset(self, obs: np.ndarray) -> None:
        """Reset all per-episode state for the next episode.

        Source L2147–L2153 (trajectory buffers) + L2265–L2292
        (stop flags / readiness trackers / motion-metric accumulators
        + ``_reset_motion_reward_state`` re-anchor). Order: clear flags
        first, re-anchor last.
        """
        self._obs = obs
        # Trajectory buffers (currently cleared at HDF5-save time;
        # consolidated here per plan §6.2).
        self._episode_rows = []
        self._episode_puck_detection_latency_ms = []
        self._episode_model_inference_latency_ms = []
        self._episode_block_sleep_latency_ms = []
        self._episode_other_latency_ms = []
        self._episode_images = []
        self._episode_camera_null_frames = 0

        # Stop / end flags.
        self._stop_flags = StopFlagsSnapshot()

        # Readiness trackers (per-episode).
        self._episode_readiness_first_fail_step_idx = None
        self._episode_readiness_first_fail_reason = None

        # Readiness streak (per-step state seeded at episode start).
        self._readiness_fail_streak = 0
        self._readiness_fail_first_episode_step_idx = None
        self._readiness_fail_first_total_step = None
        self._readiness_fail_prev = False
        self._readiness_fail_prev_reason = "none"

        # Motion metric accumulators.
        self._episode_motion_metric_sums = {name: 0.0 for name in MOTION_METRIC_NAMES}
        self._episode_motion_metric_count = 0

        # Re-anchor motion_reward_state from current post-reset env state
        # (must run last; depends on env state being settled).
        env = self._env
        current_state_info = getattr(env, "current_state", None)
        if not isinstance(current_state_info, dict):
            simulator = getattr(env, "simulator", None)
            if simulator is not None and hasattr(simulator, "get_current_state"):
                try:
                    current_state_info = simulator.get_current_state()
                except Exception:
                    current_state_info = None
        current_paddle_xy, current_puck_xy = _extract_motion_positions_from_state_info(
            current_state_info
        )
        _reset_motion_reward_state(
            self._motion_reward_state,
            anchor_paddle_xy=current_paddle_xy,
            anchor_puck_xy=current_puck_xy,
        )

    # ------------------------------------------------------------------
    # Episode loop.
    # ------------------------------------------------------------------

    def run_episode(self) -> PolicyEpisodeResult:
        """Run one policy episode until terminal; return the result."""
        if self._obs is None:
            raise RuntimeError(
                "PolicyRunner.run_episode called before seed_initial / seed_after_reset"
            )

        env = self._env
        device = self._device
        args = self._args
        train_args = self._train_args
        ctx = self._ctx
        transition_hold = self._transition_hold
        primitive_selector = self._primitive_selector
        action_low = self._action_low
        action_high = self._action_high

        delta_total_steps = 0
        delta_protective_stop_steps = 0
        delta_controller_disconnect_steps = 0
        delta_readiness_fail_steps = 0
        delta_human_interrupt_steps = 0
        delta_transition_hold_steps_at_start = transition_hold.steps_total
        delta_interval_primitive_env_steps = 0
        delta_interval_primitive_horizontal_env_steps = 0
        delta_interval_target_position_directional_env_steps = 0

        # Episode-end fields populated when `dones` becomes True.
        terminal: TerminalInfo | None = None
        readiness_fail_dropped_steps = 0

        while True:
            # ----------------------------- Readiness check (L1557–1606) ---
            step_ready, step_ready_reason = self._readiness_fn(env)
            if not step_ready:
                self._readiness_fail_steps_total += 1
                delta_readiness_fail_steps += 1
                if self._readiness_fail_streak == 0:
                    self._readiness_fail_first_episode_step_idx = int(len(self._episode_rows))
                    self._readiness_fail_first_total_step = int(self._total_steps)
                    self._episode_readiness_first_fail_step_idx = (
                        self._readiness_fail_first_episode_step_idx
                    )
                    self._episode_readiness_first_fail_reason = str(step_ready_reason)
                self._readiness_fail_streak += 1
                if (not self._readiness_fail_prev) or (
                    step_ready_reason != self._readiness_fail_prev_reason
                ):
                    print(
                        "[collector_safety] "
                        f"robot_step_ready=False reason={step_ready_reason}; continuing collection "
                        f"(consecutive_failures={self._readiness_fail_streak}/{self._readiness_fail_window})"
                    )
                elif self._readiness_fail_streak <= self._readiness_fail_window:
                    print(
                        "[collector_safety] "
                        f"robot_step_ready still false reason={step_ready_reason}; "
                        f"consecutive_failures={self._readiness_fail_streak}/{self._readiness_fail_window}"
                    )
                self._readiness_fail_prev = True
                self._readiness_fail_prev_reason = str(step_ready_reason)
            else:
                if self._readiness_fail_prev:
                    recovered_from_reason = str(self._readiness_fail_prev_reason)
                    recovered_streak = int(self._readiness_fail_streak)
                    had_triggered_window = recovered_streak >= self._readiness_fail_window
                    print(
                        "[collector_safety] "
                        f"robot step readiness restored after reason={recovered_from_reason}; "
                        f"consecutive_failures={recovered_streak} "
                        f"window_triggered={int(had_triggered_window)}"
                    )
                self._readiness_fail_streak = 0
                self._readiness_fail_first_episode_step_idx = None
                self._readiness_fail_first_total_step = None
                self._episode_readiness_first_fail_step_idx = None
                self._episode_readiness_first_fail_reason = None
                self._readiness_fail_prev = False
                self._readiness_fail_prev_reason = "none"
            if (
                self._readiness_fail_prev
                and self._readiness_fail_streak == self._readiness_fail_window
            ):
                print(
                    "[collector_safety] "
                    f"readiness failure window reached ({self._readiness_fail_window} consecutive); "
                    f"will terminate episode at first failure step "
                    f"(episode_step_idx={self._readiness_fail_first_episode_step_idx}, "
                    f"total_step={self._readiness_fail_first_total_step})"
                )

            # ----------------------------- Action selection (L1607–1652) ---
            collector_step_start_s = (
                time.perf_counter() if args.enable_latency_profiling else 0.0
            )
            transition_hold_active = transition_hold.active()
            obs_tensor = torch.as_tensor(self._obs, dtype=torch.float32, device=device).unsqueeze(0)
            policy_obs = self._augment_policy_observation(
                obs_tensor,
                ctx.last_action_for_policy,
                train_args.use_last_action_in_policy_state,
            )
            model_inference_ms = 0.0
            primitive_step_stats = {
                "primitive_applied_count": 0,
                "primitive_horizontal_dominant_count": 0,
                "target_position_directional_applied_count": 0,
            }
            with torch.no_grad():
                inference_start_s = (
                    time.perf_counter() if args.enable_latency_profiling else 0.0
                )
                action_tensor = self._deterministic_actor_action(self._actor, policy_obs)
                disable_noise_for_transition = bool(
                    transition_hold_active and args.transition_disable_exploration_noise
                )
                if args.exploration_noise > 0 and not disable_noise_for_transition:
                    action_tensor = action_tensor + torch.randn_like(action_tensor) * float(
                        args.exploration_noise
                    )
                action_tensor = torch.clamp(action_tensor, action_low, action_high)
                if not transition_hold_active and not args.collector_policy_stand_still:
                    primitive_selector.chance = float(
                        self._primitive_exploration_chance_for_step(args, self._total_steps)
                    )
                    current_paddle_pos, current_puck_pos, current_puck_vel = (
                        self._extract_primitive_state_tensors(env, device=device)
                    )
                    if torch.all(current_puck_vel == 0):
                        current_puck_vel = (
                            current_puck_pos - ctx.previous_puck_position_for_primitive
                        )
                    y_alignment_sign = torch.sign(
                        current_puck_pos[:, 1] - current_paddle_pos[:, 1]
                    )
                    action_tensor, primitive_step_stats = primitive_selector.apply(
                        action_tensor,
                        action_low=action_low,
                        action_high=action_high,
                        y_alignment_sign=y_alignment_sign,
                        current_paddle_position=current_paddle_pos,
                        current_puck_position=current_puck_pos,
                        current_puck_velocity=current_puck_vel,
                        return_stats=True,
                    )
                if transition_hold_active:
                    action_tensor = torch.zeros_like(action_tensor)
                if args.collector_policy_stand_still:
                    action_tensor = torch.zeros_like(action_tensor)

            env_action = action_tensor.squeeze(0).detach().cpu().numpy()
            if args.enable_latency_profiling:
                model_inference_ms = self._safe_nonnegative_ms(
                    (time.perf_counter() - inference_start_s) * 1000.0
                )

            prev_action = ctx.last_action_for_policy.clone()
            next_obs, task_reward, terminations, truncations, step_info = env.step(env_action)

            if args.enable_latency_profiling:
                collector_step_end_s = time.perf_counter()
                step_total_ms = self._safe_nonnegative_ms(
                    (collector_step_end_s - collector_step_start_s) * 1000.0
                )
                timing_info = self._env_timing_info(env)
                camera_received_s = float(timing_info.get("camera_frame_received_s", float("nan")))
                puck_done_s = float(timing_info.get("puck_detection_done_s", float("nan")))
                if np.isfinite(camera_received_s) and np.isfinite(puck_done_s):
                    puck_detection_ms = self._safe_nonnegative_ms(
                        (puck_done_s - camera_received_s) * 1000.0
                    )
                else:
                    puck_detection_ms = 0.0
                block_sleep_ms = self._safe_nonnegative_ms(
                    float(timing_info.get("sleep_before_step_s", 0.0)) * 1000.0
                )
                other_ms = self._safe_nonnegative_ms(
                    step_total_ms - model_inference_ms - puck_detection_ms - block_sleep_ms
                )
                self._episode_model_inference_latency_ms.append(model_inference_ms)
                self._episode_puck_detection_latency_ms.append(puck_detection_ms)
                self._episode_block_sleep_latency_ms.append(block_sleep_ms)
                self._episode_other_latency_ms.append(other_ms)

            camera_frame = self._latest_camera_frame(env)
            if camera_frame is not None:
                self._episode_images.append(camera_frame)
            else:
                self._episode_camera_null_frames += 1

            stop_state = _classify_stop_event(env, step_info=step_info)
            readiness_fail_stop_now = bool(
                self._readiness_fail_streak >= self._readiness_fail_window
                and self._episode_readiness_first_fail_step_idx is not None
            )
            stop_now = bool(stop_state.active and step_ready)
            if readiness_fail_stop_now:
                stop_now = True
            dones = bool(np.logical_or(terminations, truncations) or stop_now)
            # E-stops (`stop_now`) end the rollout loop but are stored as
            # truncations for the learner: `done=0` so bootstrapping continues
            # at the e-stop transition. See
            # notes/docs/environments/real-world/episode-lifecycle.md.
            terminations_tensor = torch.tensor(
                float(bool(terminations)),
                dtype=torch.float32,
                device=device,
            )
            if stop_state.protective_stop:
                delta_protective_stop_steps += 1
                self._stop_flags.had_protective_stop = True
            if stop_state.controller_disconnected:
                delta_controller_disconnect_steps += 1
                self._stop_flags.had_controller_disconnect = True
            if stop_state.human_interrupt:
                delta_human_interrupt_steps += 1
                self._stop_flags.had_human_interrupt = True
            if readiness_fail_stop_now:
                self._stop_flags.had_readiness_fail_estop = True
            if stop_now:
                self._stop_flags.had_stop = True

            # ----------------------------- Motion reward (L1711–1729) ---
            next_state_info = getattr(env, "current_state", None)
            next_paddle_xy, next_puck_xy = _extract_motion_positions_from_state_info(next_state_info)
            velocity_mag, _, jerk_mag = _extract_motion_magnitudes_from_step_info(
                step_info, self._motion_reward_state
            )
            motion_components = _compute_motion_reward_components(
                args=args,
                motion_state=self._motion_reward_state,
                paddle_xy=next_paddle_xy,
                puck_xy=next_puck_xy,
                velocity_mag=velocity_mag,
                jerk_mag=jerk_mag,
            )
            motion_reward = float(motion_components["motion_reward_total"])
            for metric_name in MOTION_METRIC_NAMES:
                self._episode_motion_metric_sums[metric_name] += float(
                    motion_components[metric_name]
                )
            self._episode_motion_metric_count += 1

            # ----------------------------- Episode-row append (L1730–1746) ---
            # `task_reward`, `motion_reward`, `done` (terminations_tensor) are
            # the same values pushed into the replay buffer below — recording
            # them on the HDF5 row makes the trajectory file self-sufficient
            # for offline policy replay / re-evaluation without needing the
            # runtime replay buffer.
            self._episode_rows.append(
                self._build_split_episode_row(
                    env=env,
                    action_xy=env_action,
                    episode_id=self._next_episode_artifact_id,
                    episode_step_idx=len(self._episode_rows),
                    protective_stop_active=stop_state.protective_stop,
                    controller_disconnected=stop_state.controller_disconnected,
                    task_reward=float(task_reward),
                    motion_reward=float(motion_reward),
                    done=float(terminations_tensor.item()),
                )
            )
            delta_interval_primitive_env_steps += int(primitive_step_stats["primitive_applied_count"])
            delta_interval_primitive_horizontal_env_steps += int(
                primitive_step_stats["primitive_horizontal_dominant_count"]
            )
            delta_interval_target_position_directional_env_steps += int(
                primitive_step_stats["target_position_directional_applied_count"]
            )

            self._episode_trajectory.append_step(
                obs=obs_tensor[0],
                next_obs=torch.as_tensor(next_obs, dtype=torch.float32, device=device),
                action=action_tensor[0],
                task_reward=torch.tensor(float(task_reward), dtype=torch.float32, device=device),
                motion_reward=torch.tensor(float(motion_reward), dtype=torch.float32, device=device),
                done=terminations_tensor,
                prev_action=prev_action[0],
            )

            self._total_steps += 1
            delta_total_steps += 1
            ctx.last_executed_action = action_tensor.detach().clone()
            ctx.previous_puck_position_for_primitive = self._extract_primitive_state_tensors(
                env, device=device
            )[1]
            primitive_selector.reset(torch.tensor([dones], dtype=torch.bool, device=device))

            if train_args.use_last_action_in_policy_state:
                if not (transition_hold_active and transition_hold.last_action_mode == "keep"):
                    ctx.last_action_for_policy = ctx.last_executed_action.clone()
            self._obs = next_obs
            transition_hold.tick()

            if dones:
                # ----------------------------- Episode finalize (L1779–1873) ---
                if self._stop_flags.had_readiness_fail_estop and (
                    self._episode_readiness_first_fail_step_idx is not None
                ):
                    (
                        readiness_fail_dropped_steps,
                        self._episode_rows,
                        self._episode_images,
                        self._episode_puck_detection_latency_ms,
                        self._episode_model_inference_latency_ms,
                        self._episode_block_sleep_latency_ms,
                        self._episode_other_latency_ms,
                        self._episode_camera_null_frames,
                    ) = truncate_collector_episode_for_readiness_fail(
                        episode_trajectory=self._episode_trajectory,
                        episode_readiness_first_fail_step_idx=
                            self._episode_readiness_first_fail_step_idx,
                        episode_rows=self._episode_rows,
                        episode_images=self._episode_images,
                        episode_puck_detection_latency_ms=
                            self._episode_puck_detection_latency_ms,
                        episode_model_inference_latency_ms=
                            self._episode_model_inference_latency_ms,
                        episode_block_sleep_latency_ms=
                            self._episode_block_sleep_latency_ms,
                        episode_other_latency_ms=self._episode_other_latency_ms,
                        episode_camera_null_frames=self._episode_camera_null_frames,
                    )
                    print(
                        "[collector_safety] "
                        f"episode_id={self._next_episode_artifact_id} readiness_fail_estop=1 "
                        f"first_fail_step_idx={self._episode_readiness_first_fail_step_idx} "
                        f"dropped_post_fail_steps={readiness_fail_dropped_steps} "
                        f"reason={self._episode_readiness_first_fail_reason}"
                    )

                # Compose terminal step_info-derived fields (L1831–1857).
                episode_success = (
                    bool(step_info.get("success", False)) if isinstance(step_info, dict) else False
                )
                episode_end_type = (
                    str(step_info.get("episode_end_type"))
                    if isinstance(step_info, dict) and step_info.get("episode_end_type") is not None
                    else None
                )
                episode_end_reasons = (
                    list(step_info.get("episode_end_reasons", []))
                    if isinstance(step_info, dict)
                    and isinstance(step_info.get("episode_end_reasons", []), list)
                    else []
                )
                episode_end_reason = (
                    str(step_info.get("episode_end_reason"))
                    if isinstance(step_info, dict) and step_info.get("episode_end_reason") is not None
                    else None
                )
                if stop_now:
                    episode_end_type = stop_state.episode_end_type
                    episode_end_reasons = [str(stop_state.episode_end_reason)]
                    episode_end_reason = str(stop_state.episode_end_reason)
                episode_stop_artifact_label = stop_state.artifact_label if stop_now else None
                if self._stop_flags.had_readiness_fail_estop:
                    episode_end_type = "estop"
                    episode_end_reasons = ["collector_readiness_fail_5steps"]
                    episode_end_reason = "collector_readiness_fail_5steps"
                    episode_stop_artifact_label = "estop"

                terminal = TerminalInfo(
                    dones=True,
                    truncated=bool(truncations),
                    success=episode_success,
                    protective_stop=self._stop_flags.had_protective_stop,
                    controller_disconnect=self._stop_flags.had_controller_disconnect,
                    readiness_fail_estop=self._stop_flags.had_readiness_fail_estop,
                    first_readiness_fail_step_idx=self._episode_readiness_first_fail_step_idx,
                    first_readiness_fail_reason=self._episode_readiness_first_fail_reason,
                    episode_success=episode_success,
                    episode_end_type=episode_end_type,
                    episode_end_reasons=episode_end_reasons,
                    episode_end_reason=episode_end_reason,
                    readiness_fail_dropped_steps=int(readiness_fail_dropped_steps),
                    stop_state_reason=stop_state.reason,
                    stop_state_artifact_label=episode_stop_artifact_label,
                    stop_state_episode_end_type=stop_state.episode_end_type,
                    stop_state_episode_end_reason=stop_state.episode_end_reason,
                    stop_now=stop_now,
                    stop_flags=StopFlagsSnapshot(**self._stop_flags.__dict__),
                )
                break

        # ----------------------------- Compose metrics ----------------
        episode_return = float(self._episode_trajectory.episode_return)
        episode_length = float(len(self._episode_trajectory.observations))
        episode_task_reward = float(
            torch.stack(self._episode_trajectory.task_rewards, dim=0).sum().item()
        )
        episode_motion_reward = float(
            torch.stack(self._episode_trajectory.motion_rewards, dim=0).sum().item()
        )
        episode_estop_flag = (
            1.0
            if (
                self._stop_flags.had_protective_stop
                or self._stop_flags.had_readiness_fail_estop
            )
            else 0.0
        )

        if self._episode_motion_metric_count > 0:
            motion_metric_means = {
                name: float(self._episode_motion_metric_sums[name]) / float(self._episode_motion_metric_count)
                for name in MOTION_METRIC_NAMES
            }
        else:
            motion_metric_means = {}

        delta_transition_hold_steps = (
            int(transition_hold.steps_total) - int(delta_transition_hold_steps_at_start)
        )

        metrics = EpisodeMetrics(
            episode_return=episode_return,
            episode_length=episode_length,
            episode_task_reward=episode_task_reward,
            episode_motion_reward=episode_motion_reward,
            episode_estop_flag=episode_estop_flag,
            motion_metric_means=motion_metric_means,
            motion_metric_count=int(self._episode_motion_metric_count),
            puck_detection_latency_ms=list(self._episode_puck_detection_latency_ms),
            model_inference_latency_ms=list(self._episode_model_inference_latency_ms),
            block_sleep_latency_ms=list(self._episode_block_sleep_latency_ms),
            other_latency_ms=list(self._episode_other_latency_ms),
            camera_null_frames=int(self._episode_camera_null_frames),
            delta_total_steps=int(delta_total_steps),
            delta_protective_stop_steps=int(delta_protective_stop_steps),
            delta_controller_disconnect_steps=int(delta_controller_disconnect_steps),
            delta_readiness_fail_steps=int(delta_readiness_fail_steps),
            delta_readiness_fail_estop_dropped_steps=int(readiness_fail_dropped_steps),
            delta_transition_hold_steps=int(delta_transition_hold_steps),
            delta_interval_primitive_env_steps=int(delta_interval_primitive_env_steps),
            delta_interval_primitive_horizontal_env_steps=int(delta_interval_primitive_horizontal_env_steps),
            delta_interval_target_position_directional_env_steps=int(
                delta_interval_target_position_directional_env_steps
            ),
            delta_human_interrupt_steps=int(delta_human_interrupt_steps),
            had_protective_stop=self._stop_flags.had_protective_stop,
            had_controller_disconnect=self._stop_flags.had_controller_disconnect,
            had_human_interrupt=self._stop_flags.had_human_interrupt,
        )

        # The trajectory and rows are already truncated; orchestrator pushes
        # them to replay unconditionally and saves HDF5/GIF/video.
        # We hand them out by reference; subsequent calls to
        # `seed_after_reset` allocate fresh per-episode buffers, so the
        # orchestrator's reference remains valid until it's done with them.
        result = PolicyEpisodeResult(
            trajectory=self._episode_trajectory,
            rows=self._episode_rows,
            images=self._episode_images,
            total_env_steps=int(delta_total_steps),
            terminal=terminal,
            metrics=metrics,
        )
        # Reset trajectory for the next episode (matches L1891).
        # The orchestrator already holds a reference for replay push;
        # `EpisodeTrajectory.reset()` mutates the held instance, which
        # matches current source semantics — but `_add_episode_to_shared_replay`
        # serializes via `_episode_to_tensors` BEFORE any mutation occurs.
        # To make the result safe to consume after we return, we hand the
        # finalized trajectory out and start a fresh one for the next episode.
        self._episode_trajectory = EpisodeTrajectory.empty()
        self._reset_primitive_rollout_state(self._primitive_selector)
        return result

    # ------------------------------------------------------------------
    # Mutable inputs the orchestrator updates between episodes.
    # ------------------------------------------------------------------

    @property
    def _next_episode_artifact_id(self) -> int:
        return int(self._artifact_episode_id)

    def set_artifact_episode_id(self, episode_id: int) -> None:
        """Episode id to stamp on every row (matches `next_episode_file_id`)."""
        self._artifact_episode_id = int(episode_id)
