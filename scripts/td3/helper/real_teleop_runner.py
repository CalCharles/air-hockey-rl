"""TeleopRunner — runs one human-controlled episode against the env.

Counterpart to ``PolicyRunner`` for the human-baseline teleop eval. The
contract matches PolicyRunner closely (``seed_initial`` /
``seed_after_reset`` / ``set_artifact_episode_id`` / ``run_episode``)
so the orchestrator and ``_save_episode_artifacts_and_pending_reset``
can drive it the same way the policy eval drives PolicyRunner.

Differences:

* No actor, primitive selector, transition-hold, or motion-reward —
  the human's cursor is the action source.
* Toggles ``simulator.set_human_control(True)`` at episode start and
  back to ``False`` at episode end so the surrounding autonomous
  reset FSM can drive the env via ``env.step(action)``.
* The recorded per-step action comes from
  ``simulator._last_teleop_policy_action`` (the cursor-derived
  displacement the simulator already computes inside ``get_transition``).
* Motion reward is reported as 0.0 — a human-baseline run is not
  evaluated against motion-shaping metrics.

Returns a ``PolicyEpisodeResult`` populated with the same fields the
artifact saver consumes; trajectory is built fresh per episode but is
not pushed to replay (no learner here).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from airhockey import AirHockeyEnv

from .real_policy_runner import (
    EpisodeMetrics,
    PolicyEpisodeResult,
    StopFlagsSnapshot,
    TerminalInfo,
)
from .real_stop_state import _classify_stop_event
from .td3_episode_collection import EpisodeTrajectory


class TeleopRunner:
    """Run one human-controlled episode at a time against the env.

    The runner expects the simulator to have been built with
    ``control_mode='mouse'`` so the camera+mouse subprocess is up and
    ``simulator.set_human_control(...)`` is meaningful. Anything else
    (autonomous reset, e-stop classification, artifact saving) flows
    through the same modules the policy eval uses.
    """

    def __init__(
        self,
        env: AirHockeyEnv,
        *,
        device: torch.device,
        build_split_episode_row: Callable,
        latest_camera_frame: Callable,
        readiness_fn: Callable,
        on_step: Callable[[int, dict], None] | None = None,
    ) -> None:
        self._env = env
        self._device = device
        self._build_split_episode_row = build_split_episode_row
        self._latest_camera_frame = latest_camera_frame
        self._readiness_fn = readiness_fn
        # Optional per-step hook so the orchestrator can update its
        # cv2 status banner without the runner owning UI state.
        self._on_step = on_step

        self._obs: np.ndarray | None = None
        self._artifact_episode_id: int = 0

        self._episode_trajectory = EpisodeTrajectory.empty()
        self._episode_rows: list = []
        self._episode_images: list = []
        self._episode_camera_null_frames: int = 0
        self._stop_flags = StopFlagsSnapshot()

        self._episode_readiness_first_fail_step_idx: int | None = None
        self._episode_readiness_first_fail_reason: str | None = None
        self._readiness_fail_streak: int = 0
        self._readiness_fail_window: int = 5

        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks called by the orchestrator.
    # ------------------------------------------------------------------

    def seed_initial(self, obs: np.ndarray) -> None:
        self._obs = obs

    def seed_after_reset(self, obs: np.ndarray) -> None:
        self._obs = obs
        self._episode_rows = []
        self._episode_images = []
        self._episode_camera_null_frames = 0
        self._stop_flags = StopFlagsSnapshot()
        self._episode_readiness_first_fail_step_idx = None
        self._episode_readiness_first_fail_reason = None
        self._readiness_fail_streak = 0

    def set_artifact_episode_id(self, episode_id: int) -> None:
        self._artifact_episode_id = int(episode_id)

    @property
    def total_steps(self) -> int:
        return int(self._total_steps)

    # ------------------------------------------------------------------
    # Episode loop.
    # ------------------------------------------------------------------

    def run_episode(self) -> PolicyEpisodeResult:
        if self._obs is None:
            raise RuntimeError(
                "TeleopRunner.run_episode called before seed_initial / seed_after_reset"
            )

        env = self._env
        device = self._device
        simulator = getattr(env, "simulator", None)
        if simulator is None or not hasattr(simulator, "set_human_control"):
            raise RuntimeError(
                "TeleopRunner requires AirHockeyReal with set_human_control(); "
                "make sure control_mode='mouse' was set at construction."
            )

        delta_total_steps = 0
        delta_protective_stop_steps = 0
        delta_controller_disconnect_steps = 0
        delta_readiness_fail_steps = 0

        terminal: TerminalInfo | None = None

        # Hand control to the human for the duration of this episode.
        # The reset runner already toggled it back to action-driven before
        # the FSM ran; we flip it on here and off in the ``finally`` so a
        # mid-episode exception still leaves the env in a state where the
        # FSM can drive the robot on the next reset.
        print("[teleop_run] human_control=ON (episode start)")
        simulator.set_human_control(True)
        try:
            while True:
                step_ready, step_ready_reason = self._readiness_fn(env)
                if not step_ready:
                    delta_readiness_fail_steps += 1
                    if self._readiness_fail_streak == 0:
                        self._episode_readiness_first_fail_step_idx = int(
                            len(self._episode_rows)
                        )
                        self._episode_readiness_first_fail_reason = str(step_ready_reason)
                    self._readiness_fail_streak += 1
                else:
                    self._readiness_fail_streak = 0

                # The action argument is ignored by the simulator while
                # _human_control_active=True (cursor drives the paddle).
                # We pass zeros for clarity.
                next_obs, task_reward, terminations, truncations, step_info = env.step(
                    np.zeros(2, dtype=np.float32)
                )
                recorded_action = np.asarray(
                    getattr(simulator, "_last_teleop_policy_action", np.zeros(2)),
                    dtype=np.float64,
                )

                stop_state = _classify_stop_event(env, step_info=step_info)
                readiness_fail_stop_now = bool(
                    self._readiness_fail_streak >= self._readiness_fail_window
                    and self._episode_readiness_first_fail_step_idx is not None
                )
                stop_now = bool(stop_state.active and step_ready) or readiness_fail_stop_now
                if stop_state.protective_stop:
                    delta_protective_stop_steps += 1
                    self._stop_flags.had_protective_stop = True
                if stop_state.controller_disconnected:
                    delta_controller_disconnect_steps += 1
                    self._stop_flags.had_controller_disconnect = True
                if stop_state.human_interrupt:
                    self._stop_flags.had_human_interrupt = True
                if readiness_fail_stop_now:
                    self._stop_flags.had_readiness_fail_estop = True
                if stop_now:
                    self._stop_flags.had_stop = True

                camera_frame = self._latest_camera_frame(env)
                if camera_frame is not None:
                    self._episode_images.append(camera_frame)
                else:
                    self._episode_camera_null_frames += 1

                self._episode_rows.append(
                    self._build_split_episode_row(
                        env=env,
                        action_xy=recorded_action,
                        episode_id=int(self._artifact_episode_id),
                        episode_step_idx=len(self._episode_rows),
                        protective_stop_active=stop_state.protective_stop,
                        controller_disconnected=stop_state.controller_disconnected,
                        reward=float(task_reward),
                        done=float(bool(terminations)),
                    )
                )

                obs_tensor = torch.as_tensor(self._obs, dtype=torch.float32, device=device)
                next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                action_tensor = torch.as_tensor(recorded_action, dtype=torch.float32, device=device)
                self._episode_trajectory.append_step(
                    obs=obs_tensor,
                    next_obs=next_obs_tensor,
                    action=action_tensor,
                    reward=torch.tensor(float(task_reward), dtype=torch.float32, device=device),
                    done=torch.tensor(float(bool(terminations)), dtype=torch.float32, device=device),
                    prev_action=action_tensor,
                )

                self._total_steps += 1
                delta_total_steps += 1
                self._obs = next_obs

                if self._on_step is not None:
                    try:
                        self._on_step(
                            len(self._episode_rows),
                            {
                                "reward": float(task_reward),
                                "stop_now": bool(stop_now),
                            },
                        )
                    except Exception:
                        # Banner failures must never take down an episode.
                        pass

                dones = bool(np.logical_or(terminations, truncations) or stop_now)
                if dones:
                    episode_success = (
                        bool(step_info.get("success", False))
                        if isinstance(step_info, dict)
                        else False
                    )
                    if isinstance(step_info, dict) and step_info.get("episode_end_type"):
                        episode_end_type = str(step_info["episode_end_type"])
                    else:
                        episode_end_type = "terminated" if bool(terminations) else "truncated"
                    if isinstance(step_info, dict) and step_info.get("episode_end_reason"):
                        episode_end_reason = str(step_info["episode_end_reason"])
                    else:
                        episode_end_reason = None
                    if isinstance(step_info, dict) and isinstance(
                        step_info.get("episode_end_reasons", []), list
                    ):
                        episode_end_reasons = list(step_info.get("episode_end_reasons", []))
                    else:
                        episode_end_reasons = []

                    artifact_label: str | None = None
                    if stop_now:
                        episode_end_type = stop_state.episode_end_type or episode_end_type
                        episode_end_reason = stop_state.episode_end_reason or episode_end_reason
                        episode_end_reasons = [str(episode_end_reason)] if episode_end_reason else []
                        artifact_label = stop_state.artifact_label
                    if self._stop_flags.had_readiness_fail_estop:
                        episode_end_type = "estop"
                        episode_end_reason = "collector_readiness_fail_5steps"
                        episode_end_reasons = [episode_end_reason]
                        artifact_label = "estop"

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
                        readiness_fail_dropped_steps=0,
                        stop_state_reason=stop_state.reason,
                        stop_state_artifact_label=artifact_label,
                        stop_state_episode_end_type=stop_state.episode_end_type,
                        stop_state_episode_end_reason=stop_state.episode_end_reason,
                        stop_now=stop_now,
                        stop_flags=StopFlagsSnapshot(**self._stop_flags.__dict__),
                    )
                    break
        finally:
            simulator.set_human_control(False)
            print("[teleop_run] human_control=OFF (episode end / finally)")

        episode_return = float(self._episode_trajectory.episode_return)
        episode_length = float(len(self._episode_trajectory.observations))
        episode_reward = float(
            torch.stack(self._episode_trajectory.rewards, dim=0).sum().item()
        )
        episode_estop_flag = (
            1.0
            if (
                self._stop_flags.had_protective_stop
                or self._stop_flags.had_readiness_fail_estop
            )
            else 0.0
        )

        metrics = EpisodeMetrics(
            episode_return=episode_return,
            episode_length=episode_length,
            episode_reward=episode_reward,
            episode_estop_flag=episode_estop_flag,
            puck_detection_latency_ms=[],
            model_inference_latency_ms=[],
            block_sleep_latency_ms=[],
            other_latency_ms=[],
            camera_null_frames=int(self._episode_camera_null_frames),
            delta_total_steps=int(delta_total_steps),
            delta_protective_stop_steps=int(delta_protective_stop_steps),
            delta_controller_disconnect_steps=int(delta_controller_disconnect_steps),
            delta_readiness_fail_steps=int(delta_readiness_fail_steps),
            delta_readiness_fail_estop_dropped_steps=0,
            delta_transition_hold_steps=0,
            delta_interval_primitive_env_steps=0,
            delta_interval_primitive_horizontal_env_steps=0,
            delta_human_interrupt_steps=int(self._stop_flags.had_human_interrupt),
            had_protective_stop=self._stop_flags.had_protective_stop,
            had_controller_disconnect=self._stop_flags.had_controller_disconnect,
            had_human_interrupt=self._stop_flags.had_human_interrupt,
        )

        result = PolicyEpisodeResult(
            trajectory=self._episode_trajectory,
            rows=self._episode_rows,
            images=self._episode_images,
            total_env_steps=int(delta_total_steps),
            terminal=terminal,
            metrics=metrics,
        )
        self._episode_trajectory = EpisodeTrajectory.empty()
        return result
