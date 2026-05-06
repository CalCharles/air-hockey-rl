"""Human-baseline teleoperation evaluation for the air-hockey paper.

Replaces the frozen actor in ``async_td3_real_eval.py`` with a human user
driving the paddle via the mouse, while preserving the rest of the eval
protocol so policy and human numbers are directly comparable:

  * same task config (``puck_juggle_upper_half_reward`` and friends),
  * same termination conditions (``terminate_on_puck_*``, max_timesteps),
  * same juggle counter (``count_juggles_from_rows``),
  * same per-episode + summary JSONL/JSON outputs.

Mode handoff is shown to the user in a dedicated cv2 status banner with
a thick colored border, so they always know whether the system is in
RESET, HANDOFF (countdown), USER CONTROL, or POST-EPISODE phase. The
mouse teleop window itself is the existing ``image`` window spawned by
``camera_callback`` — i.e. the same UX as ``scripts/real/teleoperate.py``.

Outputs land under the standard run-data dir created by
``_setup_run_data_dir`` (same layout as a real-world training run):

  ``eval_per_episode.jsonl``  — one row per *kept* episode (the eval set).
  ``eval_summary.json``       — aggregate stats + run metadata + per-episode.
  ``episode_summaries.jsonl`` — every episode (kept *and* discarded).
  ``run_events.jsonl``        — ``run_start`` / ``eval_done`` events.
  ``episode_hdf5/<bucket>/trajectory_data*.hdf5``.

NOTE on the reset path: the autonomous reset FSM used by the policy eval
drives the robot via ``env.step(action)``, which only honors the action
when ``simulator_params.control_mode != 'mouse'``. Because the human
must keep mouse control during episodes, we run the simulator in
``control_mode='mouse'`` end-to-end and use a *visual* reset phase
instead: the script asks the user to push the puck back into the upper
half with the paddle, watches the puck's x position, and auto-advances
once it has been in the upper half for a few consecutive frames. The
phase boundaries are fully signposted in the status banner so the
human protocol mirrors the policy protocol as closely as the mouse
control mode allows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import tyro
import yaml

from airhockey import AirHockeyEnv
from airhockey.sims.real.multiprocessing import NonBlockingConsole

from scripts.smooth_policy.amp_history.amp_training.td3.helper.episode_artifacts import (
    clean_episode_hdf5,
    save_split_episode_hdf5,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.juggle_counter import (
    count_juggles_from_rows,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_eval_stats import (
    compute_eval_aggregate,
    format_eval_summary_console,
    write_eval_summary_json,
)
from scripts.smooth_policy.amp_history.amp_training.td3.helper.run_event_log import (
    append_episode_summary,
    append_run_event,
    episode_summaries_path,
    run_data_dir_from_args,
    run_events_path,
)

from scripts.smooth_policy.amp_history.amp_training.td3.helper.real_td3_runtime import (
    Args,
    _build_args_file_defaults,
    _build_split_episode_row,
    _next_available_episode_id,
    _prepare_air_hockey_config,
    _setup_run_data_dir,
)


# ---------------------------------------------------------------------------
# Phase / banner constants. The status window draws a thick colored border
# matching ``BORDER_COLOR_BGR`` and a header line matching ``BANNER_TEXT`` so
# the user can identify the current phase from across the room without
# reading small text.
# ---------------------------------------------------------------------------


PHASE_RESET = "reset"
PHASE_HANDOFF = "handoff"
PHASE_USER = "user_control"
PHASE_POST = "post_episode"

BORDER_COLOR_BGR: Dict[str, Tuple[int, int, int]] = {
    PHASE_RESET: (255, 80, 0),     # blue
    PHASE_HANDOFF: (0, 200, 255),  # yellow / amber
    PHASE_USER: (0, 200, 0),       # green
    PHASE_POST: (0, 0, 220),       # red
}

BANNER_TEXT: Dict[str, str] = {
    PHASE_RESET: "RESET PHASE",
    PHASE_HANDOFF: "GET READY",
    PHASE_USER: "USER CONTROL",
    PHASE_POST: "EPISODE OVER",
}

INSTRUCTION_TEXT: Dict[str, str] = {
    PHASE_RESET: "Push the puck back to the UPPER HALF with the paddle.",
    PHASE_HANDOFF: "Place cursor on the paddle. Episode starts at 0.",
    PHASE_USER: "Juggle the puck. Episode ends on terminate/truncate.",
    PHASE_POST: "Episode complete. Preparing next reset...",
}


# ---------------------------------------------------------------------------
# Eval-specific args (mirrors ``EvalSpecificArgs`` in async_td3_real_eval.py).
# ---------------------------------------------------------------------------


@dataclass
class TeleopEvalSpecificArgs:
    """Teleop-eval-only knobs. All env-side flags live on ``Args``."""

    eval_episodes: int = 20
    eval_max_attempts: int = 0
    eval_summary_filename: str = "eval_summary.json"
    eval_per_episode_filename: str = "eval_per_episode.jsonl"

    # Reset phase: how long (seconds) the user has to push the puck back into
    # the upper half before the script will advance even if it never sees a
    # qualifying puck position. 0 = wait forever.
    reset_max_wait_s: float = 30.0

    # Reset phase: minimum seconds to stay in the BLUE banner before allowing
    # the auto-advance. Stops a stray puck reading from skipping the reset.
    reset_min_wait_s: float = 2.5

    # Reset phase: how many *consecutive* frames the puck must be observed in
    # the upper half (and not occluded) before auto-advancing to handoff.
    reset_puck_upper_half_frames: int = 20

    # Reset phase: an optional safety margin (m) past the table midpoint that
    # the puck must clear toward the far edge to count as "in the upper half".
    # Larger = stricter (puck must be deeper into the upper half).
    reset_upper_half_margin_m: float = 0.05

    # Handoff phase: seconds for the 3-2-1 countdown banner.
    handoff_countdown_s: float = 3.0

    # Post-episode phase: seconds the RED banner stays up before the next
    # reset phase begins. Gives the user a moment to recover before resetting.
    post_episode_pause_s: float = 1.5

    # Status banner window dimensions (px). Draw it large so the user can see
    # the current phase from across the room.
    banner_window_width: int = 720
    banner_window_height: int = 420
    banner_window_name: str = "teleop_status"


# ---------------------------------------------------------------------------
# Status banner. cv2 window owned by this process — separate from the existing
# ``image`` window spawned by ``camera_callback`` (which the user looks at to
# move their mouse / control the paddle).
# ---------------------------------------------------------------------------


class StatusBanner:
    """Draws the current phase in a large, color-bordered cv2 window.

    The user keeps this window in their peripheral vision while looking at
    the main ``image`` (camera teleop) window. The thick colored border is
    visible from across the room.
    """

    def __init__(
        self,
        *,
        window_name: str,
        width: int,
        height: int,
        border_thickness: int = 24,
    ) -> None:
        self._window_name = str(window_name)
        self._width = int(width)
        self._height = int(height)
        self._border = int(border_thickness)
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        except Exception:
            pass

    def draw(
        self,
        phase: str,
        *,
        episode_idx: int,
        target_episodes: int,
        attempts: int,
        countdown_s: float | None = None,
        episode_step: int | None = None,
        episode_juggles: int | None = None,
        episode_contacts: int | None = None,
        extra_lines: List[str] | None = None,
    ) -> None:
        color = BORDER_COLOR_BGR.get(phase, (200, 200, 200))
        canvas = np.full((self._height, self._width, 3), 30, dtype=np.uint8)
        cv2.rectangle(
            canvas,
            (0, 0),
            (self._width - 1, self._height - 1),
            color,
            thickness=self._border,
        )

        header = BANNER_TEXT.get(phase, phase.upper())
        if phase == PHASE_HANDOFF and countdown_s is not None and countdown_s > 0:
            header = f"GET READY — {int(np.ceil(countdown_s))}"
        self._put_text(canvas, header, y=110, scale=2.6, color=color, thickness=5)

        instruction = INSTRUCTION_TEXT.get(phase, "")
        if instruction:
            self._put_text(canvas, instruction, y=170, scale=0.7, color=(220, 220, 220), thickness=2)

        progress = f"Episode {min(episode_idx, target_episodes)}/{target_episodes}"
        self._put_text(canvas, progress, y=230, scale=0.9, color=(240, 240, 240), thickness=2)

        attempts_line = f"Attempts: {attempts}"
        self._put_text(canvas, attempts_line, y=265, scale=0.7, color=(180, 180, 180), thickness=1)

        live = []
        if episode_step is not None:
            live.append(f"step={episode_step}")
        if episode_juggles is not None:
            live.append(f"juggles={episode_juggles}")
        if episode_contacts is not None:
            live.append(f"contacts={episode_contacts}")
        if live:
            self._put_text(canvas, " ".join(live), y=305, scale=0.8, color=(220, 220, 220), thickness=2)

        if extra_lines:
            for i, line in enumerate(extra_lines[:3]):
                self._put_text(
                    canvas,
                    line,
                    y=345 + 28 * i,
                    scale=0.6,
                    color=(180, 180, 180),
                    thickness=1,
                )

        try:
            cv2.imshow(self._window_name, canvas)
            cv2.waitKey(1)
        except Exception:
            pass

    def _put_text(
        self,
        canvas: np.ndarray,
        text: str,
        *,
        y: int,
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(self._border + 8, (self._width - text_w) // 2)
        cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1 — auto-detected reset. Wait until the puck is back in the upper
# half (away from the paddle's bottom edge) for N consecutive frames.
# ---------------------------------------------------------------------------


def _puck_state_from_env(env: AirHockeyEnv) -> Tuple[float, float, bool]:
    """Return (puck_x, puck_y, occluded). Falls back to (NaN, NaN, True) on
    any introspection error so callers treat unknown state as 'not ready'.
    """
    simulator = getattr(env, "simulator", None)
    if simulator is None or not hasattr(simulator, "get_current_state"):
        return (float("nan"), float("nan"), True)
    try:
        state = simulator.get_current_state()
        puck = state["pucks"][0]
        pos = np.asarray(puck.get("position", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        occ_arr = np.asarray(puck.get("occluded", [0.0]), dtype=np.float64).reshape(-1)
        occluded = bool(occ_arr[0] > 0.5) if occ_arr.size > 0 else False
        return (float(pos[0]), float(pos[1]), occluded)
    except Exception:
        return (float("nan"), float("nan"), True)


def _puck_in_upper_half(
    env: AirHockeyEnv,
    *,
    margin_m: float,
) -> bool:
    """Puck x is closer to ``table_x_top`` than to ``table_x_bot``.

    The real-world env's ``table_x_top`` is the far edge and ``table_x_bot``
    is the paddle side. The midpoint divides the table; the upper-half
    condition adds ``margin_m`` toward ``table_x_top`` so a noisy puck
    barely past the centerline doesn't accidentally satisfy it.
    """
    table_x_top = float(getattr(env, "table_x_top", float("nan")))
    table_x_bot = float(getattr(env, "table_x_bot", float("nan")))
    if not (np.isfinite(table_x_top) and np.isfinite(table_x_bot)):
        return False
    puck_x, _, occluded = _puck_state_from_env(env)
    if occluded or not np.isfinite(puck_x):
        return False
    midpoint = 0.5 * (table_x_top + table_x_bot)
    if table_x_top < table_x_bot:
        return puck_x <= midpoint - float(margin_m)
    return puck_x >= midpoint + float(margin_m)


def _step_no_action(env: AirHockeyEnv) -> Tuple[Any, ...]:
    """Step with a zero action. In ``control_mode='mouse'`` the action is
    ignored — the simulator reads the user's cursor — so this just advances
    one tick of the simulator while the user is positioning the puck."""
    return env.step(np.zeros(2, dtype=np.float32))


def _run_reset_phase(
    *,
    env: AirHockeyEnv,
    banner: StatusBanner,
    nbc: NonBlockingConsole | None,
    eval_args: TeleopEvalSpecificArgs,
    episode_idx: int,
    target_episodes: int,
    attempts: int,
) -> str:
    """Show the BLUE banner until the puck is back in the upper half.

    Returns the reason the phase ended: ``"puck_in_upper_half"``,
    ``"max_wait_reached"``, or ``"user_skipped"``.
    """
    consecutive = 0
    needed = max(1, int(eval_args.reset_puck_upper_half_frames))
    deadline = time.time() + max(0.0, float(eval_args.reset_max_wait_s))
    min_until = time.time() + max(0.0, float(eval_args.reset_min_wait_s))

    print(
        f"[teleop_eval] reset phase start "
        f"(min_wait={eval_args.reset_min_wait_s:.1f}s, "
        f"max_wait={eval_args.reset_max_wait_s:.1f}s, "
        f"puck_upper_half_frames={needed}, "
        f"margin_m={eval_args.reset_upper_half_margin_m:.3f})"
    )
    while True:
        _step_no_action(env)
        puck_x, puck_y, occluded = _puck_state_from_env(env)
        in_upper = _puck_in_upper_half(env, margin_m=eval_args.reset_upper_half_margin_m)
        consecutive = consecutive + 1 if in_upper else 0

        extra = [
            f"puck=({puck_x:+.3f},{puck_y:+.3f}) occ={int(occluded)}",
            f"upper_half_consecutive={consecutive}/{needed}",
        ]
        banner.draw(
            PHASE_RESET,
            episode_idx=episode_idx,
            target_episodes=target_episodes,
            attempts=attempts,
            extra_lines=extra,
        )

        now = time.time()
        if now >= min_until and consecutive >= needed:
            return "puck_in_upper_half"
        if eval_args.reset_max_wait_s > 0 and now >= deadline:
            print(
                f"[teleop_eval] reset phase max wait reached "
                f"({eval_args.reset_max_wait_s:.1f}s); advancing anyway"
            )
            return "max_wait_reached"
        if nbc is not None:
            key = nbc.get_data()
            if key in (" ", "s", "S"):
                print("[teleop_eval] reset phase skipped by user (key)")
                return "user_skipped"
            if key in ("x", "X"):
                raise KeyboardInterrupt("user requested exit during reset")


def _run_handoff_phase(
    *,
    env: AirHockeyEnv,
    banner: StatusBanner,
    eval_args: TeleopEvalSpecificArgs,
    episode_idx: int,
    target_episodes: int,
    attempts: int,
) -> None:
    """YELLOW countdown banner. The user uses this window to move their
    cursor onto the paddle's current position so the policy/handoff doesn't
    cause a sudden jerk when user control begins."""
    countdown_total = max(0.5, float(eval_args.handoff_countdown_s))
    end = time.time() + countdown_total
    while True:
        _step_no_action(env)
        remaining = end - time.time()
        if remaining <= 0:
            return
        banner.draw(
            PHASE_HANDOFF,
            episode_idx=episode_idx,
            target_episodes=target_episodes,
            attempts=attempts,
            countdown_s=remaining,
        )


def _run_post_episode_pause(
    *,
    env: AirHockeyEnv,
    banner: StatusBanner,
    eval_args: TeleopEvalSpecificArgs,
    episode_idx: int,
    target_episodes: int,
    attempts: int,
    end_reason: str,
) -> None:
    """RED banner before the next reset begins."""
    end = time.time() + max(0.0, float(eval_args.post_episode_pause_s))
    extra = [f"end_reason={end_reason}"]
    while time.time() < end:
        _step_no_action(env)
        banner.draw(
            PHASE_POST,
            episode_idx=episode_idx,
            target_episodes=target_episodes,
            attempts=attempts,
            extra_lines=extra,
        )


# ---------------------------------------------------------------------------
# Phase 3 — user-controlled episode. Mirrors PolicyRunner.run_episode but
# without an actor: the action passed to env.step is a zero vector because
# in mouse mode the simulator reads the user's cursor instead.
# ---------------------------------------------------------------------------


@dataclass
class TeleopEpisodeResult:
    rows: list
    images: list
    episode_return: float
    episode_length: int
    episode_task_reward: float
    episode_success: bool
    episode_end_type: str | None
    episode_end_reason: str | None


def _run_user_episode(
    *,
    env: AirHockeyEnv,
    banner: StatusBanner,
    nbc: NonBlockingConsole | None,
    eval_args: TeleopEvalSpecificArgs,
    artifact_episode_id: int,
    episode_idx: int,
    target_episodes: int,
    attempts: int,
) -> TeleopEpisodeResult:
    """Step the env until terminations/truncations fire. Records HDF5 rows
    in the same split-schema format as the policy runner, so downstream
    analysis tools (juggle_counter, sysid, replay_real_in_sim) work
    unchanged on the resulting trajectory."""
    rows: list = []
    images: list = []
    episode_return = 0.0
    task_reward_sum = 0.0
    episode_step_idx = 0
    episode_success = False
    episode_end_type: str | None = None
    episode_end_reason: str | None = None
    last_juggle_refresh = 0
    cached_juggles = 0
    cached_contacts = 0

    user_aborted = False

    while True:
        _, task_reward, terminations, truncations, step_info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        episode_step_idx += 1
        recorded_action = getattr(
            getattr(env, "simulator", None),
            "_last_teleop_policy_action",
            np.zeros(2, dtype=np.float32),
        )
        rows.append(
            _build_split_episode_row(
                env=env,
                action_xy=np.asarray(recorded_action, dtype=np.float64),
                episode_id=int(artifact_episode_id),
                episode_step_idx=episode_step_idx - 1,
                protective_stop_active=False,
                controller_disconnected=False,
                task_reward=float(task_reward),
                motion_reward=0.0,
                done=float(bool(terminations)),
            )
        )
        latest_image = None
        sim_images = getattr(getattr(env, "simulator", None), "images", None)
        if isinstance(sim_images, list) and sim_images:
            latest_image = sim_images[-1]
        if latest_image is not None:
            images.append(latest_image)
        task_reward_f = float(task_reward)
        episode_return += task_reward_f
        task_reward_sum += task_reward_f

        # Refresh the live juggle/contact count every 10 steps so the banner
        # stays current without paying the full count_juggles_from_rows cost
        # on every step.
        if episode_step_idx - last_juggle_refresh >= 10:
            counts = count_juggles_from_rows(rows)
            cached_juggles = int(counts.n_juggles)
            cached_contacts = int(counts.n_contacts)
            last_juggle_refresh = episode_step_idx

        banner.draw(
            PHASE_USER,
            episode_idx=episode_idx,
            target_episodes=target_episodes,
            attempts=attempts,
            episode_step=episode_step_idx,
            episode_juggles=cached_juggles,
            episode_contacts=cached_contacts,
            extra_lines=[f"return={episode_return:+.2f}"],
        )

        if nbc is not None:
            key = nbc.get_data()
            if key in ("x", "X"):
                user_aborted = True
                break
            if key in ("q", "Q"):
                # Discard the current attempt without tagging it as a kept
                # episode; mirrors the existing teleop's 'q' = reset behavior.
                episode_end_type = "user_abort"
                episode_end_reason = "user_pressed_q"
                break

        if bool(terminations) or bool(truncations):
            episode_success = (
                bool(step_info.get("success", False)) if isinstance(step_info, dict) else False
            )
            if isinstance(step_info, dict):
                if step_info.get("episode_end_type") is not None:
                    episode_end_type = str(step_info.get("episode_end_type"))
                if step_info.get("episode_end_reason") is not None:
                    episode_end_reason = str(step_info.get("episode_end_reason"))
            if episode_end_type is None:
                episode_end_type = "terminated" if bool(terminations) else "truncated"
            break

    if user_aborted:
        raise KeyboardInterrupt("user pressed x")

    return TeleopEpisodeResult(
        rows=rows,
        images=images,
        episode_return=episode_return,
        episode_length=episode_step_idx,
        episode_task_reward=task_reward_sum,
        episode_success=episode_success,
        episode_end_type=episode_end_type,
        episode_end_reason=episode_end_reason,
    )


# ---------------------------------------------------------------------------
# Artifact saving. Lighter-weight than ``_save_episode_artifacts_and_pending_reset``
# (no GIF / camera video / pending reset flush — the FSM reset path doesn't
# exist in mouse mode) but writes the same on-disk HDF5 layout so the
# resulting trajectories slot into the existing analysis tooling.
# ---------------------------------------------------------------------------


def _episode_length_bucket(n_steps: int) -> str:
    if n_steps < 50:
        return "<50"
    if n_steps <= 100:
        return "50-100"
    if n_steps <= 200:
        return "100-200"
    return ">200"


def _save_teleop_episode(
    *,
    args: Args,
    result: TeleopEpisodeResult,
    episode_id: int,
    counters: dict,
) -> Tuple[bool, str, Path | None]:
    n_steps = len(result.rows)
    bucket_dir = (
        Path(args.episode_artifact_dir).expanduser().resolve() / _episode_length_bucket(n_steps)
    )
    artifact_path = save_split_episode_hdf5(
        output_dir=bucket_dir,
        episode_id=int(episode_id),
        episode_rows=result.rows,
        episode_images=result.images if result.images else None,
    )
    counters["episodes_saved"] += 1
    clean_result = clean_episode_hdf5(artifact_path, min_timesteps=1)
    if not clean_result.kept:
        if clean_result.reason == "short_episode":
            counters["episodes_removed_short"] += 1
        else:
            counters["episodes_removed_invalid"] += 1
        return False, str(clean_result.reason), None
    counters["successful_online_episodes_kept"] += 1
    return True, "kept", Path(clean_result.path)


def _append_teleop_per_episode_row(
    args: Args, eval_args: TeleopEvalSpecificArgs, record: Dict[str, Any]
) -> None:
    import json

    path = Path(run_data_dir_from_args(args)) / eval_args.eval_per_episode_filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, default=str)
            f.write("\n")
    except Exception:
        print(
            f"[teleop_eval] failed to append per-episode row to {path}:\n"
            f"{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Top-level eval loop.
# ---------------------------------------------------------------------------


def _force_teleop_mode(args: Args) -> None:
    """Lock down the training-side knobs so this script never accidentally
    mutates training state. The actor / replay / checkpointing paths are
    not exercised at all by this script, but we still flip the flags so a
    stray import doesn't trigger anything."""
    args.exploration_noise = 0.0
    args.exploration_primitive_chance = 0.0
    args.exploration_primitive_chance_start = 0.0
    args.collector_policy_stand_still = False
    args.enable_periodic_checkpointing = False
    args.load_replay_from_checkpoint = False


def _force_mouse_control(air_hockey_config: dict) -> None:
    """Force ``simulator_params.control_mode='mouse'`` and disable async
    rendering / wait-for-space so the human teleop UX matches the existing
    ``scripts/real/teleoperate.py`` exactly. Mutates ``air_hockey_config``
    (the *inner* air_hockey dict returned by ``_prepare_air_hockey_config``)
    in place.
    """
    sim = air_hockey_config.setdefault("simulator_params", {})
    if not isinstance(sim, dict):
        sim = {}
        air_hockey_config["simulator_params"] = sim
    sim["control_mode"] = "mouse"
    sim["wait_for_space_to_start"] = False
    sim["async_render_enabled"] = False
    sim["async_render_sim_view_enabled"] = False


def run_teleop_eval(
    *,
    args: Args,
    eval_args: TeleopEvalSpecificArgs,
) -> Dict[str, Any]:
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    collector_config = _prepare_air_hockey_config(config, seed=args.seed)
    _force_mouse_control(collector_config)
    env = AirHockeyEnv(collector_config)

    banner = StatusBanner(
        window_name=eval_args.banner_window_name,
        width=eval_args.banner_window_width,
        height=eval_args.banner_window_height,
    )

    # Initial env reset so simulator.images / mouse subprocess come up.
    env.reset(seed=None, write_traj=False)

    next_episode_file_id = _next_available_episode_id(args.episode_artifact_dir)
    target_kept = max(1, int(eval_args.eval_episodes))
    max_attempts = (
        int(eval_args.eval_max_attempts) if int(eval_args.eval_max_attempts) > 0 else None
    )

    counters: dict = {
        "episodes_saved": 0,
        "episodes_removed_short": 0,
        "episodes_removed_invalid": 0,
        "successful_online_episodes_kept": 0,
    }

    per_episode_records: List[Dict[str, Any]] = []
    total_attempts = 0
    cumulative_env_steps = 0
    eval_start_time = time.time()
    eval_started_iso = datetime.fromtimestamp(eval_start_time, tz=timezone.utc).isoformat()

    print(
        "[teleop_eval] starting human-baseline eval: "
        f"target_kept={target_kept} "
        f"max_attempts={max_attempts} "
        f"reset_puck_upper_half_frames={eval_args.reset_puck_upper_half_frames}"
    )
    print(
        "[teleop_eval] press 'q' during USER CONTROL to discard the current "
        "attempt and re-reset; press 'x' at any time to exit."
    )

    with NonBlockingConsole() as nbc:
        while len(per_episode_records) < target_kept:
            if max_attempts is not None and total_attempts >= max_attempts:
                print(
                    f"[teleop_eval] attempt cap reached "
                    f"({max_attempts}); kept "
                    f"{len(per_episode_records)}/{target_kept}; stopping."
                )
                break
            episode_idx_for_banner = len(per_episode_records) + 1

            # 1) RESET — wait for the user to push the puck back.
            reset_reason = _run_reset_phase(
                env=env,
                banner=banner,
                nbc=nbc,
                eval_args=eval_args,
                episode_idx=episode_idx_for_banner,
                target_episodes=target_kept,
                attempts=total_attempts,
            )

            # 2) HANDOFF countdown.
            _run_handoff_phase(
                env=env,
                banner=banner,
                eval_args=eval_args,
                episode_idx=episode_idx_for_banner,
                target_episodes=target_kept,
                attempts=total_attempts,
            )

            # 3) USER CONTROL — runs until env terminates / truncates.
            saved_episode_id = next_episode_file_id
            episode_start_wall_time = time.time()
            try:
                episode_result = _run_user_episode(
                    env=env,
                    banner=banner,
                    nbc=nbc,
                    eval_args=eval_args,
                    artifact_episode_id=saved_episode_id,
                    episode_idx=episode_idx_for_banner,
                    target_episodes=target_kept,
                    attempts=total_attempts + 1,
                )
            except KeyboardInterrupt:
                raise
            episode_end_wall_time = time.time()
            total_attempts += 1
            cumulative_env_steps += int(episode_result.episode_length)

            # 4) SAVE artifacts.
            episode_kept, clean_reason, artifact_path = _save_teleop_episode(
                args=args,
                result=episode_result,
                episode_id=saved_episode_id,
                counters=counters,
            )
            next_episode_file_id += 1

            episode_juggle_counts = count_juggles_from_rows(episode_result.rows)
            episode_juggle_success = bool(episode_juggle_counts.juggle_success)
            timestamp_iso = datetime.fromtimestamp(
                episode_end_wall_time, tz=timezone.utc
            ).isoformat()

            print(
                f"[teleop_eval] attempt={total_attempts} "
                f"episode_id={saved_episode_id} "
                f"len={episode_result.episode_length} "
                f"return={episode_result.episode_return:+.3f} "
                f"juggles={episode_juggle_counts.n_juggles} "
                f"contacts={episode_juggle_counts.n_contacts} "
                f"end={episode_result.episode_end_type}/{episode_result.episode_end_reason} "
                f"kept={episode_kept} reason={clean_reason} "
                f"reset={reset_reason}"
            )

            # 5) PER-EPISODE record (kept episodes only count toward
            #    eval_per_episode.jsonl). Every episode (kept or not) goes
            #    into episode_summaries.jsonl so the run is fully auditable.
            episode_summary = {
                "episode_id": int(saved_episode_id),
                "run_episode_index": int(total_attempts),
                "wall_time_s": float(episode_end_wall_time),
                "wall_time_s_start": float(episode_start_wall_time),
                "timestamp_iso": timestamp_iso,
                "kept": bool(episode_kept),
                "clean_reason": clean_reason,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
                "n_steps": int(len(episode_result.rows)),
                "episode_length": float(episode_result.episode_length),
                "episode_return": float(episode_result.episode_return),
                "episode_task_reward": float(episode_result.episode_task_reward),
                "episode_motion_reward": 0.0,
                "episode_success": bool(episode_result.episode_success),
                "episode_juggles": int(episode_juggle_counts.n_juggles),
                "episode_contacts": int(episode_juggle_counts.n_contacts),
                "episode_juggle_success": episode_juggle_success,
                "episode_estop_flag": 0.0,
                "had_protective_stop": False,
                "had_controller_disconnect": False,
                "readiness_fail_estop": False,
                "episode_end_type": episode_result.episode_end_type,
                "episode_end_reason": episode_result.episode_end_reason,
                "stop_state_artifact_label": None,
                "replay_partition": None,
                "episode_return_success_threshold": None,
                "total_steps": int(cumulative_env_steps),
                "actor_version": 0,
                "run_elapsed_total_s": float(time.time() - eval_start_time),
                "exploration_primitive_chance_runtime": 0.0,
                "reset_phase_reason": reset_reason,
                "control_mode": "mouse",
                "operator": "human",
            }
            append_episode_summary(args, episode_summary)

            if episode_kept:
                per_episode_records.append({
                    "episode_id": int(saved_episode_id),
                    "kept_index": int(len(per_episode_records) + 1),
                    "wall_time_s": float(episode_end_wall_time),
                    "timestamp_iso": timestamp_iso,
                    "n_steps": int(len(episode_result.rows)),
                    "episode_length": float(episode_result.episode_length),
                    "episode_return": float(episode_result.episode_return),
                    "episode_task_reward": float(episode_result.episode_task_reward),
                    "episode_motion_reward": 0.0,
                    "episode_juggles": int(episode_juggle_counts.n_juggles),
                    "episode_contacts": int(episode_juggle_counts.n_contacts),
                    "episode_juggle_success": episode_juggle_success,
                    "episode_success": bool(episode_result.episode_success),
                    "episode_estop_flag": 0.0,
                    "had_protective_stop": False,
                    "had_controller_disconnect": False,
                    "readiness_fail_estop": False,
                    "episode_end_type": episode_result.episode_end_type,
                    "episode_end_reason": episode_result.episode_end_reason,
                    "stop_state_artifact_label": None,
                    "artifact_path": str(artifact_path) if artifact_path is not None else None,
                })
                _append_teleop_per_episode_row(args, eval_args, per_episode_records[-1])

            # 6) POST-EPISODE pause before next reset.
            _run_post_episode_pause(
                env=env,
                banner=banner,
                eval_args=eval_args,
                episode_idx=episode_idx_for_banner,
                target_episodes=target_kept,
                attempts=total_attempts,
                end_reason=str(episode_result.episode_end_reason or episode_result.episode_end_type or "done"),
            )

    # ---- Aggregate + write summary -------------------------------------
    eval_finished_time = time.time()
    eval_finished_iso = datetime.fromtimestamp(eval_finished_time, tz=timezone.utc).isoformat()
    aggregate = compute_eval_aggregate(per_episode_records)

    run_meta: Dict[str, Any] = {
        "model_path": None,
        "config": str(args.config),
        "args_file": str(args.args_file) if args.args_file is not None else None,
        "train_args_file": str(args.train_args) if args.train_args is not None else None,
        "run_data_dir": str(run_data_dir_from_args(args)),
        "seed": int(args.seed),
        "n_target_episodes": int(target_kept),
        "n_attempts": int(total_attempts),
        "n_kept": int(len(per_episode_records)),
        "n_discarded": int(total_attempts - len(per_episode_records)),
        "started_iso": eval_started_iso,
        "finished_iso": eval_finished_iso,
        "elapsed_s": float(eval_finished_time - eval_start_time),
        "control_mode": "mouse",
        "operator": "human",
        "counters_at_finish": dict(counters),
    }
    summary_path = Path(run_data_dir_from_args(args)) / eval_args.eval_summary_filename
    write_eval_summary_json(
        summary_path,
        run_meta=run_meta,
        aggregate=aggregate,
        per_episode=per_episode_records,
    )
    print(f"[teleop_eval] wrote summary: {summary_path}")
    print(format_eval_summary_console(
        aggregate,
        n_target=target_kept,
        n_attempts=total_attempts,
        n_discarded=total_attempts - len(per_episode_records),
    ))

    banner.close()
    env.close()
    return {"run_meta": run_meta, "aggregate": aggregate, "per_episode": per_episode_records}


# ---------------------------------------------------------------------------
# CLI entry.
# ---------------------------------------------------------------------------


def _parse_teleop_eval_specific_args() -> TeleopEvalSpecificArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-attempts", type=int, default=0)
    parser.add_argument("--eval-summary-filename", type=str, default="eval_summary.json")
    parser.add_argument("--eval-per-episode-filename", type=str, default="eval_per_episode.jsonl")
    parser.add_argument("--reset-max-wait-s", type=float, default=30.0)
    parser.add_argument("--reset-min-wait-s", type=float, default=2.5)
    parser.add_argument("--reset-puck-upper-half-frames", type=int, default=20)
    parser.add_argument("--reset-upper-half-margin-m", type=float, default=0.05)
    parser.add_argument("--handoff-countdown-s", type=float, default=3.0)
    parser.add_argument("--post-episode-pause-s", type=float, default=1.5)
    parser.add_argument("--banner-window-width", type=int, default=720)
    parser.add_argument("--banner-window-height", type=int, default=420)
    parser.add_argument("--banner-window-name", type=str, default="teleop_status")
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return TeleopEvalSpecificArgs(
        eval_episodes=int(parsed.eval_episodes),
        eval_max_attempts=int(parsed.eval_max_attempts),
        eval_summary_filename=str(parsed.eval_summary_filename),
        eval_per_episode_filename=str(parsed.eval_per_episode_filename),
        reset_max_wait_s=float(parsed.reset_max_wait_s),
        reset_min_wait_s=float(parsed.reset_min_wait_s),
        reset_puck_upper_half_frames=int(parsed.reset_puck_upper_half_frames),
        reset_upper_half_margin_m=float(parsed.reset_upper_half_margin_m),
        handoff_countdown_s=float(parsed.handoff_countdown_s),
        post_episode_pause_s=float(parsed.post_episode_pause_s),
        banner_window_width=int(parsed.banner_window_width),
        banner_window_height=int(parsed.banner_window_height),
        banner_window_name=str(parsed.banner_window_name),
    )


def main(args: Args, eval_args: TeleopEvalSpecificArgs) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _force_teleop_mode(args)
    if eval_args.eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be > 0, got {eval_args.eval_episodes}")

    print(
        "[teleop_eval_run] "
        f"eval_episodes={eval_args.eval_episodes} "
        f"eval_max_attempts={eval_args.eval_max_attempts} "
        f"config={args.config} "
        f"seed={args.seed}"
    )
    print(
        "[run_event_log] writing per-run JSONL streams to:\n"
        f"    episodes : {episode_summaries_path(args)}\n"
        f"    events   : {run_events_path(args)}"
    )
    append_run_event(
        args,
        "run_start",
        run_data_dir=str(run_data_dir_from_args(args)),
        run_name=str(getattr(args, "run_name", "")),
        seed=int(getattr(args, "seed", 0)),
        config=str(getattr(args, "config", "")),
        mode="teleop_eval",
        eval_episodes=int(eval_args.eval_episodes),
        eval_max_attempts=int(eval_args.eval_max_attempts),
        operator="human",
        control_mode="mouse",
    )

    eval_outcome_reason = "completed"
    payload: Dict[str, Any] = {}
    try:
        payload = run_teleop_eval(args=args, eval_args=eval_args)
    except KeyboardInterrupt:
        print("[teleop_eval] interrupted by user; partial summary may be on disk.")
        eval_outcome_reason = "keyboard_interrupt"
    except BaseException:
        eval_outcome_reason = "exception"
        raise
    finally:
        append_run_event(
            args,
            "eval_done",
            reason=eval_outcome_reason,
            n_target_episodes=int(eval_args.eval_episodes),
            n_kept=int(len(payload.get("per_episode", []))) if payload else 0,
            n_attempts=int(payload.get("run_meta", {}).get("n_attempts", 0)) if payload else 0,
            elapsed_s=float(payload.get("run_meta", {}).get("elapsed_s", 0.0)) if payload else 0.0,
        )


if __name__ == "__main__":
    eval_args = _parse_teleop_eval_specific_args()

    temp_args = tyro.cli(Args)
    if temp_args.args_file is None:
        raise SystemExit(
            "async_td3_real_teleop_eval.py requires --args-file pointing to an "
            "online-behavior YAML (e.g. td3_online.yaml). The same args-file the "
            "policy eval uses is fine — exploration knobs and learner toggles are "
            "all forced off regardless."
        )
    mapped_defaults, applied_keys, ignored_keys = _build_args_file_defaults(temp_args.args_file)
    mapped_defaults["args_file"] = temp_args.args_file
    if temp_args.train_args is not None:
        mapped_defaults["train_args"] = temp_args.train_args
    default_args = Args(**mapped_defaults)

    args = tyro.cli(Args, default=default_args)
    print(f"[args_file] loaded defaults from: {args.args_file}")
    if applied_keys:
        print("[args_file] applied keys:", ", ".join(applied_keys))
    if ignored_keys:
        print("[args_file] ignored unsupported keys:", ", ".join(ignored_keys))
    _setup_run_data_dir(args, run_note="")
    main(args, eval_args)
