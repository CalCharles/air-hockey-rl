"""Human-baseline teleoperation evaluation for the air-hockey paper.

Mirrors the structure of ``async_td3_real_eval.py`` but swaps the frozen
TD3 actor for a human user driving the paddle via the mouse. The reset
between episodes is the same autonomous ``ResetPolicyFSM`` the policy
eval and training runs use — the human only takes over during the
policy-controlled portion of each episode.

Pieces shared with the policy eval (so the human-vs-policy numbers are
directly comparable):

  * task config (``puck_juggle_upper_half_reward`` and friends),
  * termination conditions (``terminate_on_puck_*``, max_timesteps),
  * juggle counter (``count_juggles_from_rows``),
  * autonomous reset (``ResetRunner`` + ``ResetPolicyFSM``),
  * artifact saving / per-episode JSONL / eval summary
    (``_save_episode_artifacts_and_pending_reset``,
    ``compute_eval_aggregate``, ``write_eval_summary_json``).

Pieces specific to the human-baseline run:

  * ``simulator.set_human_control(True/False)`` flipping inside
    ``TeleopRunner`` so the cursor drives the paddle during episodes
    and the FSM drives it during resets.
  * Phase banner overlay (RESET / HANDOFF / USER CONTROL / EPISODE
    OVER) drawn directly on the live ``image`` window by the camera
    subprocess, so the participant has exactly ONE cv2 window to look
    at and exactly ONE cv2 window to drag the cursor on. Earlier
    revisions used a separate ``teleop_status`` window which the
    participant naturally hovered over — cursor events never reached
    the ``image`` window's mouse callback, so the paddle would not
    follow.
  * Clean camera-callback monkey-patch that drops all overlay drawings
    (edges, region, target marker, paddle/puck circles) so the user
    sees a clean live camera feed underneath the phase banner.
  * Optional ``--participant-id`` that nests the run-data dir under a
    sanitized subfolder for user-study session grouping.

Outputs land under the standard run-data dir (matching the policy eval):

  ``eval_per_episode.jsonl``  — one row per *kept* episode (the eval set).
  ``eval_summary.json``       — aggregate stats + run metadata + per-episode.
  ``episode_summaries.jsonl`` — every episode (kept *and* discarded).
  ``reset_summaries.jsonl``   — every reset event (success/failure).
  ``run_events.jsonl``        — ``run_start`` / ``eval_done`` events.
  ``episode_hdf5/<bucket>/trajectory_data*.hdf5``.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import tyro
import yaml

from airhockey import AirHockeyEnv

from scripts.td3.helper.episode_artifacts import (
    clean_episode_hdf5,
    save_split_episode_hdf5,
)
from scripts.td3.helper.human_interrupt import (
    HumanInterruptListener,
)
from scripts.td3.helper.juggle_counter import (
    count_juggles_from_rows,
)
from scripts.td3.helper.real_eval_stats import (
    compute_eval_aggregate,
    format_eval_summary_console,
    write_eval_summary_json,
)
from scripts.td3.helper.real_reset_runner import (
    ResetKind,
    ResetRunner,
    StopFlags,
    pick_reset_kind,
)
from scripts.td3.helper.real_teleop_runner import (
    TeleopRunner,
)
from scripts.td3.helper.run_event_log import (
    append_episode_summary,
    append_run_event,
    episode_summaries_path,
    reset_summaries_path,
    run_data_dir_from_args,
    run_events_path,
)
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM

from scripts.td3.helper.real_td3_runtime import (
    Args,
    _build_args_file_defaults,
    _build_split_episode_row,
    _extract_primitive_state_tensors,
    _latest_camera_frame,
    _next_available_episode_id,
    _prepare_air_hockey_config,
    _setup_run_data_dir,
    _simulator_step_readiness,
    install_quiet_print_filter,
)
from scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real import (
    _save_episode_artifacts_and_pending_reset,
)


# ---------------------------------------------------------------------------
# Phase / banner constants. The camera-subprocess overlay draws a colored
# border matching ``BORDER_COLOR_BGR`` and a header line matching
# ``BANNER_TEXT`` directly on top of the live camera image so the
# participant can identify the current phase from across the room without
# reading small text — and so they only have ONE cv2 window to interact
# with (the same one their cursor needs to be over for paddle control).
# ---------------------------------------------------------------------------


PHASE_RESET = "reset"
PHASE_HANDOFF = "handoff"
PHASE_USER = "user_control"
PHASE_POST = "post_episode"

# Internal id mapping shared between the main process (``_publish_phase``)
# and the camera subprocess (``_draw_phase_overlay``). The shared array is
# integer-typed for portability across fork/spawn, so we encode the phase
# as an int and decode it back into the string label inside the subprocess.
_PHASE_TO_ID: Dict[str, int] = {
    PHASE_RESET: 0,
    PHASE_HANDOFF: 1,
    PHASE_USER: 2,
    PHASE_POST: 3,
}
_ID_TO_PHASE: Dict[int, str] = {v: k for k, v in _PHASE_TO_ID.items()}

# ``phase_state`` shared-array layout (multiprocessing.Array("i", N)):
#   [0] phase_id (see _PHASE_TO_ID)
#   [1] episode_idx
#   [2] target_episodes
#   [3] attempts
#   [4] countdown_ds (countdown seconds * 10, integer)
#   [5] episode_step
#   [6] juggles
#   [7] contacts
#   [8] version_counter (incremented on every publish — handy when reading
#       the prints to see whether updates are propagating)
_PHASE_STATE_LEN = 9

BORDER_COLOR_BGR: Dict[str, Tuple[int, int, int]] = {
    PHASE_RESET: (255, 80, 0),     # blue
    PHASE_HANDOFF: (0, 200, 255),  # yellow / amber
    PHASE_USER: (0, 200, 0),       # green
    PHASE_POST: (0, 0, 220),       # red
}

BANNER_TEXT: Dict[str, str] = {
    PHASE_RESET: "ROBOT RESETTING",
    PHASE_HANDOFF: "GET READY",
    PHASE_USER: "USER CONTROL",
    PHASE_POST: "EPISODE OVER",
}

INSTRUCTION_TEXT: Dict[str, str] = {
    PHASE_RESET: "Stand back. The robot is running its reset routine.",
    PHASE_HANDOFF: "Place cursor on the paddle. Episode starts at 0.",
    PHASE_USER: "Juggle the puck. Episode ends on terminate/truncate.",
    PHASE_POST: "Episode complete. Preparing next reset...",
}


# ---------------------------------------------------------------------------
# Clean camera callback (teleop-only). Drop-in replacement for
# ``airhockey.sims.real.control_parameters.camera_callback`` that skips ALL
# of the original overlay drawing (robot edge limits, region overlays,
# target marker, green/red puck/paddle markers) so participants see a clean
# camera feed.
#
# In addition: this version draws the phase / progress banner directly on
# the ``image`` window (read from a shared ``phase_state`` array). That
# unifies the participant's view — there is exactly one cv2 window to
# look at and one cv2 window to drag the cursor on. The previous
# two-window setup put the GET READY / USER CONTROL banner in a separate
# window, which the participant naturally hovered over, so cursor events
# never reached the ``image`` window's mouse callback and the paddle
# would not follow.
#
# Same shared-array protocol as the original so AirHockeyReal reads mouse
# / puck positions unchanged. The factory ``_make_teleop_camera_callback``
# closes over ``phase_state`` so we can keep the on-disk callback signature
# (which ``start_callbacks`` invokes positionally) untouched. Patched into
# ``airhockey.sims.air_hockey_real`` from ``run_teleop_eval`` BEFORE
# ``AirHockeyEnv`` is constructed; effect is scoped to the teleop eval
# process. All other entrypoints (training, policy eval, plain
# teleoperate.py) keep the original overlays.
# ---------------------------------------------------------------------------


def _draw_phase_overlay(showdst: np.ndarray, phase_state) -> None:
    """Render the phase border + header line on top of ``showdst``.

    Reads the latest published values from ``phase_state``. Designed to be
    cheap (a single rectangle + 1-3 putText calls) so it doesn't slow down
    the camera-detection loop.
    """
    try:
        phase_id = int(phase_state[0])
        episode_idx = int(phase_state[1])
        target_episodes = int(phase_state[2])
        attempts = int(phase_state[3])
        countdown_ds = int(phase_state[4])
        episode_step = int(phase_state[5])
        juggles = int(phase_state[6])
        contacts = int(phase_state[7])
    except Exception:
        return

    phase = _ID_TO_PHASE.get(phase_id, PHASE_RESET)
    color = BORDER_COLOR_BGR.get(phase, (200, 200, 200))
    h, w = showdst.shape[:2]
    border_thick = 10
    cv2.rectangle(showdst, (0, 0), (w - 1, h - 1), color, thickness=border_thick)

    header = BANNER_TEXT.get(phase, phase.upper())
    if phase == PHASE_HANDOFF and countdown_ds > 0:
        header = f"GET READY -- {int(np.ceil(countdown_ds / 10.0))}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    header_scale = 1.2
    header_thickness = 3
    (text_w, text_h), _ = cv2.getTextSize(header, font, header_scale, header_thickness)
    bar_top = border_thick
    bar_bottom = bar_top + text_h + 28
    cv2.rectangle(
        showdst,
        (border_thick, bar_top),
        (w - border_thick - 1, bar_bottom),
        (30, 30, 30),
        thickness=-1,
    )
    text_x = max(border_thick + 12, (w - text_w) // 2)
    text_y = bar_top + text_h + 6
    cv2.putText(
        showdst,
        header,
        (text_x, text_y),
        font,
        header_scale,
        color,
        header_thickness,
        cv2.LINE_AA,
    )

    stats = f"Ep {min(max(episode_idx, 0), max(target_episodes, 1))}/{max(target_episodes, 1)}  attempts={attempts}"
    if phase == PHASE_USER:
        stats += f"  step={episode_step}  juggles={juggles}  contacts={contacts}"
    stats_y = bar_bottom + 22
    if 0 < stats_y < h - border_thick:
        cv2.putText(
            showdst,
            stats,
            (border_thick + 12, stats_y),
            font,
            0.6,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    instruction = INSTRUCTION_TEXT.get(phase, "")
    inst_y = stats_y + 22
    if instruction and 0 < inst_y < h - border_thick:
        cv2.putText(
            showdst,
            instruction,
            (border_thick + 12, inst_y),
            font,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def _make_teleop_camera_callback(phase_state):
    """Build the camera-subprocess callback with ``phase_state`` captured.

    The returned function preserves the positional signature
    ``start_callbacks`` calls with, so we can bind it to
    ``airhockey.sims.air_hockey_real.camera_callback`` without touching
    the env construction path.
    """

    def _teleop_camera_callback(
        shared_array,
        save_image_check,
        puck_array,
        paddle_info,
        target_info,
        region_info,
        goal_info,
        lims=None,
        edge_lims=None,
        puck_detector=None,
        puck_detector_kwargs=None,
        puck_radius=0.03175,
        region_x_offset=1.0,
    ):
        import imageio

        from airhockey.sims.real import control_parameters as _cp
        from airhockey.sims.real.image_detection import find_red_hockey_puck

        if puck_detector is None:
            puck_detector = find_red_hockey_puck
        detector_kwargs = puck_detector_kwargs if puck_detector_kwargs is not None else {}

        # Realize the cv2 window once up-front so the very first
        # ``setMouseCallback`` succeeds. Without this, on some cv2
        # builds the first call in the loop is a silent no-op while
        # HighGUI lazily creates the window, and the mouse-callback
        # plumbing only kicks in on the next iteration.
        try:
            cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("image", _cp.move_event)
        except Exception:
            pass

        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Diagnostic: print mousepos + shared cursor every ~3 s (60 frames
        # at the ~20 Hz camera loop) so we can verify cursor events are
        # actually firing on this window. Cheap; safe to leave on while
        # debugging the participant-view setup.
        frame_idx = 0
        diag_every = 60
        while True:
            ret, image = cap.read()
            save_image_id = save_image_check[0] == 1
            showdst, save_image = _cp.homography_transform(image, get_save=save_image_id)
            if save_image_id:
                imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", save_image)
            puck = puck_detector(showdst, rotate=False, **detector_kwargs)

            _draw_phase_overlay(showdst, phase_state)

            cv2.imshow("image", showdst)
            cv2.setMouseCallback("image", _cp.move_event)

            puck_array[0] = puck[0]
            puck_array[1] = puck[1]
            puck_array[2] = puck[2]
            shared_array[0] = _cp.mousepos[0] * _cp.visual_downscale_constant
            shared_array[1] = _cp.mousepos[1] * _cp.visual_downscale_constant
            shared_array[2] = _cp.mousepos[2] * _cp.visual_downscale_constant

            frame_idx += 1
            if frame_idx % diag_every == 0:
                print(
                    "[teleop_cam_dbg] "
                    f"frame={frame_idx} "
                    f"mousepos={tuple(_cp.mousepos)} "
                    f"shared_cursor=({shared_array[0]:.1f},{shared_array[1]:.1f},{shared_array[2]:.1f}) "
                    f"phase_id={int(phase_state[0])} version={int(phase_state[8])}"
                )

            cv2.waitKey(1)

    return _teleop_camera_callback


def _install_clean_camera_callback(phase_state) -> None:
    """Monkey-patch ``camera_callback`` inside ``airhockey.sims.air_hockey_real``
    to point at the closure built by ``_make_teleop_camera_callback``. Must
    run BEFORE the ``AirHockeyEnv`` is constructed; ``start_callbacks``
    reads the binding off the module by name when spawning the camera
    subprocess.
    """
    import airhockey.sims.air_hockey_real as _ahr

    _ahr.camera_callback = _make_teleop_camera_callback(phase_state)


def _publish_phase(
    phase_state,
    phase: str,
    *,
    episode_idx: int,
    target_episodes: int,
    attempts: int,
    countdown_s: float | None = None,
    episode_step: int | None = None,
    episode_juggles: int | None = None,
    episode_contacts: int | None = None,
) -> None:
    """Push the current phase + progress numbers into the shared array so
    the camera subprocess can render the overlay on its next frame.

    Cheap (a handful of int writes); safe to call from the orchestrator's
    main loop and from the per-step ``_on_step`` hook.
    """
    if phase_state is None:
        return
    try:
        phase_state[0] = int(_PHASE_TO_ID.get(phase, 0))
        phase_state[1] = int(episode_idx)
        phase_state[2] = int(target_episodes)
        phase_state[3] = int(attempts)
        phase_state[4] = int(round(max(0.0, float(countdown_s) if countdown_s is not None else 0.0) * 10.0))
        phase_state[5] = int(episode_step) if episode_step is not None else 0
        phase_state[6] = int(episode_juggles) if episode_juggles is not None else 0
        phase_state[7] = int(episode_contacts) if episode_contacts is not None else 0
        phase_state[8] = (int(phase_state[8]) + 1) % 1_000_000
    except Exception:
        # Phase publishing must never take down the orchestrator.
        pass


# ---------------------------------------------------------------------------
# Eval-specific args. Mirrors ``EvalSpecificArgs`` in async_td3_real_eval.py
# plus a few teleop-only knobs (banner geometry + handoff/post-episode
# pacing + participant id).
# ---------------------------------------------------------------------------


@dataclass
class TeleopEvalSpecificArgs:
    """Teleop-eval-only knobs. All env-side flags live on ``Args``."""

    eval_episodes: int = 20
    eval_max_attempts: int = 0
    eval_summary_filename: str = "eval_summary.json"
    eval_per_episode_filename: str = "eval_per_episode.jsonl"

    # User-study participant id. When set, the run-data dir is nested under a
    # subfolder named after the (sanitized) id, so each subject's session
    # lands at ``<data_root_dir>/<participant_id>/data_<TIMESTAMP>/`` and the
    # ``eval_summary.json`` is automatically grouped by participant.
    participant_id: str = ""

    # Handoff phase: seconds for the 3-2-1 countdown banner before USER
    # CONTROL begins. The reset FSM has already finished by this point;
    # this gives the participant a moment to put their cursor on the paddle.
    handoff_countdown_s: float = 3.0

    # Post-episode phase: seconds the RED banner stays up before the next
    # reset begins. Gives the user a moment to recover.
    post_episode_pause_s: float = 1.5

    # NOTE: the banner_window_* fields below are retained for CLI
    # compatibility but are unused now that the phase banner is drawn
    # directly on the camera-subprocess ``image`` window (single-window
    # design — see module docstring). Leaving them defaulted so any
    # existing wrapper scripts that pass --banner-window-* still parse.
    banner_window_width: int = 720
    banner_window_height: int = 420
    banner_window_name: str = "teleop_status"

    # When True (default), suppress noisy per-step debug prints from the
    # training-side helpers (mirrors ``--quiet`` on the policy eval).
    quiet: bool = True


# ---------------------------------------------------------------------------
# Eval-mode arg overrides. Counterpart to ``_force_eval_mode`` in
# ``async_td3_real_eval.py``; the teleop eval doesn't run an actor at all,
# but the same training-side flags need to stay off so a stray import path
# doesn't accidentally tick replay / checkpoints / exploration.
# ---------------------------------------------------------------------------


# Canonical real-world sim config the teleop eval always uses. Same task
# (``puck_juggle_upper_half_reward``), same termination conditions, same
# ``obs_type=history`` as the policy eval, and ``simulator: real`` so the
# real robot drives. The args-file's ``config:`` field is ignored because
# in practice ``td3_online.yaml`` (and friends) ship with a Box2D path
# for sim-mirror runs — the teleop user study is fixed-purpose, so we
# pin the sim config the same way ``teleoperate.py`` defaults ``--cfg``
# to ``configs/real_configs/mouse_config.yaml``.
TELEOP_EVAL_SIM_CONFIG = "configs/real_configs/rollout_td3_config.yaml"


def _force_teleop_mode(args: Args) -> None:
    args.exploration_noise = 0.0
    args.exploration_primitive_chance = 0.0
    args.exploration_primitive_chance_start = 0.0
    args.collector_policy_stand_still = False
    args.enable_periodic_checkpointing = False
    args.load_replay_from_checkpoint = False
    # Pin the sim config to the canonical real-world rollout YAML.
    # ``td3_online.yaml`` (the recommended ``--args-file`` for the policy
    # path) sets ``config:`` to a Box2D file for sim-mirror runs; the
    # teleop user study only ever runs on the real robot, so we override
    # whatever the args-file provided.
    if args.config != TELEOP_EVAL_SIM_CONFIG:
        print(
            "[teleop_eval] overriding args.config "
            f"{args.config!r} -> {TELEOP_EVAL_SIM_CONFIG!r} "
            "(canonical real-world rollout config; matches policy eval)."
        )
        args.config = TELEOP_EVAL_SIM_CONFIG


def _force_teleop_sim_params(air_hockey_config: dict) -> None:
    """Set the env-side knobs the teleop eval needs.

    Keep ``control_mode='mouse'`` so the camera+mouse subprocess and cv2
    window come up. Force ``simulator='real'`` belt-and-suspenders so a
    future change to ``TELEOP_EVAL_SIM_CONFIG`` can't accidentally route
    the teleop eval through Box2D. Disable any space-bar prompts and
    async render UIs so the run is unattended apart from the human
    episode itself. The runtime cursor-vs-action toggle is then handled
    by ``simulator.set_human_control(...)`` inside ``TeleopRunner`` and
    the orchestrator.
    """
    air_hockey_config["simulator"] = "real"
    sim = air_hockey_config.setdefault("simulator_params", {})
    if not isinstance(sim, dict):
        sim = {}
        air_hockey_config["simulator_params"] = sim
    sim["control_mode"] = "mouse"
    sim["wait_for_space_to_start"] = False
    sim["async_render_enabled"] = False
    sim["async_render_sim_view_enabled"] = False


# ---------------------------------------------------------------------------
# Eval loop. Mirrors ``run_eval`` in ``async_td3_real_eval.py``.
# ---------------------------------------------------------------------------


def run_teleop_eval(
    args: Args,
    eval_args: TeleopEvalSpecificArgs,
) -> Dict[str, Any]:
    np.random.seed(args.seed)
    device = torch.device(args.collector_device)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    collector_config = _prepare_air_hockey_config(config, seed=args.seed)
    _force_teleop_sim_params(collector_config)
    sim_params = collector_config.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["transition_hold_steps_on_estop_enter"] = int(
            args.transition_hold_steps_post_estop_enter
        )
        sim_params["transition_hold_steps_on_estop_clear"] = int(
            args.transition_hold_steps_post_estop_clear
        )
        sim_params["transition_hold_steps_on_safety_rearm"] = int(
            args.transition_hold_steps_post_safety_rearm
        )

    # Shared phase / progress array consumed by the camera subprocess to
    # render the banner directly on the live ``image`` window. Created
    # BEFORE ``_install_clean_camera_callback`` so the closure inside the
    # patched callback captures it; passed by reference into the
    # subprocess so updates from the orchestrator propagate without any
    # IPC plumbing of our own.
    phase_state = multiprocessing.Array("i", _PHASE_STATE_LEN)
    _publish_phase(
        phase_state,
        PHASE_RESET,
        episode_idx=0,
        target_episodes=max(1, int(eval_args.eval_episodes)),
        attempts=0,
    )

    _install_clean_camera_callback(phase_state)
    env = AirHockeyEnv(collector_config)
    simulator = getattr(env, "simulator", None)
    if simulator is None or not hasattr(simulator, "set_human_control"):
        raise RuntimeError(
            "Teleop eval requires AirHockeyReal with set_human_control(); "
            "make sure the env was constructed with simulator='real' and "
            "control_mode='mouse'."
        )
    # FSM drives the robot; flip cursor control off until the first user
    # episode begins. ``TeleopRunner.run_episode`` toggles it on for each
    # episode and back off in its ``finally`` block.
    simulator.set_human_control(False)
    print("[teleop_run] human_control=OFF (initial; FSM mode)")

    # Human interrupt listener — same singleton pipeline async_td3_real
    # uses. Start before the startup reset so the operator can hit 's'
    # to STOP / 'r' to RESET at any time, including during the FSM. The
    # listener feeds ``human_interrupt_state``, which ``_classify_stop_event``
    # reads inside both ``TeleopRunner`` (truncates the episode) and
    # ``run_reset_fsm`` (waits for clear before continuing). Daemon
    # thread, so a KeyboardInterrupt / unhandled exception still tears
    # it down with the process even though we don't wrap it in
    # try/finally — same pattern as ``async_td3_real``.
    human_interrupt_listener = HumanInterruptListener()
    human_interrupt_listener.start()

    # Reset runner. Identical to the policy eval — autonomous FSM between
    # human episodes.
    next_reset_file_id = _next_available_episode_id(args.reset_artifact_dir)
    reset_rng = np.random.default_rng(args.seed)
    reset_runner = ResetRunner(
        env,
        device=device,
        reset_rng=reset_rng,
        reset_policy_fsm_cls=ResetPolicyFSM,
        build_split_episode_row=_build_split_episode_row,
        latest_camera_frame=_latest_camera_frame,
        extract_primitive_state_tensors=_extract_primitive_state_tensors,
    )
    pending_reset_artifact = None

    target_kept = max(1, int(eval_args.eval_episodes))
    max_attempts = (
        int(eval_args.eval_max_attempts) if int(eval_args.eval_max_attempts) > 0 else None
    )

    counters: dict = {
        "reset_fsm_steps_total": 0,
        "protective_stop_episodes": 0,
        "protective_stop_steps": 0,
        "controller_disconnect_episodes": 0,
        "controller_disconnect_steps": 0,
        "readiness_fail_steps_total": 0,
        "readiness_fail_estop_episodes": 0,
        "readiness_fail_estop_dropped_steps_total": 0,
        "episodes_saved": 0,
        "episodes_removed_short": 0,
        "episodes_removed_invalid": 0,
        "episodes_gif_generated": 0,
        "episodes_gif_failed": 0,
        "episodes_camera_video_generated": 0,
        "episodes_camera_video_failed": 0,
        "successful_online_episodes_kept": 0,
    }

    # Startup reset.
    _publish_phase(
        phase_state,
        PHASE_RESET,
        episode_idx=1,
        target_episodes=target_kept,
        attempts=0,
    )
    startup_result = reset_runner.run(
        kind=ResetKind.STARTUP,
        artifact_episode_id=next_reset_file_id,
        episode_had_stop_flags=StopFlags(),
        episode_end_wall_time=time.time(),
        pending_reset_artifact=pending_reset_artifact,
        next_reset_file_id=next_reset_file_id,
    )
    pending_reset_artifact = startup_result.pending_reset_artifact
    next_reset_file_id = startup_result.next_reset_file_id
    counters["reset_fsm_steps_total"] += int(startup_result.total_fsm_steps)

    # Teleop runner. Owns the cursor-vs-action toggle and the per-step
    # episode-row append. The on_step hook just keeps the banner live.
    teleop_runner = TeleopRunner(
        env,
        device=device,
        build_split_episode_row=_build_split_episode_row,
        latest_camera_frame=_latest_camera_frame,
        readiness_fn=_simulator_step_readiness,
    )
    teleop_runner.seed_initial(startup_result.obs)

    next_episode_file_id = _next_available_episode_id(args.episode_artifact_dir)
    per_episode_records: List[Dict[str, Any]] = []
    total_attempts = 0
    eval_start_time = time.time()
    eval_started_iso = datetime.fromtimestamp(eval_start_time, tz=timezone.utc).isoformat()

    print(
        "[teleop_eval] starting human-baseline eval: "
        f"target_kept={target_kept} "
        f"max_attempts={max_attempts}"
    )

    # Wire a per-step banner update into the teleop runner so the
    # USER CONTROL banner reflects step / juggles live. Juggle counts
    # are recomputed every 10 steps and cached between refreshes so
    # the banner stays informative without paying full-trajectory
    # juggle-counting cost on every step.
    live_counts = {"juggles": 0, "contacts": 0}

    def _on_step(step_idx: int, _meta: dict) -> None:
        if step_idx == 1 or step_idx % 10 == 0:
            counts = count_juggles_from_rows(teleop_runner._episode_rows)  # type: ignore[attr-defined]
            live_counts["juggles"] = int(counts.n_juggles)
            live_counts["contacts"] = int(counts.n_contacts)
        _publish_phase(
            phase_state,
            PHASE_USER,
            episode_idx=len(per_episode_records) + 1,
            target_episodes=target_kept,
            attempts=total_attempts + 1,
            episode_step=step_idx,
            episode_juggles=live_counts["juggles"],
            episode_contacts=live_counts["contacts"],
        )

    teleop_runner._on_step = _on_step  # type: ignore[attr-defined]

    while len(per_episode_records) < target_kept:
        if max_attempts is not None and total_attempts >= max_attempts:
            print(
                f"[teleop_eval] attempt cap reached ({max_attempts}); kept "
                f"{len(per_episode_records)}/{target_kept}; stopping."
            )
            break
        episode_idx_for_banner = len(per_episode_records) + 1

        # 1. USER CONTROL — handoff is immediate (no countdown). The
        # reset FSM has just finished and the participant is already
        # watching the ``image`` window, so we drop straight into the
        # green USER CONTROL phase to keep the session snappy.
        # 2. USER CONTROL — TeleopRunner flips ``set_human_control(True)``
        # for the duration of the episode and back to False on exit. An
        # e-stop / controller disconnect / human-interrupt during the
        # episode is detected by ``_classify_stop_event`` inside the
        # runner; the runner truncates the episode and surfaces the
        # stop flags via ``result.terminal.stop_flags`` so ``pick_reset_kind``
        # routes through the ``HARD_WITH_FSM`` path on the next reset —
        # same as ``async_td3_real`` / ``async_td3_real_eval``.
        teleop_runner.set_artifact_episode_id(next_episode_file_id)
        episode_start_wall_time = time.time()
        result = teleop_runner.run_episode()
        episode_end_wall_time = time.time()
        total_attempts += 1
        counters["protective_stop_steps"] += result.metrics.delta_protective_stop_steps
        counters["controller_disconnect_steps"] += (
            result.metrics.delta_controller_disconnect_steps
        )
        counters["readiness_fail_steps_total"] += result.metrics.delta_readiness_fail_steps
        counters["readiness_fail_estop_dropped_steps_total"] += (
            result.metrics.delta_readiness_fail_estop_dropped_steps
        )
        if result.metrics.had_protective_stop:
            counters["protective_stop_episodes"] += 1
        if result.metrics.had_controller_disconnect:
            counters["controller_disconnect_episodes"] += 1
        if result.terminal.readiness_fail_estop:
            counters["readiness_fail_estop_episodes"] += 1

        if result.terminal.stop_flags.had_stop:
            print(
                "[teleop_eval] STOP detected during USER CONTROL: "
                f"protective_stop={int(result.metrics.had_protective_stop)} "
                f"controller_disconnect={int(result.metrics.had_controller_disconnect)} "
                f"readiness_fail_estop={int(result.terminal.readiness_fail_estop)} "
                f"human_interrupt={int(result.metrics.had_human_interrupt)} "
                "→ truncating episode and routing to HARD_WITH_FSM reset."
            )

        # 3. SAVE artifacts (HDF5 + GIF + camera video) and flush
        # pending reset. Same helper as the policy eval — identical
        # on-disk layout.
        saved_episode_id = next_episode_file_id
        (
            next_episode_file_id,
            episode_kept,
            clean_reason,
            artifact_path,
        ) = _save_episode_artifacts_and_pending_reset(
            args=args,
            result=result,
            next_episode_file_id=next_episode_file_id,
            pending_reset_artifact=pending_reset_artifact,
            latency_output_dir=None,
            counters=counters,
        )
        pending_reset_artifact = None

        episode_juggle_counts = count_juggles_from_rows(result.rows)
        timestamp_iso = datetime.fromtimestamp(
            episode_end_wall_time, tz=timezone.utc
        ).isoformat()

        print(
            f"[teleop_eval] attempt={total_attempts} "
            f"episode_id={saved_episode_id} "
            f"len={int(result.metrics.episode_length)} "
            f"return={result.metrics.episode_return:+.3f} "
            f"juggles={episode_juggle_counts.n_juggles} "
            f"contacts={episode_juggle_counts.n_contacts} "
            f"end={result.terminal.episode_end_type}/{result.terminal.episode_end_reason} "
            f"kept={episode_kept} reason={clean_reason}"
        )

        if episode_kept:
            record = {
                "episode_id": int(saved_episode_id),
                "kept_index": int(len(per_episode_records) + 1),
                "wall_time_s": float(episode_end_wall_time),
                "wall_time_s_start": float(episode_start_wall_time),
                "timestamp_iso": timestamp_iso,
                "n_steps": int(len(result.rows)),
                "episode_length": float(result.metrics.episode_length),
                "episode_return": float(result.metrics.episode_return),
                "episode_task_reward": float(result.metrics.episode_task_reward),
                "episode_motion_reward": float(result.metrics.episode_motion_reward),
                "episode_juggles": int(episode_juggle_counts.n_juggles),
                "episode_contacts": int(episode_juggle_counts.n_contacts),
                "episode_juggle_success": bool(episode_juggle_counts.juggle_success),
                "episode_success": bool(result.terminal.episode_success),
                "episode_estop_flag": float(result.metrics.episode_estop_flag),
                "had_protective_stop": bool(result.metrics.had_protective_stop),
                "had_controller_disconnect": bool(result.metrics.had_controller_disconnect),
                "readiness_fail_estop": bool(result.terminal.readiness_fail_estop),
                "episode_end_type": result.terminal.episode_end_type,
                "episode_end_reason": result.terminal.episode_end_reason,
                "stop_state_artifact_label": result.terminal.stop_state_artifact_label,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
            }
            per_episode_records.append(record)
            _append_teleop_per_episode_row(args, eval_args, record)

        append_episode_summary(
            args,
            {
                "episode_id": int(saved_episode_id),
                "run_episode_index": int(total_attempts),
                "wall_time_s": float(episode_end_wall_time),
                "wall_time_s_start": float(episode_start_wall_time),
                "timestamp_iso": timestamp_iso,
                "kept": bool(episode_kept),
                "clean_reason": clean_reason,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
                "n_steps": int(len(result.rows)),
                "episode_length": float(result.metrics.episode_length),
                "episode_return": float(result.metrics.episode_return),
                "episode_task_reward": float(result.metrics.episode_task_reward),
                "episode_motion_reward": float(result.metrics.episode_motion_reward),
                "episode_success": bool(result.terminal.episode_success),
                "episode_juggles": int(episode_juggle_counts.n_juggles),
                "episode_contacts": int(episode_juggle_counts.n_contacts),
                "episode_juggle_success": bool(episode_juggle_counts.juggle_success),
                "episode_estop_flag": float(result.metrics.episode_estop_flag),
                "had_protective_stop": bool(result.metrics.had_protective_stop),
                "had_controller_disconnect": bool(result.metrics.had_controller_disconnect),
                "readiness_fail_estop": bool(result.terminal.readiness_fail_estop),
                "episode_end_type": result.terminal.episode_end_type,
                "episode_end_reason": result.terminal.episode_end_reason,
                "stop_state_artifact_label": result.terminal.stop_state_artifact_label,
                "replay_partition": None,
                "episode_return_success_threshold": None,
                "total_steps": int(teleop_runner.total_steps),
                "actor_version": 0,
                "run_elapsed_total_s": float(time.time() - eval_start_time),
                "exploration_primitive_chance_runtime": 0.0,
                "control_mode": "mouse",
                "operator": "human",
            },
        )

        # 4. POST-EPISODE pause before the next reset begins.
        post_end = time.time() + max(0.0, float(eval_args.post_episode_pause_s))
        while time.time() < post_end:
            _publish_phase(
                phase_state,
                PHASE_POST,
                episode_idx=episode_idx_for_banner,
                target_episodes=target_kept,
                attempts=total_attempts,
                episode_step=int(result.metrics.episode_length),
                episode_juggles=int(episode_juggle_counts.n_juggles),
                episode_contacts=int(episode_juggle_counts.n_contacts),
            )
            time.sleep(0.05)

        # 5. AUTONOMOUS RESET (same path as the policy eval).
        # TeleopRunner's finally block already toggled cursor control off
        # when the episode ended (including on e-stop). Re-asserting it
        # here is belt-and-suspenders: even if a future code path bypasses
        # the runner's finally, the FSM never runs while cursor mode is
        # active. ``pick_reset_kind`` returns ``HARD_WITH_FSM`` whenever
        # ``had_stop`` is set so e-stops always get the full hard-reset
        # treatment (env.reset() — which waits for the operator to clear
        # the protective stop — followed by the FSM).
        simulator.set_human_control(False)
        print("[teleop_run] human_control=OFF (post-episode; back to FSM mode)")
        _publish_phase(
            phase_state,
            PHASE_RESET,
            episode_idx=episode_idx_for_banner + 1,
            target_episodes=target_kept,
            attempts=total_attempts,
        )
        kind = pick_reset_kind(
            total_attempts,
            StopFlags(
                had_stop=result.terminal.stop_flags.had_stop,
                had_protective_stop=result.terminal.stop_flags.had_protective_stop,
                had_controller_disconnect=result.terminal.stop_flags.had_controller_disconnect,
            ),
        )
        reset_result = reset_runner.run(
            kind=kind,
            artifact_episode_id=next_reset_file_id,
            episode_had_stop_flags=StopFlags(
                had_stop=result.terminal.stop_flags.had_stop,
                had_protective_stop=result.terminal.stop_flags.had_protective_stop,
                had_controller_disconnect=result.terminal.stop_flags.had_controller_disconnect,
            ),
            episode_end_wall_time=episode_end_wall_time,
            pending_reset_artifact=pending_reset_artifact,
            next_reset_file_id=next_reset_file_id,
        )
        counters["reset_fsm_steps_total"] += reset_result.total_fsm_steps
        pending_reset_artifact = reset_result.pending_reset_artifact
        next_reset_file_id = reset_result.next_reset_file_id
        teleop_runner.seed_after_reset(reset_result.obs)

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
        "participant_id": str(eval_args.participant_id),
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

    human_interrupt_listener.stop()
    env.close()
    return {"run_meta": run_meta, "aggregate": aggregate, "per_episode": per_episode_records}


# ---------------------------------------------------------------------------
# JSONL helper local to this entrypoint — mirrors the policy eval's
# ``_append_eval_per_episode_row`` so analysts can find the eval set with
# one ls.
# ---------------------------------------------------------------------------


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
# CLI entry. Matches async_td3_real_eval.py's argparse + tyro split so the
# same args-file format works for both entry points.
# ---------------------------------------------------------------------------


def _parse_teleop_eval_specific_args() -> TeleopEvalSpecificArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-max-attempts", type=int, default=0)
    parser.add_argument("--eval-summary-filename", type=str, default="eval_summary.json")
    parser.add_argument(
        "--eval-per-episode-filename", type=str, default="eval_per_episode.jsonl"
    )
    parser.add_argument("--handoff-countdown-s", type=float, default=3.0)
    parser.add_argument("--post-episode-pause-s", type=float, default=1.5)
    parser.add_argument("--banner-window-width", type=int, default=720)
    parser.add_argument("--banner-window-height", type=int, default=420)
    parser.add_argument("--banner-window-name", type=str, default="teleop_status")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Restore noisy per-step debug prints from training-side helpers.",
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return TeleopEvalSpecificArgs(
        eval_episodes=int(parsed.eval_episodes),
        eval_max_attempts=int(parsed.eval_max_attempts),
        eval_summary_filename=str(parsed.eval_summary_filename),
        eval_per_episode_filename=str(parsed.eval_per_episode_filename),
        handoff_countdown_s=float(parsed.handoff_countdown_s),
        post_episode_pause_s=float(parsed.post_episode_pause_s),
        banner_window_width=int(parsed.banner_window_width),
        banner_window_height=int(parsed.banner_window_height),
        banner_window_name=str(parsed.banner_window_name),
        # ``participant_id`` is NOT a CLI flag — it's prompted at runtime
        # in ``__main__`` so each session uses the same command and only
        # the operator-typed id changes.
        quiet=not bool(parsed.verbose),
    )


def main(args: Args, eval_args: TeleopEvalSpecificArgs) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if eval_args.quiet:
        prefixes, substrs = install_quiet_print_filter()
        print(
            "[teleop_quiet] suppressing per-step/per-reset debug prints "
            "(pass --verbose to restore). "
            f"prefixes={list(prefixes)} substrings={list(substrs)}"
        )
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
        f"    resets   : {reset_summaries_path(args)}\n"
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
        participant_id=str(eval_args.participant_id),
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


def _sanitize_participant_id(raw: str) -> str:
    """Filesystem-safe participant id: keep [A-Za-z0-9_.-], collapse the rest
    into single underscores, strip leading/trailing underscores. Empty input
    yields empty output."""
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", raw.strip()).strip("_")
    return cleaned


def _prompt_participant_id(data_root_dir: str) -> str:
    """Interactive prompt for the participant id.

    Loops until the operator types a non-empty id that survives sanitization.
    Returns the sanitized id; the caller nests ``args.data_root_dir`` under
    it so each session lands at ``<data_root_dir>/<participant_id>/data_<TS>/``.

    Echoes the parent directory so the operator sees where the session is
    going to land before confirming.
    """
    print(
        "\n=== teleop user-study session ==="
        f"\nparent directory: {data_root_dir}"
        "\nallowed characters: letters / digits / _ / - / ."
        "\n(other characters get collapsed into underscores)"
    )
    while True:
        try:
            raw = input("participant id (e.g., p01, alice, S07): ").strip()
        except EOFError:
            raise SystemExit(
                "no participant id entered (stdin closed). "
                "Re-run in an interactive terminal."
            )
        sanitized = _sanitize_participant_id(raw)
        if not sanitized:
            print(
                "[teleop_eval] participant id sanitized to empty — please "
                "include at least one letter, digit, underscore, hyphen, or dot."
            )
            continue
        if sanitized != raw:
            print(f"[teleop_eval] participant id sanitized: {raw!r} -> {sanitized!r}")
        confirm = input(f"confirm participant id {sanitized!r}? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return sanitized
        print("[teleop_eval] re-entering participant id...")


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

    # Always prompt for the participant id at runtime so the launch
    # command stays identical across sessions and only --data-root-dir
    # selects the parent directory. Each session lands under
    # ``<data_root_dir>/<participant_id>/data_<TIMESTAMP>/``.
    sanitized_participant = _prompt_participant_id(args.data_root_dir)
    eval_args.participant_id = sanitized_participant
    args.data_root_dir = str(Path(args.data_root_dir) / sanitized_participant)
    print(
        f"[teleop_eval] participant_id={sanitized_participant!r} -> "
        f"data_root_dir={args.data_root_dir}"
    )

    _setup_run_data_dir(args, run_note="")
    main(args, eval_args)
