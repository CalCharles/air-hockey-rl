"""Real-world TD3 entrypoint — sole runnable async-real script.

Orchestrator that drives ``PolicyRunner`` (one episode at a time) and
``ResetRunner`` (one reset at a time, including the four-way kind table).

Sibling modules under ``helper/`` own everything else: the shared runtime
library (``helper/real_td3_runtime.py``: ``Args`` / ``TrainArgs`` /
``LearnerRuntimeState``, args-file parsing, checkpoint helpers,
episode/replay utilities, the synchronous learner step) and the per-concern
runners (``real_policy_runner``, ``real_reset_runner``,
``real_transition_hold``, ``real_stop_state``,
``real_collector_metrics``, ``real_warm_start``, ``real_episode_buffers``,
…). This file is the only ``__main__`` for real-world async TD3 runs.

See ``notes/scratch/async_td3_real_modularization_plan.md`` for the
contracts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.td3.helper.episode_artifacts import (
    clean_episode_hdf5,
    generate_episode_camera_video,
    generate_episode_gif,
    generate_episode_homography_gif,
    save_split_episode_hdf5,
)
from scripts.td3.helper.shared_replay import SharedTD3Replay
from scripts.td3.helper.real_collector_factories import (
    build_primitive_exploration_selector_for_real_collector,
)
from scripts.td3.helper.real_collector_metrics import (
    ROLLING_WINDOW_SIZES,
    compute_rolling_window_metrics_multi,
    format_rolling_window_console_line,
    rolling_mean,
    update_stats_dict_rolling_windows,
    write_rolling_windows_tensorboard_scalars,
)
from scripts.td3.helper.run_event_log import (
    append_episode_summary,
    append_reset_summary,
    append_run_event,
    episode_summaries_path,
    reset_summaries_path,
    run_data_dir_from_args,
    run_events_path,
    utc_timestamps,
)
from scripts.td3.helper.juggle_counter import (
    JUGGLE_SUCCESS_THRESHOLD,
    JuggleCounts,
    count_juggles_from_rows,
)
from scripts.td3.helper.real_warm_start import (
    _warm_start_replay_from_hdf5,
)
from scripts.td3.helper.real_policy_runner import PolicyRunner
from scripts.td3.helper.real_reset_runner import (
    ResetKind,
    ResetRunner,
    StopFlags,
    pick_reset_kind,
)
from scripts.td3.helper.real_transition_hold import (
    RolloutContext,
    TransitionHoldState,
    normalize_transition_last_action_mode,
)
from scripts.td3.helper.human_interrupt import (
    HumanInterruptListener,
    human_interrupt_state,
)
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM
from scripts.real.rollout_reset_policy_hybrid import (
    ResetPolicyHybridFSM,
    build_juggle_actor,
)

# Single-symbol toggle for the reset FSM. The legacy programmatic-strike
# ``ResetPolicyFSM`` is the canonical default; the policy-handoff hybrid is
# kept available for opt-in but is currently less reliable in practice. To
# enable the hybrid path, change this to ``ResetPolicyHybridFSM``.
_DEFAULT_RESET_FSM_CLS = ResetPolicyFSM


# Minimum timesteps required for a *policy* episode to be retained by
# ``clean_episode_hdf5``. Episodes shorter than this — e.g. a 1-step puck-
# hits-bottom or a quick puck_passed_paddle — are noisy and uninformative for
# both critic learning and offline analysis, so we discard them at the
# validator. Reset trajectories use a separate ``min_timesteps=1`` since a
# clean reset can legitimately be a handful of steps.
#
# Owned exclusively by this run file (intentionally not surfaced via
# ``Args`` / args-files / configs) so the contract is fixed across all
# invocations: training (``async_td3_real.py``) and eval
# (``async_td3_real_eval.py``, which imports the shared helper from here)
# share this threshold by construction.
EPISODE_MIN_TIMESTEPS = 50

# Shared init / teardown / learner / config helpers live in the runtime
# library under helper/. This file is the orchestrator only.
from scripts.td3.helper.real_td3_runtime import (
    Args,
    LearnerRuntimeState,
    ROLLING_PERF_WINDOW_EPISODES,
    TrainArgs,
    _add_episode_to_shared_replay,
    _bucketed_output_dir,
    _build_args_file_defaults,
    _build_collector_actor,
    _coerce_float_list,
    _copy_to_stop_dir,
    _env_timing_info,
    _finalize_sync_learner_state,
    _init_sync_learner_state,
    _latest_camera_frame,
    _load_train_args,
    _load_training_state_checkpoint,
    _next_available_episode_id,
    _normalize_replay_source_priority,
    _prepare_air_hockey_config,
    _prompt_optional_run_note,
    _reset_output_dir,
    _reset_primitive_rollout_state,
    _run_sync_learner_iteration,
    _safe_nonnegative_ms,
    _setup_run_data_dir,
    _simulator_step_readiness,
    _stop_output_dir,
    _build_split_episode_row,
    _write_latency_profile_episode,
    augment_policy_observation,
    deterministic_actor_action,
    install_quiet_print_filter,
    primitive_exploration_chance_for_step,
)


def _save_episode_artifacts_and_pending_reset(
    *,
    args: Args,
    result,
    next_episode_file_id: int,
    pending_reset_artifact,
    latency_output_dir: Path | None,
    counters: dict,
    min_timesteps: int = EPISODE_MIN_TIMESTEPS,
) -> tuple[int, bool, str, Path | None]:
    """HDF5 + GIF + camera video + pending reset flush — matches L1953–L2154.

    Mutates ``counters`` (dict of named tally counters); returns
    ``(next_episode_file_id, episode_kept, clean_reason, artifact_path)``:

    * ``episode_kept`` — True iff ``clean_episode_hdf5`` accepted the
      trajectory. Callers use it to suppress training, replay, and
      performance-metric updates for trajectories that the validator
      discarded.
    * ``clean_reason`` — ``"kept"`` when the file was retained, otherwise
      the rejection reason from ``clean_episode_hdf5`` (e.g.,
      ``"short_episode"``, ``"non_finite_pose"``).
    * ``artifact_path`` — absolute path to the retained HDF5 trajectory
      file, or ``None`` if it was discarded (the file is unlinked by
      the validator in that case).

    ``min_timesteps`` is the floor passed to ``clean_episode_hdf5``;
    callers (the eval entrypoint) override it for tasks whose episodes
    routinely end in fewer than the juggle-default 50 steps.

    The pending-reset flush is included here because it lives in the
    same artifact-save block in the source.
    """
    n_episode_steps = len(result.rows)
    n_camera_frames = len(result.images)
    has_camera_images = n_camera_frames > 0

    if n_camera_frames == 0 and result.metrics.camera_null_frames > 0:
        sim_images_len = -1
        sim_cap = "N/A"
        print(
            f"[collector] WARNING: zero camera frames captured this episode. "
            f"simulator.images len={sim_images_len} simulator.cap={sim_cap}"
        )

    if args.enable_latency_profiling and latency_output_dir is not None:
        try:
            latency_json_path, latency_plot_path, latency_summary = _write_latency_profile_episode(
                output_dir=latency_output_dir,
                episode_id=next_episode_file_id,
                puck_detection_ms=result.metrics.puck_detection_latency_ms,
                model_inference_ms=result.metrics.model_inference_latency_ms,
                block_sleep_ms=result.metrics.block_sleep_latency_ms,
                other_ms=result.metrics.other_latency_ms,
                hist_bins=args.latency_profile_hist_bins,
            )
            bucket_summary = latency_summary["buckets"]
            print(
                "[latency] "
                f"episode_id={next_episode_file_id} "
                f"puck_p50={bucket_summary['puck_detection_ms']['median_ms']:.3f} "
                f"model_p50={bucket_summary['model_inference_ms']['median_ms']:.3f} "
                f"sleep_p50={bucket_summary['block_sleep_ms']['median_ms']:.3f} "
                f"other_p50={bucket_summary['other_ms']['median_ms']:.3f} "
                f"json={latency_json_path} plot={latency_plot_path}"
            )
        except Exception:
            print(
                f"[latency] episode_id={next_episode_file_id} "
                f"latency output FAILED:\n{traceback.format_exc()}"
            )

    artifact_path = save_split_episode_hdf5(
        output_dir=_bucketed_output_dir(args.episode_artifact_dir, n_episode_steps),
        episode_id=next_episode_file_id,
        episode_rows=result.rows,
        episode_images=result.images if has_camera_images else None,
    )
    counters["episodes_saved"] += 1

    clean_result = clean_episode_hdf5(artifact_path, min_timesteps=int(min_timesteps))
    episode_stop_artifact_label = result.terminal.stop_state_artifact_label
    if not clean_result.kept:
        print(
            f"[collector] episode_id={next_episode_file_id} "
            f"removed: reason={clean_result.reason} timesteps={clean_result.timesteps}"
        )
        if clean_result.reason == "short_episode":
            counters["episodes_removed_short"] += 1
        else:
            counters["episodes_removed_invalid"] += 1
    else:
        counters["successful_online_episodes_kept"] += 1
        counters["_clean_path"] = clean_result.path
        if episode_stop_artifact_label is not None:
            stop_copy_path = _copy_to_stop_dir(
                clean_result.path,
                _stop_output_dir(args.episode_artifact_dir, episode_stop_artifact_label),
            )
            print(
                f"[collector] episode_id={next_episode_file_id} "
                f"{episode_stop_artifact_label} HDF5 copied to {stop_copy_path}"
            )
        if args.enable_episode_gif:
            try:
                gif_path = generate_episode_gif(
                    episode_hdf5_path=clean_result.path,
                    gif_root=_bucketed_output_dir(args.episode_gif_dir, n_episode_steps),
                    fps=args.episode_gif_fps,
                    max_frames=(args.episode_gif_max_frames if args.episode_gif_max_frames > 0 else None),
                    subsample=args.episode_gif_subsample,
                    require_puck=args.episode_gif_require_puck,
                )
                counters["episodes_gif_generated"] += 1
                if episode_stop_artifact_label is not None:
                    stop_gif_path = _copy_to_stop_dir(
                        gif_path,
                        _stop_output_dir(args.episode_gif_dir, episode_stop_artifact_label),
                    )
                    print(
                        f"[collector] episode_id={next_episode_file_id} "
                        f"{episode_stop_artifact_label} GIF copied to {stop_gif_path}"
                    )
            except Exception:
                counters["episodes_gif_failed"] += 1
                print(
                    f"[collector] episode_id={next_episode_file_id} "
                    f"GIF generation FAILED:\n{traceback.format_exc()}"
                )
        if args.enable_episode_camera_video:
            try:
                camera_video_path = generate_episode_camera_video(
                    episode_hdf5_path=clean_result.path,
                    video_root=_bucketed_output_dir(
                        args.episode_camera_video_dir,
                        n_episode_steps,
                    ),
                    fps=args.episode_camera_video_fps,
                    max_frames=(
                        args.episode_camera_video_max_frames
                        if args.episode_camera_video_max_frames > 0
                        else None
                    ),
                    subsample=args.episode_camera_video_subsample,
                    codec=args.episode_camera_video_codec,
                )
                counters["episodes_camera_video_generated"] += 1
                if episode_stop_artifact_label is not None:
                    stop_camera_video_path = _copy_to_stop_dir(
                        camera_video_path,
                        _stop_output_dir(
                            args.episode_camera_video_dir,
                            episode_stop_artifact_label,
                        ),
                    )
                    print(
                        f"[collector] episode_id={next_episode_file_id} "
                        f"{episode_stop_artifact_label} camera video copied to "
                        f"{stop_camera_video_path}"
                    )
                print(
                    f"[collector] episode_id={next_episode_file_id} camera video OK"
                )
            except Exception:
                counters["episodes_camera_video_failed"] += 1
                print(
                    f"[collector] episode_id={next_episode_file_id} "
                    f"camera video FAILED:\n{traceback.format_exc()}"
                )
        else:
            print(
                f"[collector] episode_id={next_episode_file_id} "
                f"camera video SKIPPED (enable_episode_camera_video=False)"
            )

        if args.enable_episode_homography_gif:
            try:
                with h5py.File(clean_result.path, "r") as h5_file:
                    has_goal_dataset = "goal" in h5_file
                    has_train_img = "train_img" in h5_file
                if has_goal_dataset and has_train_img:
                    homography_gif_path = generate_episode_homography_gif(
                        episode_hdf5_path=clean_result.path,
                        gif_root=args.episode_homography_gif_dir,
                        fps=args.episode_homography_gif_fps,
                        max_frames=(
                            args.episode_homography_gif_max_frames
                            if args.episode_homography_gif_max_frames > 0
                            else None
                        ),
                        subsample=args.episode_homography_gif_subsample,
                    )
                    counters["episodes_homography_gif_generated"] += 1
                    if episode_stop_artifact_label is not None:
                        stop_homography_gif_path = _copy_to_stop_dir(
                            homography_gif_path,
                            _stop_output_dir(
                                args.episode_homography_gif_dir,
                                episode_stop_artifact_label,
                            ),
                        )
                        print(
                            f"[collector] episode_id={next_episode_file_id} "
                            f"{episode_stop_artifact_label} homography GIF copied to "
                            f"{stop_homography_gif_path}"
                        )
                    print(
                        f"[collector] episode_id={next_episode_file_id} "
                        f"homography GIF OK -> {homography_gif_path}"
                    )
            except Exception:
                counters["episodes_homography_gif_failed"] += 1
                print(
                    f"[collector] episode_id={next_episode_file_id} "
                    f"homography GIF FAILED:\n{traceback.format_exc()}"
                )

    # Pending reset artifact flush — same block as L2106–2146.
    if pending_reset_artifact is not None:
        reset_camera_frames = len(pending_reset_artifact.images)
        has_reset_images = reset_camera_frames > 0
        print(
            "[collector_reset_artifact] "
            f"episode_id={pending_reset_artifact.episode_id} "
            f"partition={pending_reset_artifact.partition} "
            f"reason={pending_reset_artifact.done_reason} "
            f"steps={pending_reset_artifact.step_count} "
            f"camera_frames={reset_camera_frames} "
            f"null_frames={pending_reset_artifact.camera_null_frames} "
            f"flush_after_policy_episode={next_episode_file_id}"
        )
        reset_artifact_path = save_split_episode_hdf5(
            output_dir=_reset_output_dir(
                args.reset_artifact_dir,
                pending_reset_artifact.partition,
                pending_reset_artifact.step_count,
            ),
            episode_id=pending_reset_artifact.episode_id,
            episode_rows=pending_reset_artifact.rows,
            episode_images=pending_reset_artifact.images if has_reset_images else None,
        )
        reset_clean_result = clean_episode_hdf5(reset_artifact_path, min_timesteps=1)
        if not reset_clean_result.kept:
            print(
                "[collector_reset_artifact] "
                f"episode_id={pending_reset_artifact.episode_id} "
                f"removed: reason={reset_clean_result.reason} "
                f"timesteps={reset_clean_result.timesteps}"
            )
        else:
            print(
                "[collector_reset_artifact] "
                f"episode_id={pending_reset_artifact.episode_id} "
                f"saved to {reset_clean_result.path}"
            )
        # One JSONL row per reset event (success and failure) so the reset
        # timeline is queryable in the same shape as episode_summaries.
        # Wall time is the moment the reset FSM finished (captured by
        # ResetRunner), not flush time, so resets and episodes interleave
        # correctly when sorted by wall_time_s.
        wall_time_s_end = float(getattr(pending_reset_artifact, "wall_time_s_end", 0.0))
        timestamp_iso_end = (
            datetime.fromtimestamp(wall_time_s_end, tz=timezone.utc).isoformat()
            if wall_time_s_end > 0.0
            else None
        )
        append_reset_summary(
            args,
            {
                "reset_id": int(pending_reset_artifact.episode_id),
                "wall_time_s_end": wall_time_s_end,
                "timestamp_iso_end": timestamp_iso_end,
                "partition": pending_reset_artifact.partition,
                "done_reason": pending_reset_artifact.done_reason,
                "step_count": int(pending_reset_artifact.step_count),
                "camera_null_frames": int(pending_reset_artifact.camera_null_frames),
                "kept": bool(reset_clean_result.kept),
                "clean_reason": str(reset_clean_result.reason),
                "artifact_path": str(reset_clean_result.path) if reset_clean_result.kept else None,
                "flush_after_policy_episode": int(next_episode_file_id),
            },
        )

    return (
        next_episode_file_id + 1,
        bool(clean_result.kept),
        str(clean_result.reason),
        Path(clean_result.path) if clean_result.kept else None,
    )


def _periodic_log(
    *,
    args: Args,
    writer: SummaryWriter,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    learner_state: LearnerRuntimeState,
    primitive_selector,
    transition_hold: TransitionHoldState,
    total_steps: int,
    total_episodes: int,
    counters: dict,
    rolling_state: dict,
    elapsed_offset_s: float,
    collector_start_time: float,
    episodic_returns: list,
    episodic_lengths: list,
    success_rates: list,
    episodic_juggles: list,
    episodic_contacts: list,
    interval_state: dict,
    last_log_time: float,
    episode_return_success_threshold: float,
) -> float:
    """Periodic stats / TB log block (matches source L2294–L2478).

    Returns the new ``last_log_time``.
    """
    now = time.time()
    snapshot = replay.state_snapshot()
    elapsed_s = max(0.0, elapsed_offset_s + (now - collector_start_time))
    rolling_multi = compute_rolling_window_metrics_multi(
        reward_values=rolling_state["reward"],
        episode_length_values=rolling_state["length"],
        estop_episode_flags=rolling_state["estop"],
        episode_return_values=rolling_state["return"],
        episode_juggles_values=rolling_state["juggles"],
        episode_contacts_values=rolling_state["contacts"],
    )
    rolling50_m = rolling_multi[50]
    stats["collector_steps"] = float(total_steps)
    stats["collector_total_steps"] = float(total_steps)
    stats["collector_episodes"] = float(total_episodes)
    stats["collector_actor_version"] = float(learner_state.total_actor_updates)
    stats["transition_hold_events_total"] = float(transition_hold.events_total)
    stats["transition_hold_active"] = float(1.0 if transition_hold.active() else 0.0)
    stats["transition_hold_steps_remaining"] = float(transition_hold.steps_remaining)
    stats["replay_success_size"] = float(snapshot["success"]["size"])
    stats["replay_failure_size"] = float(snapshot["failure"]["size"])
    stats["episodes_saved"] = float(counters["episodes_saved"])
    stats["episodes_removed_short"] = float(counters["episodes_removed_short"])
    stats["episodes_removed_invalid"] = float(counters["episodes_removed_invalid"])
    stats["episodes_gif_generated"] = float(counters["episodes_gif_generated"])
    stats["episodes_gif_failed"] = float(counters["episodes_gif_failed"])
    stats["episodes_camera_video_generated"] = float(counters["episodes_camera_video_generated"])
    stats["episodes_camera_video_failed"] = float(counters["episodes_camera_video_failed"])
    stats["successful_online_episodes_kept"] = float(counters["successful_online_episodes_kept"])
    stats["estop_steps"] = float(counters["protective_stop_steps"])
    stats["estop_episodes"] = float(counters["protective_stop_episodes"])
    stats["controller_disconnect_steps"] = float(counters["controller_disconnect_steps"])
    stats["controller_disconnect_episodes"] = float(counters["controller_disconnect_episodes"])
    stats["human_interrupt_steps"] = float(counters["human_interrupt_steps"])
    stats["human_interrupt_episodes"] = float(counters["human_interrupt_episodes"])
    stats["reset_fsm_steps"] = float(counters["reset_fsm_steps_total"])
    stats["transition_hold_steps"] = float(transition_hold.steps_total)
    stats["primitive_chance"] = float(primitive_selector.chance)
    stats["interval_primitive_env_steps"] = float(interval_state["primitive"])
    stats["run_elapsed_total_s"] = float(elapsed_s)
    update_stats_dict_rolling_windows(
        stats,
        rolling_multi,
        raw_reward_values=rolling_state["reward"],
        raw_episode_length_values=rolling_state["length"],
        raw_estop_episode_flags=rolling_state["estop"],
        raw_episode_return_values=rolling_state["return"],
        raw_episode_juggles_values=rolling_state["juggles"],
        raw_episode_contacts_values=rolling_state["contacts"],
    )
    writer.add_scalar("replay/success_buffer_size", float(snapshot["success"]["size"]), total_steps)
    writer.add_scalar("replay/failure_buffer_size", float(snapshot["failure"]["size"]), total_steps)
    writer.add_scalar("exploration/primitive_chance", float(primitive_selector.chance), total_steps)
    writer.add_scalar("exploration/primitive_env_steps", float(interval_state["primitive"]), total_steps)
    writer.add_scalar(
        "exploration/primitive_horizontal_env_steps",
        float(interval_state["primitive_horizontal"]),
        total_steps,
    )
    writer.add_scalar(
        "replay/episode_return_success_threshold",
        float(episode_return_success_threshold),
        total_steps,
    )
    writer.add_scalar(
        "replay/recent_episode_window_count",
        float(rolling_state["recent_episode_window_count"]),
        total_steps,
    )
    writer.add_scalar("artifacts/episodes_saved", float(counters["episodes_saved"]), total_steps)
    writer.add_scalar(
        "artifacts/episodes_removed_short", float(counters["episodes_removed_short"]), total_steps
    )
    writer.add_scalar(
        "artifacts/episodes_removed_invalid", float(counters["episodes_removed_invalid"]), total_steps
    )
    writer.add_scalar(
        "artifacts/episodes_gif_generated", float(counters["episodes_gif_generated"]), total_steps
    )
    writer.add_scalar("artifacts/episodes_gif_failed", float(counters["episodes_gif_failed"]), total_steps)
    writer.add_scalar(
        "artifacts/episodes_camera_video_generated",
        float(counters["episodes_camera_video_generated"]),
        total_steps,
    )
    writer.add_scalar(
        "artifacts/episodes_camera_video_failed",
        float(counters["episodes_camera_video_failed"]),
        total_steps,
    )
    writer.add_scalar("safety/estop_steps", float(counters["protective_stop_steps"]), total_steps)
    writer.add_scalar("safety/estop_episodes", float(counters["protective_stop_episodes"]), total_steps)
    writer.add_scalar(
        "safety/controller_disconnect_steps",
        float(counters["controller_disconnect_steps"]),
        total_steps,
    )
    writer.add_scalar(
        "safety/controller_disconnect_episodes",
        float(counters["controller_disconnect_episodes"]),
        total_steps,
    )
    writer.add_scalar(
        "operator/human_interrupt_steps",
        float(counters["human_interrupt_steps"]),
        total_steps,
    )
    writer.add_scalar(
        "operator/human_interrupt_episodes",
        float(counters["human_interrupt_episodes"]),
        total_steps,
    )
    writer.add_scalar(
        "transitions/hold_active", float(1.0 if transition_hold.active() else 0.0), total_steps
    )
    writer.add_scalar(
        "transitions/hold_steps_remaining", float(transition_hold.steps_remaining), total_steps
    )
    writer.add_scalar(
        "transitions/hold_events_total", float(transition_hold.events_total), total_steps
    )
    writer.add_scalar(
        "charts/SPS",
        float(total_steps) / max((now - collector_start_time), 1e-6),
        total_steps,
    )
    writer.add_scalar("runtime/elapsed_total_s", float(elapsed_s), total_steps)
    write_rolling_windows_tensorboard_scalars(writer, rolling_multi, total_steps)
    if episodic_returns:
        writer.add_scalar("charts/avg_episodic_return", float(np.mean(episodic_returns)), total_steps)
        writer.add_scalar("charts/min_episodic_return", float(np.min(episodic_returns)), total_steps)
        writer.add_scalar("charts/max_episodic_return", float(np.max(episodic_returns)), total_steps)
        writer.add_scalar(
            "charts/avg_success_rate",
            float(np.mean(success_rates)) if success_rates else 0.0,
            total_steps,
        )
        writer.add_scalar(
            "charts/avg_episodic_length", float(np.mean(episodic_lengths)), total_steps
        )
        episodic_returns.clear()
        episodic_lengths.clear()
        success_rates.clear()
    if episodic_juggles:
        writer.add_scalar(
            "charts/avg_episodic_juggles", float(np.mean(episodic_juggles)), total_steps
        )
        writer.add_scalar(
            "charts/min_episodic_juggles", float(np.min(episodic_juggles)), total_steps
        )
        writer.add_scalar(
            "charts/max_episodic_juggles", float(np.max(episodic_juggles)), total_steps
        )
        writer.add_scalar(
            "charts/avg_juggle_success_rate",
            float(np.mean([1.0 if j >= JUGGLE_SUCCESS_THRESHOLD else 0.0 for j in episodic_juggles])),
            total_steps,
        )
        writer.add_scalar(
            "charts/avg_episodic_contacts",
            float(np.mean(episodic_contacts)) if episodic_contacts else 0.0,
            total_steps,
        )
        episodic_juggles.clear()
        episodic_contacts.clear()

    target = int(args.total_timesteps)
    target_str = f" / {target}" if target > 0 else ""
    print(
        f"[progress] steps={total_steps}{target_str} episodes={total_episodes} "
        f"elapsed={elapsed_s:.0f}s"
    )
    print(
        "[collector] "
        f"steps={total_steps} episodes={total_episodes} "
        f"actor_version={learner_state.total_actor_updates} "
        f"success_rb={snapshot['success']['size']} failure_rb={snapshot['failure']['size']} "
        f"saved={counters['episodes_saved']} short_removed={counters['episodes_removed_short']} "
        f"invalid_removed={counters['episodes_removed_invalid']} "
        f"gif_ok={counters['episodes_gif_generated']} gif_fail={counters['episodes_gif_failed']} "
        f"cam_video_ok={counters['episodes_camera_video_generated']} "
        f"cam_video_fail={counters['episodes_camera_video_failed']} "
        f"estop_steps={counters['protective_stop_steps']} "
        f"estop_episodes={counters['protective_stop_episodes']} "
        f"disconnect_steps={counters['controller_disconnect_steps']} "
        f"disconnect_episodes={counters['controller_disconnect_episodes']} "
        f"readiness_fail_steps={counters['readiness_fail_steps_total']} "
        f"readiness_fail_estop_episodes={counters['readiness_fail_estop_episodes']} "
        f"readiness_fail_dropped_steps={counters['readiness_fail_estop_dropped_steps_total']} "
        f"reset_fsm_steps={counters['reset_fsm_steps_total']} "
        f"transition_hold_steps={transition_hold.steps_total} "
        f"primitive_chance={primitive_selector.chance:.4f} "
        f"primitive_steps={interval_state['primitive']} "
        f"transition_hold_active={int(transition_hold.active())} "
        f"transition_hold_remaining={transition_hold.steps_remaining} "
        f"transition_events_total={transition_hold.events_total} "
        f"transition_reason={transition_hold.reason} "
        f"elapsed_total_s={elapsed_s:.1f} "
        f"rolling50_juggles_avg={rolling50_m.episode_juggles.avg:.2f} "
        f"rolling50_reward_avg={rolling50_m.reward_avg:.4f} "
        f"rolling50_len_avg={rolling50_m.episode_length_avg:.2f} "
        f"rolling50_estops={rolling50_m.estop_episode_count:.0f}"
    )
    for window_size in ROLLING_WINDOW_SIZES:
        m = rolling_multi[window_size]
        print(f"[collector_rolling{window_size}] {format_rolling_window_console_line(m)}")
    if transition_hold.reason_counts:
        print(
            "[collector_transition] "
            f"reason_counts={dict(sorted(transition_hold.reason_counts.items()))}"
        )
    interval_state["primitive"] = 0
    interval_state["primitive_horizontal"] = 0
    return now


def _write_per_episode_tb(
    writer: SummaryWriter,
    *,
    result,
    total_steps: int,
    stats: Dict[str, object],
    juggle_counts: JuggleCounts,
) -> None:
    """Per-episode TB scalars (matches L1860–1867)."""
    writer.add_scalar("charts/episodic_return", result.metrics.episode_return, total_steps)
    writer.add_scalar("charts/episodic_length", result.metrics.episode_length, total_steps)
    writer.add_scalar(
        "charts/episodic_success", float(1.0 if result.terminal.episode_success else 0.0), total_steps
    )
    writer.add_scalar("charts/episodic_juggles", float(juggle_counts.n_juggles), total_steps)
    writer.add_scalar("charts/episodic_contacts", float(juggle_counts.n_contacts), total_steps)
    writer.add_scalar(
        "charts/episodic_juggle_success",
        float(1.0 if juggle_counts.juggle_success else 0.0),
        total_steps,
    )


def _print_episode_progress(
    *,
    result,
    next_episode_file_id: int,
    counters: dict,
    total_steps: int,
    total_episodes: int,
    transition_hold: TransitionHoldState,
    elapsed_offset_s: float,
    collector_start_time: float,
    rolling_state: dict,
    juggle_counts: JuggleCounts,
) -> None:
    """Per-episode progress lines (matches L1921–1952)."""
    n_episode_steps = len(result.rows)
    n_camera_frames = len(result.images)
    has_camera_images = n_camera_frames > 0
    elapsed_s = max(0.0, elapsed_offset_s + (time.time() - collector_start_time))
    elapsed_min = elapsed_s / 60.0
    elapsed_hr = elapsed_s / 3600.0
    rolling_multi = compute_rolling_window_metrics_multi(
        reward_values=rolling_state["reward"],
        episode_length_values=rolling_state["length"],
        estop_episode_flags=rolling_state["estop"],
        episode_return_values=rolling_state["return"],
        episode_juggles_values=rolling_state["juggles"],
        episode_contacts_values=rolling_state["contacts"],
    )
    rolling50_m = rolling_multi[50]
    stop_state_reason = (
        result.terminal.stop_state_reason if result.terminal.stop_now else "none"
    )
    print(
        f"[collector] episode_id={next_episode_file_id} "
        f"steps={n_episode_steps} camera_frames={n_camera_frames} "
        f"null_frames={result.metrics.camera_null_frames} "
        f"has_images={'yes' if has_camera_images else 'NO'} "
        f"end_type={result.terminal.episode_end_type} "
        f"end_reason={result.terminal.episode_end_reason} "
        f"stop_reason={stop_state_reason} "
        f"protective_stop={int(result.terminal.protective_stop)} "
        f"controller_disconnected={int(result.terminal.controller_disconnect)} "
        f"readiness_fail_estop={int(result.terminal.readiness_fail_estop)} "
        f"human_interrupt={int(result.terminal.stop_flags.had_human_interrupt)} "
        f"end_reasons={result.terminal.episode_end_reasons}"
    )
    print(
        "[collector_progress] "
        f"episode_policy_steps={n_episode_steps} "
        f"policy_steps={total_steps} "
        f"reset_fsm_steps={counters['reset_fsm_steps_total']} "
        f"transition_hold_steps={transition_hold.steps_total} "
        f"estop_steps={counters['protective_stop_steps']} "
        f"disconnect_steps={counters['controller_disconnect_steps']} "
        f"readiness_fail_steps={counters['readiness_fail_steps_total']} "
        f"readiness_fail_estop_episodes={counters['readiness_fail_estop_episodes']} "
        f"readiness_fail_dropped_steps={counters['readiness_fail_estop_dropped_steps_total']} "
        f"episodes={total_episodes} "
        f"elapsed_s={elapsed_s:.1f} "
        f"elapsed_min={elapsed_min:.2f} "
        f"elapsed_hr={elapsed_hr:.3f} "
        f"juggles={juggle_counts.n_juggles} "
        f"contacts={juggle_counts.n_contacts} "
        f"juggle_success={int(juggle_counts.juggle_success)} "
        f"rolling50_juggles_avg={rolling50_m.episode_juggles.avg:.2f} "
        f"rolling50_reward_avg={rolling50_m.reward_avg:.4f} "
        f"rolling50_len_avg={rolling50_m.episode_length_avg:.2f} "
        f"rolling50_estops={rolling50_m.estop_episode_count:.0f}"
    )
    for window_size in ROLLING_WINDOW_SIZES:
        m = rolling_multi[window_size]
        print(f"[collector_rolling{window_size}] {format_rolling_window_console_line(m)}")


# ---------------------------------------------------------------------------
# The orchestrator. ~250 lines including all state plumbing.
# ---------------------------------------------------------------------------


def collector_process_modular(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    learner_state: LearnerRuntimeState,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
) -> None:
    """Orchestrator. Drives PolicyRunner + ResetRunner around the learner,
    replay push, and artifact saves. Replaces the original monolithic
    ``collector_process`` that the refactor split into per-concern runners."""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.collector_device)

    # ----------------------------- Env / writer / config -----------------
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    collector_config = _prepare_air_hockey_config(config, seed=args.seed)
    sim_params = collector_config.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
        sim_params["transition_hold_steps_on_estop_enter"] = int(
            args.transition_hold_steps_post_estop_enter
        )
        sim_params["transition_hold_steps_on_estop_clear"] = int(
            args.transition_hold_steps_post_estop_clear
        )
        sim_params["transition_hold_steps_on_safety_rearm"] = int(
            args.transition_hold_steps_post_safety_rearm
        )
    env = AirHockeyEnv(collector_config)
    writer = SummaryWriter(tb_log_dir)
    latency_output_dir: Path | None = None
    if args.enable_latency_profiling:
        if args.latency_profile_output_dir is not None:
            latency_output_dir = Path(args.latency_profile_output_dir).expanduser().resolve()
        else:
            latency_output_dir = Path(tb_log_dir).resolve().parent / "latency_profiles"
        latency_output_dir.mkdir(parents=True, exist_ok=True)
    run_data_dir = run_data_dir_from_args(args)
    print(
        "[run_event_log] writing per-run JSONL streams to:\n"
        f"    episodes : {episode_summaries_path(args)}\n"
        f"    resets   : {reset_summaries_path(args)}\n"
        f"    events   : {run_events_path(args)}"
    )
    # ----------------------------- Run-start event ------------------------
    # Stamp the start of this run into the events stream. Anchors the
    # chronological log: every later wall_time_s in the JSONLs can be
    # offset against this row to yield run-elapsed seconds.
    append_run_event(
        args,
        "run_start",
        run_data_dir=str(run_data_dir),
        run_name=str(getattr(args, "run_name", "")),
        seed=int(getattr(args, "seed", 0)),
        args_file=str(getattr(args, "args_file", "")) if getattr(args, "args_file", None) else None,
        train_args=str(getattr(args, "train_args", "")) if getattr(args, "train_args", None) else None,
        config=str(getattr(args, "config", "")) if getattr(args, "config", None) else None,
        model_path=str(getattr(args, "model_path", "")) if getattr(args, "model_path", None) else None,
        # This entrypoint always trains a TD3 (single-head critic + transformed
        # Bellman target) actor — pinned here for parity with the eval log
        # schema so downstream tooling can branch uniformly on ``agent``.
        agent="td3",
        mode="train",
        smoke_test_seconds=float(getattr(args, "smoke_test_seconds", 0.0)),
    )

    # ----------------------------- Actor / primitive selector -------------
    actor = _build_collector_actor(
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
    )
    action_low = torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0)
    action_high = torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0)
    primitive_selector = build_primitive_exploration_selector_for_real_collector(
        args, device, initial_total_steps=0
    )
    primitive_selector.set_primitive_weights(
        stand_still=float(args.exploration_primitive_weight_stand_still),
        same_direction=float(args.exploration_primitive_weight_same_direction),
    )
    actor.load_state_dict(
        {key: value.detach().cpu() for key, value in learner_state.actor.state_dict().items()},
        strict=False,
    )

    # ----------------------------- Shared dataclasses ---------------------
    ctx = RolloutContext(
        last_action_for_policy=torch.zeros((1, act_dim), dtype=torch.float32, device=device),
        last_executed_action=torch.zeros((1, act_dim), dtype=torch.float32, device=device),
    )
    transition_hold = TransitionHoldState(
        last_action_mode=normalize_transition_last_action_mode(args.transition_last_action_mode),
        log_every_step=bool(args.transition_hold_log_every_step),
    )

    # ----------------------------- Reset runner ---------------------------
    next_reset_file_id = _next_available_episode_id(args.reset_artifact_dir)
    if next_reset_file_id > 0:
        print(
            f"[collector] continuing reset artifact ids from {next_reset_file_id} "
            f"(existing data found in {args.reset_artifact_dir})"
        )
    reset_rng = np.random.default_rng(args.seed)
    if _DEFAULT_RESET_FSM_CLS is ResetPolicyHybridFSM:
        # Build the frozen juggle actor once on the collector process and
        # close over it in the FSM factory. The ``ResetRunner`` factory
        # contract is ``(env, rng) -> fsm``, so any extra dependencies (the
        # actor + device + use-last-action flag) ride in via the closure.
        juggle_actor, juggle_device, juggle_uses_last_action = build_juggle_actor(device)

        def _hybrid_reset_fsm_factory(env, rng):
            return ResetPolicyHybridFSM(
                env,
                rng,
                juggle_actor=juggle_actor,
                juggle_device=juggle_device,
                use_last_action_in_policy_state=juggle_uses_last_action,
            )

        reset_fsm_factory = _hybrid_reset_fsm_factory
    else:
        reset_fsm_factory = _DEFAULT_RESET_FSM_CLS
    reset_runner = ResetRunner(
        env,
        device=device,
        reset_rng=reset_rng,
        reset_policy_fsm_cls=reset_fsm_factory,
        build_split_episode_row=_build_split_episode_row,
        latest_camera_frame=_latest_camera_frame,
    )
    pending_reset_artifact = None

    # ----------------------------- Human interrupt listener ---------------
    # Start before the startup reset so the operator can interrupt during
    # init / first FSM run too. Always-on; stopped after the main loop.
    # See helper/human_interrupt.py.
    human_interrupt_listener = HumanInterruptListener()
    human_interrupt_listener.start()

    # ----------------------------- Startup reset --------------------------
    startup_result = reset_runner.run(
        kind=ResetKind.STARTUP,
        artifact_episode_id=next_reset_file_id,
        episode_had_stop_flags=StopFlags(),
        episode_end_wall_time=time.time(),  # unused for STARTUP
        pending_reset_artifact=pending_reset_artifact,
        next_reset_file_id=next_reset_file_id,
    )
    pending_reset_artifact = startup_result.pending_reset_artifact
    next_reset_file_id = startup_result.next_reset_file_id
    counters: dict = {
        "reset_fsm_steps_total": int(startup_result.total_fsm_steps),
        "protective_stop_episodes": 0,
        "protective_stop_steps": 0,
        "controller_disconnect_episodes": 0,
        "controller_disconnect_steps": 0,
        "readiness_fail_steps_total": 0,
        "readiness_fail_estop_episodes": 0,
        "readiness_fail_estop_dropped_steps_total": 0,
        "human_interrupt_episodes": 0,
        "human_interrupt_steps": 0,
        "episodes_saved": 0,
        "episodes_removed_short": 0,
        "episodes_removed_invalid": 0,
        "episodes_gif_generated": 0,
        "episodes_gif_failed": 0,
        "episodes_homography_gif_generated": 0,
        "episodes_homography_gif_failed": 0,
        "episodes_camera_video_generated": 0,
        "episodes_camera_video_failed": 0,
        "successful_online_episodes_kept": int(stats.get("successful_online_episodes_kept", 0)),
    }

    # ----------------------------- Policy runner --------------------------
    policy_runner = PolicyRunner(
        env,
        actor,
        device=device,
        args=args,
        train_args=train_args,
        action_low=action_low,
        action_high=action_high,
        primitive_selector=primitive_selector,
        transition_hold=transition_hold,
        ctx=ctx,
        reset_primitive_rollout_state=_reset_primitive_rollout_state,
        deterministic_actor_action=deterministic_actor_action,
        augment_policy_observation=augment_policy_observation,
        primitive_exploration_chance_for_step=primitive_exploration_chance_for_step,
        latest_camera_frame=_latest_camera_frame,
        env_timing_info=_env_timing_info,
        safe_nonnegative_ms=_safe_nonnegative_ms,
        build_split_episode_row=_build_split_episode_row,
        readiness_fn=_simulator_step_readiness,
    )
    policy_runner.seed_initial(startup_result.obs)
    total_steps = int(stats.get("collector_total_steps", stats.get("collector_steps", 0.0)))
    policy_runner.set_total_steps(total_steps)
    # Fresh-step counter base. `total_steps` may be non-zero on resume — we
    # want `fresh_collector_steps_this_run` to count post-launch steps only,
    # so the learning_starts_fresh_steps gate has run-relative semantics.
    run_start_total_steps = total_steps
    stats["fresh_collector_steps_this_run"] = float(0)

    # Startup transition hold (reset → policy). Source L1438.
    transition_hold.begin(
        reason=startup_result.transition_reason,
        hold_steps=int(args.transition_hold_steps_post_reset),
        sim_hold=True,
        env=env,
        ctx=ctx,
        primitive_selector=primitive_selector,
        reset_primitive_rollout_state=_reset_primitive_rollout_state,
        use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
        device=device,
    )

    # ----------------------------- Per-run state --------------------------
    next_episode_file_id = _next_available_episode_id(args.episode_artifact_dir)
    if next_episode_file_id > 0:
        print(
            f"[collector] continuing episode artifact ids from {next_episode_file_id} "
            f"(existing data found in {args.episode_artifact_dir})"
        )
    total_episodes = 0
    last_log_time = time.time()
    recent_episode_returns: deque[float] = deque(maxlen=args.recent_episode_window_size)
    episode_return_success_threshold = 0.0
    rolling_state = {
        "reward": deque(
            _coerce_float_list(
                stats.get("rolling50_reward_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            ),
            maxlen=ROLLING_PERF_WINDOW_EPISODES,
        ),
        "length": deque(
            _coerce_float_list(
                stats.get("rolling50_episode_length_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            ),
            maxlen=ROLLING_PERF_WINDOW_EPISODES,
        ),
        "estop": deque(
            _coerce_float_list(
                stats.get("rolling50_estop_episode_flags", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            ),
            maxlen=ROLLING_PERF_WINDOW_EPISODES,
        ),
        # Total episode return distribution (avg/min/max/std/median over
        # the last 50 episodes). Defaults to empty when resuming from a
        # checkpoint that predates this series.
        "return": deque(
            _coerce_float_list(
                stats.get("rolling50_episode_return_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            ),
            maxlen=ROLLING_PERF_WINDOW_EPISODES,
        ),
        # Juggle / contact counts per episode (paddle-puck contact + clear
        # long-term direction flip — see helper/juggle_counter.py). Treated
        # as a sliding-window evaluation metric in parallel with `return`.
        "juggles": deque(
            _coerce_float_list(
                stats.get("rolling50_episode_juggles_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            ),
            maxlen=ROLLING_PERF_WINDOW_EPISODES,
        ),
        "contacts": deque(
            _coerce_float_list(
                stats.get("rolling50_episode_contacts_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            ),
            maxlen=ROLLING_PERF_WINDOW_EPISODES,
        ),
        "recent_episode_window_count": 0,
    }
    interval_state = {
        "primitive": 0,
        "primitive_horizontal": 0,
    }
    episodic_returns: list = []
    episodic_lengths: list = []
    success_rates: list = []
    episodic_juggles: list = []
    episodic_contacts: list = []
    elapsed_offset_s = float(stats.get("run_elapsed_total_s", 0.0))
    collector_start_time = time.time()
    checkpoint_save_request_id = int(stats.get("checkpoint_save_request_id", 0))
    # Tracks the last checkpoint request id the orchestrator has already
    # surfaced as a run_events.jsonl row. Initialized from the stats so a
    # resumed run does not double-emit checkpoint events.
    last_seen_checkpoint_request_id = int(stats.get("last_checkpoint_request_id", 0))

    # ----------------------------- Main loop ------------------------------
    while True:
        if (
            args.smoke_test_seconds > 0
            and (time.time() - collector_start_time) >= float(args.smoke_test_seconds)
        ):
            print(
                f"[collector] smoke-test duration reached ({args.smoke_test_seconds:.1f}s), "
                "stopping."
            )
            append_run_event(
                args,
                "smoke_test_done",
                smoke_test_seconds=float(args.smoke_test_seconds),
                elapsed_s=float(time.time() - collector_start_time),
            )
            break

        # total_timesteps cap. `total_steps` only counts steps from kept
        # episodes (invalid-episode steps are rolled back below), so the cap
        # measures valid policy data only. On resume `total_steps` is
        # initialized from `collector_total_steps` in the checkpoint, so a
        # 50k-step resume with cap=100k runs for 50k more.
        if int(args.total_timesteps) > 0 and total_steps >= int(args.total_timesteps):
            print(
                f"[collector] total_timesteps reached ({total_steps} >= "
                f"{int(args.total_timesteps)}), stopping."
            )
            append_run_event(
                args,
                "total_timesteps_reached",
                total_steps=int(total_steps),
                target=int(args.total_timesteps),
            )
            break

        # 1. Run one policy episode.
        policy_runner.set_artifact_episode_id(next_episode_file_id)
        result = policy_runner.run_episode()
        episode_end_wall_time = time.time()
        total_steps = policy_runner.total_steps
        total_episodes += 1
        # Operational/safety counters: always update — these reflect what
        # physically happened in the world (e-stops, disconnects, readiness
        # faults, reset-FSM steps), not what the trained policy learned, so
        # they must remain accurate even when the trajectory is discarded.
        counters["protective_stop_steps"] += result.metrics.delta_protective_stop_steps
        counters["controller_disconnect_steps"] += result.metrics.delta_controller_disconnect_steps
        counters["readiness_fail_steps_total"] += result.metrics.delta_readiness_fail_steps
        counters["readiness_fail_estop_dropped_steps_total"] += (
            result.metrics.delta_readiness_fail_estop_dropped_steps
        )
        counters["human_interrupt_steps"] += result.metrics.delta_human_interrupt_steps
        if result.metrics.had_protective_stop:
            counters["protective_stop_episodes"] += 1
        if result.metrics.had_controller_disconnect:
            counters["controller_disconnect_episodes"] += 1
        if result.terminal.readiness_fail_estop:
            counters["readiness_fail_estop_episodes"] += 1
        if result.metrics.had_human_interrupt:
            counters["human_interrupt_episodes"] += 1

        # 2. Save artifacts + flush pending reset FIRST so we know whether
        # the trajectory passed validation. Trajectories rejected by
        # `clean_episode_hdf5` (non-finite values, inconsistent dataset
        # lengths, zero timesteps, etc.) must not contribute to replay,
        # learner updates, or any performance metric.
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
            latency_output_dir=latency_output_dir,
            counters=counters,
        )
        pending_reset_artifact = None
        # Replay partition + success threshold are only assigned for kept
        # episodes (inside the block below); seed defaults so the
        # episode summary record has consistent fields for discarded
        # episodes too.
        replay_partition: str | None = None
        replay_threshold_at_episode: float | None = None

        # Roll back step counter for discarded episodes — we count only steps
        # from kept (valid) trajectories toward training progress, the
        # checkpoint cadence, and the total_timesteps cap. Use
        # `delta_total_steps` (the per-episode env-step delta the runner
        # accumulated into `_total_steps`) so the rollback exactly cancels
        # the in-loop increments, even for episodes that were post-hoc
        # truncated for readiness-fail e-stops.
        if not episode_kept:
            n_invalid = int(result.metrics.delta_total_steps)
            policy_runner.rollback_invalid_episode_steps(n_invalid)
            total_steps = policy_runner.total_steps

        # Wall-clock + step counters track elapsed time for the run; safe
        # to update regardless of whether this episode was kept.
        elapsed_s = max(0.0, elapsed_offset_s + (time.time() - collector_start_time))
        stats["run_elapsed_total_s"] = float(elapsed_s)
        stats["collector_steps"] = float(total_steps)
        stats["collector_total_steps"] = float(total_steps)

        # Step-based periodic checkpoint trigger. Fires on the first episode
        # boundary after `total_steps` crosses the next multiple of the
        # configured interval. `total_steps` is already valid-only after
        # the rollback above, so we count only kept-episode steps. The
        # cross-state lives in stats["last_checkpoint_collector_steps"]
        # and is persisted in the checkpoint payload to survive resumes.
        last_step_ckpt = int(float(stats.get("last_checkpoint_collector_steps", 0)))
        ckpt_interval = int(args.checkpoint_every_collector_steps)
        if (
            args.enable_periodic_checkpointing
            and ckpt_interval > 0
            and total_steps // ckpt_interval > last_step_ckpt // ckpt_interval
        ):
            checkpoint_save_request_id += 1
            stats["checkpoint_save_request_id"] = float(checkpoint_save_request_id)
            stats["checkpoint_reason"] = "periodic_collector_steps"
            stats["checkpoint_trigger_total_steps"] = float(total_steps)
            stats["last_checkpoint_collector_steps"] = float(total_steps)
        # Run-relative fresh-step counter — read by the learner's
        # `learning_starts_fresh_steps` gate. On resume `total_steps`
        # carries over from prior runs but `run_start_total_steps` is
        # snapshot at the top of this orchestrator call, so the gate
        # auto-skips when the prior run already crossed it.
        stats["fresh_collector_steps_this_run"] = float(
            max(0, total_steps - run_start_total_steps)
        )

        # 3. Apply the kept-trajectory updates: replay push, learner step,
        # rolling/episodic perf metrics, per-episode TB scalars, and the
        # periodic-checkpoint trigger. All are skipped if the validator
        # rejected the trajectory.
        # Compute juggle/contact counts for kept *and* discarded episodes —
        # cheap (<1ms over a few hundred rows) and lets the JSONL summary
        # carry them for offline analysis even on rejected trajectories.
        episode_juggle_counts = count_juggles_from_rows(result.rows)
        if episode_kept:
            interval_state["primitive"] += result.metrics.delta_interval_primitive_env_steps
            interval_state["primitive_horizontal"] += (
                result.metrics.delta_interval_primitive_horizontal_env_steps
            )
            rolling_state["reward"].append(result.metrics.episode_reward)
            rolling_state["length"].append(result.metrics.episode_length)
            rolling_state["estop"].append(result.metrics.episode_estop_flag)
            rolling_state["return"].append(result.metrics.episode_return)
            rolling_state["juggles"].append(float(episode_juggle_counts.n_juggles))
            rolling_state["contacts"].append(float(episode_juggle_counts.n_contacts))
            episodic_returns.append(result.metrics.episode_return)
            episodic_lengths.append(result.metrics.episode_length)
            success_rates.append(1.0 if result.terminal.episode_success else 0.0)
            episodic_juggles.append(float(episode_juggle_counts.n_juggles))
            episodic_contacts.append(float(episode_juggle_counts.n_contacts))

            _write_per_episode_tb(
                writer,
                result=result,
                total_steps=total_steps,
                stats=stats,
                juggle_counts=episode_juggle_counts,
            )

            partition, ep_return, episode_return_success_threshold, _ = _add_episode_to_shared_replay(
                replay=replay,
                episode_trajectory=result.trajectory,
                recent_episode_returns=recent_episode_returns,
                success_top_fraction=args.success_top_fraction,
            )
            replay_partition = partition
            replay_threshold_at_episode = float(episode_return_success_threshold)
            rolling_state["recent_episode_window_count"] = len(recent_episode_returns)
            actor_updated = _run_sync_learner_iteration(
                args=args,
                train_args=train_args,
                replay=replay,
                stats=stats,
                state=learner_state,
            )
            # If the learner just finished a periodic checkpoint as a side
            # effect, surface it as a row in run_events.jsonl. We detect
            # the save by polling stats["last_checkpoint_request_id"]
            # (set inside _run_sync_learner_iteration on a successful
            # save) against the value we saw on the previous iteration.
            new_ckpt_request_id = int(stats.get("last_checkpoint_request_id", 0))
            if new_ckpt_request_id > last_seen_checkpoint_request_id:
                append_run_event(
                    args,
                    "checkpoint_saved",
                    request_id=new_ckpt_request_id,
                    checkpoint_dir=str(stats.get("last_checkpoint_dir", "")),
                    total_steps=int(float(stats.get("last_checkpoint_collector_steps", 0.0))),
                    q_updates=int(float(stats.get("last_checkpoint_q_updates", 0.0))),
                    trigger=str(stats.get("checkpoint_reason", "")),
                )
                last_seen_checkpoint_request_id = new_ckpt_request_id
            if actor_updated:
                actor.load_state_dict(
                    {k: v.detach().cpu() for k, v in learner_state.actor.state_dict().items()},
                    strict=False,
                )
                transition_hold.begin(
                    reason="actor_sync_update",
                    hold_steps=int(args.transition_hold_steps_post_actor_sync),
                    sim_hold=False,
                    env=env,
                    ctx=ctx,
                    primitive_selector=primitive_selector,
                    reset_primitive_rollout_state=_reset_primitive_rollout_state,
                    use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
                    device=device,
                )

            rolling_multi = compute_rolling_window_metrics_multi(
                reward_values=rolling_state["reward"],
                episode_length_values=rolling_state["length"],
                estop_episode_flags=rolling_state["estop"],
                episode_return_values=rolling_state["return"],
                episode_juggles_values=rolling_state["juggles"],
                episode_contacts_values=rolling_state["contacts"],
            )
            update_stats_dict_rolling_windows(
                stats,
                rolling_multi,
                raw_reward_values=rolling_state["reward"],
                raw_episode_length_values=rolling_state["length"],
                raw_estop_episode_flags=rolling_state["estop"],
                raw_episode_return_values=rolling_state["return"],
                raw_episode_juggles_values=rolling_state["juggles"],
                raw_episode_contacts_values=rolling_state["contacts"],
            )
            write_rolling_windows_tensorboard_scalars(writer, rolling_multi, total_steps)
        else:
            print(
                f"[collector] episode_id={saved_episode_id} discarded by validation; "
                "skipping replay push, learner update, and performance logging."
            )

        stats["successful_online_episodes_kept"] = float(
            counters["successful_online_episodes_kept"]
        )

        # Per-episode progress lines (always print; uses the id the
        # discarded artifact would have taken so logs stay greppable).
        _print_episode_progress(
            result=result,
            next_episode_file_id=saved_episode_id,
            counters=counters,
            total_steps=total_steps,
            total_episodes=total_episodes,
            transition_hold=transition_hold,
            elapsed_offset_s=elapsed_offset_s,
            collector_start_time=collector_start_time,
            rolling_state=rolling_state,
            juggle_counts=episode_juggle_counts,
        )

        # Append one JSONL row covering this episode (kept *or* discarded)
        # so the full episode-by-episode return / metric history is on
        # disk for offline analysis. The trajectory itself is at
        # ``artifact_path`` for kept episodes; discarded trajectories
        # are unlinked by the validator and have ``artifact_path=None``.
        append_episode_summary(
            args,
            {
                "episode_id": int(saved_episode_id),
                "run_episode_index": int(total_episodes),
                "timestamp_iso": datetime.fromtimestamp(
                    episode_end_wall_time, tz=timezone.utc
                ).isoformat(),
                "wall_time_s": float(episode_end_wall_time),
                "kept": bool(episode_kept),
                "clean_reason": clean_reason,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
                "n_steps": int(len(result.rows)),
                "episode_length": float(result.metrics.episode_length),
                "episode_return": float(result.metrics.episode_return),
                "episode_reward": float(result.metrics.episode_reward),
                "episode_success": bool(result.terminal.episode_success),
                "episode_juggles": int(episode_juggle_counts.n_juggles),
                "episode_contacts": int(episode_juggle_counts.n_contacts),
                "episode_juggle_success": bool(episode_juggle_counts.juggle_success),
                "episode_estop_flag": float(result.metrics.episode_estop_flag),
                "had_protective_stop": bool(result.metrics.had_protective_stop),
                "had_controller_disconnect": bool(result.metrics.had_controller_disconnect),
                "had_human_interrupt": bool(result.metrics.had_human_interrupt),
                "readiness_fail_estop": bool(result.terminal.readiness_fail_estop),
                "episode_end_type": result.terminal.episode_end_type,
                "episode_end_reason": result.terminal.episode_end_reason,
                "stop_state_artifact_label": result.terminal.stop_state_artifact_label,
                "replay_partition": replay_partition,
                "episode_return_success_threshold": replay_threshold_at_episode,
                "total_steps": int(total_steps),
                "actor_version": int(learner_state.total_actor_updates),
                "run_elapsed_total_s": float(elapsed_s),
                # Annealed exploration-primitive chance at the moment this
                # episode ended — captures the runtime value that was
                # actually applied during the episode, so future analysis
                # doesn't need to recompute from start/end/horizon args.
                "exploration_primitive_chance_runtime": float(primitive_selector.chance),
            },
        )

        # 4. Pick reset kind and run reset to completion.
        # OR in the live singleton state so a human_interrupt that lands
        # *during* bookkeeping (artifact save / replay push / learner step
        # / JSONL append) — i.e., after the previous episode's terminal
        # info was frozen — still routes to HARD_WITH_FSM. This guarantees
        # the operator gets a safe-pose handover (via _hard_reset_with_pause)
        # regardless of when between-episodes the press lands.
        live_human_interrupt = bool(human_interrupt_state.is_active())
        had_human_interrupt_now = bool(
            result.terminal.stop_flags.had_human_interrupt or live_human_interrupt
        )
        had_stop_now = bool(result.terminal.stop_flags.had_stop or live_human_interrupt)
        kind = pick_reset_kind(
            total_episodes,
            StopFlags(
                had_stop=had_stop_now,
                had_protective_stop=result.terminal.stop_flags.had_protective_stop,
                had_controller_disconnect=result.terminal.stop_flags.had_controller_disconnect,
                had_human_interrupt=had_human_interrupt_now,
            ),
        )
        reset_result = reset_runner.run(
            kind=kind,
            artifact_episode_id=next_reset_file_id,
            episode_had_stop_flags=StopFlags(
                had_stop=had_stop_now,
                had_protective_stop=result.terminal.stop_flags.had_protective_stop,
                had_controller_disconnect=result.terminal.stop_flags.had_controller_disconnect,
                had_human_interrupt=had_human_interrupt_now,
            ),
            episode_end_wall_time=episode_end_wall_time,
            pending_reset_artifact=pending_reset_artifact,
            next_reset_file_id=next_reset_file_id,
        )
        counters["reset_fsm_steps_total"] += reset_result.total_fsm_steps
        pending_reset_artifact = reset_result.pending_reset_artifact
        next_reset_file_id = reset_result.next_reset_file_id

        transition_hold.begin(
            reason=reset_result.transition_reason,
            hold_steps=int(args.transition_hold_steps_post_reset),
            sim_hold=True,
            env=env,
            ctx=ctx,
            primitive_selector=primitive_selector,
            reset_primitive_rollout_state=_reset_primitive_rollout_state,
            use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
            device=device,
        )
        policy_runner.seed_after_reset(reset_result.obs)

        # 5. Periodic logging (between episodes only — see plan §6.1).
        now = time.time()
        if now - last_log_time >= float(args.collector_log_interval_sec):
            last_log_time = _periodic_log(
                args=args,
                writer=writer,
                replay=replay,
                stats=stats,
                learner_state=learner_state,
                primitive_selector=primitive_selector,
                transition_hold=transition_hold,
                total_steps=total_steps,
                total_episodes=total_episodes,
                counters=counters,
                rolling_state=rolling_state,
                elapsed_offset_s=elapsed_offset_s,
                collector_start_time=collector_start_time,
                episodic_returns=episodic_returns,
                episodic_lengths=episodic_lengths,
                success_rates=success_rates,
                episodic_juggles=episodic_juggles,
                episodic_contacts=episodic_contacts,
                interval_state=interval_state,
                last_log_time=last_log_time,
                episode_return_success_threshold=episode_return_success_threshold,
            )

    human_interrupt_listener.stop()
    env.close()
    writer.close()


# ---------------------------------------------------------------------------
# main + __main__ — same shape as async_td3_real, just calls our orchestrator.
# ---------------------------------------------------------------------------


def _parse_modular_specific_args() -> argparse.Namespace:
    """Strip modular-only flags from ``sys.argv`` before tyro sees it.

    Currently just ``--no-quiet`` (opt out of the default per-step /
    per-reset debug suppression — see ``install_quiet_print_filter``).
    Mirrors the eval entrypoint's small argparse pre-pass so the rest
    of the CLI flows through tyro on ``Args`` unchanged.

    Quiet mode is *on by default* so a long-running training session
    shows the qualitatively useful per-episode signals
    (``[collector_rolling50]`` rolling-window summaries, checkpoints,
    warm-start, errors) without the high-rate control-gate /
    force-wrench / reset-FSM chatter.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--no-quiet",
        dest="quiet",
        action="store_false",
        default=True,
        help=(
            "Restore per-step / per-reset debug prints from real-robot "
            "helpers (control-gate moveL/servoStop, force-wrench worker, "
            "second-hit capture, reset FSM, transition holds, etc.). "
            "Use when investigating a real-robot fault."
        ),
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    return parsed


def main(args: Args, train_args: TrainArgs, *, quiet: bool = True) -> None:
    if quiet:
        prefixes, substrs = install_quiet_print_filter()
        print(
            "[main_quiet] suppressing per-step/per-reset debug prints "
            "(rolling-window summaries / checkpoints / warm-start / errors "
            "still shown — pass --no-quiet to restore full chatter). "
            f"prefixes={list(prefixes)} substrings={list(substrs)}"
        )
    if not (0.0 < args.success_top_fraction < 1.0):
        raise ValueError("success_top_fraction must be in (0, 1).")
    if args.q_updates <= 0:
        raise ValueError("q_updates must be > 0.")
    if args.actor_updates_per_iteration <= 0:
        raise ValueError("actor_updates_per_iteration must be > 0.")
    if args.target_network_frequency <= 0:
        raise ValueError("target_network_frequency must be > 0.")
    if abs(float(args.critic_success_sample_fraction + args.critic_failure_sample_fraction) - 1.0) > 1e-6:
        raise ValueError(
            "critic_success_sample_fraction + critic_failure_sample_fraction must equal 1.0."
        )
    if (
        args.enable_periodic_checkpointing
        and int(args.checkpoint_every_collector_steps) <= 0
    ):
        raise ValueError(
            "checkpoint_every_collector_steps must be > 0 when checkpointing is enabled."
        )
    # Loud, non-fatal warning: writing periodic checkpoints without the
    # non-vital fields produces files that cannot be cleanly resumed
    # (no optimizer state, no learner-update counters, no
    # collector_total_steps / run_elapsed_total_s, no rolling-window
    # deques). Resume from such a checkpoint loses Adam momentum,
    # restarts the TB step axis at zero, and starts the rolling-50
    # statistics cold. See notes/docs/training/real-world-resume.md.
    if args.enable_periodic_checkpointing and not args.include_non_vital_training_state_fields:
        print(
            "[main] WARNING: enable_periodic_checkpointing=True but "
            "include_non_vital_training_state_fields=False — checkpoints will OMIT "
            "optimizer state, learner counters, collector_total_steps, "
            "run_elapsed_total_s, and rolling-window deques. Resume from these "
            "checkpoints will be lossy (Adam momentum reset, TB step axis "
            "restarts at 0, rolling-50 stats start cold). Set "
            "include_non_vital_training_state_fields=true in your args YAML "
            "(or pass --include-non-vital-training-state-fields) for "
            "fully-resumable checkpoints. See notes/docs/training/real-world-resume.md."
        )
    normalized_replay_priority = _normalize_replay_source_priority(args.replay_source_priority)
    if normalized_replay_priority != str(args.replay_source_priority).strip().lower():
        print("[main] replay_source_priority normalized to 'warmstart_only' due to invalid input.")
    if normalize_transition_last_action_mode(args.transition_last_action_mode) != str(
        args.transition_last_action_mode
    ).strip().lower():
        print("[main] transition_last_action_mode normalized to 'zero' due to invalid input.")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    probe_config = _prepare_air_hockey_config(config, seed=args.seed, return_goal_obs=False)
    probe_config["simulator"] = "box2d"
    probe_sim_params = dict(probe_config.get("simulator_params", {}))
    for key in ("control_mode", "wait_for_space_to_start", "save_path", "debug_control", "debug_control_every"):
        probe_sim_params.pop(key, None)
    probe_config["simulator_params"] = probe_sim_params
    probe_env = AirHockeyEnv(probe_config)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    action_low_np = np.asarray(probe_env.action_space.low, dtype=np.float32)
    action_high_np = np.asarray(probe_env.action_space.high, dtype=np.float32)

    replay = SharedTD3Replay(
        success_capacity=args.success_buffer_size,
        failure_capacity=args.failure_buffer_size,
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
    )
    stats: Dict[str, object] = {}
    stats["successful_online_episodes_kept"] = float(0)
    stats["checkpoint_save_request_id"] = float(0)
    stats["last_checkpoint_collector_steps"] = float(0.0)
    stats["collector_total_steps"] = float(0.0)
    stats["run_elapsed_total_s"] = float(0.0)
    stats["rolling50_window_size"] = float(ROLLING_PERF_WINDOW_EPISODES)
    stats["rolling50_window_count"] = float(0.0)
    stats["rolling50_reward_avg"] = float(0.0)
    stats["rolling50_episode_length_avg"] = float(0.0)
    stats["rolling50_estop_episode_count"] = float(0.0)
    stats["rolling50_reward_values"] = []
    stats["rolling50_episode_length_values"] = []
    stats["rolling50_estop_episode_flags"] = []
    stats["rolling50_episode_return_values"] = []
    stats["rolling50_episode_juggles_values"] = []
    stats["rolling50_episode_contacts_values"] = []
    training_state_checkpoint: Dict[str, object] | None = None
    if args.model_path is not None:
        training_state_checkpoint = _load_training_state_checkpoint(args.model_path)
        if "collector_total_steps" in training_state_checkpoint:
            loaded_reward_values = _coerce_float_list(
                training_state_checkpoint.get(
                    "rolling50_reward_values",
                    training_state_checkpoint.get("rolling50_task_reward_values", []),
                ),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            )
            loaded_length_values = _coerce_float_list(
                training_state_checkpoint["rolling50_episode_length_values"],
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            )
            loaded_estop_flags = _coerce_float_list(
                training_state_checkpoint["rolling50_estop_episode_flags"],
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            )
            # Use .get(...) so checkpoints saved before the return series
            # was tracked still load — the resume just starts with an
            # empty return window and refills as new episodes arrive.
            loaded_return_values = _coerce_float_list(
                training_state_checkpoint.get("rolling50_episode_return_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            )
            # Same .get(...) treatment for juggle/contact rolling values:
            # checkpoints written before this metric existed resume cleanly
            # with empty windows that refill as new episodes arrive.
            loaded_juggles_values = _coerce_float_list(
                training_state_checkpoint.get("rolling50_episode_juggles_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            )
            loaded_contacts_values = _coerce_float_list(
                training_state_checkpoint.get("rolling50_episode_contacts_values", []),
                max_items=ROLLING_PERF_WINDOW_EPISODES,
            )
            stats["collector_total_steps"] = float(training_state_checkpoint["collector_total_steps"])
            # Old checkpoints (pre step-cadence) won't have this field; default
            # to the resumed `collector_total_steps` so the first new trigger
            # fires at the next clean multiple of `checkpoint_every_collector_steps`,
            # not redundantly at the resumed step itself.
            stats["last_checkpoint_collector_steps"] = float(
                training_state_checkpoint.get(
                    "last_checkpoint_collector_steps",
                    training_state_checkpoint["collector_total_steps"],
                )
            )
            stats["run_elapsed_total_s"] = float(training_state_checkpoint["run_elapsed_total_s"])
            stats["rolling50_reward_values"] = loaded_reward_values
            stats["rolling50_episode_length_values"] = loaded_length_values
            stats["rolling50_estop_episode_flags"] = loaded_estop_flags
            stats["rolling50_episode_return_values"] = loaded_return_values
            stats["rolling50_episode_juggles_values"] = loaded_juggles_values
            stats["rolling50_episode_contacts_values"] = loaded_contacts_values
            stats["rolling50_window_count"] = float(
                max(
                    len(loaded_reward_values),
                    len(loaded_length_values),
                    len(loaded_estop_flags),
                    len(loaded_return_values),
                )
            )
            stats["rolling50_reward_avg"] = float(rolling_mean(loaded_reward_values))
            stats["rolling50_episode_length_avg"] = float(rolling_mean(loaded_length_values))
            stats["rolling50_episode_return_avg"] = float(rolling_mean(loaded_return_values))
            stats["rolling50_episode_juggles_avg"] = float(rolling_mean(loaded_juggles_values))
            stats["rolling50_episode_contacts_avg"] = float(rolling_mean(loaded_contacts_values))
            stats["rolling50_estop_episode_count"] = float(sum(loaded_estop_flags))

    warm_start_requested = len(args.warm_start_hdf5_dirs) > 0
    checkpoint_replay_loaded = False
    if args.load_replay_from_checkpoint and training_state_checkpoint is not None:
        if warm_start_requested and normalized_replay_priority == "warmstart_only":
            print("[resume_replay] skipping checkpoint replay because warmstart_only is active.")
        else:
            replay.load_state_dict(
                {
                    "success": training_state_checkpoint["success_replay_buffer"],
                    "failure": training_state_checkpoint["failure_replay_buffer"],
                }
            )
            snapshot = replay.state_snapshot()
            print(
                "[resume_replay] loaded from checkpoint "
                f"success_rb={snapshot['success']['size']} failure_rb={snapshot['failure']['size']}"
            )
            checkpoint_replay_loaded = True
            stats["resume_replay_loaded"] = float(1)
    try:
        if warm_start_requested:
            if checkpoint_replay_loaded and normalized_replay_priority == "checkpoint_only":
                print("[warm_start] skipped because replay_source_priority=checkpoint_only")
            else:
                if checkpoint_replay_loaded and normalized_replay_priority == "checkpoint_then_append":
                    print("[warm_start] appending warm-start data on top of checkpoint replay.")
                warm_start_summary = _warm_start_replay_from_hdf5(
                    args=args,
                    replay=replay,
                    env=probe_env,
                    add_episode_to_shared_replay=_add_episode_to_shared_replay,
                )
                for key, value in warm_start_summary.items():
                    stats[f"warm_start/{key}"] = float(value)
    finally:
        probe_env.close()
    # `_setup_run_data_dir` already populated `checkpoint_root_dir` with the
    # unified run folder, so TB logs land alongside the episode data.
    base_log_dir = str(Path(args.checkpoint_root_dir).expanduser().resolve())
    collector_tb_dir = os.path.join(base_log_dir, "collector_tb")
    learner_tb_dir = os.path.join(base_log_dir, "learner_tb")
    os.makedirs(collector_tb_dir, exist_ok=True)
    os.makedirs(learner_tb_dir, exist_ok=True)
    print(f"TensorBoard logs: {base_log_dir}")

    learner_state = _init_sync_learner_state(
        args=args,
        train_args=train_args,
        replay=replay,
        stats=stats,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        tb_log_dir=learner_tb_dir,
        resume_checkpoint=training_state_checkpoint,
    )
    run_end_reason = "completed"
    try:
        collector_process_modular(
            args,
            train_args,
            replay,
            stats,
            learner_state,
            obs_dim,
            act_dim,
            action_low_np,
            action_high_np,
            collector_tb_dir,
        )
    except KeyboardInterrupt:
        print("[main] interrupted by user; shutting down.")
        run_end_reason = "keyboard_interrupt"
    except BaseException:
        run_end_reason = "exception"
        raise
    finally:
        _finalize_sync_learner_state(
            args=args,
            train_args=train_args,
            replay=replay,
            stats=stats,
            state=learner_state,
        )
        # Emit run_end AFTER finalize so the post-finalize checkpoint
        # request id (if any) is captured. The orchestrator already
        # handles its own final-checkpoint event detection during the
        # shutdown path, but if the finalize wrote a fresh "final_*"
        # checkpoint we emit one more event here for symmetry.
        final_request_id = int(stats.get("last_checkpoint_request_id", 0))
        if final_request_id > 0 and stats.get("last_checkpoint_dir"):
            append_run_event(
                args,
                "checkpoint_saved",
                request_id=final_request_id,
                checkpoint_dir=str(stats.get("last_checkpoint_dir", "")),
                total_steps=int(float(stats.get("last_checkpoint_collector_steps", 0.0))),
                q_updates=int(float(stats.get("last_checkpoint_q_updates", 0.0))),
                trigger="final_on_shutdown",
            )
        append_run_event(
            args,
            "run_end",
            reason=run_end_reason,
            collector_total_steps=int(float(stats.get("collector_total_steps", 0.0))),
            run_elapsed_total_s=float(stats.get("run_elapsed_total_s", 0.0)),
            successful_online_episodes_kept=int(
                float(stats.get("successful_online_episodes_kept", 0.0))
            ),
            episodes_saved=int(float(stats.get("episodes_saved", 0.0))),
            episodes_removed_short=int(float(stats.get("episodes_removed_short", 0.0))),
            episodes_removed_invalid=int(float(stats.get("episodes_removed_invalid", 0.0))),
            last_checkpoint_dir=str(stats.get("last_checkpoint_dir", "")) or None,
        )
        print("Final stats:", dict(stats))


if __name__ == "__main__":
    modular_extra_args = _parse_modular_specific_args()
    temp_args = tyro.cli(Args)
    if temp_args.train_args is None:
        raise SystemExit(
            "async_td3_real.py requires --train-args pointing to the "
            "training run's args.yaml (produced by td3_training.py). It supplies "
            "the actor/critic architecture and use_last_action_in_policy_state "
            "flag that must match the saved checkpoint."
        )
    if temp_args.args_file is None:
        raise SystemExit(
            "async_td3_real.py requires --args-file pointing to an "
            "online-behavior YAML (e.g. td3_residual.yaml). Architecture comes "
            "from --train-args; this file supplies online training/collection "
            "defaults only."
        )
    train_args = _load_train_args(temp_args.train_args)
    mapped_defaults, applied_keys, ignored_keys = _build_args_file_defaults(temp_args.args_file)
    mapped_defaults["args_file"] = temp_args.args_file
    mapped_defaults["train_args"] = temp_args.train_args
    default_args = Args(**mapped_defaults)

    args = tyro.cli(Args, default=default_args)
    print(f"[train_args] loaded architecture from: {args.train_args}")
    print(
        f"[train_args] "
        f"agent_hidden_layer_size={train_args.agent_hidden_layer_size} "
        f"agent_num_hidden_layers={train_args.agent_num_hidden_layers} "
        f"q_hidden_layer_size={train_args.q_hidden_layer_size} "
        f"q_num_hidden_layers={train_args.q_num_hidden_layers} "
        f"use_last_action_in_policy_state={train_args.use_last_action_in_policy_state} "
        f"num_critics={train_args.num_critics} "
        f"target_critic_subset_size={train_args.target_critic_subset_size}"
    )
    print(f"[args_file] loaded defaults from: {args.args_file}")
    if applied_keys:
        print("[args_file] applied keys:", ", ".join(applied_keys))
    else:
        print("[args_file] applied keys: none")
    if ignored_keys:
        print("[args_file] ignored unsupported keys:", ", ".join(ignored_keys))
    run_note = _prompt_optional_run_note()
    _setup_run_data_dir(args, run_note)
    main(args, train_args, quiet=bool(modular_extra_args.quiet))
