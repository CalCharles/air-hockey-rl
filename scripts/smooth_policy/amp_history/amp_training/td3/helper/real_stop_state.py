"""Stop / e-stop event classification for real robot collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airhockey import AirHockeyEnv


@dataclass(frozen=True)
class StopEventState:
    protective_stop: bool = False
    controller_disconnected: bool = False
    active: bool = False
    reason: str = "none"
    episode_end_type: str | None = None
    episode_end_reason: str | None = None
    artifact_label: str | None = None


def _build_stop_event_state(
    *,
    protective_stop: bool = False,
    controller_connected: bool | None = None,
    legacy_estop_signal: bool = False,
) -> StopEventState:
    controller_disconnected = bool(controller_connected is False)
    if bool(protective_stop):
        return StopEventState(
            protective_stop=True,
            controller_disconnected=controller_disconnected,
            active=True,
            reason="protective_stop",
            episode_end_type="estop",
            episode_end_reason="collector_protective_stop",
            artifact_label="estop",
        )
    if controller_disconnected:
        return StopEventState(
            protective_stop=False,
            controller_disconnected=True,
            active=True,
            reason="controller_disconnected",
            episode_end_type="controller_disconnected",
            episode_end_reason="collector_controller_disconnected",
            artifact_label="controller_disconnected",
        )
    if bool(legacy_estop_signal):
        return StopEventState(
            protective_stop=True,
            controller_disconnected=False,
            active=True,
            reason="legacy_estop_signal",
            episode_end_type="estop",
            episode_end_reason="collector_estop_legacy_signal",
            artifact_label="estop",
        )
    return StopEventState()


def _stop_state_from_saved_row(
    train_vals: np.ndarray,
    optional_data: dict[str, np.ndarray],
    row_idx: int,
) -> StopEventState:
    stop_flags = optional_data.get("stop_flags")
    protective_stop = bool(float(train_vals[row_idx, 3]) > 0.5)
    controller_connected: bool | None = None
    if stop_flags is not None and stop_flags.shape[0] > row_idx:
        stop_row = np.asarray(stop_flags[row_idx], dtype=np.float64).reshape(-1)
        if stop_row.size > 0:
            protective_stop = protective_stop or bool(float(stop_row[0]) > 0.5)
        if stop_row.size > 1:
            controller_connected = not bool(float(stop_row[1]) > 0.5)
    return _build_stop_event_state(
        protective_stop=protective_stop,
        controller_connected=controller_connected,
    )


def _classify_stop_event(
    env: AirHockeyEnv,
    step_info: dict | None = None,
) -> StopEventState:
    """Classify collector stop conditions without conflating them with readiness summaries."""
    if isinstance(step_info, dict):
        protective_stop_present = "protective_stop" in step_info
        controller_connected_present = "controller_connected" in step_info
        if protective_stop_present or controller_connected_present:
            return _build_stop_event_state(
                protective_stop=bool(step_info.get("protective_stop", False)),
                controller_connected=(
                    bool(step_info.get("controller_connected", True))
                    if controller_connected_present
                    else None
                ),
            )
        if "estop" in step_info:
            estop_value = np.asarray(step_info["estop"], dtype=np.float64).reshape(-1)
            if estop_value.size > 0 and bool(estop_value[0] > 0.5):
                return _build_stop_event_state(legacy_estop_signal=True)

    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return StopEventState()

    readiness_fn = getattr(simulator, "robot_command_readiness", None)
    if callable(readiness_fn):
        try:
            readiness = readiness_fn()
            if isinstance(readiness, dict):
                protective_stop_present = "protective_stop" in readiness
                controller_connected_present = "controller_connected" in readiness
                if protective_stop_present or controller_connected_present:
                    return _build_stop_event_state(
                        protective_stop=bool(readiness.get("protective_stop", False)),
                        controller_connected=(
                            bool(readiness.get("controller_connected", True))
                            if controller_connected_present
                            else None
                        ),
                    )
        except Exception:
            pass

    rcv = getattr(simulator, "rcv", None)
    if rcv is not None and hasattr(rcv, "isProtectiveStopped"):
        try:
            return _build_stop_event_state(protective_stop=bool(rcv.isProtectiveStopped()))
        except Exception:
            # Telemetry failure is unsafe; treat as a disconnected controller path.
            return _build_stop_event_state(controller_connected=False)

    vals = getattr(simulator, "vals", None)
    if isinstance(vals, list) and len(vals) > 0:
        try:
            latest = np.asarray(vals[-1], dtype=np.float64).reshape(-1)
            if latest.shape[0] > 3 and bool(latest[3] > 0.5):
                return _build_stop_event_state(legacy_estop_signal=True)
        except Exception:
            return StopEventState()
    return StopEventState()
