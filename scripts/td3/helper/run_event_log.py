"""Append-only JSONL event log for async real-world TD3 runs.

One run produces three sibling JSONL files in ``<run_data_dir>/``:

* ``episode_summaries.jsonl`` — one row per policy episode (kept and discarded).
* ``reset_summaries.jsonl``   — one row per reset event (success and failure).
* ``run_events.jsonl``        — operational events: run_start, run_end,
                                checkpoint_saved.

Each row is a self-contained JSON object with a ``wall_time_s`` /
``timestamp_iso`` pair plus event-type-specific fields. Append failures
are caught and logged to stdout; they never abort the live run.

The three logs together let an analyst reconstruct the run on a high
level — interleaving episodes and resets by ``wall_time_s`` and overlaying
operational events — without having to parse captured stdout.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def run_data_dir_from_args(args: Any) -> Path:
    """Return the unified run-data directory for ``args``.

    ``_setup_run_data_dir`` makes ``args.episode_artifact_dir`` a child of
    the run-data dir; we recover the parent here so all three JSONLs live
    next to ``episode_hdf5/`` / ``checkpoint_*`` / TB logs / etc.
    """
    return Path(args.episode_artifact_dir).expanduser().resolve().parent


def episode_summaries_path(args: Any) -> Path:
    return run_data_dir_from_args(args) / "episode_summaries.jsonl"


def reset_summaries_path(args: Any) -> Path:
    return run_data_dir_from_args(args) / "reset_summaries.jsonl"


def run_events_path(args: Any) -> Path:
    return run_data_dir_from_args(args) / "run_events.jsonl"


# ---------------------------------------------------------------------------
# Append primitives
# ---------------------------------------------------------------------------


def utc_timestamps() -> tuple[float, str]:
    """Return ``(wall_time_s, timestamp_iso)`` paired off the same ``time.time()``."""
    wall = time.time()
    return wall, datetime.fromtimestamp(wall, tz=timezone.utc).isoformat()


def append_jsonl_row(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a line to ``path``.

    Best-effort: any IO/serialization failure prints a one-line warning
    and returns. A live real-world run must never abort because a
    summary row could not be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, default=str)
            f.write("\n")
    except Exception:
        print(
            f"[run_event_log] failed to append to {path}:\n"
            f"{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Convenience writers (per log type)
# ---------------------------------------------------------------------------


def append_episode_summary(args: Any, record: Mapping[str, Any]) -> None:
    """Append one row to ``episode_summaries.jsonl``."""
    append_jsonl_row(episode_summaries_path(args), record)


def append_reset_summary(args: Any, record: Mapping[str, Any]) -> None:
    """Append one row to ``reset_summaries.jsonl``."""
    append_jsonl_row(reset_summaries_path(args), record)


def append_run_event(args: Any, event_type: str, **fields: Any) -> None:
    """Append one row to ``run_events.jsonl`` with a stamped header.

    ``event_type`` is the discriminator (e.g. ``"run_start"``,
    ``"run_end"``, ``"checkpoint_saved"``). All ``fields`` are merged
    into the row alongside the auto-injected ``wall_time_s`` /
    ``timestamp_iso``.
    """
    wall, iso = utc_timestamps()
    record: dict[str, Any] = {
        "event_type": event_type,
        "wall_time_s": wall,
        "timestamp_iso": iso,
    }
    record.update(fields)
    append_jsonl_row(run_events_path(args), record)
