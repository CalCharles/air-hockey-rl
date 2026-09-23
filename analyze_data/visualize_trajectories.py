#!/usr/bin/env python3
"""
Batch-visualize real-world air-hockey HDF5 trajectories in the Box2D simulator.

For every `.hdf5` file found (recursively) under a data folder, this script:

  1. Loads the real-world trajectory (split HDF5 schema — see
     `scripts/visualization/visualize_real_trajectory_split.py:SPLIT_DATASETS`).
     Real paddle/puck positions (`pose`, `puck`) and the policy's commanded
     target (`desired_pose`) are stored in table-frame metres.
  2. Reconstructs the normalized `[-1, 1]` actions the policy executed
     (`actions = clip((desired_pose - pose) / move_lims, -1, 1)`) and replays
     them in the Box2D simulator, starting from the real episode's initial
     paddle/puck position + velocity
     (`scripts/visualization/replay_real_in_sim.py:replay_episode`).
  3. Renders a side-by-side GIF: REAL (left) | SIM (right), with a semi-
     transparent gray "ghost" overlay drawn on the REAL panel showing where
     the Box2D sim currently thinks the paddle/puck are. This makes drift
     between what actually happened on the robot and what the same actions
     "translate" to in Box2D visible in two ways at once: panel misalignment
     and ghost offset.
  4. Writes `<name>.gif` (+ a `<name>.json` per-step position-error summary)
     into `analyze_data/replay_gifs/<last folder name in data_dir>/`, mirroring
     the folder structure of the input data folder *relative to the path you
     pass in* underneath that (e.g. passing `.../real_runs/episode_hdf5` for a
     file at `<data_dir>/100-200/trajectory_data451.hdf5` writes to
     `analyze_data/replay_gifs/episode_hdf5/100-200/trajectory_data451.gif`).

This script is a thin batch wrapper around the existing single-episode replay
primitive in `scripts/visualization/replay_real_in_sim.py` — see that file
(and `notes/docs/environments/real-world/replay-real-in-sim.md`) for the full
replay/ghost-overlay implementation details.

Pass `--puck-source real` to pin the sim puck to the recorded real-world
puck trajectory at every step instead of letting Box2D compute it from
collisions — isolates paddle-position drift (from action/observation delay)
from confounds where sim-computed puck dynamics have diverged from real.
Batches with this flag are written to a sibling `<name>_realpuck/` output
folder so they don't collide with default (`puck_source="sim"`) batches.

Usage:
    python analyze_data/visualize_trajectories.py /path/to/data_folder
    python analyze_data/visualize_trajectories.py real_runs/online_run/episode_hdf5 --max-steps 200
    python analyze_data/visualize_trajectories.py sysid/teleop --workers 4 --overwrite
    python analyze_data/visualize_trajectories.py sysid/teleop --limit 2   # quick smoke test

Only HDF5 files matching the split schema can be replayed; files that don't
match (too short, legacy flat `train_vals` schema, corrupted, etc.) are
skipped with a logged reason and do NOT stop the rest of the batch.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure repo root is importable when running as `python analyze_data/visualize_trajectories.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

from scripts.visualization.replay_real_in_sim import replay_episode  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "configs" / "new_juggle" / "sysid_best_params_hist4.yaml"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "replay_gifs"


@dataclass
class ReplayJob:
    hdf5_path: Path
    output_gif_path: Path
    config_path: str
    enable_noise: bool
    max_steps: Optional[int]
    fps: int
    frame_width: int
    start_frame: int
    puck_vel_fit: bool
    puck_vel_half_window: int
    puck_source: str


@dataclass
class ReplayResult:
    hdf5_path: Path
    output_gif_path: Path
    status: str  # "ok" | "failed"
    message: str = ""


def discover_hdf5_files(data_dir: Path, pattern: str) -> list[Path]:
    """Recursively find HDF5 files under `data_dir`, sorted for reproducible order."""
    return sorted(p for p in data_dir.rglob(pattern) if p.is_file())


def mirrored_output_path(hdf5_path: Path, data_dir: Path, output_root: Path) -> Path:
    """Map a source hdf5 path to its output GIF path, mirroring `data_dir`'s subfolder
    structure (relative to `data_dir`) underneath `output_root`.

    Deliberately does NOT call `.resolve()` on `hdf5_path`: if an individual file
    (or an intermediate folder) inside `data_dir` is itself a symlink to somewhere
    outside `data_dir` (common on this machine, e.g. `data/` entries symlinked to
    `/data2/...`), resolving it would follow the symlink target and break
    `relative_to`. `hdf5_path` comes from `data_dir.rglob(...)`, so it is always
    already prefixed with `data_dir` verbatim — no resolution needed here.
    """
    rel = hdf5_path.relative_to(data_dir)
    return (output_root / rel).with_suffix(".gif")


def _run_one(job: ReplayJob) -> ReplayResult:
    """Run a single episode replay. Never raises — failures are captured so a bad
    file doesn't kill the rest of the batch (used both in-process and as the
    ProcessPoolExecutor worker function)."""
    try:
        job.output_gif_path.parent.mkdir(parents=True, exist_ok=True)
        replay_episode(
            episode_path=str(job.hdf5_path),
            config_path=job.config_path,
            output_path=str(job.output_gif_path),
            enable_noise=job.enable_noise,
            max_steps=job.max_steps,
            fps=job.fps,
            frame_width=job.frame_width,
            start_frame=job.start_frame,
            puck_vel_fit=job.puck_vel_fit,
            puck_vel_half_window=job.puck_vel_half_window,
            puck_source=job.puck_source,
        )
        return ReplayResult(job.hdf5_path, job.output_gif_path, "ok")
    except Exception as exc:  # noqa: BLE001 - keep the batch alive on per-file failure
        return ReplayResult(
            job.hdf5_path,
            job.output_gif_path,
            "failed",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-replay real-world HDF5 trajectories in the Box2D sim and render "
            "sim-vs-real GIFs (with a ghost overlay) into analyze_data/replay_gifs/, "
            "mirroring the input folder's structure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Folder containing real-world .hdf5 trajectory files (searched recursively).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output folder; mirrors data_dir's subfolder structure underneath it.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Box2D sim YAML config used for the replay.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hdf5",
        help="Glob pattern (applied recursively via rglob) for selecting HDF5 files.",
    )
    parser.add_argument(
        "--enable-noise",
        action="store_true",
        help="Use the config's noise/delay/occlusion/termination settings verbatim "
        "(default: disabled, for a clean deterministic replay).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on replay length (in steps) per episode.",
    )
    parser.add_argument("--fps", type=int, default=20, help="GIF playback frame rate.")
    parser.add_argument(
        "--frame-width",
        type=int,
        default=160,
        help="Width (px) each panel (REAL/SIM) is resized to before side-by-side concat.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index in each real episode at which to begin the comparison.",
    )
    parser.add_argument(
        "--puck-vel-fit",
        action="store_true",
        help="Estimate initial puck velocity via a gravity-linear LSQ fit instead of "
        "a two-point finite difference (see replay_real_in_sim.py).",
    )
    parser.add_argument("--puck-vel-half-window", type=int, default=5)
    parser.add_argument(
        "--puck-source",
        type=str,
        choices=["sim", "real"],
        default="sim",
        help=(
            "'sim' (default): Box2D computes puck dynamics normally from collisions "
            "with the sim paddle. 'real': pin the sim puck's position+velocity to the "
            "recorded real-world trajectory at every step, bypassing sim puck physics. "
            "Isolates paddle-position drift (from action/observation delay) from any "
            "confound where sim-computed puck dynamics have diverged from real — see "
            "scripts/visualization/replay_real_in_sim.py for details. When 'real', "
            "output batches go into a sibling '<data_dir.name>_realpuck' output folder."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render GIFs that already exist at the output path (default: skip them).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of files to process — use for a quick smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (each episode replay is independent).",
    )
    return parser.parse_args()


def _log_result(n_done: int, n_total: int, result: ReplayResult) -> None:
    if result.status == "ok":
        print(f"[{n_done}/{n_total}] OK     {result.hdf5_path} -> {result.output_gif_path}")
    else:
        first_line = result.message.splitlines()[0] if result.message else ""
        print(f"[{n_done}/{n_total}] FAILED {result.hdf5_path}: {first_line}")


def _print_summary(results: list[ReplayResult], n_skipped_existing: int) -> None:
    n_ok = sum(1 for r in results if r.status == "ok")
    n_failed = sum(1 for r in results if r.status == "failed")
    print("\n" + "=" * 60)
    print(
        f"Done. {n_ok} succeeded, {n_failed} failed, "
        f"{n_skipped_existing} skipped (GIF already existed)."
    )
    if n_failed:
        print("\nFailures:")
        for r in results:
            if r.status == "failed":
                print(f"  - {r.hdf5_path}")
                print(f"      {r.message.splitlines()[0]}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")
    # Nest outputs under a folder named after `data_dir` itself (the last path
    # component you passed in), then mirror data_dir's own subfolder structure
    # underneath that — so e.g. passing `.../real_runs/episode_hdf5` writes to
    # `<output_root>/episode_hdf5/...` rather than dumping straight into
    # `<output_root>/...`.
    dir_suffix = "_realpuck" if args.puck_source == "real" else ""
    output_root = Path(args.output_root).expanduser().resolve() / f"{data_dir.name}{dir_suffix}"

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str((_REPO_ROOT / config_path).resolve())

    hdf5_files = discover_hdf5_files(data_dir, args.pattern)
    if args.limit is not None:
        hdf5_files = hdf5_files[: args.limit]
    if not hdf5_files:
        print(f"No files matching '{args.pattern}' found under {data_dir}.")
        return

    jobs: list[ReplayJob] = []
    n_skipped_existing = 0
    for hdf5_path in hdf5_files:
        out_path = mirrored_output_path(hdf5_path, data_dir, output_root)
        if out_path.exists() and not args.overwrite:
            n_skipped_existing += 1
            continue
        jobs.append(
            ReplayJob(
                hdf5_path=hdf5_path,
                output_gif_path=out_path,
                config_path=config_path,
                enable_noise=args.enable_noise,
                max_steps=args.max_steps,
                fps=args.fps,
                frame_width=args.frame_width,
                start_frame=args.start_frame,
                puck_vel_fit=args.puck_vel_fit,
                puck_vel_half_window=args.puck_vel_half_window,
                puck_source=args.puck_source,
            )
        )

    print(
        f"Found {len(hdf5_files)} file(s) under {data_dir}; "
        f"{n_skipped_existing} already have GIFs (use --overwrite to redo), "
        f"{len(jobs)} to process. Output root: {output_root}"
    )

    results: list[ReplayResult] = []
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, job): job for job in jobs}
            for n_done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                _log_result(n_done, len(jobs), result)
    else:
        for n_done, job in enumerate(jobs, start=1):
            result = _run_one(job)
            results.append(result)
            _log_result(n_done, len(jobs), result)

    _print_summary(results, n_skipped_existing)


if __name__ == "__main__":
    main()
