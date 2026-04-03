"""
Remove TD3 sweep run directories with too few periodic checkpoints.

A "run" is an immediate child of the sweep log directory that contains ``args.yaml``.
Periodic checkpoints are subdirectories named ``checkpoint_<global_step>`` that contain
``model.pth`` (same definition as ``collect_policy_data.discover_td3_policy_snapshots``).

Runs with fewer than ``--min-checkpoints`` such folders are candidates for deletion.
By default only a dry-run report is printed; pass ``--apply`` to remove directories.

Optional ``--aggregate-gifs`` copies one rollout GIF per *kept* run (those with enough
checkpoints) into ``<sweep-dir>/<gifs-subdir>/``, named ``<run-folder>.gif``, and writes
``missing_evaluation_gifs.txt`` there listing sweep tasks that have no run dir, too few
checkpoints, or no ``eval_0.gif`` under ``rollout/``.

``generate_sweep.py`` can run this automatically when the sweep YAML sets
``post_min_checkpoints`` (see that script's docstring).

Usage:
    uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/cleanup_sweep_runs.py \\
        --sweep-dir /data2/calebc/air_hockey/reward_sweep \\
        --min-checkpoints 5 --aggregate-gifs

    uv run .../cleanup_sweep_runs.py \\
        --sweep-file scripts/smooth_policy/amp_history/configs/td3/example_reward_sweep.yaml \\
        --min-checkpoints 5 --aggregate-gifs --apply
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Iterator

import yaml

_CHECKPOINT_SUBDIR_RE = re.compile(r"^checkpoint_(\d+)$")


def resolve_run_dir(
    sweep_dir: str,
    tag: str,
    *,
    eval_run_dir_suffix: str | None = None,
    eval_resolve_run_dir: bool = True,
) -> str | None:
    """
    Match training layout: LOG_BASE/<tag> with args.yaml, else <tag>r1..r50.
    If eval_run_dir_suffix is set (e.g. 'r1'), only that path is tried.
    """
    if eval_run_dir_suffix is not None:
        d = os.path.join(sweep_dir, f"{tag}{eval_run_dir_suffix}")
        if os.path.isfile(os.path.join(d, "args.yaml")):
            return d
        return None
    d = os.path.join(sweep_dir, tag)
    if os.path.isfile(os.path.join(d, "args.yaml")):
        return d
    if not eval_resolve_run_dir:
        return None
    for i in range(1, 51):
        d2 = os.path.join(sweep_dir, f"{tag}r{i}")
        if os.path.isfile(os.path.join(d2, "args.yaml")):
            return d2
    return None


def iter_run_dirs(sweep_dir: str) -> Iterator[tuple[str, str]]:
    """Yield (basename, full_path) for each immediate child that has args.yaml."""
    try:
        names = sorted(os.listdir(sweep_dir))
    except OSError as e:
        print(f"ERROR: cannot list {sweep_dir}: {e}", file=sys.stderr)
        sys.exit(1)
    for name in names:
        full = os.path.join(sweep_dir, name)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "args.yaml")):
            yield name, full


def count_periodic_checkpoints(run_dir: str) -> int:
    """Number of checkpoint_<step>/ subdirs under run_dir that contain model.pth."""
    n = 0
    try:
        names = os.listdir(run_dir)
    except OSError:
        return 0
    for name in names:
        if not _CHECKPOINT_SUBDIR_RE.match(name):
            continue
        if os.path.isfile(os.path.join(run_dir, name, "model.pth")):
            n += 1
    return n


def find_one_rollout_eval_gif(run_dir: str) -> str | None:
    """
    Pick a single eval GIF under rollout/: prefer final/, then flat eval_0.gif,
    else highest checkpoint_*/eval_0.gif.
    """
    rollout = os.path.join(run_dir, "rollout")
    if not os.path.isdir(rollout):
        return None
    final_gif = os.path.join(rollout, "final", "eval_0.gif")
    if os.path.isfile(final_gif):
        return final_gif
    root_gif = os.path.join(rollout, "eval_0.gif")
    if os.path.isfile(root_gif):
        return root_gif
    best_step = -1
    best_path: str | None = None
    try:
        names = os.listdir(rollout)
    except OSError:
        return None
    for name in names:
        m = _CHECKPOINT_SUBDIR_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        p = os.path.join(rollout, name, "eval_0.gif")
        if os.path.isfile(p) and step > best_step:
            best_step = step
            best_path = p
    return best_path


def _write_missing_report(
    path: str,
    *,
    no_run_dir: list[str],
    low_checkpoints: list[tuple[str, int, str | None]],
    no_gif: list[tuple[str, str]],
    copied_ok: list[str],
) -> None:
    lines = [
        "# Sweep evaluation GIF aggregation report",
        "#",
        "# Sections list sweep task tags (folder basename keys from generate_sweep).",
        "",
        "## Copied OK (one GIF per line)",
    ]
    if copied_ok:
        lines.extend(f"  {t}" for t in copied_ok)
    else:
        lines.append("  (none)")
    lines += [
        "",
        "## No training run directory (args.yaml not found at expected path)",
    ]
    if no_run_dir:
        lines.extend(f"  {t}" for t in no_run_dir)
    else:
        lines.append("  (none)")
    lines += [
        "",
        "## Run dir exists but periodic checkpoint count < min (no GIF copied; may be deleted if cleanup_apply)",
    ]
    if low_checkpoints:
        for tag, n_ck, resolved in low_checkpoints:
            extra = f" -> {resolved}" if resolved else ""
            lines.append(f"  {tag}  ({n_ck} checkpoints){extra}")
    else:
        lines.append("  (none)")
    lines += [
        "",
        "## Enough checkpoints but no eval_0.gif found under rollout/",
    ]
    if no_gif:
        for tag, rdir in no_gif:
            lines.append(f"  {tag}  ({rdir})")
    else:
        lines.append("  (none)")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def postprocess_reward_sweep(
    sweep_dir: str,
    min_checkpoints: int,
    *,
    sweep_cfg: dict | None = None,
    aggregate_gifs: bool = False,
    gifs_subdir: str = "full_evaluation_gifs",
    missing_report_name: str = "missing_evaluation_gifs.txt",
    cleanup_apply: bool = False,
    eval_run_dir_suffix: str | None = None,
    eval_resolve_run_dir: bool = True,
    verbose: bool = False,
) -> None:
    """
    Classify runs by checkpoint count; optionally copy GIFs from kept runs then delete low runs.

    If sweep_cfg is set, imports sweep_run_tags from generate_sweep for the missing report.
    eval_* options must match the eval bash script resolution when locating each tag's run dir.
    """
    sweep_dir = os.path.abspath(sweep_dir)
    if not os.path.isdir(sweep_dir):
        print(f"ERROR: sweep dir not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)
    if min_checkpoints < 0:
        print("ERROR: min_checkpoints must be >= 0.", file=sys.stderr)
        sys.exit(1)

    suffix = eval_run_dir_suffix
    resolve_flag = eval_resolve_run_dir
    if sweep_cfg is not None:
        if suffix is None and "eval_run_dir_suffix" in sweep_cfg:
            suffix = sweep_cfg.get("eval_run_dir_suffix")
        resolve_flag = bool(sweep_cfg.get("eval_resolve_run_dir", True))

    to_remove: list[tuple[str, str, int]] = []
    kept: list[tuple[str, str, int]] = []

    for name, full in iter_run_dirs(sweep_dir):
        n_ckpt = count_periodic_checkpoints(full)
        if n_ckpt < min_checkpoints:
            to_remove.append((name, full, n_ckpt))
        else:
            kept.append((name, full, n_ckpt))

    mode = "APPLY" if cleanup_apply else "DRY RUN"
    print(f"{mode}: sweep_dir={sweep_dir}  min_checkpoints={min_checkpoints}")
    print(f"Runs scanned: {len(to_remove) + len(kept)}  remove: {len(to_remove)}  keep: {len(kept)}")

    if verbose and kept:
        print("\nKeeping (on-disk):")
        for name, _full, n_ckpt in kept:
            print(f"  {name}  ({n_ckpt} checkpoints)")

    out_dir: str | None = None
    if aggregate_gifs:
        out_dir = os.path.join(sweep_dir, gifs_subdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nAggregate GIFs -> {out_dir}")
        for name, full, n_ckpt in kept:
            gif_src = find_one_rollout_eval_gif(full)
            if not gif_src:
                print(f"  skip (no eval_0.gif): {name}")
                continue
            dest = os.path.join(out_dir, f"{name}.gif")
            try:
                shutil.copy2(gif_src, dest)
                print(f"  copied {name}.gif")
            except OSError as e:
                print(f"  ERROR copy {name}: {e}", file=sys.stderr)

    expected_tags: list[str] | None = None
    if sweep_cfg is not None:
        from scripts.smooth_policy.amp_history.amp_training.td3.helper.generate_sweep import sweep_run_tags

        expected_tags = sweep_run_tags(sweep_cfg)

    if aggregate_gifs and out_dir is not None and expected_tags is not None:
        no_run_dir: list[str] = []
        low_checkpoints: list[tuple[str, int, str | None]] = []
        no_gif: list[tuple[str, str]] = []
        copied_ok: list[str] = []

        for tag in expected_tags:
            resolved = resolve_run_dir(
                sweep_dir,
                tag,
                eval_run_dir_suffix=suffix,
                eval_resolve_run_dir=resolve_flag,
            )
            if resolved is None:
                no_run_dir.append(tag)
                continue
            n_ck = count_periodic_checkpoints(resolved)
            if n_ck < min_checkpoints:
                low_checkpoints.append((tag, n_ck, resolved))
                continue
            gif_path = find_one_rollout_eval_gif(resolved)
            dest_gif = os.path.join(out_dir, f"{os.path.basename(resolved)}.gif")
            if gif_path and os.path.isfile(dest_gif):
                copied_ok.append(tag)
            elif not gif_path:
                no_gif.append((tag, resolved))
            else:
                no_gif.append((tag, resolved))

        report_path = os.path.join(out_dir, missing_report_name)
        _write_missing_report(
            report_path,
            no_run_dir=no_run_dir,
            low_checkpoints=low_checkpoints,
            no_gif=no_gif,
            copied_ok=copied_ok,
        )
        print(f"Wrote {report_path}")
    elif aggregate_gifs and out_dir is not None:
        # No sweep YAML: report only on-disk kept runs missing GIFs
        missing_only = [name for name, full, _ in kept if not find_one_rollout_eval_gif(full)]
        report_path = os.path.join(out_dir, missing_report_name)
        _write_missing_report(
            report_path,
            no_run_dir=[],
            low_checkpoints=[],
            no_gif=[(n, os.path.join(sweep_dir, n)) for n in missing_only],
            copied_ok=[name for name, _, _ in kept if name not in missing_only],
        )
        print(f"Wrote {report_path}")

    if not to_remove:
        print("\nNothing to remove (checkpoint cleanup).")
        return

    print("\nRemove (checkpoint count < min):")
    for name, full, n_ckpt in to_remove:
        print(f"  {name}  ({n_ckpt} checkpoints) -> {full}")

    if not cleanup_apply:
        print("\nNo directories deleted. Re-run with --apply to remove them.")
        return

    errors = 0
    for name, full, n_ckpt in to_remove:
        try:
            shutil.rmtree(full)
            print(f"Removed {name}")
        except OSError as e:
            print(f"ERROR removing {full}: {e}", file=sys.stderr)
            errors += 1
    if errors:
        sys.exit(1)


def _sweep_dir_and_cfg(args: argparse.Namespace) -> tuple[str, dict | None]:
    if args.sweep_file:
        if not os.path.isfile(args.sweep_file):
            print(f"ERROR: sweep file not found: {args.sweep_file}", file=sys.stderr)
            sys.exit(1)
        with open(args.sweep_file) as f:
            sweep_cfg = yaml.safe_load(f) or {}
        if "log_parent_dir" not in sweep_cfg:
            print("ERROR: sweep YAML must contain 'log_parent_dir'.", file=sys.stderr)
            sys.exit(1)
        return sweep_cfg["log_parent_dir"], sweep_cfg
    assert args.sweep_dir is not None
    return args.sweep_dir, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete sweep run dirs with fewer than N periodic TD3 checkpoints; optional GIF aggregation."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--sweep-file",
        help="Sweep YAML with log_parent_dir (same as aggregate_sweep.py).",
    )
    src.add_argument(
        "--sweep-dir",
        help="Path to the sweep log parent directory (contains one subdir per run).",
    )
    parser.add_argument(
        "--min-checkpoints",
        type=int,
        required=True,
        metavar="N",
        help="Treat runs with fewer than N checkpoint_*/model.pth as low; remove if --apply.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete low-checkpoint directories. Without this, only print what would be removed.",
    )
    parser.add_argument(
        "--aggregate-gifs",
        action="store_true",
        help="Copy one eval_0.gif per kept run into <sweep-dir>/<gifs-subdir>/; write missing report.",
    )
    parser.add_argument(
        "--gifs-subdir",
        default="full_evaluation_gifs",
        help="Subdirectory under sweep dir for aggregated GIFs (default: full_evaluation_gifs).",
    )
    parser.add_argument(
        "--missing-report-name",
        default="missing_evaluation_gifs.txt",
        help="Written inside the GIF output folder when --aggregate-gifs is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print runs that are kept (with checkpoint counts).",
    )
    args = parser.parse_args()

    sweep_dir, sweep_cfg = _sweep_dir_and_cfg(args)
    postprocess_reward_sweep(
        sweep_dir=sweep_dir,
        min_checkpoints=args.min_checkpoints,
        sweep_cfg=sweep_cfg,
        aggregate_gifs=args.aggregate_gifs,
        gifs_subdir=args.gifs_subdir,
        missing_report_name=args.missing_report_name,
        cleanup_apply=args.apply,
        eval_run_dir_suffix=sweep_cfg.get("eval_run_dir_suffix") if sweep_cfg else None,
        eval_resolve_run_dir=bool(sweep_cfg.get("eval_resolve_run_dir", True)) if sweep_cfg else True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
