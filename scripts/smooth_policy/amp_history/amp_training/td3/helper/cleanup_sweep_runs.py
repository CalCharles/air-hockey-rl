"""
TD3 sweep log cleanup under ``log_parent_dir``.

**Pruning policy (two reasons to remove a run folder)**

1. **Too few periodic checkpoints** — Any immediate child run dir that contains
   ``args.yaml`` and has fewer than ``min_checkpoints`` subdirs ``checkpoint_<step>/``
   with ``model.pth`` is removed when ``cleanup_apply`` is true (not only the
   eval-canonical path; e.g. a leftover ``tagr1/`` with a failed short run is removed too).

2. **No ``args.yaml``** — Hollow dirs, stale ``tag/`` when ``tagrN/`` holds the run,
   abandoned expected slots (with ``--sweep-file``), and bumped ``tagrN/`` dirs without
   ``args.yaml`` are removed by the corresponding prune passes (see below).

**Canonical dirs** (same ``<tag>`` vs ``<tag>r1`` resolution as eval) drive **GIF
aggregation** only. Non-canonical dirs with ``args.yaml`` are still subject to the
low-checkpoint rule above; use ``--remove-noncanonical-run-dirs`` with ``--apply`` to
also delete non-canonical dirs that *do* have enough checkpoints.

With ``prune_hollow_dirs``, immediate children with no ``args.yaml`` and **no files**
anywhere under them are removed.

With ``prune_stale_tag_dirs``, removes ``tag/`` with no ``args.yaml`` when ``tagrN/`` has
the real run.

With ``prune_abandoned_tag_dirs`` and ``--sweep-file``, removes expected ``tag/`` with no
``args.yaml`` at ``tag/`` or any ``tagrN/``.

With ``prune_r_suffix_without_args``, removes any ``tagrN/`` child with no ``args.yaml``.

With ``promote_r_dirs``, renames ``tagrN/`` -> ``tag/`` when appropriate.

Periodic checkpoints are ``checkpoint_<global_step>/model.pth`` (same as
``collect_policy_data.discover_td3_policy_snapshots``).

``generate_sweep.py`` can run this when the sweep YAML sets ``post_min_checkpoints``.

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
# TD3 uses <tag>r1, r2 with no separator before ``r``.
_STEM_FROM_RUN_BASENAME_RE = re.compile(r"^(.*)r(\d+)$")


def sweep_stem_from_run_basename(basename: str) -> str:
    """``motion_gamma_0p7r3`` -> ``motion_gamma_0p7``; ``motion_gamma_0p7`` unchanged."""
    m = _STEM_FROM_RUN_BASENAME_RE.match(basename)
    if m:
        return m.group(1)
    return basename


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


def _is_dir_empty(path: str) -> bool:
    try:
        return os.path.isdir(path) and len(os.listdir(path)) == 0
    except OSError:
        return False


def _dir_tree_contains_any_file(root: str) -> bool:
    """True if any regular file exists anywhere under ``root``."""
    try:
        for _dirpath, _dirnames, filenames in os.walk(root):
            if filenames:
                return True
    except OSError:
        return True
    return False


def prune_hollow_sweep_subdirs(
    sweep_dir: str,
    *,
    skip_basenames: set[str],
) -> list[str]:
    """
    Remove immediate child directories that have no ``args.yaml`` and contain **no files**
    anywhere beneath them (only empty or nested-empty folders).

    Does not remove dirs that have any file (including under subfolders). Returns basenames
    removed.
    """
    removed: list[str] = []
    try:
        names = sorted(os.listdir(sweep_dir))
    except OSError:
        return removed
    for name in names:
        if name in skip_basenames:
            continue
        full = os.path.join(sweep_dir, name)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "args.yaml")):
            continue
        if _dir_tree_contains_any_file(full):
            continue
        try:
            shutil.rmtree(full)
            removed.append(name)
            print(f"  pruned hollow dir (no args.yaml, no files): {name}/")
        except OSError as e:
            print(f"ERROR pruning {full}: {e}", file=sys.stderr)
    return removed


def prune_stale_tag_dirs_without_args(
    sweep_dir: str,
    *,
    skip_basenames: set[str],
) -> list[str]:
    """
    Remove immediate child dirs whose basename is **not** ``tagrN`` form, have no
    ``args.yaml``, but ``tagr1``..``tagr50`` exists with ``args.yaml``.

    This clears placeholder ``motion_reward_weight_0p5/`` trees (often with stray files)
    when the real run lives in ``motion_reward_weight_0p5r1/``. Skips basenames matching
    ``...r<number>`` (those are the real bumped dirs, not stale bases).
    """
    removed: list[str] = []
    try:
        names = sorted(os.listdir(sweep_dir))
    except OSError:
        return removed
    for name in names:
        if name in skip_basenames:
            continue
        full = os.path.join(sweep_dir, name)
        if not os.path.isdir(full):
            continue
        if _STEM_FROM_RUN_BASENAME_RE.match(name):
            continue
        if os.path.isfile(os.path.join(full, "args.yaml")):
            continue
        if _lowest_r_suffix_run_dir(sweep_dir, name) is None:
            continue
        try:
            shutil.rmtree(full)
            removed.append(name)
            print(f"  removed stale {name}/ (no args.yaml; training under {name}rN/)")
        except OSError as e:
            print(f"ERROR removing stale {full}: {e}", file=sys.stderr)
    return removed


def prune_abandoned_expected_tag_dirs(
    sweep_dir: str,
    expected_tags: list[str],
    *,
    skip_basenames: set[str],
) -> list[str]:
    """
    For each tag in the sweep YAML list: if ``tag/`` exists, has no ``args.yaml``, and no
    ``tagrN/args.yaml`` exists, remove ``tag/`` (failed launch, partial logs, never started).

    Only runs when sweep tags are known (``--sweep-file``). Does not remove dirs whose
    basename is ``tagrN`` form.
    """
    removed: list[str] = []
    for tag in dict.fromkeys(expected_tags):
        if tag in skip_basenames:
            continue
        if _STEM_FROM_RUN_BASENAME_RE.match(tag):
            continue
        base = os.path.join(sweep_dir, tag)
        if not os.path.isdir(base):
            continue
        if os.path.isfile(os.path.join(base, "args.yaml")):
            continue
        if _lowest_r_suffix_run_dir(sweep_dir, tag) is not None:
            continue
        try:
            shutil.rmtree(base)
            removed.append(tag)
            print(f"  removed abandoned {tag}/ (no args.yaml and no {tag}rN/ with args)")
        except OSError as e:
            print(f"ERROR removing abandoned {base}: {e}", file=sys.stderr)
    return removed


def prune_r_suffix_dirs_without_args(
    sweep_dir: str,
    *,
    skip_basenames: set[str],
) -> list[str]:
    """
    Remove immediate children whose basename matches ``...r<number>`` (TD3 bumped dir) but
    contain no ``args.yaml`` (crash before config write).
    """
    removed: list[str] = []
    try:
        names = sorted(os.listdir(sweep_dir))
    except OSError:
        return removed
    for name in names:
        if name in skip_basenames:
            continue
        if not _STEM_FROM_RUN_BASENAME_RE.match(name):
            continue
        full = os.path.join(sweep_dir, name)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "args.yaml")):
            continue
        try:
            shutil.rmtree(full)
            removed.append(name)
            print(f"  removed {name}/ (bumped run dir, no args.yaml)")
        except OSError as e:
            print(f"ERROR removing {full}: {e}", file=sys.stderr)
    return removed


def _lowest_r_suffix_run_dir(sweep_dir: str, tag: str) -> tuple[int, str] | None:
    """Smallest N>=1 such that ``<tag>rN/args.yaml`` exists. Returns (N, abs_path)."""
    for i in range(1, 51):
        r_path = os.path.join(sweep_dir, f"{tag}r{i}")
        if os.path.isfile(os.path.join(r_path, "args.yaml")):
            return i, os.path.abspath(r_path)
    return None


def promote_r_suffix_into_base_tag(sweep_dir: str, tag: str) -> bool:
    """
    If ``tag/args.yaml`` is missing and some ``tagrN`` has ``args.yaml``:
    rename ``tagrN`` -> ``tag`` when ``tag`` is absent, or replace an empty ``tag`` dir.

    If ``tag`` exists, is non-empty, and has no ``args.yaml``, logs a skip (no merge).

    Returns True if a promotion was performed.
    """
    base = os.path.join(sweep_dir, tag)
    if os.path.isfile(os.path.join(base, "args.yaml")):
        return False
    found = _lowest_r_suffix_run_dir(sweep_dir, tag)
    if found is None:
        return False
    _n, r_abs = found
    r_bn = os.path.basename(r_abs)
    try:
        if not os.path.exists(base):
            shutil.move(r_abs, base)
            print(f"  promoted: {r_bn}/ -> {tag}/ (renamed)")
            return True
        if _is_dir_empty(base):
            os.rmdir(base)
            shutil.move(r_abs, base)
            print(f"  promoted: {r_bn}/ -> {tag}/ (replaced empty {tag}/)")
            return True
        print(
            f"  skip promote {tag}: {tag}/ exists, is not empty, and has no args.yaml "
            f"(would use {r_bn}/; empty or remove {tag}/ manually)",
            file=sys.stderr,
        )
        return False
    except OSError as e:
        print(f"ERROR promote {tag} ({r_bn} -> {tag}/): {e}", file=sys.stderr)
        return False


def _tags_for_promotion(sweep_dir: str, expected_tags: list[str] | None) -> list[str]:
    if expected_tags is not None:
        return list(expected_tags)
    stems: set[str] = set()
    for name, _full in iter_run_dirs(sweep_dir):
        stems.add(sweep_stem_from_run_basename(name))
    return sorted(stems)


def collect_canonical_run_paths(
    sweep_dir: str,
    *,
    sweep_tags: list[str] | None,
    eval_run_dir_suffix: str | None,
    eval_resolve_run_dir: bool,
) -> tuple[set[str], list[str]]:
    """
    Canonical abs paths for eval (``tag`` if ``args.yaml``, else ``tagr1``..``r50``),
    or from inferred stems when ``sweep_tags`` is None.
    """
    if sweep_tags is not None:
        ordered = list(sweep_tags)
        paths: set[str] = set()
        for tag in ordered:
            p = resolve_run_dir(
                sweep_dir,
                tag,
                eval_run_dir_suffix=eval_run_dir_suffix,
                eval_resolve_run_dir=eval_resolve_run_dir,
            )
            if p is not None:
                paths.add(os.path.abspath(p))
        return paths, ordered

    stems: set[str] = set()
    for name, _full in iter_run_dirs(sweep_dir):
        stems.add(sweep_stem_from_run_basename(name))
    paths = set()
    ordered_stems = sorted(stems)
    for stem in ordered_stems:
        p = resolve_run_dir(
            sweep_dir,
            stem,
            eval_run_dir_suffix=eval_run_dir_suffix,
            eval_resolve_run_dir=eval_resolve_run_dir,
        )
        if p is not None:
            paths.add(os.path.abspath(p))
    return paths, ordered_stems


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
    remove_noncanonical_run_dirs: bool = False,
    promote_r_dirs: bool = False,
    prune_hollow_dirs: bool = False,
    prune_stale_tag_dirs: bool = False,
    prune_abandoned_tag_dirs: bool = False,
    prune_r_suffix_without_args: bool = False,
    eval_run_dir_suffix: str | None = None,
    eval_resolve_run_dir: bool = True,
    verbose: bool = False,
) -> None:
    """
    **Order of operations**

    1. **Prune** — Remove dirs with no ``args.yaml`` (hollow, stale base, abandoned expected
       slots, ``tagrN/`` without args), per flags.
    2. **Promote** — ``tagrN/`` -> ``tag/`` when needed.
    3. **GIF aggregation** — Copy eval GIFs from kept canonical runs; write
       ``missing_evaluation_gifs.txt`` recording sweep tasks that are missing a run dir,
       below min checkpoints, or missing a GIF (only when ``aggregate_gifs``).
    4. **Removal** — Delete low-checkpoint run dirs (and optional non-canonical) when
       ``cleanup_apply``.

    If sweep_cfg is set, imports sweep_run_tags from generate_sweep for the missing report.
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

    expected_tags: list[str] | None = None
    if sweep_cfg is not None:
        from scripts.smooth_policy.amp_history.amp_training.td3.helper.generate_sweep import sweep_run_tags

        expected_tags = sweep_run_tags(sweep_cfg)

    skip_prune = {os.path.basename(gifs_subdir.strip("/"))} if gifs_subdir else set()
    skip_prune.discard("")
    skip_prune.add("full_evaluation_gifs")

    # ----- Phase 1: Prune (no args.yaml / invalid run shells) -----
    _any_prune = (
        prune_hollow_dirs
        or prune_stale_tag_dirs
        or prune_abandoned_tag_dirs
        or prune_r_suffix_without_args
    )
    print("\n=== Phase 1: Prune (no args.yaml) ===")
    if not _any_prune:
        print("(all prune_* steps disabled)")
    if prune_hollow_dirs:
        print("— Hollow dirs (no args.yaml, no files under tree)")
        prune_hollow_sweep_subdirs(sweep_dir, skip_basenames=skip_prune)
    if prune_stale_tag_dirs:
        if suffix is not None:
            print("— Stale tag/: skipped (eval_run_dir_suffix is set)")
        elif not resolve_flag:
            print("— Stale tag/: skipped (eval_resolve_run_dir is false)")
        else:
            print("— Stale tag/ (no args; real run is tagrN/)")
            prune_stale_tag_dirs_without_args(sweep_dir, skip_basenames=skip_prune)
    if prune_abandoned_tag_dirs:
        if expected_tags is None:
            print("— Abandoned tag/: skipped (need --sweep-file)")
        else:
            print("— Abandoned expected tag/ (no args at tag/ or tagrN/)")
            prune_abandoned_expected_tag_dirs(sweep_dir, expected_tags, skip_basenames=skip_prune)
    if prune_r_suffix_without_args:
        print("— Bumped tagrN/ with no args.yaml")
        prune_r_suffix_dirs_without_args(sweep_dir, skip_basenames=skip_prune)

    # ----- Phase 2: Promote tagrN -> tag -----
    print("\n=== Phase 2: Promote (tagrN -> tag) ===")
    if promote_r_dirs:
        if suffix is not None:
            print("skipped (eval_run_dir_suffix is set)")
        elif not resolve_flag:
            print("skipped (eval_resolve_run_dir is false)")
        else:
            for t in _tags_for_promotion(sweep_dir, expected_tags):
                promote_r_suffix_into_base_tag(sweep_dir, t)
    else:
        print("(promote_r_dirs disabled)")

    # After prune+promote: canonical layout and checkpoint classification
    canonical_paths, _ordering_key = collect_canonical_run_paths(
        sweep_dir,
        sweep_tags=expected_tags,
        eval_run_dir_suffix=suffix,
        eval_resolve_run_dir=resolve_flag,
    )

    all_run_paths = {os.path.abspath(full) for _name, full in iter_run_dirs(sweep_dir)}
    noncanonical_paths = sorted(all_run_paths - canonical_paths)

    to_remove_low: list[tuple[str, str, int]] = []
    for name, full in sorted(list(iter_run_dirs(sweep_dir)), key=lambda x: x[1]):
        n_ckpt = count_periodic_checkpoints(full)
        if n_ckpt < min_checkpoints:
            to_remove_low.append((name, full, n_ckpt))

    kept: list[tuple[str, str, int]] = []
    for full in sorted(canonical_paths):
        name = os.path.basename(full)
        n_ckpt = count_periodic_checkpoints(full)
        if n_ckpt >= min_checkpoints:
            kept.append((name, full, n_ckpt))

    if verbose and kept:
        print("\nCanonical runs with enough checkpoints (GIF sources):")
        for name, _full, n_ckpt in kept:
            print(f"  {name}  ({n_ckpt} checkpoints)")

    # ----- Phase 3: Aggregate GIFs + missing report -----
    out_dir: str | None = None
    if aggregate_gifs:
        print("\n=== Phase 3: Aggregate GIFs + missing report ===")
        out_dir = os.path.join(sweep_dir, gifs_subdir)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output: {out_dir}")
        for name, full, n_ckpt in kept:
            gif_src = find_one_rollout_eval_gif(full)
            if not gif_src:
                print(f"  skip (no eval_0.gif under rollout/): {name}")
                continue
            dest = os.path.join(out_dir, f"{name}.gif")
            try:
                shutil.copy2(gif_src, dest)
                print(f"  copied {name}.gif")
            except OSError as e:
                print(f"  ERROR copy {name}: {e}", file=sys.stderr)

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
        print(f"Recorded missing / status -> {report_path}")
    elif aggregate_gifs and out_dir is not None:
        missing_only = [name for name, full, _ in kept if not find_one_rollout_eval_gif(full)]
        report_path = os.path.join(out_dir, missing_report_name)
        _write_missing_report(
            report_path,
            no_run_dir=[],
            low_checkpoints=[],
            no_gif=[(n, os.path.join(sweep_dir, n)) for n in missing_only],
            copied_ok=[name for name, _, _ in kept if name not in missing_only],
        )
        print(f"Recorded missing / status -> {report_path}")
    elif not aggregate_gifs:
        print("\n=== Phase 3: Aggregate GIFs (skipped) ===")

    # ----- Phase 4: Remove low-checkpoint / non-canonical dirs -----
    print("\n=== Phase 4: Low-checkpoint removal ===")
    mode = "APPLY" if cleanup_apply else "DRY RUN"
    print(f"{mode}: sweep_dir={sweep_dir}  min_checkpoints={min_checkpoints}")
    print(
        f"Canonical run dirs: {len(canonical_paths)}  eligible for GIFs: {len(kept)}  "
        f"run dirs below min checkpoints: {len(to_remove_low)}"
    )
    if noncanonical_paths:
        print(f"Non-canonical run dirs on disk: {len(noncanonical_paths)}")
        if verbose:
            for p in noncanonical_paths:
                print(f"  {os.path.basename(p)}  -> {p}")

    if not to_remove_low and not (remove_noncanonical_run_dirs and noncanonical_paths):
        print("\nNothing to remove (checkpoint cleanup).")
        if noncanonical_paths and not remove_noncanonical_run_dirs:
            print(
                "Non-canonical dirs skipped for deletion (use --remove-noncanonical-run-dirs with --apply)."
            )
        return

    print("\nRemove (low checkpoint — any run dir with args.yaml):")
    for name, full, n_ckpt in to_remove_low:
        print(f"  {name}  ({n_ckpt} checkpoints) -> {full}")
    removed_by_low_ckpt = {os.path.abspath(full) for _name, full, _ in to_remove_low}
    if remove_noncanonical_run_dirs and noncanonical_paths:
        extra_nc = [p for p in noncanonical_paths if os.path.abspath(p) not in removed_by_low_ckpt]
        if extra_nc:
            print("\nRemove (non-canonical, sufficient checkpoints):")
            for p in extra_nc:
                print(f"  {os.path.basename(p)}  -> {p}")

    if not cleanup_apply:
        print("\nNo directories deleted. Re-run with --apply to remove them.")
        if noncanonical_paths and not remove_noncanonical_run_dirs:
            print("Tip: --remove-noncanonical-run-dirs deletes extra tagrN sibling folders.")
        return

    errors = 0
    for name, full, n_ckpt in to_remove_low:
        try:
            shutil.rmtree(full)
            print(f"Removed {name}")
        except OSError as e:
            print(f"ERROR removing {full}: {e}", file=sys.stderr)
            errors += 1
    if remove_noncanonical_run_dirs:
        for p in noncanonical_paths:
            if os.path.abspath(p) in removed_by_low_ckpt:
                continue
            try:
                shutil.rmtree(p)
                print(f"Removed {os.path.basename(p)} (non-canonical)")
            except OSError as e:
                print(f"ERROR removing {p}: {e}", file=sys.stderr)
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
    parser.add_argument(
        "--remove-noncanonical-run-dirs",
        action="store_true",
        help="With --apply, also delete args.yaml dirs that are not the eval-resolved path "
        "(e.g. tagr1 when tag wins).",
    )
    parser.add_argument(
        "--promote-r-dirs",
        action="store_true",
        help="Before cleanup, rename tagrN/ -> tag/ when tag has no args.yaml and tag is "
        "missing or an empty directory (lowest N wins).",
    )
    parser.add_argument(
        "--prune-hollow-dirs",
        action="store_true",
        help="Remove sweep subdirs with no args.yaml and no files anywhere beneath (before promote).",
    )
    parser.add_argument(
        "--prune-stale-tag-dirs",
        action="store_true",
        help="Remove tag/ with no args.yaml when tagrN/ has args.yaml (before promote).",
    )
    parser.add_argument(
        "--prune-abandoned-tag-dirs",
        action="store_true",
        help="With --sweep-file: remove expected tag/ with no args at tag/ or any tagrN/ (failed slots).",
    )
    parser.add_argument(
        "--prune-r-suffix-without-args",
        action="store_true",
        help="Remove tagrN/ immediate children that have no args.yaml.",
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
        remove_noncanonical_run_dirs=args.remove_noncanonical_run_dirs,
        promote_r_dirs=args.promote_r_dirs,
        prune_hollow_dirs=args.prune_hollow_dirs,
        prune_stale_tag_dirs=args.prune_stale_tag_dirs,
        prune_abandoned_tag_dirs=args.prune_abandoned_tag_dirs,
        prune_r_suffix_without_args=args.prune_r_suffix_without_args,
        eval_run_dir_suffix=sweep_cfg.get("eval_run_dir_suffix") if sweep_cfg else None,
        eval_resolve_run_dir=bool(sweep_cfg.get("eval_resolve_run_dir", True)) if sweep_cfg else True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
