"""Per-episode puck-observation defects vs episode performance for real eval runs.

For each run folder (``eval_per_episode.jsonl`` + ``episode_hdf5/``), counts per kept episode the
logged-visible, logged-occluded (held last value; the reset placeholder before the first sighting
is excluded) and phantom frames, classified exactly as in
``scripts/visualization/plot_real_puck_tracks.py``. Writes one table per run, rows sorted by
episode return (highest first), plus Spearman correlations of return with the counts and with
the per-frame rates (counts scale with episode length, and so does return).

Usage:
    python scripts/analysis/puck_defects_vs_performance.py \
        --run-dirs <run_dir> [<run_dir> ...] --out defects_vs_performance.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.visualization.plot_real_puck_tracks import (  # noqa: E402
    camera_track,
    classify,
    fit_homography,
    load_episode,
)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    # average ranks for ties
    for x, r in ((a, ra), (b, rb)):
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def run_rows(run_dir: Path, phantom_dist: float, max_speed: float):
    records = [json.loads(line) for line in open(run_dir / "eval_per_episode.jsonl") if line.strip()]
    paths = []
    for r in records:
        art = Path(r["artifact_path"])
        paths.append(run_dir / "episode_hdf5" / art.parent.name / art.name)
    episodes = [load_episode(p) for p in paths]
    H = fit_homography(episodes)
    rows = []
    for r, p, ep in zip(records, paths, episodes):
        track = camera_track(ep, H)
        occ, phantom, placeholder = classify(ep, track, phantom_dist, max_speed)
        vis = ~occ & ~phantom
        held = occ & ~placeholder
        T = len(occ)
        rows.append(dict(
            episode=r["episode_id"], file=p.stem, ret=float(r["episode_return"]), steps=T,
            visible=int(vis.sum()), occluded=int(held.sum()), phantom=int(phantom.sum()),
            occ_pct=100.0 * held.sum() / T, phantom_pct=100.0 * phantom.sum() / T,
            end=r.get("episode_end_reason", ""),
        ))
    rows.sort(key=lambda d: -d["ret"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="Markdown output path (also printed)")
    ap.add_argument("--phantom-dist", type=float, default=0.08)
    ap.add_argument("--max-speed", type=float, default=5.0)
    args = ap.parse_args()

    lines = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        rows = run_rows(run_dir, args.phantom_dist, args.max_speed)
        lines.append(f"## {run_dir.parent.name} / {run_dir.name}\n")
        lines.append("| episode | file | return | steps | logged visible | logged occluded | phantom | occluded % | phantom % | end reason |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in rows:
            lines.append(f"| {d['episode']} | {d['file']} | {d['ret']:.1f} | {d['steps']} | {d['visible']} | {d['occluded']} | "
                         f"{d['phantom']} | {d['occ_pct']:.1f} | {d['phantom_pct']:.1f} | {d['end']} |")
        ret = [d["ret"] for d in rows]
        lines.append("")
        lines.append("Spearman ρ with return (n = %d): " % len(rows) + ", ".join(
            f"{k} {spearman(ret, [d[k] for d in rows]):+.2f}"
            for k in ("steps", "occluded", "phantom", "occ_pct", "phantom_pct")))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
