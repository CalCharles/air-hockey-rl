"""Where on the table does the real puck detector lose the puck?

For every episode HDF5 under the given run folders, each occlusion run (logged ``occluded`` frames,
plus phantom frames — logged visible at a wrong position, see ``plot_real_puck_tracks.py`` —
which are treated as lost too) becomes a segment from the last good logged puck position before
the run to the first good position after it. Runs that last until the episode ends have no end
point and are skipped (counted in the title).

Writes two PNGs on the Box2D table (robot / paddle end on the right, as in the overhead camera):

1. ``occlusion_segments.png`` — every segment (start ● green, end ● red) over a heat map of how
   many segments cross each cell.
2. ``occlusion_probability.png`` — P(occluded | puck in cell) = occluded frames in the cell / all
   frames in the cell. Visible frames sit at their logged position; occluded frames are placed by
   linear interpolation along their segment (frame k of an n-frame run at fraction k / (n + 1)).
   Cells with fewer than ``--min-frames`` frames are greyed out; each cell shows its frame count.

Usage:
    python scripts/visualization/plot_occlusion_heatmaps.py \
        --run-dirs results/eval_4_tasks_of_policies_hist4_2026-09-19/juggle_sysid/data_20261014-201743 \
        --out-dir results/eval_4_tasks_of_policies_hist4_2026-09-19/juggle_sysid/data_20261014-201743/occlusion_heatmaps
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.visualization.plot_real_puck_tracks import (  # noqa: E402
    TABLE_LENGTH,
    TABLE_WIDTH,
    camera_track,
    classify,
    fit_homography,
    load_episode,
)
from scripts.visualization.visualize_real_trajectory import RealTrajectoryRenderer  # noqa: E402

EXTENT = (-TABLE_LENGTH / 2, TABLE_LENGTH / 2, -TABLE_WIDTH / 2, TABLE_WIDTH / 2)


def episode_segments(ep, track, phantom_dist, max_speed):
    """Return (segments, visible_xy, occluded_xy, n_open) for one episode."""
    xy = ep["puck"][:, :2]
    occ, phantom, placeholder = classify(ep, track, phantom_dist, max_speed)
    lost = (occ | phantom) & ~placeholder
    good = ~occ & ~phantom
    T = len(xy)
    segments, occ_xy, n_open = [], [], 0
    t = 0
    while t < T:
        if not lost[t]:
            t += 1
            continue
        j = t
        while j < T and lost[j]:
            j += 1
        start = t - 1
        if start < 0 or not good[start]:
            t = j
            continue
        if j >= T or not good[j]:
            n_open += 1
            t = j
            continue
        p0, p1, n = xy[start], xy[j], j - t
        segments.append((p0, p1, n))
        for k in range(1, n + 1):
            occ_xy.append(p0 + (p1 - p0) * k / (n + 1))
        t = j
    return segments, xy[good], np.array(occ_xy).reshape(-1, 2), n_open


def draw_table(ax, table_img):
    ax.imshow(cv2.cvtColor(cv2.rotate(table_img, cv2.ROTATE_180), cv2.COLOR_BGR2RGB), extent=EXTENT, alpha=0.6, zorder=0)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xlabel("table x (m)  —  robot / paddle end →")
    ax.set_ylabel("table y (m)")


def shade_excluded(ax, x_min, x_max):
    for lo, hi in ((EXTENT[0], x_min), (x_max, EXTENT[1])):
        if lo is not None and hi is not None and hi > lo:
            ax.axvspan(lo, hi, color="0.15", alpha=0.35, zorder=4, lw=0)
    for x in (x_min, x_max):
        if x is not None:
            ax.axvline(x, color="k", ls="--", lw=1.0, zorder=4)


def segment_cell_counts(segments, xe, ye):
    """Number of segments passing through each cell (each segment counted once per cell)."""
    counts = np.zeros((len(xe) - 1, len(ye) - 1))
    cell = min(xe[1] - xe[0], ye[1] - ye[0])
    for p0, p1, _ in segments:
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / (cell / 4))) + 1)
        pts = p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
        ix = np.clip(np.searchsorted(xe, pts[:, 0]) - 1, 0, len(xe) - 2)
        iy = np.clip(np.searchsorted(ye, pts[:, 1]) - 1, 0, len(ye) - 2)
        for a, b in set(zip(ix, iy)):
            counts[a, b] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cell", type=float, default=0.10, help="heat-map cell size (m)")
    ap.add_argument("--min-frames", type=int, default=5, help="grey out probability cells with fewer frames")
    ap.add_argument("--phantom-dist", type=float, default=0.08)
    ap.add_argument("--max-speed", type=float, default=5.0)
    ap.add_argument("--x-min", type=float, default=None,
                    help="ignore the table beyond this x (far end); e.g. -0.80 = outer edge of the far face-off circles")
    ap.add_argument("--save-map", default=None,
                    help="write the P(occluded | cell) grid as JSON (consumed by the Box2D sim's occlusion_map_path)")
    ap.add_argument("--x-max", type=float, default=None,
                    help="ignore the table beyond this x (robot end); e.g. 0.78 = outer edge of the robot-end face-off circles")
    args = ap.parse_args()

    segments, vis_xy, occ_xy, n_open, n_eps = [], [], [], 0, 0
    seen_hashes, n_dupes = set(), 0
    for rd in args.run_dirs:
        # The recorder can save the same episode into two buckets (e.g. `<50/` and `estop/`); count it once.
        paths = []
        for p in sorted(glob.glob(os.path.join(rd, "episode_hdf5", "*", "*.hdf5"))):
            digest = hashlib.md5(Path(p).read_bytes()).hexdigest()
            if digest in seen_hashes:
                n_dupes += 1
                continue
            seen_hashes.add(digest)
            paths.append(p)
        episodes = [load_episode(p) for p in paths]
        H = fit_homography(episodes)
        for ep in episodes:
            s, v, o, n = episode_segments(ep, camera_track(ep, H), args.phantom_dist, args.max_speed)
            segments += s
            vis_xy.append(v)
            occ_xy.append(o)
            n_open += n
            n_eps += 1
    vis_xy = np.concatenate(vis_xy)
    x_lo = -np.inf if args.x_min is None else args.x_min
    x_hi = np.inf if args.x_max is None else args.x_max
    inside = lambda x: (x >= x_lo) & (x <= x_hi)  # noqa: E731
    # End regions: drop visible frames there and every occlusion with an end point there
    # (its frames leave both the numerator and the denominator).
    n_seg_all = len(segments)
    segments = [s for s in segments if inside(s[0][0]) and inside(s[1][0])]
    n_seg_dropped = n_seg_all - len(segments)
    vis_xy = vis_xy[inside(vis_xy[:, 0])]
    occ_xy = np.array([p0 + (p1 - p0) * k / (n + 1) for p0, p1, n in segments for k in range(1, n + 1)]).reshape(-1, 2)
    occ_xy = occ_xy[inside(occ_xy[:, 0])]
    band = "" if args.x_min is None and args.x_max is None else f"\nonly {x_lo:+.2f} ≤ x ≤ {x_hi:+.2f} analysed (shaded ends excluded; {n_seg_dropped} segments with an end point there dropped)"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_img = RealTrajectoryRenderer(render_size=600, paddle_input_frame="table", quiet=True).table_img

    # Cells span the analysed band exactly, so none straddles a cut-off line.
    gx0 = EXTENT[0] if args.x_min is None else args.x_min
    gx1 = EXTENT[1] if args.x_max is None else args.x_max
    nx = max(1, int(round((gx1 - gx0) / args.cell)))
    ny = int(np.ceil(TABLE_WIDTH / args.cell))
    xe = np.linspace(gx0, gx1, nx + 1)
    ye = np.linspace(EXTENT[2], EXTENT[3], ny + 1)
    n_occ_frames = sum(n for _, _, n in segments)

    # 1. segments over a crossing-count heat map
    counts = segment_cell_counts(segments, xe, ye)
    fig, ax = plt.subplots(figsize=(13, 6.6))
    draw_table(ax, table_img)
    hm = ax.pcolormesh(xe, ye, np.ma.masked_equal(counts, 0).T, cmap="inferno_r", alpha=0.55, zorder=1)
    cb = fig.colorbar(hm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("occlusion segments crossing the cell")
    for p0, p1, n in segments:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color="tab:blue", lw=0.6 + 0.25 * min(n, 10), alpha=0.55, zorder=2)
    if segments:
        s0 = np.array([s[0] for s in segments])
        s1 = np.array([s[1] for s in segments])
        ax.scatter(s0[:, 0], s0[:, 1], s=18, c="limegreen", edgecolors="k", linewidths=0.4, zorder=3, label="lost (last good position)")
        ax.scatter(s1[:, 0], s1[:, 1], s=18, c="red", edgecolors="k", linewidths=0.4, zorder=3, label="re-acquired (first good position)")
    shade_excluded(ax, args.x_min, args.x_max)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(f"{len(segments)} occlusion segments ({n_occ_frames} frames) from {n_eps} episodes; "
                 f"line width ∝ frames lost; {n_open} runs lasting to episode end not drawn{band}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "occlusion_segments.png", dpi=130)
    plt.close(fig)

    # 2. P(occluded | cell)
    n_vis, _, _ = np.histogram2d(vis_xy[:, 0], vis_xy[:, 1], bins=[xe, ye])
    n_occ = np.histogram2d(occ_xy[:, 0], occ_xy[:, 1], bins=[xe, ye])[0] if len(occ_xy) else np.zeros_like(n_vis)
    total = n_vis + n_occ
    prob = np.where(total > 0, n_occ / np.maximum(total, 1), np.nan)
    shown = np.ma.masked_where(total < args.min_frames, prob)
    if args.save_map:
        # Cells with < min_frames frames (or none) get p = 0; everything outside the grid is p = 0 too.
        p_map = np.where(total >= args.min_frames, np.nan_to_num(prob), 0.0)
        Path(args.save_map).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_map).write_text(json.dumps({
            "description": "P(puck observation occluded | true puck in cell), real-robot detector, table/observation frame "
                           "(robot / paddle end = +x). p[i][j] is the cell x_edges[i]..x_edges[i+1], y_edges[j]..y_edges[j+1].",
            "source_run_dirs": [str(r) for r in args.run_dirs],
            "x_edges": [float(v) for v in xe],
            "y_edges": [float(v) for v in ye],
            "p": [[round(float(v), 4) for v in row] for row in p_map],
            "frames": [[int(v) for v in row] for row in total],
            "min_frames": args.min_frames,
        }, indent=1))
        print(f"wrote map {args.save_map}")
    fig, ax = plt.subplots(figsize=(13, 6.6))
    draw_table(ax, table_img)
    ax.pcolormesh(xe, ye, np.ma.masked_where((total == 0) | (total >= args.min_frames), np.ones_like(total)).T,
                  cmap="Greys", vmin=0, vmax=3, alpha=0.5, zorder=1)
    hm = ax.pcolormesh(xe, ye, shown.T, cmap="magma_r", vmin=0, vmax=max(0.5, float(np.nanmax(shown)) if shown.count() else 0.5),
                       alpha=0.7, zorder=1, edgecolors="w", linewidth=0.3)
    cb = fig.colorbar(hm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("P(occluded | puck in cell)")
    for a in range(nx):
        for b in range(ny):
            if total[a, b] > 0:
                txt = f"{prob[a, b]*100:.0f}%\n{int(total[a, b])}" if total[a, b] >= args.min_frames else f"{int(total[a, b])}"
                ax.text((xe[a] + xe[a + 1]) / 2, (ye[b] + ye[b + 1]) / 2, txt, ha="center", va="center", fontsize=6,
                        color="k", zorder=3)
    ax.set_title(f"P(occluded | position), {args.cell*100:.0f} cm cells: occluded frames / all frames "
                 f"({len(occ_xy)} occluded of {len(occ_xy) + len(vis_xy)}; label = % and frame count; "
                 f"grey = < {args.min_frames} frames){band}", fontsize=9)
    shade_excluded(ax, args.x_min, args.x_max)
    fig.tight_layout()
    fig.savefig(out_dir / "occlusion_probability.png", dpi=130)
    plt.close(fig)
    print(f"{n_dupes} duplicate episode files skipped")
    print(f"{n_eps} episodes, {len(segments)} segments ({n_occ_frames} occluded frames), {n_open} open runs skipped")
    print(f"wrote {out_dir / 'occlusion_segments.png'} and {out_dir / 'occlusion_probability.png'}")


if __name__ == "__main__":
    main()
