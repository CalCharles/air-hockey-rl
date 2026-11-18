"""Plot the observed puck track of real-robot episodes on the Box2D table.

One PNG per episode HDF5 (split schema, e.g. ``<run>/episode_hdf5/*/trajectory_data*.hdf5``):

- logged puck positions (``puck[:, :2]``, observation/table frame) as dots coloured by time step;
- frames the detector logged as occluded (``puck[:, 2] == 1``; the position is the held last
  value) as hollow squares;
- "phantom" detections: logged visible but > ``--phantom-dist`` from the puck actually seen in
  the camera frame, or implying a jump faster than ``--max-speed``, as red stars;
- the puck as seen by an independent red-blob detector on the saved camera frames
  (``train_img``), mapped to the table by a homography fitted on the run's agreeing frames,
  as a thin grey reference line;
- the paddle (``pose[:, :2]``) as a faint dashed line.

Usage:
    python scripts/visualization/plot_real_puck_tracks.py \
        --run-dir results/eval_4_tasks_of_policies_hist4_2026-09-19/juggle_sysid/data_20261014-201743 \
        --out-dir results/eval_4_tasks_of_policies_hist4_2026-09-19/juggle_sysid/data_20261014-201743/puck_track_plots
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.visualization.visualize_real_trajectory import RealTrajectoryRenderer  # noqa: E402

TABLE_LENGTH = 1.9304
TABLE_WIDTH = 0.8636
STEP_S = 0.049


def red_blob(img):
    """Single red blob centroid (px) in a BGR frame, or None if there is not exactly one."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 80)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    blobs = [cents[k] for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 15]
    return blobs[0] if len(blobs) == 1 else None


def load_episode(path):
    with h5py.File(path, "r") as f:
        ep = {
            "puck": f["puck"][:],
            "pose": f["pose"][:, :2],
            "img": f["train_img"][:] if "train_img" in f else None,
        }
    ep["blob_px"] = [red_blob(im) if ep["img"] is not None else None for im in (ep["img"] if ep["img"] is not None else [])]
    return ep


def fit_homography(episodes):
    """Pixel -> table homography from frames where the logged detection is visible."""
    src, dst = [], []
    for ep in episodes:
        for px, pk in zip(ep["blob_px"], ep["puck"]):
            if px is not None and pk[2] < 0.5:
                src.append(px)
                dst.append(pk[:2])
    if len(src) < 20:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 0.02)
    return H


def camera_track(ep, H):
    track = np.full((len(ep["puck"]), 2), np.nan)
    if H is None:
        return track
    for t, px in enumerate(ep["blob_px"]):
        if px is not None:
            track[t] = cv2.perspectiveTransform(np.float32([[px]]), H)[0, 0]
    return track


def classify(ep, track, phantom_dist, max_speed):
    xy, occ = ep["puck"][:, :2], ep["puck"][:, 2] > 0.5
    T = len(xy)
    phantom = np.zeros(T, dtype=bool)
    placeholder = np.zeros(T, dtype=bool)
    last_vis = None
    for t in range(T):
        if occ[t]:
            # reset sentinel (no puck seen yet this episode): x = -0.8, y = 0
            if last_vis is None:
                placeholder[t] = True
            continue
        if not np.isnan(track[t, 0]) and np.linalg.norm(xy[t] - track[t]) > phantom_dist:
            phantom[t] = True
        elif last_vis is not None and np.linalg.norm(xy[t] - xy[last_vis]) / (STEP_S * (t - last_vis)) > max_speed:
            phantom[t] = True
        if not phantom[t]:
            last_vis = t
    return occ, phantom, placeholder


def puck_speed(xy, good):
    """Speed (m/s) at each ``good`` frame: central difference over the neighbouring good frames
    (one-sided at the ends); NaN elsewhere. Occluded / phantom frames are skipped, so a gap is
    bridged by the real elapsed time."""
    idx = np.flatnonzero(good)
    speed = np.full(len(xy), np.nan)
    for n, t in enumerate(idx):
        a = idx[n - 1] if n > 0 else t
        b = idx[n + 1] if n + 1 < len(idx) else t
        if b > a:
            speed[t] = np.linalg.norm(xy[b] - xy[a]) / (STEP_S * (b - a))
    return speed


def plot_episode(ep, track, table_img, title, out_path, phantom_dist, max_speed,
                 color_by="time", speed_max=3.0, last_steps=0):
    xy = ep["puck"][:, :2]
    T = len(xy)
    occ, phantom, placeholder = classify(ep, track, phantom_dist, max_speed)
    t0 = max(0, T - last_steps) if last_steps else 0
    # Classify over the whole episode (phantom checks need the history), then plot the window only.
    win = np.arange(T) >= t0
    vis = ~occ & ~phantom
    speed = puck_speed(xy, vis)
    # A phantom's colour is the speed its jump implies from the previous good frame.
    for k in np.flatnonzero(phantom):
        prev = np.flatnonzero(vis[:k])
        if len(prev):
            speed[k] = np.linalg.norm(xy[k] - xy[prev[-1]]) / (STEP_S * (k - prev[-1]))
    held = occ & ~placeholder
    placeholder = placeholder | ~win
    vis, held, phantom = vis & win, held & win, phantom & win
    track = np.where(win[:, None], track, np.nan)
    ep = dict(ep, pose=np.where(win[:, None], ep["pose"], np.nan))
    if color_by == "speed":
        t = speed
        cmap = plt.get_cmap("plasma")
        norm = plt.Normalize(0, speed_max)
    else:
        t = np.arange(T)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(t0, max(T - 1, t0 + 1))

    fig, ax = plt.subplots(figsize=(13, 6.6))
    # Rotate 180 deg so the robot end (+x) is on the right and +y is up, as in the overhead camera.
    ax.imshow(cv2.cvtColor(cv2.rotate(table_img, cv2.ROTATE_180), cv2.COLOR_BGR2RGB),
              extent=(-TABLE_LENGTH / 2, TABLE_LENGTH / 2, -TABLE_WIDTH / 2, TABLE_WIDTH / 2), alpha=0.55)

    ax.plot(ep["pose"][:, 0], ep["pose"][:, 1], "--", color="0.35", lw=0.9, alpha=0.6, zorder=2)
    ax.plot(track[:, 0], track[:, 1], "-", color="0.25", lw=1.0, alpha=0.6, zorder=2)
    ax.plot(xy[~placeholder, 0], xy[~placeholder, 1], "-", color="0.6", lw=0.6, alpha=0.7, zorder=3)

    ax.scatter(xy[vis, 0], xy[vis, 1], c=t[vis], cmap=cmap, norm=norm, s=30, edgecolors="k", linewidths=0.3, zorder=5)
    held_color = "0.15" if color_by == "speed" else cmap(norm(t[held]))  # no speed while occluded
    ax.scatter(xy[held, 0], xy[held, 1], facecolors="none", edgecolors=held_color, marker="s", s=90, linewidths=1.6, zorder=4)
    ax.scatter(xy[phantom, 0], xy[phantom, 1], c=np.nan_to_num(t[phantom], nan=speed_max), cmap=cmap, norm=norm,
               marker="*", s=260, edgecolors="red", linewidths=1.4, zorder=6)

    # Held frames (and repeated phantoms) stack on one spot: label each run with its length in frames.
    for mask, offset in ((held, (6, -11)), (phantom, (8, 6))):
        k = 0
        while k < T:
            if mask[k]:
                j = k
                while j < T and mask[j] and np.allclose(xy[j], xy[k], atol=0.01):
                    j += 1
                if mask is held or j - k > 1:
                    ax.annotate(f"×{j - k}", xy[k], textcoords="offset points", xytext=offset, fontsize=8,
                                fontweight="bold", color="crimson", zorder=7)
                k = j
            else:
                k += 1

    # Step labels every 10 steps on the logged track.
    for k in range(t0 - t0 % 10, T, 10):
        if k >= t0 and not placeholder[k]:
            ax.annotate(str(k), xy[k], textcoords="offset points", xytext=(4, 4), fontsize=7, color="k", zorder=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    if color_by == "speed":
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, extend="max")
        cb.set_label("puck speed (m/s, from logged positions)")
    else:
        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(f"time step (≈{STEP_S * 1000:.0f} ms each)")

    held_mec = "0.15" if color_by == "speed" else cmap(0.6)
    legend = [
        Line2D([], [], marker="o", ls="", mfc=cmap(0.6), mec="k", ms=6, label=f"logged visible ({int(vis.sum())})"),
        Line2D([], [], marker="s", ls="", mfc="none", mec=held_mec, mew=1.6, ms=9, label=f"logged occluded, held last value ({int(held.sum())}; ×n = frames frozen)"),
        Line2D([], [], marker="*", ls="", mfc=cmap(0.6), mec="red", mew=1.4, ms=14, label=f"phantom: logged visible, wrong position ({int(phantom.sum())})"),
        Line2D([], [], color="0.25", lw=1.0, label="puck seen in camera frame (independent detector)"),
        Line2D([], [], color="0.35", lw=0.9, ls="--", label="paddle (telemetry)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_xlim(-TABLE_LENGTH / 2, TABLE_LENGTH / 2)
    ax.set_ylim(-TABLE_WIDTH / 2, TABLE_WIDTH / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("table x (m)  —  robot / paddle end →")
    ax.set_ylabel("table y (m)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return dict(steps=T, visible=int(vis.sum()), held=int(held.sum()), phantom=int(phantom.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Run folder containing episode_hdf5/")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: <run-dir>/puck_track_plots)")
    ap.add_argument("--phantom-dist", type=float, default=0.08, help="m between logged and camera puck to call a phantom")
    ap.add_argument("--max-speed", type=float, default=5.0, help="m/s implied jump between visible frames to call a phantom")
    ap.add_argument("--color-by", choices=["time", "speed"], default="time", help="dot colour: time step or puck speed")
    ap.add_argument("--speed-max", type=float, default=3.0, help="top of the speed colour scale (m/s)")
    ap.add_argument("--last-steps", type=int, default=0, help="plot only the final N steps of each episode (0 = all)")
    ap.add_argument("--subdir", default="*", help="episode_hdf5 sub-folder to read (e.g. '>200'); '*' = all")
    ap.add_argument("--episodes", nargs="*", default=None, help="only plot these stems (e.g. trajectory_data3); "
                    "the camera homography is still fitted on every episode of the run")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.run_dir, "episode_hdf5", args.subdir, "*.hdf5")),
                   key=lambda p: int(Path(p).stem.replace("trajectory_data", "") or 0))
    if not paths:
        sys.exit(f"No episode HDF5s under {args.run_dir}/episode_hdf5")
    out_dir = Path(args.out_dir or os.path.join(args.run_dir, "puck_track_plots"))
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [load_episode(p) for p in paths]
    H = fit_homography(episodes)
    renderer = RealTrajectoryRenderer(render_size=600, paddle_input_frame="table", quiet=True)

    n_written = 0
    for p, ep in zip(paths, episodes):
        name = Path(p).stem
        if args.episodes and name not in args.episodes:
            continue
        track = camera_track(ep, H)
        title = f"{Path(args.run_dir).parent.name}/{Path(args.run_dir).name} — {name}"
        if args.last_steps:
            title += f" — last {min(args.last_steps, len(ep['puck']))} of {len(ep['puck'])} steps"
        stats = plot_episode(ep, track, renderer.table_img, title, out_dir / f"{name}.png", args.phantom_dist,
                             args.max_speed, color_by=args.color_by, speed_max=args.speed_max,
                             last_steps=args.last_steps)
        print(f"{name}: {stats}")
        n_written += 1
    print(f"wrote {n_written} plots to {out_dir}")


if __name__ == "__main__":
    main()
