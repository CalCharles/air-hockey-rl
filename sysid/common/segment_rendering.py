"""Rendering for automatic trajectory segmentation results.

* ``render_gif``  — per-trajectory GIF: top-down table view (sim frame,
  robot on the right, +y up — same orientation as the overhead camera) with a
  label-coloured puck trail, the raw camera image when the HDF5 carries one,
  a banner with the current label / event metrics and a colour-coded timeline.
* ``render_plot`` — per-trajectory PNG: x(t), y(t), local speed and the
  split-fit velocity jump ``dv`` with label bands.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

from .trajectory_segmentation import EVENT_LABELS, LABELS, SegmentationResult

# BGR colours (cv2) per label.
LABEL_COLORS_BGR = {
    "free_fall": (60, 180, 60),
    "wall_collision": (255, 140, 0),
    "paddle_collision": (40, 40, 230),
    "opponent_hit": (200, 60, 200),
    "unknown_impulse": (0, 200, 255),
    "rest": (150, 150, 150),
    "occluded": (40, 40, 40),
}
LABEL_COLORS_RGB = {k: (v[2] / 255, v[1] / 255, v[0] / 255) for k, v in LABEL_COLORS_BGR.items()}


class TopDownRenderer:
    """Top-down table drawing in the sim frame, oriented like the camera:
    x (long axis) horizontal with the robot end (x < 0) on the RIGHT, +y up."""

    def __init__(self, cfg, ppm: float = 210.0, margin: int = 14):
        self.cfg = cfg
        self.ppm = ppm
        self.margin = margin
        self.w = int(cfg.table_length * ppm) + 2 * margin
        self.h = int(cfg.table_width * ppm) + 2 * margin

    def to_px(self, xy) -> tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        u = self.margin + (0.5 * self.cfg.table_length - x) * self.ppm
        v = self.margin + (0.5 * self.cfg.table_width - y) * self.ppm
        return int(round(u)), int(round(v))

    def blank(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 235, dtype=np.uint8)
        m = self.margin
        cv2.rectangle(img, (m, m), (self.w - m, self.h - m), (250, 250, 250), -1)
        cv2.rectangle(img, (m, m), (self.w - m, self.h - m), (80, 80, 80), 2)
        cx = self.to_px((0.0, 0.0))[0]
        cv2.line(img, (cx, m), (cx, self.h - m), (200, 200, 200), 1)
        cv2.putText(img, "robot >", (self.w - m - 52, self.h - m - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(img, "+y", (m + 3, m + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        return img

    def draw_frame(self, result: SegmentationResult, t: int, trail: int = 12) -> np.ndarray:
        traj, labels = result.trajectory, result.labels
        img = self.blank()
        pr = max(2, int(self.cfg.puck_radius * self.ppm))
        padr = max(3, int(self.cfg.paddle_radius * self.ppm))
        # Paddle + contact circle.
        pp = self.to_px(traj.paddle_xy[t])
        cv2.circle(img, pp, int(self.cfg.contact_distance * self.ppm), (200, 200, 200), 1)
        cv2.circle(img, pp, padr, (220, 160, 60), -1)
        cv2.circle(img, pp, padr, (120, 80, 20), 1)
        # Trail (valid frames only), coloured by label.
        for k in range(max(0, t - trail), t):
            if not traj.valid[k]:
                continue
            col = LABEL_COLORS_BGR[str(labels[k])]
            cv2.circle(img, self.to_px(traj.puck_xy[k]), max(1, pr - 3), col, -1)
        # Current puck.
        lab = str(labels[t])
        col = LABEL_COLORS_BGR[lab]
        if traj.valid[t]:
            q = self.to_px(traj.puck_xy[t])
            cv2.circle(img, q, pr, (30, 30, 30), -1)
            cv2.circle(img, q, pr + 3, col, 2)
        else:
            # last known position, hollow
            prev = np.flatnonzero(traj.valid[:t + 1])
            if len(prev):
                q = self.to_px(traj.puck_xy[prev[-1]])
                cv2.circle(img, q, pr, (30, 30, 30), 1)
                cv2.putText(img, "occl", (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
        # Event markers: an X at the impact frame, shown for a few frames.
        for ev in result.events:
            if ev.split <= t <= ev.split + 4:
                frame_idx = ev.split if traj.valid[ev.split] else ev.pre_end
                q = self.to_px(traj.puck_xy[frame_idx])
                c = LABEL_COLORS_BGR[ev.label]
                cv2.drawMarker(img, q, c, cv2.MARKER_TILTED_CROSS, 18, 2)
        return img


def _timeline(labels: np.ndarray, width: int, height: int, t: int) -> np.ndarray:
    n = len(labels)
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    edges = np.linspace(0, width, n + 1).astype(int)
    for k in range(n):
        strip[:, edges[k]:max(edges[k] + 1, edges[k + 1])] = LABEL_COLORS_BGR[str(labels[k])]
    u = int((t + 0.5) / n * width)
    cv2.line(strip, (u, 0), (u, height - 1), (255, 255, 255), 2)
    return strip


def _legend(width: int, height: int = 18) -> np.ndarray:
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    x = 6
    for lab in LABELS:
        cv2.rectangle(img, (x, 4), (x + 10, 14), LABEL_COLORS_BGR[lab], -1)
        cv2.putText(img, lab, (x + 13, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (40, 40, 40), 1)
        x += 13 + int(6.3 * len(lab)) + 10
    return img


def _banner(result: SegmentationResult, t: int, width: int, height: int = 40) -> np.ndarray:
    traj, labels = result.trajectory, result.labels
    lab = str(labels[t])
    col = LABEL_COLORS_BGR[lab]
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), col, 3)
    seg = next(s for s in result.segments if s.start <= t <= s.end)
    line1 = f"{traj.path.stem}  t={t:3d}/{traj.n - 1}  {traj.t_rel[t]:5.2f}s   {lab}  [{seg.start}-{seg.end}, {seg.n_frames} fr]"
    cv2.putText(img, line1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    line2 = ""
    if seg.label in EVENT_LABELS and seg.meta.get("events"):
        ev = seg.meta["events"][0]
        wall = f" wall={ev['wall']}" if ev.get("wall") else ""
        line2 = (f"|v| {ev['speed_pre']:.2f} -> {ev['speed_post']:.2f} m/s (x{ev['speed_ratio']:.2f})"
                 f"  dv={ev['dv']:.2f}{wall}  paddle_d={ev['paddle_min_dist']:.3f}  gap={ev['gap_frames']}")
    elif seg.label in ("free_fall", "rest") and "fit_rms_m" in seg.meta:
        sp = result.stats["speed_local"][t]
        sp_txt = f"{sp:.2f}" if np.isfinite(sp) else "  -"
        line2 = f"|v|={sp_txt} m/s   segment fit rms={seg.meta['fit_rms_m'] * 100:.1f} cm  (n_usable={seg.meta['n_usable']})"
    elif seg.label == "occluded":
        line2 = "puck not tracked (gap longer than max_occlusion_gap)"
    cv2.putText(img, line2, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
    return img


def render_gif(result: SegmentationResult, out_path: str | Path, fps: int = 10,
               ppm: float = 210.0, camera: bool = True, max_frames: int | None = None) -> Path:
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj = result.trajectory
    rend = TopDownRenderer(result.config, ppm=ppm)
    images = None
    if camera and traj.has_image:
        with h5py.File(traj.path, "r") as f:
            images = f["image"][:]
    frames = []
    n = traj.n if max_frames is None else min(traj.n, max_frames)
    for t in range(n):
        top = rend.draw_frame(result, t)
        panels = [top]
        if images is not None:
            im = images[t]
            scale = top.shape[0] / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), top.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.putText(im, "camera", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            panels.append(im)
        body = np.concatenate(panels, axis=1)
        width = body.shape[1]
        frame = np.concatenate([
            _banner(result, t, width),
            body,
            _timeline(result.labels, width, 16, t),
            _legend(width),
        ], axis=0)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path


def render_plot(result: SegmentationResult, out_path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj, cfg = result.trajectory, result.config
    t = np.arange(traj.n)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, traj.n / 25), 8), sharex=True)
    for ax in axes:
        for seg in result.segments:
            ax.axvspan(seg.start - 0.5, seg.end + 0.5, color=LABEL_COLORS_RGB[seg.label], alpha=0.25, lw=0)
    px = np.where(traj.valid, traj.puck_xy[:, 0], np.nan)
    py = np.where(traj.valid, traj.puck_xy[:, 1], np.nan)
    axes[0].plot(t, px, ".-", ms=3, lw=0.8, color="tab:blue", label="puck x")
    axes[0].plot(t, traj.paddle_xy[:, 0], "-", lw=0.8, color="tab:orange", label="paddle x")
    axes[0].axhline(cfg.x_wall, color="k", lw=0.5, ls=":"); axes[0].axhline(-cfg.x_wall, color="k", lw=0.5, ls=":")
    axes[0].set_ylabel("x [m]"); axes[0].legend(loc="upper right", fontsize=7)
    axes[1].plot(t, py, ".-", ms=3, lw=0.8, color="tab:blue", label="puck y")
    axes[1].plot(t, traj.paddle_xy[:, 1], "-", lw=0.8, color="tab:orange", label="paddle y")
    axes[1].axhline(cfg.y_wall, color="k", lw=0.5, ls=":"); axes[1].axhline(-cfg.y_wall, color="k", lw=0.5, ls=":")
    axes[1].set_ylabel("y [m]"); axes[1].legend(loc="upper right", fontsize=7)
    axes[2].plot(t, result.stats["speed_local"], "-", lw=0.8, color="tab:blue", label="|v| (post-fit)")
    axes[2].plot(t, result.stats["dv"], "-", lw=0.8, color="tab:red", label="dv (split-fit jump)")
    axes[2].axhline(cfg.dv_threshold, color="tab:red", lw=0.5, ls="--")
    d = np.linalg.norm(traj.puck_xy - traj.paddle_xy, axis=1)
    axes[2].plot(t, np.where(traj.valid, d, np.nan), "-", lw=0.8, color="tab:gray", label="puck-paddle dist")
    axes[2].axhline(cfg.contact_distance, color="tab:gray", lw=0.5, ls="--")
    axes[2].set_ylabel("m/s | m"); axes[2].set_xlabel("frame"); axes[2].legend(loc="upper right", fontsize=7)
    axes[2].set_ylim(0, 3.5)
    handles = [Patch(color=LABEL_COLORS_RGB[l], alpha=0.4, label=l) for l in LABELS]
    axes[0].legend(handles=handles + axes[0].get_legend_handles_labels()[0], loc="upper right", fontsize=6, ncol=3)
    fig.suptitle(f"{traj.path.name}  (puck_x_sign={traj.puck_x_sign}, paddle x = {traj.paddle_x_sign:+d}*pose_x {traj.paddle_x_offset:+.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
