"""Overlay real and replayed paddle trajectories on the Box2D scene.

The real paddle is drawn as the canonical paddle sprite (it is the ground truth); every sim
replay is a translucent coloured ghost with a trail; the recorded target (``desired_pose``) is a
small cross. Frames use ``AirHockeyRenderer`` on the replayer's env, are cropped to the
paddle workspace, and are exported BGR→RGB (GIF: `imageio`, PNG: `cv2`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset import PaddleTrial
from .replay import PaddleReplayer, ReplayResult

# BGR. Fixed order: real = black, first sim (canonical) = orange, second (fitted) = blue, third = aqua.
COLOR_REAL = (20, 20, 20)
SIM_COLORS = [(52, 104, 235), (214, 120, 42), (122, 175, 27)]
COLOR_TARGET = (110, 110, 110)


def _hex_to_rgb(bgr):
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


class SceneRenderer:
    """Draws frames of one trial on the Box2D table using the replayer's env / renderer."""

    def __init__(self, replayer: PaddleReplayer, margin_m: float = 0.10):
        from airhockey.renderers.render import AirHockeyRenderer
        self.rep = replayer
        self.r = AirHockeyRenderer(replayer.env, show_target_position=False, show_acceleration_arrow=False)
        self.offset = replayer.offset
        self.paddle_radius = float(replayer.sim.paddle_radius)
        self.ppm = float(replayer.sim.ppm)
        # crop rows to the robot workspace (robot-frame x range) + margin
        x_min, x_max = float(replayer.sim.lims[0]), float(replayer.sim.lims[1])
        rows = [self.px((x, 0.0))[1] for x in (x_min - margin_m, x_max + margin_m)]
        self.row0, self.row1 = max(0, min(rows)), max(rows)

    def px(self, pose_robot) -> tuple[int, int]:
        base = np.asarray(pose_robot, float) + self.offset
        u, v = self.r.world_xy_to_output_pixel(float(base[0]), float(base[1]))
        return int(round(u)), int(round(v))

    def table_with_real_paddle(self, real_pose) -> np.ndarray:
        self.r.frame = self.r.air_hockey_table_img.copy()
        base = np.asarray(real_pose, float) + self.offset
        self.r.draw_circle_with_image(self.r.convert_to_render_coords_sys(base), self.paddle_radius, circle_type="paddle")
        return cv2.rotate(self.r.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def draw_ghost(self, frame, pose_robot, color, alpha: float = 0.45) -> None:
        c = self.px(pose_robot)
        r = max(1, int(round(self.paddle_radius * self.ppm)))
        overlay = frame.copy()
        cv2.circle(overlay, c, r, color, thickness=-1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
        cv2.circle(frame, c, r, color, thickness=2, lineType=cv2.LINE_AA)

    def draw_trail(self, frame, poses, color, thickness: int = 2, dots: bool = True) -> None:
        pts = np.array([self.px(p) for p in poses], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, color, thickness, lineType=cv2.LINE_AA)
        if dots:
            for p in pts:
                cv2.circle(frame, tuple(int(v) for v in p), 2, color, -1, lineType=cv2.LINE_AA)

    def draw_target(self, frame, pose_robot, size: int = 6) -> None:
        u, v = self.px(pose_robot)
        cv2.line(frame, (u - size, v), (u + size, v), COLOR_TARGET, 2, cv2.LINE_AA)
        cv2.line(frame, (u, v - size), (u, v + size), COLOR_TARGET, 2, cv2.LINE_AA)

    def finish(self, frame, width: Optional[int] = None, labels: Optional[list[tuple[str, tuple]]] = None) -> np.ndarray:
        """Crop to the workspace, add text lines (BGR colours), BGR→RGB, resize to ``width``."""
        f = frame[self.row0:self.row1 + 1].copy()
        if labels:                                   # white text band above the scene, never over the paddles
            band = np.full((15 * len(labels) + 6, f.shape[1], 3), 255, np.uint8)
            y = 14
            for text, color in labels:
                cv2.putText(band, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
                y += 15
            f = np.vstack([band, f])
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if width and width != rgb.shape[1]:
            h = max(1, int(round(rgb.shape[0] * width / rgb.shape[1])))
            rgb = cv2.resize(rgb, (int(width), h), interpolation=cv2.INTER_AREA)
        return rgb

    # -- products -----------------------------------------------------------------------
    def trial_frames(self, trial: PaddleTrial, results: dict[str, ReplayResult], width: Optional[int] = None,
                     hold_last: int = 6) -> list[np.ndarray]:
        """One RGB frame per step: real paddle + sim ghosts + trails up to that step."""
        frames = []
        names = list(results)
        for k in range(trial.n_steps):
            f = self.table_with_real_paddle(trial.pose[k])
            self.draw_trail(f, trial.pose[: k + 1], COLOR_REAL, 2)
            for i, name in enumerate(names):
                self.draw_trail(f, results[name].sim_pose[: k + 1], SIM_COLORS[i], 1, dots=False)
                self.draw_ghost(f, results[name].sim_pose[k], SIM_COLORS[i])
            self.draw_target(f, trial.desired[k])
            labels = [(f"{trial.name}  step {k:2d}{'  (settle)' if trial.settle[k] else ''}", COLOR_REAL)]
            for i, name in enumerate(names):
                labels.append((f"{name}: {results[name].pos_err_mm[k]:5.1f} mm  (mean {results[name].pos_err_mm[1:].mean():5.1f})", SIM_COLORS[i]))
            frames.append(self.finish(f, width, labels))
        frames += [frames[-1]] * hold_last
        return frames

    def trial_summary_image(self, trial: PaddleTrial, results: dict[str, ReplayResult], width: Optional[int] = None) -> np.ndarray:
        """Static overlay: full trails, real paddle at the end, sim ghosts at the end."""
        f = self.table_with_real_paddle(trial.pose[-1])
        self.draw_ghost(f, trial.pose[0], (160, 160, 160), alpha=0.25)
        self.draw_trail(f, trial.pose, COLOR_REAL, 2)
        for i, name in enumerate(results):
            self.draw_trail(f, results[name].sim_pose, SIM_COLORS[i], 2)
            self.draw_ghost(f, results[name].sim_pose[-1], SIM_COLORS[i])
        self.draw_target(f, trial.desired[-1])
        labels = [(f"{trial.condition} trial {trial.repeat}", COLOR_REAL)]
        for i, name in enumerate(results):
            e = results[name].pos_err_mm[1:]
            labels.append((f"{name}: mean {e.mean():5.1f}  max {e.max():5.1f}  final {e[-1]:5.1f} mm", SIM_COLORS[i]))
        return self.finish(f, width, labels)


def write_gif(frames: list[np.ndarray], path, fps: int = 10) -> None:
    import imageio
    imageio.mimsave(str(path), frames, duration=1000.0 / fps, loop=0)


def write_png(rgb: np.ndarray, path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def mosaic(images: list[np.ndarray], ncols: int, pad: int = 4, bg: int = 255) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), np.uint8)
    h = max(im.shape[0] for im in images); w = max(im.shape[1] for im in images)
    nrows = int(np.ceil(len(images) / ncols))
    out = np.full((nrows * (h + pad) + pad, ncols * (w + pad) + pad, 3), bg, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, ncols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


def legend_strip(names: list[str], width: int, height: int = 22) -> np.ndarray:
    """A one-line RGB legend: real (black) + each sim set in its colour."""
    strip = np.full((height, width, 3), 255, np.uint8)
    x = 6
    entries = [("real", COLOR_REAL)] + [(n, SIM_COLORS[i]) for i, n in enumerate(names)]
    for text, bgr in entries:
        rgb = (bgr[2], bgr[1], bgr[0])
        cv2.circle(strip, (x + 5, height // 2), 5, rgb, -1, cv2.LINE_AA)
        cv2.putText(strip, text, (x + 14, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rgb, 1, cv2.LINE_AA)
        x += 14 + 8 * len(text) + 18
    return strip
