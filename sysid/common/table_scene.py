"""Draw real and simulated puck trajectories on the Box2D table (shared by the puck and wall overlays).

Points are given in the **sim frame** of the sysid sections (x = long axis, robot at x < 0); the
scene converts to the env's base frame (``x_base = −x_sim``) and draws with the env's own
``AirHockeyRenderer`` in the horizontal layout, mirrored so the robot is on the right, as in the
camera images. Real puck = the env's puck sprite, sim puck = a translucent coloured ghost, the
analytic model = a thin line. Frames are exported BGR→RGB.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# BGR. real = black, Box2D sim = blue, analytic model = grey, marker = green.
C_REAL, C_SIM, C_MODEL, C_MARK, C_MUTED = (20, 20, 20), (214, 120, 42), (150, 150, 150), (122, 175, 27), (140, 140, 140)


def sim_to_base(xy) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.array([-xy[0], xy[1]])


def base_to_sim(xy) -> np.ndarray:
    return sim_to_base(xy)


class TableScene:
    """Frames of the Box2D table with sprites, ghosts, trails and text bands."""

    def __init__(self, env, x_range=(-0.98, 0.98)):
        from airhockey.renderers.render import AirHockeyRenderer
        self.env = env
        self.sim = env.simulator
        self.r = AirHockeyRenderer(env, show_target_position=False, show_acceleration_arrow=False)
        self.paddle_radius, self.puck_radius = float(self.sim.paddle_radius), float(self.sim.puck_radius)
        self.ppm = float(self.sim.ppm)
        self.width_px = int(self.r.air_hockey_table_img.shape[1])
        cols = sorted(self._px_base((x, 0.0))[0] for x in x_range)
        self.col0, self.col1 = max(0, cols[0]), min(self.width_px - 1, cols[1])

    # -- geometry -----------------------------------------------------------------------
    def _px_base(self, base_xy) -> tuple[int, int]:
        x, y = self.r._base_to_prerotate_pixel(float(base_xy[0]), float(base_xy[1]))
        return int(round(self.width_px - 1 - x)), int(round(y))

    def px(self, sim_xy) -> tuple[int, int]:
        return self._px_base(sim_to_base(sim_xy))

    def table(self) -> np.ndarray:
        return cv2.flip(self.r.air_hockey_table_img.copy(), 1)

    # -- drawing (all inputs in the sim frame) -------------------------------------------
    def draw_sprite(self, frame, sim_xy, kind: str = "puck") -> None:
        self.r.frame = cv2.flip(frame, 1)
        self.r.draw_circle_with_image(self.r.convert_to_render_coords_sys(sim_to_base(sim_xy)),
                                      self.paddle_radius if kind == "paddle" else self.puck_radius, circle_type=kind)
        frame[:] = cv2.flip(self.r.frame, 1)

    def draw_ghost(self, frame, sim_xy, color=C_SIM, radius_m: Optional[float] = None, alpha: float = 0.35) -> None:
        c = self.px(sim_xy)
        r = max(1, int(round((radius_m if radius_m is not None else self.puck_radius) * self.ppm)))
        overlay = frame.copy()
        cv2.circle(overlay, c, r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
        cv2.circle(frame, c, r, color, 2, cv2.LINE_AA)

    def draw_trail(self, frame, pts_sim, color, thickness: int = 2, dots: bool = True, dot_radius: int = 2) -> None:
        pts = np.array([self.px(p) for p in pts_sim], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, color, thickness, lineType=cv2.LINE_AA)
        if dots:
            for p in pts:
                cv2.circle(frame, tuple(int(v) for v in p), dot_radius, color, -1, lineType=cv2.LINE_AA)

    def draw_marker(self, frame, sim_xy, color=C_MARK, size: int = 7) -> None:
        u, v = self.px(sim_xy)
        cv2.line(frame, (u - size, v - size), (u + size, v + size), color, 2, cv2.LINE_AA)
        cv2.line(frame, (u - size, v + size), (u + size, v - size), color, 2, cv2.LINE_AA)

    def draw_wall_line(self, frame, wall: str, color=C_MUTED) -> None:
        """The sim's contact line of a wall (puck-centre position at contact), sim frame."""
        L, W, r = float(self.sim.length), float(self.sim.width), self.puck_radius
        if wall[0] == "x":
            x = (0.5 * L - r) * (1 if wall[1] == "+" else -1)
            a, b = self.px((x, -0.5 * W)), self.px((x, 0.5 * W))
        else:
            y = (0.5 * W - r) * (1 if wall[1] == "+" else -1)
            a, b = self.px((-0.5 * L, y)), self.px((0.5 * L, y))
        cv2.line(frame, a, b, color, 1, cv2.LINE_AA)

    # -- export --------------------------------------------------------------------------
    def finish(self, frame, width: Optional[int] = None, labels: Optional[list[tuple[str, tuple]]] = None) -> np.ndarray:
        """Crop to the table, add a text band (BGR colours), BGR→RGB, resize to ``width``."""
        f = frame[:, self.col0:self.col1 + 1].copy()
        if labels:
            f = np.vstack([band(f.shape[1], labels), f])
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if width and width != rgb.shape[1]:
            h = max(1, int(round(rgb.shape[0] * width / rgb.shape[1])))
            rgb = cv2.resize(rgb, (int(width), h), interpolation=cv2.INTER_AREA)
        return rgb


def band(width: int, lines: list[tuple[str, tuple]], scale: float = 0.42) -> np.ndarray:
    out = np.full((15 * len(lines) + 6, width, 3), 255, np.uint8)
    y = 14
    for text, color in lines:
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 15
    return out


def legend_strip(entries: list[tuple[str, tuple]], width: int, height: int = 22) -> np.ndarray:
    """One-line RGB legend from (name, BGR colour) entries."""
    strip = np.full((height, width, 3), 255, np.uint8)
    x = 6
    for text, bgr in entries:
        rgb = (bgr[2], bgr[1], bgr[0])
        cv2.circle(strip, (x + 5, height // 2), 5, rgb, -1, cv2.LINE_AA)
        cv2.putText(strip, text, (x + 14, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rgb, 1, cv2.LINE_AA)
        x += 14 + 8 * len(text) + 18
    return strip


def write_gif(frames: list[np.ndarray], path, fps: int = 10) -> None:
    import imageio
    imageio.mimsave(str(path), frames, duration=1000.0 / fps, loop=0)


def write_mp4(frames: list[np.ndarray], path, fps: int = 10) -> None:
    import imageio
    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w - w % 2, h - h % 2)) if (w % 2 or h % 2) else f for f in frames]
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as wr:
        for f in frames:
            wr.append_data(f)


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


def pick_percentiles(values: np.ndarray, n: int) -> list[int]:
    """Indices of the candidates at evenly spaced percentiles of ``values`` (n of them, no
    repeats): 10 / 30 / 50 / 70 / 90 % for n = 5 — a spread over the error distribution rather
    than the best or the worst cases."""
    order = np.argsort(values)
    if len(order) <= n:
        return [int(i) for i in order]
    qs = (np.arange(n) + 0.5) / n
    picks, used = [], set()
    for q in qs:
        k = int(round(q * (len(order) - 1)))
        while k in used and k + 1 < len(order):
            k += 1
        used.add(k); picks.append(int(order[k]))
    return picks
