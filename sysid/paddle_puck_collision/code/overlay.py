"""Side-by-side frames of a real collision and its sim replication.

Three panels per step, all on the camera's clock: the stored camera image (``train_img``), the
real trajectory drawn on the Box2D table with the env's own sprites (puck from the tracker,
paddle from the robot pose shifted by the calibrated camera lag, the fitted pre / post
free-flight models as trails, the contact point), and the sim replay (``HeadOnCollider.run``
with a free-flying puck, the paddle started so that it is at the real contact position at the
real contact step). The scene is drawn in the renderer's horizontal layout and mirrored so the
robot is on the right, as in the camera image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sysid.common.trajectory_segmentation import model_state
from sysid.paddle_pid.code.overlay import write_gif, write_png, mosaic  # noqa: F401  (re-exported)
from .dataset import CollisionTrial
from .speeds import CollisionMeasurement, SpeedConfig
from .sim_collision import HeadOnCollider, CollisionParams

# BGR. real = black, pre-model = blue, post-model = orange, sim = blue ghost, contact = green.
C_REAL, C_PRE, C_POST, C_SIM, C_CONTACT, C_MUTED = (20, 20, 20), (214, 120, 42), (52, 104, 235), (214, 120, 42), (122, 175, 27), (140, 140, 140)


def write_mp4(frames: list[np.ndarray], path, fps: int = 10) -> None:
    import imageio
    h, w = frames[0].shape[:2]
    frames = [cv2.resize(f, (w - w % 2, h - h % 2)) if (w % 2 or h % 2) else f for f in frames]   # h264 needs even dims
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as wr:
        for f in frames:
            wr.append_data(f)


class CollisionScene:
    """Draws base-frame states on the Box2D table using the collider's env renderer."""

    def __init__(self, collider: HeadOnCollider, x_range=(-0.98, 0.98)):
        from airhockey.renderers.render import AirHockeyRenderer
        self.col = collider
        self.r = AirHockeyRenderer(collider.env, show_target_position=False, show_acceleration_arrow=False)
        self.paddle_radius, self.puck_radius = collider.paddle_radius, collider.puck_radius
        self.ppm = float(collider.sim.ppm)
        self.width_px = int(self.r.air_hockey_table_img.shape[1])
        cols = sorted(self.px((x, 0.0))[0] for x in x_range)
        self.col0, self.col1 = max(0, cols[0]), min(self.width_px - 1, cols[1])

    # pixel of a base-frame point on the *mirrored* horizontal table (robot on the right)
    def px(self, base_xy) -> tuple[int, int]:
        x, y = self.r._base_to_prerotate_pixel(float(base_xy[0]), float(base_xy[1]))
        return int(round(self.width_px - 1 - x)), int(round(y))

    def table(self) -> np.ndarray:
        return cv2.flip(self.r.air_hockey_table_img.copy(), 1)

    def draw_sprite(self, frame, base_xy, kind: str) -> None:
        self.r.frame = cv2.flip(frame, 1)
        self.r.draw_circle_with_image(self.r.convert_to_render_coords_sys(np.asarray(base_xy, float)),
                                      self.paddle_radius if kind == "paddle" else self.puck_radius, circle_type=kind)
        frame[:] = cv2.flip(self.r.frame, 1)

    def draw_ghost(self, frame, base_xy, color, radius_m: float, alpha: float = 0.35) -> None:
        c = self.px(base_xy)
        r = max(1, int(round(radius_m * self.ppm)))
        overlay = frame.copy()
        cv2.circle(overlay, c, r, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
        cv2.circle(frame, c, r, color, 2, cv2.LINE_AA)

    def draw_trail(self, frame, pts_base, color, thickness: int = 2, dots: bool = True) -> None:
        pts = np.array([self.px(p) for p in pts_base], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, color, thickness, lineType=cv2.LINE_AA)
        if dots:
            for p in pts:
                cv2.circle(frame, tuple(int(v) for v in p), 2, color, -1, lineType=cv2.LINE_AA)

    def draw_marker(self, frame, base_xy, color=C_CONTACT, size: int = 7) -> None:
        u, v = self.px(base_xy)
        cv2.line(frame, (u - size, v - size), (u + size, v + size), color, 2, cv2.LINE_AA)
        cv2.line(frame, (u - size, v + size), (u + size, v - size), color, 2, cv2.LINE_AA)

    def crop(self, frame) -> np.ndarray:
        return frame[:, self.col0:self.col1 + 1]


def _band(width: int, lines: list[tuple[str, tuple]], scale: float = 0.42) -> np.ndarray:
    band = np.full((15 * len(lines) + 6, width, 3), 255, np.uint8)
    y = 14
    for text, color in lines:
        cv2.putText(band, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 15
    return band


def _fit_height(img: np.ndarray, height: int) -> np.ndarray:
    w = max(1, int(round(img.shape[1] * height / img.shape[0])))
    return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)


def _panel(img_bgr: np.ndarray, title_lines, height: int) -> np.ndarray:
    body = _fit_height(img_bgr, height)
    return np.vstack([_band(body.shape[1], title_lines), body])


def _hstack(panels: list[np.ndarray], pad: int = 6) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)])
        out += [p, np.full((h, pad, 3), 255, np.uint8)]
    return np.hstack(out[:-1])


class CollisionRenderer:
    """Per-trial side-by-side products for one parameter set."""

    def __init__(self, collider: HeadOnCollider, cfg: SpeedConfig, params: CollisionParams, params_label: str,
                 panel_height: int = 240, tail_steps: int = 12, offset: bool = False):
        """``offset=True``: the sim puck is launched in the real lateral lane (``m.dy`` from the paddle
        centre) instead of head-on, so the contact position is matched and the exit angle is meaningful."""
        self.col, self.cfg, self.params, self.label = collider, cfg, params, params_label
        self.offset = bool(offset)
        self.scene = CollisionScene(collider)
        self.h, self.tail = int(panel_height), int(tail_steps)
        self.seg = cfg.seg_cfg()
        collider.set_params(params)

    # -- the sim run aligned to the real contact step ----------------------------------
    def sim_run(self, trial: CollisionTrial, m: CollisionMeasurement, n_render: int) -> dict:
        dt = self.col.dt
        first_post = int(np.searchsorted(trial.pad_t - trial.t0, m.t_c, side="right"))   # first real step after t_c
        pre_roll = max(1, first_post - 1)
        k_off = first_post - (pre_roll + 1)                                               # sim frame j ↔ real step j + k_off
        t_c_sim = pre_roll * dt + self.col.contact_tau
        pad_start = min(0.90, m.pad_x + m.u_p * t_c_sim)
        post = max(1, n_render - k_off - pre_roll - 1)
        frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        r = self.col.run(m.u_p, m.speed_in, pre_roll_steps=pre_roll, post_steps=post, paddle_start_x=pad_start,
                         y=float(np.clip(m.pad_y, -0.3, 0.3)), puck_launch="free", puck_physics=True,
                         dy=float(m.dy) if self.offset else 0.0,
                         on_frame=lambda j, p, q: frames.__setitem__(j, (np.array(p), np.array(q))))
        r.update({"frames": frames, "k_off": k_off, "pre_roll": pre_roll, "first_post": first_post, "pad_start": pad_start})
        return r

    def frames(self, trial: CollisionTrial, m: CollisionMeasurement, camera_images: Optional[np.ndarray],
               hold_last: int = 8) -> tuple[list[np.ndarray], dict]:
        """RGB frames (camera | real scene | sim scene) for real steps 0 .. first_post + tail."""
        n_steps = trial.pad_t.shape[0]
        first_post = int(np.searchsorted(trial.pad_t - trial.t0, m.t_c, side="right"))
        n_render = min(n_steps, first_post + self.tail)
        sim = self.sim_run(trial, m, n_render)
        lag = self.cfg.camera_lag_s
        t0 = m.pre_fit["t0"]
        fpre = {"p0": np.array(m.pre_fit["p0"]), "u": np.array(m.pre_fit["u"])}
        fpost = {"p0": np.array(m.post_fit["p0"]), "u": np.array(m.post_fit["u"])}
        pre_trail = [model_state(fpre, s - t0, self.seg)[0] for s in np.linspace(trial.t[m.pre_idx[0]], trial.t0 + m.t_c, 30)]
        post_trail = [model_state(fpost, s - t0, self.seg)[0] for s in np.linspace(trial.t0 + m.t_c, trial.t[m.post_idx[-1]], 30)]
        contact = np.array([m.x_c, m.y_c])
        out = []
        real_puck_trail, sim_puck_trail = [], []
        for k in range(n_render):
            t_k = float(trial.pad_t[k] - trial.t0)
            row = trial.n_arm + k
            puck_xy, puck_ok = trial.puck_xy[row], bool(trial.puck_valid[row])
            pad_xy = np.array([np.interp(trial.pad_t[k] - lag, trial.pad_t, trial.pad_xy[:, 0]),
                               np.interp(trial.pad_t[k] - lag, trial.pad_t, trial.pad_xy[:, 1])])
            after = t_k >= m.t_c
            # real scene
            f = self.scene.table()
            self.scene.draw_trail(f, pre_trail, C_PRE, 1, dots=False)
            if after:
                self.scene.draw_trail(f, post_trail, C_POST, 1, dots=False)
                self.scene.draw_marker(f, contact)
            if puck_ok:
                real_puck_trail.append(puck_xy)
            self.scene.draw_trail(f, real_puck_trail, C_REAL, 1)
            self.scene.draw_sprite(f, pad_xy, "paddle")
            if puck_ok:
                self.scene.draw_sprite(f, puck_xy, "puck")
            else:                                   # occluded: ghost at the fitted model's position
                fit_, = (fpost,) if after else (fpre,)
                self.scene.draw_ghost(f, model_state(fit_, trial.pad_t[k] - t0, self.seg)[0], C_MUTED, self.puck_radius_m(), 0.25)
            real_lines = [(f"REAL  step {k:2d}  t = {t_k:5.2f} s{'' if puck_ok else '   puck occluded (ghost = model)'}", C_REAL),
                          (f"in {m.speed_in:.2f}  out {m.speed_out:.2f}  paddle {m.u_p:.2f} m/s  exit angle {np.degrees(np.arctan2(m.vy_out, m.vx_out_away)):+.0f} deg  offset dy {100 * m.dy:+.1f} cm", C_REAL),
                          ("red = paddle, green = puck, blue/orange = pre/post fit, X = contact", C_MUTED)]
            real_panel = _panel(self.scene.crop(f), real_lines, self.h)
            # sim scene
            j = k - sim["k_off"]
            g = self.scene.table()
            if j in sim["frames"]:
                pad_s, puck_s = sim["frames"][j]
                sim_puck_trail.append(puck_s)
                self.scene.draw_trail(g, sim_puck_trail, C_SIM, 1)
                self.scene.draw_sprite(g, pad_s, "paddle")
                self.scene.draw_sprite(g, puck_s, "puck")
                if after:
                    self.scene.draw_marker(g, contact)
            sim_lines = [(f"SIM  {self.label}: e {self.params.restitution:.3f}  m_pad/m_puck {self.params.mass_ratio:.1f}  contacts {sim['n_contacts']}", C_REAL),
                         (f"in {sim['u_k_actual']:.2f}  out {sim['speed_out']:.2f} (real {m.speed_out:.2f})  paddle {sim['u_p_actual']:.2f} m/s  "
                          f"angle {sim['angle_out_deg']:+.0f} (real {np.degrees(np.arctan2(m.vy_out, m.vx_out_away)):+.0f}) deg", C_REAL),
                         ((f"straight launch in the real lane, dy {100 * sim['dy']:+.1f} cm; " if self.offset else "head-on launch; ")
                          + "sysid gravity+damping; paddle const. speed", C_MUTED)]
            sim_panel = _panel(self.scene.crop(g), sim_lines, self.h)
            panels = []
            if camera_images is not None and k < camera_images.shape[0]:
                cam_lines = [(f"CAMERA  {trial.name.replace('collision_', '')}", C_REAL),
                             (f"{trial.condition}   take {trial.repeat}   {'valid' if m.valid else m.reason[:48]}", C_REAL),
                             (f"frame ~{1000 * lag:.0f} ms behind the robot clock; all panels on camera clock", C_MUTED)]
                panels.append(_panel(np.ascontiguousarray(camera_images[k]), cam_lines, self.h))
            panels += [real_panel, sim_panel]
            out.append(cv2.cvtColor(_hstack(panels), cv2.COLOR_BGR2RGB))
        out += [out[-1]] * hold_last
        return out, sim

    def puck_radius_m(self) -> float:
        return self.scene.puck_radius
