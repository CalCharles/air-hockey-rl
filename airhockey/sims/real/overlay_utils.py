from pathlib import Path

import cv2
import numpy as np

DEFAULT_VISUAL_DOWNSCALE_CONSTANT = 2.0
DEFAULT_OFFSET_CONSTANTS = np.array((2100.0, 500.0), dtype=float)
DEFAULT_SIM_OVERLAY_ALPHA = 0.25
DEFAULT_TABLE_LENGTH = 1.9304
DEFAULT_TABLE_WIDTH = 0.8636
DEFAULT_CENTER_OFFSET = 1.2


def _coerce_offset_constants(offset_constants):
    if offset_constants is None:
        return DEFAULT_OFFSET_CONSTANTS
    return np.array(offset_constants, dtype=float).reshape(2)


def _coerce_downscale(visual_downscale_constant):
    downscale = float(visual_downscale_constant)
    if downscale <= 0:
        return DEFAULT_VISUAL_DOWNSCALE_CONSTANT
    return downscale


def meters_to_display_pixels(distance_m, visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT):
    downscale = _coerce_downscale(visual_downscale_constant)
    return (float(distance_m) * 1000.0) / downscale


def robot_to_display_pixel(
    x_m,
    y_m,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
):
    pixel_coord = np.array((float(x_m) * 1000.0, -float(y_m) * 1000.0), dtype=float)
    pixel_coord += _coerce_offset_constants(offset_constants)
    return pixel_coord / _coerce_downscale(visual_downscale_constant)


def robot_to_display_pixel_int(
    x_m,
    y_m,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
):
    px, py = robot_to_display_pixel(
        x_m,
        y_m,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )
    return int(np.round(px)), int(np.round(py))


def display_pixel_to_robot(
    px,
    py,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
):
    pixel_coord = np.array((float(px), float(py)), dtype=float) * _coerce_downscale(visual_downscale_constant)
    robot_xy = (pixel_coord - _coerce_offset_constants(offset_constants)) * 0.001
    return float(robot_xy[0]), float(-robot_xy[1])


def observation_to_robot_xy(x_obs, y_obs, x_offset):
    return float(x_obs) - float(x_offset), float(y_obs)


def draw_target_marker(
    frame,
    target_xy,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
    color=(0, 165, 255),
    marker_size=15,
    thickness=3,
):
    if frame is None or target_xy is None or len(target_xy) < 2:
        return frame

    center = robot_to_display_pixel_int(
        target_xy[0],
        target_xy[1],
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )

    cv2.circle(frame, center, marker_size, (0, 0, 0), thickness + 2)
    cv2.circle(frame, center, marker_size, color, thickness)

    cv2.line(
        frame,
        (center[0] - marker_size, center[1]),
        (center[0] + marker_size, center[1]),
        (0, 0, 0),
        thickness + 2,
    )
    cv2.line(
        frame,
        (center[0], center[1] - marker_size),
        (center[0], center[1] + marker_size),
        (0, 0, 0),
        thickness + 2,
    )
    cv2.line(
        frame,
        (center[0] - marker_size, center[1]),
        (center[0] + marker_size, center[1]),
        color,
        thickness,
    )
    cv2.line(
        frame,
        (center[0], center[1] - marker_size),
        (center[0], center[1] + marker_size),
        color,
        thickness,
    )
    return frame


def draw_robot_circle_marker(
    frame,
    x_m,
    y_m,
    radius_m,
    color,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
    thickness=2,
    outline_color=(0, 0, 0),
    outline_thickness=2,
):
    if frame is None:
        return frame

    center = robot_to_display_pixel_int(
        x_m,
        y_m,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )
    radius_px = max(1, int(np.round(meters_to_display_pixels(radius_m, visual_downscale_constant))))
    cv2.circle(frame, center, radius_px + outline_thickness, outline_color, outline_thickness)
    cv2.circle(frame, center, radius_px, color, thickness)
    return frame


def draw_puck_marker_from_state(
    frame,
    puck_state,
    puck_radius_m,
    x_offset_for_state=0.0,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
    color=(0, 255, 0),
    require_visible=True,
):
    if frame is None or puck_state is None or len(puck_state) < 3:
        return frame
    if require_visible and int(puck_state[2]) != 0:
        return frame

    puck_x_robot, puck_y_robot = observation_to_robot_xy(
        puck_state[0], puck_state[1], x_offset_for_state
    )
    return draw_robot_circle_marker(
        frame,
        puck_x_robot,
        puck_y_robot,
        puck_radius_m,
        color,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )


def draw_paddle_marker(
    frame,
    paddle_xy,
    paddle_radius_m,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
    color=(255, 0, 0),
):
    if frame is None or paddle_xy is None or len(paddle_xy) < 2:
        return frame
    return draw_robot_circle_marker(
        frame,
        float(paddle_xy[0]),
        float(paddle_xy[1]),
        paddle_radius_m,
        color,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )


def draw_goal_marker(
    frame,
    goal_xy_robot,
    goal_radius_m=None,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
    color=(0, 255, 255),
    marker_size=10,
    thickness=2,
):
    """Draw a goal marker (success-radius ring + center crosshair) in robot frame.

    Intended for goal-conditioned tasks (e.g. ``puck_goal_position``) so the
    operator can see WHERE the puck is being asked to land. ``goal_xy_robot``
    is expected in robot frame — task code that owns the goal in table frame
    should subtract ``center_offset_constant`` before calling this.

    Distinct yellow color (BGR ``(0, 255, 255)``) to avoid clashing with the
    orange paddle-target cross and the green puck dot.
    """
    if frame is None or goal_xy_robot is None or len(goal_xy_robot) < 2:
        return frame

    if goal_radius_m is not None and float(goal_radius_m) > 0:
        draw_robot_circle_marker(
            frame,
            float(goal_xy_robot[0]),
            float(goal_xy_robot[1]),
            float(goal_radius_m),
            color,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            thickness=thickness,
        )

    center = robot_to_display_pixel_int(
        goal_xy_robot[0],
        goal_xy_robot[1],
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )
    cv2.line(
        frame,
        (center[0] - marker_size, center[1]),
        (center[0] + marker_size, center[1]),
        (0, 0, 0),
        thickness + 2,
    )
    cv2.line(
        frame,
        (center[0], center[1] - marker_size),
        (center[0], center[1] + marker_size),
        (0, 0, 0),
        thickness + 2,
    )
    cv2.line(
        frame,
        (center[0] - marker_size, center[1]),
        (center[0] + marker_size, center[1]),
        color,
        thickness,
    )
    cv2.line(
        frame,
        (center[0], center[1] - marker_size),
        (center[0], center[1] + marker_size),
        color,
        thickness,
    )
    return frame


def _assets_dir_from_overlay_utils():
    return Path(__file__).resolve().parents[3] / "assets"


def load_box2d_environment_image(assets_dir=None):
    """Load the Box2D table bitmap in the same orientation as ``AirHockeyRenderer``.

    ``AirHockeyRenderer`` rotates ``air_hockey_table.png`` 90° clockwise and then
    stretches it to the table rectangle. The four image corners therefore correspond
    to the four physical table corners; we keep the rotated image at native
    resolution and let the display homography do the stretch into camera space.
    """
    folder = Path(assets_dir) if assets_dir is not None else _assets_dir_from_overlay_utils()
    table_path = folder / "air_hockey_table.png"
    img = cv2.imread(str(table_path))
    if img is None:
        return None
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def box2d_table_src_dst_points(
    src_hw,
    table_length=DEFAULT_TABLE_LENGTH,
    table_width=DEFAULT_TABLE_WIDTH,
    center_offset=DEFAULT_CENTER_OFFSET,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
):
    """Pixel correspondences from the Box2D table image to the homography display.

    After the 90° clockwise rotate, Box2D's pre-rotate pixel map is:

        pixel_x = (-table_x + length/2) * ppm
        pixel_y = ( table_y + width/2) * ppm

    so image corners (0,0), (w-1,0), (w-1,h-1), (0,h-1) are table
    ``(+L/2, -W/2)``, ``(-L/2, -W/2)``, ``(-L/2, +W/2)``, ``(+L/2, +W/2)``.
    Destination pixels use the same robot→showdst map as the rest of the
    camera overlay stack (``robot_to_display_pixel``).
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    src = np.array(
        [
            [0.0, 0.0],
            [float(src_w - 1), 0.0],
            [float(src_w - 1), float(src_h - 1)],
            [0.0, float(src_h - 1)],
        ],
        dtype=np.float32,
    )
    half_l = float(table_length) * 0.5
    half_w = float(table_width) * 0.5
    table_corners = (
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
        (half_l, half_w),
    )
    dst = np.array(
        [
            robot_to_display_pixel(
                tx - float(center_offset),
                ty,
                offset_constants=offset_constants,
                visual_downscale_constant=visual_downscale_constant,
            )
            for tx, ty in table_corners
        ],
        dtype=np.float32,
    )
    return src, dst


def warp_box2d_environment_to_display(
    table_bgr,
    dst_hw,
    table_length=DEFAULT_TABLE_LENGTH,
    table_width=DEFAULT_TABLE_WIDTH,
    center_offset=DEFAULT_CENTER_OFFSET,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
):
    """Warp a Box2D table image into a homography-rectified camera frame.

    Returns ``(warped_bgr, mask_u8)`` at ``dst_hw``. The mask is 255 on pixels
    that came from the table image (so black fill outside the table is not
    blended onto the camera).
    """
    if table_bgr is None:
        return None, None
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    if dst_h <= 0 or dst_w <= 0:
        return None, None
    src, dst = box2d_table_src_dst_points(
        table_bgr.shape[:2],
        table_length=table_length,
        table_width=table_width,
        center_offset=center_offset,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    dsize = (dst_w, dst_h)
    warped = cv2.warpPerspective(table_bgr, homography, dsize)
    src_mask = np.full(table_bgr.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(src_mask, homography, dsize)
    return warped, mask


def blend_masked_overlay(frame, warped, mask, alpha=DEFAULT_SIM_OVERLAY_ALPHA):
    """Blend ``warped`` onto ``frame`` where ``mask`` is nonzero. Modifies ``frame``."""
    if frame is None or warped is None or mask is None:
        return frame
    alpha = float(alpha)
    if alpha <= 0.0:
        return frame
    if frame.shape[:2] != warped.shape[:2] or frame.shape[:2] != mask.shape[:2]:
        return frame
    alpha = min(1.0, alpha)
    mask_f = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    blended = frame.astype(np.float32) * (1.0 - mask_f) + warped.astype(np.float32) * mask_f
    np.copyto(frame, np.clip(np.round(blended), 0, 255).astype(np.uint8))
    return frame


class Box2DEnvironmentOverlay:
    """Cached faint Box2D table overlay aligned to the homography display frame."""

    def __init__(
        self,
        alpha=DEFAULT_SIM_OVERLAY_ALPHA,
        table_length=DEFAULT_TABLE_LENGTH,
        table_width=DEFAULT_TABLE_WIDTH,
        center_offset=DEFAULT_CENTER_OFFSET,
        offset_constants=None,
        visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
        assets_dir=None,
        table_image=None,
    ):
        self.alpha = float(alpha)
        self.table_length = float(table_length)
        self.table_width = float(table_width)
        self.center_offset = float(center_offset)
        self.offset_constants = _coerce_offset_constants(offset_constants)
        self.visual_downscale_constant = _coerce_downscale(visual_downscale_constant)
        self._table_bgr = table_image if table_image is not None else load_box2d_environment_image(assets_dir)
        self._warped = None
        self._mask = None
        self._dst_hw = None
        self._load_warned = False

    @classmethod
    def from_config(cls, sim_overlay):
        if not sim_overlay:
            return None
        if isinstance(sim_overlay, dict):
            alpha = float(sim_overlay.get("alpha", DEFAULT_SIM_OVERLAY_ALPHA))
            if alpha <= 0.0:
                return None
            return cls(
                alpha=alpha,
                table_length=sim_overlay.get("table_length", DEFAULT_TABLE_LENGTH),
                table_width=sim_overlay.get("table_width", DEFAULT_TABLE_WIDTH),
                center_offset=sim_overlay.get("center_offset", DEFAULT_CENTER_OFFSET),
                offset_constants=sim_overlay.get("offset_constants"),
                visual_downscale_constant=sim_overlay.get(
                    "visual_downscale_constant", DEFAULT_VISUAL_DOWNSCALE_CONSTANT
                ),
                assets_dir=sim_overlay.get("assets_dir"),
            )
        alpha = float(sim_overlay)
        if alpha <= 0.0:
            return None
        return cls(alpha=alpha)

    def apply(self, frame):
        if frame is None or self.alpha <= 0.0:
            return frame
        if self._table_bgr is None:
            if not self._load_warned:
                print(
                    "[sim_overlay] Could not load assets/air_hockey_table.png; "
                    "Box2D environment overlay disabled."
                )
                self._load_warned = True
            return frame
        dst_hw = (int(frame.shape[0]), int(frame.shape[1]))
        if self._dst_hw != dst_hw:
            self._warped, self._mask = warp_box2d_environment_to_display(
                self._table_bgr,
                dst_hw,
                table_length=self.table_length,
                table_width=self.table_width,
                center_offset=self.center_offset,
                offset_constants=self.offset_constants,
                visual_downscale_constant=self.visual_downscale_constant,
            )
            self._dst_hw = dst_hw
        return blend_masked_overlay(frame, self._warped, self._mask, self.alpha)
