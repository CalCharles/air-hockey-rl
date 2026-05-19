import cv2
import numpy as np

DEFAULT_VISUAL_DOWNSCALE_CONSTANT = 2.0
DEFAULT_OFFSET_CONSTANTS = np.array((2100.0, 500.0), dtype=float)


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
