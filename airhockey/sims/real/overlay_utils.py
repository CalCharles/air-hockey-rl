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


def enlarged_goal_marker_radius_m(simulator, reward_radius_m=None):
    """Visual goal-ring radius on homography overlays.

    Matches the enlarged default used in ``puck_goal_position_velocity`` /
    ``puck_goal_position_dynamic_negative_regions`` when ``goal_radius_type``
    is ``fixed`` (``(min + max) / 2 * 0.75``). Falls back to the task reward
    radius when simulator bounds are unavailable.
    """
    min_r = getattr(simulator, "min_goal_radius", None)
    max_r = getattr(simulator, "max_goal_radius", None)
    if min_r is not None and max_r is not None:
        return (float(min_r) + float(max_r)) / 2.0 * 0.75
    if reward_radius_m is None:
        return None
    try:
        reward_val = float(reward_radius_m)
    except (TypeError, ValueError):
        return None
    return reward_val if reward_val > 0 else None


def draw_homography_episode_markers(
    frame,
    *,
    target_xy_robot,
    puck_state_table,
    paddle_xy_robot,
    goal_xy_table=None,
    goal_radius_m=None,
    center_offset_constant=0.0,
    puck_radius_m=0.03175,
    paddle_radius_m=0.0508,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
):
    """Draw the live robot overlay stack onto a homography-warped BGR frame."""
    if frame is None:
        return frame

    if target_xy_robot is not None:
        draw_target_marker(
            frame,
            target_xy_robot,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
        )

    if puck_state_table is not None:
        draw_puck_marker_from_state(
            frame,
            puck_state_table,
            puck_radius_m,
            x_offset_for_state=center_offset_constant,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            color=(0, 255, 0),
            require_visible=True,
        )

    if paddle_xy_robot is not None:
        draw_paddle_marker(
            frame,
            paddle_xy_robot,
            paddle_radius_m,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            color=(255, 0, 0),
        )

    if goal_xy_table is not None:
        goal_robot_x, goal_robot_y = observation_to_robot_xy(
            goal_xy_table[0],
            goal_xy_table[1],
            center_offset_constant,
        )
        draw_goal_marker(
            frame,
            (goal_robot_x, goal_robot_y),
            goal_radius_m=goal_radius_m,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
        )

    return frame


def draw_goal_marker(
    frame,
    goal_xy_robot,
    goal_radius_m=None,
    offset_constants=None,
    visual_downscale_constant=DEFAULT_VISUAL_DOWNSCALE_CONSTANT,
    color=(0, 255, 0),
    thickness=2,
):
    """Draw the task success region as a green ring in robot frame.

    Intended for goal-conditioned tasks (e.g. ``puck_goal_position``). Matches
    the Box2D ``AirHockeyRenderer`` convention (green goal circle at true
    ``goal_radius``). ``goal_xy_robot`` is in robot frame — callers with a
    table-frame goal should subtract ``center_offset_constant`` first.
    """
    if frame is None or goal_xy_robot is None or len(goal_xy_robot) < 2:
        return frame
    if goal_radius_m is None or float(goal_radius_m) <= 0:
        return frame

    return draw_robot_circle_marker(
        frame,
        float(goal_xy_robot[0]),
        float(goal_xy_robot[1]),
        float(goal_radius_m),
        color,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
        thickness=thickness,
    )
