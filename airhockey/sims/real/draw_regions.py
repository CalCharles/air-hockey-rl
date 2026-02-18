import numpy as np
import cv2
import copy
from .overlay_utils import robot_to_display_pixel_int, meters_to_display_pixels, observation_to_robot_xy, draw_paddle_marker


def visualize_regions(
    frame,
    reward_region_info,
    goal_info,
    paddle_info,
    x_offset=1.0,
    offset_constants=None,
    visual_downscale_constant=2,
    draw_paddle=True,
):
    # frame is the image frame
    # reward regions defined: [x y rx ry, ...]
    # goals defined: x y r
    # paddle defined: x y r
    for r in reward_region_info:
        rx, ry = observation_to_robot_xy(r[0], r[1], x_offset)
        center_coordinates = robot_to_display_pixel_int(
            rx,
            ry,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
        )

        pixel_radius = (
            int(np.round(meters_to_display_pixels(r[2], visual_downscale_constant))),
            int(np.round(meters_to_display_pixels(r[3], visual_downscale_constant))),
        )
        color = (0, 0, 255)  # Red color in BGR
        thickness = 2  # Thickness of 2 px, use -1 for a filled circle
        cv2.ellipse(frame, center_coordinates, pixel_radius, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)

    goal = goal_info[:2]
    goal = copy.deepcopy(goal)
    goal_x, goal_y = observation_to_robot_xy(goal[0], goal[1], x_offset)
    goal_center_coordinates = robot_to_display_pixel_int(
        goal_x,
        goal_y,
        offset_constants=offset_constants,
        visual_downscale_constant=visual_downscale_constant,
    )
    cv2.circle(
        frame,
        goal_center_coordinates,
        radius=int(np.round(meters_to_display_pixels(goal_info[2], visual_downscale_constant))),
        color=(0, 255, 0),
        thickness=thickness,
    )

    if draw_paddle and paddle_info is not None:
        draw_paddle_marker(
            frame,
            (paddle_info[0], paddle_info[1]),
            paddle_info[2],
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            color=(255, 0, 0),
        )
    return frame
