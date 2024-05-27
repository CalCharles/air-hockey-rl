import numpy as np
import cv2
import copy

def visualize_regions(frame, reward_region_info, goal_info, paddle_info):
    # frame is the image frame
    # reward regions defined: [x y rx ry, ...]
    # goals defined: x y r
    # paddle defined: x y r
    offset_constants = np.array((2100, 500))
    for r in reward_region_info:
        # print("reward_regions", r.state, r.radius)
        # ((0.09870443) *1000  + 500)/2
        pixel_coord = (((r[0] - 1)*1000, (-r[1])*1000 ) + offset_constants) /2
        center_coordinates = (int(np.round(pixel_coord[0])), int(np.round(pixel_coord[1])))  # Example coordinates (x, y)
        
        pixel_radius = (int((r[2])*1000 /2), int((r[3])*1000 /2))
        # radius = int(np.round((pixel_radius[0]))) #, int(np.round(-pixel_radius[1])))
        color = (0, 0, 255)  # Red color in BGR
        thickness = 2  # Thickness of 2 px, use -1 for a filled circle
        # center_coordinates = (480, 360)
        center_coordinates_new = (center_coordinates[0], center_coordinates[1])
        # print("reward_regions", center_coordinates_new, pixel_radius)
        # Draw the circle on the frame
        cv2.ellipse(frame, center_coordinates_new, pixel_radius, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)
    goal = goal_info[:2]
    goal = copy.deepcopy(goal)
    goal_coord = (((goal[0] - 1)*1000, (-goal[1])*1000 ) + offset_constants) /2
    goal_center_coordinates = (int(np.round(goal_coord[0])), int(np.round(goal_coord[1])))  # Example coordinates (x, y)
    cv2.circle(frame, goal_center_coordinates, radius=int(np.round(goal_info[2]*1000/2)), color=(0,255,0), thickness=thickness)

    paddles = [copy.deepcopy(paddle_info[:2])]
    paddle_coord = (((paddles[0][0])*1000, (-paddles[0][1])*1000 ) + offset_constants) /2
    center_coordinates = (int(np.round(paddle_coord[0])), int(np.round(paddle_coord[1])))  # Example coordinates (x, y)
    cv2.circle(frame, center_coordinates, radius=int(np.round(paddle_info[2]*1000/2)), color=(255,0,0), thickness=thickness)
    return frame
