import cv2
import imageio
import time, os
import numpy as np
from .image_detection import find_red_hockey_paddle, find_red_hockey_puck
from .draw_regions import visualize_regions
from .overlay_utils import (
    robot_to_display_pixel_int,
    draw_target_marker,
    draw_puck_marker_from_state,
    draw_paddle_marker,
)


mousepos = (0,0,1)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
Mimg = np.load(os.path.join(base_dir, 'assets', 'real' ,'Mimg.npy'))

upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2
save_downscale_constant = 2
offset_constants = np.array((2250, 500))


def _effective_xmax(y_m, lims, edge_lims):
    _, x_max_lim, _, _ = lims
    top_abs, _, max_bias_p, max_bias_m = edge_lims
    return min(x_max_lim, max_bias_m - top_abs * y_m, max_bias_p + top_abs * y_m)


def draw_robot_edge_limits(frame, lims, edge_lims, color=(0, 255, 255), thickness=2):
    x_min_lim, _, y_min, y_max = lims

    def _to_int_point(x_m, y_m):
        return robot_to_display_pixel_int(
            x_m,
            y_m,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
        )

    # Left edge (x = x_min), top and bottom edges, plus effective right edge.
    left_top = _to_int_point(x_min_lim, y_max)
    left_bottom = _to_int_point(x_min_lim, y_min)
    cv2.line(frame, left_top, left_bottom, color, thickness)

    right_top_x = _effective_xmax(y_max, lims, edge_lims)
    right_bottom_x = _effective_xmax(y_min, lims, edge_lims)
    right_top = _to_int_point(right_top_x, y_max)
    right_bottom = _to_int_point(right_bottom_x, y_min)

    cv2.line(frame, left_top, right_top, color, thickness)
    cv2.line(frame, left_bottom, right_bottom, color, thickness)

    ys = np.linspace(y_min, y_max, num=64)
    right_points = np.array(
        [_to_int_point(_effective_xmax(y_val, lims, edge_lims), y_val) for y_val in ys],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(frame, [right_points], isClosed=False, color=color, thickness=thickness)

def single_point_homography(matrix, point):
    x,y = point
    return np.array([matrix[0,0] * x + matrix[0,1] * y + matrix[0,2] /
                    (matrix[2,0] * x + matrix[2,1] * y + matrix[2,2]), 
                     matrix[1,0] * x + matrix[1,1] * y + matrix[1,2] /
                    (matrix[2,0] * x + matrix[2,1] * y + matrix[2,2])])

def homography_transform(image, get_save=True, rotate=False):
    image = cv2.rotate(image, cv2.ROTATE_180)
    save_image = None
    if get_save:
        save_image = cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant)))
        # print("images", image, save_image)
    image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
    if rotate: 
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        showdst = cv2.resize(dst, (int(480*upscale_constant / visual_downscale_constant), int(640*upscale_constant / visual_downscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    else:
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    return showdst, save_image

def camera_callback(
    shared_array,
    save_image_check,
    puck_array,
    paddle_info,
    target_info,
    region_info,
    goal_info,
    lims=None,
    edge_lims=None,
    puck_detector=find_red_hockey_puck,
    puck_detector_kwargs=None,
    puck_radius=0.03175,
    region_x_offset=1.0,
):
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    detector_kwargs = puck_detector_kwargs if puck_detector_kwargs is not None else {}
    while True:
        start = time.time()
        ret, image = cap.read()
        save_image_id = save_image_check[0] == 1
        showdst, save_image = homography_transform(image, get_save=save_image_id)
        if save_image_id: 
            imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", save_image)
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant))))
        # # shared_image[:] = image.flatten()
        # image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
        #             interpolation = cv2.INTER_LINEAR)
        # dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
        # showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
        #             interpolation = cv2.INTER_LINEAR)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        puck = puck_detector(showdst, rotate=False, **detector_kwargs)
        if lims is not None and edge_lims is not None:
            draw_robot_edge_limits(showdst, lims, edge_lims)
        if region_info is not None:
            showdst = visualize_regions(
                showdst,
                region_info,
                goal_info,
                paddle_info,
                x_offset=region_x_offset,
                offset_constants=offset_constants,
                visual_downscale_constant=visual_downscale_constant,
                draw_paddle=False,
            )
        if target_info[2] > 0:
            draw_target_marker(
                showdst,
                (target_info[0], target_info[1]),
                offset_constants=offset_constants,
                visual_downscale_constant=visual_downscale_constant,
            )
        draw_puck_marker_from_state(
            showdst,
            puck,
            puck_radius,
            x_offset_for_state=0.0,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            color=(0, 255, 0),
            require_visible=True,
        )
        draw_paddle_marker(
            showdst,
            (paddle_info[0], paddle_info[1]),
            paddle_info[2],
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            color=(255, 0, 0),
        )
        cv2.imshow('image',showdst)
        cv2.setMouseCallback('image', move_event)
        puck_array[0] = puck[0]
        puck_array[1] = puck[1]
        puck_array[2] = puck[2]
        shared_array[0] = mousepos[0] * visual_downscale_constant
        shared_array[1] = mousepos[1] * visual_downscale_constant
        shared_array[2] = mousepos[2] * visual_downscale_constant
        cv2.waitKey(1)
        # print("showtime", time.time() - start)

def move_event(event, x, y, flags, params):
    global mousepos
    if event==cv2.EVENT_MOUSEMOVE:
  
        # displaying the coordinates
        # on the Shell
        # print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y)
        mousepos = (x,y,1)

# callback functions for mimic control
def mimic_control(shared_array):
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    Mimg_tele = np.load(os.path.join(base_dir, 'assets', 'real' ,'Mimg_tele.npy'))

    while True:
        start = time.time()
        ret, image = cap.read()
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # shared_image[:] = image.flatten()
        image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        dst = cv2.warpPerspective(image,Mimg_tele,original_size * upscale_constant)
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        x,y,changed_image = find_red_hockey_paddle(showdst)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        cv2.imshow('image',changed_image)
        shared_array[0] = y * visual_downscale_constant
        shared_array[1] = x * visual_downscale_constant
        cv2.waitKey(1)

def save_callback(save_image_check):
    # TODO: changed to 0 for now
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        start = time.time()
        ret, image = cap.read()
        showdst, save_image = homography_transform(image, get_save=True, rotate=True)
        if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", save_image)
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant))))
        # image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
        #             interpolation = cv2.INTER_LINEAR)
        # dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
        # dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        # showdst = cv2.resize(dst, (int(480*upscale_constant / visual_downscale_constant), int(640*upscale_constant / visual_downscale_constant)), 
        #             interpolation = cv2.INTER_LINEAR)
        cv2.imshow('showdst',showdst)
        cv2.waitKey(1)

# performs saving without multiprocessing
def save_collect(cap, paddle_info, region_info, goal_info, show=True, lims=None, edge_lims=None, region_x_offset=1.0):
    start = time.time()
    ret, image = cap.read()
    frame_received_s = time.time()
    showdst, save_image = homography_transform(image, get_save=True, rotate=False)
    if lims is not None and edge_lims is not None:
        draw_robot_edge_limits(showdst, lims, edge_lims)
    if region_info is not None:
        showdst = visualize_regions(
            showdst,
            region_info,
            goal_info,
            paddle_info,
            x_offset=region_x_offset,
            offset_constants=offset_constants,
            visual_downscale_constant=visual_downscale_constant,
            draw_paddle=False,
        )
    if show:
        cv2.imshow('showdst',showdst)
        cv2.waitKey(1)
    return showdst, save_image, frame_received_s

def observe_collect(showdst, paddle_info, region_info, goal_info, save_image=False):
    result, changed_image = find_red_hockey_paddle(showdst)
    x,y,detected = result
    showdst[x-3:x+3, y-3:y+3, :] = 0
    x,y = (np.array([y * 2,x * 2]) - offset_constants)/ 1000
    y = - y 
    print(x,y)
    if region_info is not None: showdst = visualize_regions(showdst, region_info, goal_info, (x, y, paddle_info[-1]))
    cv2.imshow('image',showdst)
    cv2.waitKey(1)
    if save_image: 
        imageio.imsave("./data/observe/img" + str(time.time()) + ".jpg", showdst)

    return x,y, detected
