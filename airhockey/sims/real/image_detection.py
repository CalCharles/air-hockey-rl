import cv2, os
import numpy as np

def find_red_hockey_paddle(image):
    # Load the image
    # image = cv2.imread(image_path)

    # Convert to HSV color space
    # start = time.time()
    # print(image.shape)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)), 
                    interpolation = cv2.INTER_LINEAR)
    # print(image.shape)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_image[:,:int(540)] = 0
    # hsv_image[int(400):,:] = 0
    hsv_image[:,:int(540 / 4)] = 0
    hsv_image[int(500 / 4):,:] = 0

    # hsv_image[:,:120] = 0
    # hsv_image[:,200:] = 0
    # hsv_image[200:,:] = 0

    # Define the range of red color in HSV
    # These values might need adjustment depending on the image
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = mask1 + mask2
    # cv2.imshow('hsv',hsv_image)
    # # cv2.imshow('mask',mask)
    # cv2.waitKey(1)



    # Blob detection parameters
#     params = cv2.SimpleBlobDetector_Params()
#     params.filterByColor = True
#     params.blobColor = 255  # Since the mask will be white where the puck is
#     params.minDistBetweenBlobs = 100


#    # Filter by Area.
#     params.filterByArea = False
#     params.minArea = 100  # Adjust based on the size of the puck in the image

#     # Filter by Circularity
#     params.filterByCircularity = False
#     params.minCircularity = 0.7  # Adjust to better match the puck's shape

#     # Filter by Convexity
#     params.filterByConvexity = False
#     params.minConvexity = 0.8

#     # Filter by Inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.5
    
    # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    # keypoints = detector.detect(mask)
    vals = np.where(mask > 0)
    if len(vals[0]) < MIN_DETECT:
        return (-2, 0, 1), image
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))

    # # Draw detected blobs as red circles
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # print(image.shape)
    
    # image_with_keypoints = cv2.drawKeypoints(image, [(x,y)], np.array([]), (0, 0, 255),
    #                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # py = int(keypoints[0].pt[0])
    # px = int(keypoints[0].pt[1])
    # width=100
    image[x-3:x+3, y-3:y+3, :] = 0
    # Save the image with keypoints
    # cv2.imwrite('output.jpg', image_with_keypoints)
    # print("inrange", time.time()-start)
    return (x*4,y*4,0), image



MIN_DETECT = 25

# Mimg = np.load('assets/real/Mimg.npy')
Mimg = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "real", "Mimg.npy"))
upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2
save_downscale_constant = 2
offset_constants = np.array((2100, 500))

def _fallback_puck(puck_history):
    if puck_history is not None and len(puck_history) > 0:
        return puck_history[-1][0], puck_history[-1][1], 1
    return -2, 0, 1


def _preprocess_puck_image(image, rotate):
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.resize(
        image,
        (int(image.shape[1] / 2), int(image.shape[0] / 2)),
        interpolation=cv2.INTER_LINEAR,
    )
    h, w = image.shape[:2]
    image[min(249, h):, :] = 0
    image[: min(9, h), :] = 0
    image[:, min(470, w):] = 0
    return image


def _pixel_to_robot_xy(x_px, y_px):
    # Detector coordinates are in the downscaled detector frame.
    # We map back by x4 to the original homography pixel frame before offset conversion.
    homo_idx = (np.array([x_px * 4, y_px * 4]) - offset_constants) * 0.001
    return homo_idx[0], -homo_idx[1]


def _dual_red_mask(hsv_image, sat_min, val_min, low_h=(0, 10), high_h=(170, 180)):
    low_1 = np.array([low_h[0], sat_min, val_min], dtype=np.uint8)
    high_1 = np.array([low_h[1], 255, 255], dtype=np.uint8)
    low_2 = np.array([high_h[0], sat_min, val_min], dtype=np.uint8)
    high_2 = np.array([high_h[1], 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_image, low_1, high_1) | cv2.inRange(hsv_image, low_2, high_2)


def _apply_antiglare_rescue_mask(
    base_mask,
    hsv_image,
    antiglare_min_x_px=None,
    antiglare_max_x_px=None,
    antiglare_min_y_px=None,
    antiglare_max_y_px=None,
):
    if None in (
        antiglare_min_x_px,
        antiglare_max_x_px,
        antiglare_min_y_px,
        antiglare_max_y_px,
    ):
        return base_mask

    h, w = base_mask.shape[:2]
    x0 = int(np.clip(np.round(float(antiglare_min_x_px)), 0, w))
    x1 = int(np.clip(np.round(float(antiglare_max_x_px)), 0, w))
    y0 = int(np.clip(np.round(float(antiglare_min_y_px)), 0, h))
    y1 = int(np.clip(np.round(float(antiglare_max_y_px)), 0, h))
    if x1 <= x0 or y1 <= y0:
        return base_mask

    rescue_mask = base_mask.copy()
    roi_hsv = hsv_image[y0:y1, x0:x1]
    roi_rescue = _dual_red_mask(
        roi_hsv,
        sat_min=20,
        val_min=120,
        low_h=(0, 15),
        high_h=(165, 180),
    )
    rescue_mask[y0:y1, x0:x1] = cv2.bitwise_or(rescue_mask[y0:y1, x0:x1], roi_rescue)
    return rescue_mask


def _history_to_detector_pixel(puck_history, center_offset_constant):
    if puck_history is None or len(puck_history) == 0:
        return None
    prev_x = float(puck_history[-1][0]) - float(center_offset_constant)
    prev_y = float(puck_history[-1][1])
    pred_x = (prev_x * 1000.0 + float(offset_constants[0])) / 4.0
    pred_y = (-prev_y * 1000.0 + float(offset_constants[1])) / 4.0
    return pred_x, pred_y


def _select_component_centroid(mask, puck_history=None, center_offset_constant=0.0):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < MIN_DETECT or area > 2500:
            continue
        cx, cy = centroids[label_idx]
        candidates.append((area, float(cx), float(cy)))
    if not candidates:
        return None

    pred = _history_to_detector_pixel(puck_history, center_offset_constant)
    if pred is None:
        _, cx, cy = max(candidates, key=lambda c: c[0])
        return int(np.round(cx)), int(np.round(cy))

    pred_x, pred_y = pred
    _, cx, cy = min(
        candidates,
        key=lambda c: (c[1] - pred_x) ** 2 + (c[2] - pred_y) ** 2,
    )
    return int(np.round(cx)), int(np.round(cy))


def find_red_hockey_puck(image, puck_history=None, rotate=True, **_ignored_kwargs):
    image = _preprocess_puck_image(image, rotate=rotate)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    refined_mask = cv2.inRange(
        hsv_image,
        np.array([0, 137, 80], dtype=np.uint8),
        np.array([8, 255, 255], dtype=np.uint8),
    )
    puck_idx = np.where(refined_mask)
    if len(puck_idx[0]) < MIN_DETECT:
        return _fallback_puck(puck_history)

    y = int(np.round(np.median(puck_idx[0])))
    x = int(np.round(np.median(puck_idx[1])))
    robot_x, robot_y = _pixel_to_robot_xy(x, y)
    return robot_x, robot_y, 0


def find_red_hockey_puck_antiglare(
    image,
    puck_history=None,
    rotate=True,
    antiglare_min_x_px=None,
    antiglare_max_x_px=None,
    antiglare_min_y_px=None,
    antiglare_max_y_px=None,
    center_offset_constant=0.0,
    **_ignored_kwargs,
):
    image = _preprocess_puck_image(image, rotate=rotate)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = _dual_red_mask(
        hsv_image,
        sat_min=90,
        val_min=45,
        low_h=(0, 10),
        high_h=(170, 180),
    )
    mask = _apply_antiglare_rescue_mask(
        mask,
        hsv_image,
        antiglare_min_x_px=antiglare_min_x_px,
        antiglare_max_x_px=antiglare_max_x_px,
        antiglare_min_y_px=antiglare_min_y_px,
        antiglare_max_y_px=antiglare_max_y_px,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    center = _select_component_centroid(
        mask,
        puck_history=puck_history,
        center_offset_constant=center_offset_constant,
    )
    if center is None:
        return _fallback_puck(puck_history)

    x, y = center
    robot_x, robot_y = _pixel_to_robot_xy(x, y)
    return robot_x, robot_y, 0