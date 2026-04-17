import cv2, os
from functools import lru_cache
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
MIN_PUCK_RADIUS_PX = 5
MAX_PUCK_RADIUS_PX = 10
MIN_PUCK_CIRCULARITY = 0.4
SIMPLE_MIN_PUCK_RADIUS_PX = 5
SIMPLE_MAX_PUCK_RADIUS_PX = 10
SIMPLE_MIN_PUCK_CIRCULARITY = 0.80
SIMPLE_MIN_PUCK_FILL_RATIO = 0.7
SIMPLE_LOOSE_MIN_PUCK_FILL_RATIO = 0.55

# Mimg = np.load('assets/real/Mimg.npy')
Mimg = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "real", "Mimg.npy"))
upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2
save_downscale_constant = 2
# offset_constants = np.array((2100, 500)) # change this number
offset_constants = np.array((2250, 500)) # change this number
MORPH_KERNEL_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def _fallback_puck(puck_history):
    if puck_history is not None and len(puck_history) > 0:
        return puck_history[-1][0], puck_history[-1][1], 1
    return -2, 0, 1


def _validated_detection(
    robot_x,
    robot_y,
    puck_history,
    *,
    validate_field_containment=True,
    # Field bounds in policy frame (post center-offset shift). Defaults match
    # the standard table (length=1.9304 m, width=0.8636 m): puck x ∈ ±length/2,
    # puck y ∈ ±width/2. Pass explicit values from the env to stay in sync
    # with the sim config.
    field_x_min=-0.9652,
    field_x_max=0.9652,
    field_y_min=-0.4318,
    field_y_max=0.4318,
    center_offset_constant=0.0,
    validator_puck_radius=0.03175,
    # Small slack past the geometric bound absorbs calibration noise so a
    # legitimate wall-hugging puck is not rejected.
    containment_slack_m=0.005,
    validator_debug=False,
    **_ignored_kwargs,
):
    # Converts the detector output (pre-shift robot frame) into a post-shift
    # policy-frame coordinate and checks that the puck is fully contained on
    # the field. If not, returns the standard occlusion fallback so every
    # downstream consumer (obs valid-flag, puck-absence gate, reward code)
    # treats the frame as "puck not detected".
    if validate_field_containment:
        policy_x = float(robot_x) + float(center_offset_constant)
        policy_y = float(robot_y)
        r = float(validator_puck_radius)
        s = float(containment_slack_m)
        inside = (
            field_x_min + r - s <= policy_x <= field_x_max - r + s
            and field_y_min + r - s <= policy_y <= field_y_max - r + s
        )
        if not inside:
            if validator_debug:
                print(
                    f"[puck_validator] rejected (out-of-field): "
                    f"policy=({policy_x:.3f}, {policy_y:.3f}) "
                    f"bounds x=[{field_x_min:.3f},{field_x_max:.3f}] "
                    f"y=[{field_y_min:.3f},{field_y_max:.3f}] "
                    f"r={r:.4f} slack={s:.4f}"
                )
            return _fallback_puck(puck_history)
    return robot_x, robot_y, 0


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


@lru_cache(maxsize=32)
def _cached_dual_red_hsv_bounds(sat_min, val_min, low_h0, low_h1, high_h0, high_h1):
    low_1 = np.array([low_h0, sat_min, val_min], dtype=np.uint8)
    high_1 = np.array([low_h1, 255, 255], dtype=np.uint8)
    low_2 = np.array([high_h0, sat_min, val_min], dtype=np.uint8)
    high_2 = np.array([high_h1, 255, 255], dtype=np.uint8)
    return low_1, high_1, low_2, high_2


def _dual_red_mask(hsv_image, sat_min, val_min, low_h=(0, 10), high_h=(170, 180)):
    low_1, high_1, low_2, high_2 = _cached_dual_red_hsv_bounds(
        int(sat_min),
        int(val_min),
        int(low_h[0]),
        int(low_h[1]),
        int(high_h[0]),
        int(high_h[1]),
    )
    return cv2.inRange(hsv_image, low_1, high_1) | cv2.inRange(hsv_image, low_2, high_2)


def _apply_antiglare_rescue_mask(
    base_mask,
    bgr_image,
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
    roi_bgr = bgr_image[y0:y1, x0:x1]
    roi_hsv = hsv_image[y0:y1, x0:x1]
    roi_rescue = _dual_red_mask(
        roi_hsv,
        sat_min=20,
        val_min=120,
        low_h=(0, 15),
        high_h=(165, 180),
    )
    # Glare makes the puck appear desaturated/grayish red. In that case hue can
    # become unreliable, so use a simple channel-dominance rescue in BGR space.
    r = roi_bgr[:, :, 2].astype(np.int16)
    g = roi_bgr[:, :, 1].astype(np.int16)
    b = roi_bgr[:, :, 0].astype(np.int16)
    roi_gray_red_rescue = (
        (r >= 160)
        & ((r - np.maximum(g, b)) >= 25)
        & (np.abs(g - b) <= 35)
    ).astype(np.uint8) * 255

    roi_rescue_combined = cv2.bitwise_or(roi_rescue, roi_gray_red_rescue)
    rescue_mask[y0:y1, x0:x1] = cv2.bitwise_or(
        rescue_mask[y0:y1, x0:x1], roi_rescue_combined
    )
    return rescue_mask


@lru_cache(maxsize=64)
def _convert_raw_bounds_to_detector_bounds(
    antiglare_min_x_px,
    antiglare_max_x_px,
    antiglare_min_y_px,
    antiglare_max_y_px,
    raw_shape,
    detector_shape,
    rotate,
):
    if None in (
        antiglare_min_x_px,
        antiglare_max_x_px,
        antiglare_min_y_px,
        antiglare_max_y_px,
    ):
        return antiglare_min_x_px, antiglare_max_x_px, antiglare_min_y_px, antiglare_max_y_px

    x0 = float(min(antiglare_min_x_px, antiglare_max_x_px))
    x1 = float(max(antiglare_min_x_px, antiglare_max_x_px))
    y0 = float(min(antiglare_min_y_px, antiglare_max_y_px))
    y1 = float(max(antiglare_min_y_px, antiglare_max_y_px))

    raw_h, raw_w = raw_shape[:2]
    det_h, det_w = detector_shape[:2]
    if raw_h <= 0 or raw_w <= 0 or det_h <= 0 or det_w <= 0:
        return antiglare_min_x_px, antiglare_max_x_px, antiglare_min_y_px, antiglare_max_y_px

    if rotate:
        # Match _preprocess_puck_image rotation: ROTATE_90_COUNTERCLOCKWISE.
        corners = np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=float,
        )
        rotated = np.zeros_like(corners)
        rotated[:, 0] = corners[:, 1]  # x' = y
        rotated[:, 1] = (raw_w - 1.0) - corners[:, 0]  # y' = W - 1 - x
        x0_r, y0_r = np.min(rotated, axis=0)
        x1_r, y1_r = np.max(rotated, axis=0)
        raw_ref_w = float(raw_h)
        raw_ref_h = float(raw_w)
    else:
        x0_r, x1_r, y0_r, y1_r = x0, x1, y0, y1
        raw_ref_w = float(raw_w)
        raw_ref_h = float(raw_h)

    scale_x = float(det_w) / max(raw_ref_w, 1.0)
    scale_y = float(det_h) / max(raw_ref_h, 1.0)
    return (
        x0_r * scale_x,
        x1_r * scale_x,
        y0_r * scale_y,
        y1_r * scale_y,
    )


def _history_to_detector_pixel(puck_history, center_offset_constant):
    if puck_history is None or len(puck_history) == 0:
        return None
    prev_x = float(puck_history[-1][0]) - float(center_offset_constant)
    prev_y = float(puck_history[-1][1])
    pred_x = (prev_x * 1000.0 + float(offset_constants[0])) / 4.0
    pred_y = (-prev_y * 1000.0 + float(offset_constants[1])) / 4.0
    return pred_x, pred_y


def _select_component_centroid(
    mask,
    puck_history=None,
    center_offset_constant=0.0,
    min_radius_px=None,
    max_radius_px=None,
    min_circularity=None,
    min_fill_ratio=None,
    loose_min_fill_ratio=None,
    max_area=2500,
):
    pred = _history_to_detector_pixel(puck_history, center_offset_constant)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strict_candidates = []
    loose_candidates = []
    min_radius_px = MIN_PUCK_RADIUS_PX if min_radius_px is None else float(min_radius_px)
    max_radius_px = MAX_PUCK_RADIUS_PX if max_radius_px is None else float(max_radius_px)
    min_circularity = (
        MIN_PUCK_CIRCULARITY if min_circularity is None else float(min_circularity)
    )

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_DETECT or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius_px or radius > max_radius_px:
            continue

        fill_ratio = area / max(np.pi * radius * radius, 1e-6)
        if loose_min_fill_ratio is not None and fill_ratio < float(loose_min_fill_ratio):
            continue
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        candidate = (area, float(cx), float(cy), circularity, float(radius), fill_ratio)
        loose_candidates.append(candidate)
        if circularity >= min_circularity and (
            min_fill_ratio is None or fill_ratio >= float(min_fill_ratio)
        ):
            strict_candidates.append(candidate)

    # Prefer clearly circular blobs, but keep a loose fallback to avoid dropouts.
    candidates = strict_candidates if len(strict_candidates) > 0 else loose_candidates
    if not candidates:
        return None

    if pred is None:
        _, cx, cy, _, _, _ = max(
            candidates, key=lambda c: (c[3], c[5], c[0])
        )
        return int(np.round(cx)), int(np.round(cy))

    pred_x, pred_y = pred
    _, cx, cy, _, _, _ = min(
        candidates,
        key=lambda c: (c[1] - pred_x) ** 2
        + (c[2] - pred_y) ** 2
        + 200.0 * (1.0 - c[3]) ** 2
        + 120.0 * (1.0 - c[5]) ** 2,
    )
    return int(np.round(cx)), int(np.round(cy))


def find_red_hockey_puck(
    image,
    puck_history=None,
    rotate=True,
    center_offset_constant=0.0,
    **_ignored_kwargs,
):
    image = _preprocess_puck_image(image, rotate=rotate)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Puck is roughly 30 px wide in raw detector input; _preprocess_puck_image
    # downsamples by 2, so we gate radius around that while keeping a wide margin.
    red_mask = _dual_red_mask(
        hsv_image,
        sat_min=45,
        val_min=45,
        low_h=(0, 13),
        high_h=(165, 180),
    )
    r = image[:, :, 2].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 0].astype(np.int16)
    gray_red_rescue = (
        (r >= 110)
        & ((r - np.maximum(g, b)) >= 12)
        & (np.abs(g - b) <= 45)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(red_mask, gray_red_rescue)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL_3X3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL_3X3, iterations=1)

    center = _select_component_centroid(
        mask,
        puck_history=puck_history,
        center_offset_constant=center_offset_constant,
        min_radius_px=SIMPLE_MIN_PUCK_RADIUS_PX,
        max_radius_px=SIMPLE_MAX_PUCK_RADIUS_PX,
        min_circularity=SIMPLE_MIN_PUCK_CIRCULARITY,
        min_fill_ratio=SIMPLE_MIN_PUCK_FILL_RATIO,
        loose_min_fill_ratio=SIMPLE_LOOSE_MIN_PUCK_FILL_RATIO,
    )
    if center is None:
        return _fallback_puck(puck_history)

    x, y = center
    robot_x, robot_y = _pixel_to_robot_xy(x, y)
    return _validated_detection(
        robot_x, robot_y, puck_history,
        center_offset_constant=center_offset_constant,
        **_ignored_kwargs,
    )


def find_red_hockey_puck_antiglare(
    image,
    puck_history=None,
    rotate=True,
    antiglare_bounds_in_raw_image=True,
    antiglare_min_x_px=None,
    antiglare_max_x_px=None,
    antiglare_min_y_px=None,
    antiglare_max_y_px=None,
    center_offset_constant=0.0,
    **_ignored_kwargs,
):
    raw_shape = image.shape
    image = _preprocess_puck_image(image, rotate=rotate)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if antiglare_bounds_in_raw_image:
        (
            antiglare_min_x_px,
            antiglare_max_x_px,
            antiglare_min_y_px,
            antiglare_max_y_px,
        ) = _convert_raw_bounds_to_detector_bounds(
            antiglare_min_x_px,
            antiglare_max_x_px,
            antiglare_min_y_px,
            antiglare_max_y_px,
            raw_shape=raw_shape,
            detector_shape=image.shape,
            rotate=rotate,
        )

    mask = _dual_red_mask(
        hsv_image,
        sat_min=80,
        val_min=30,
        low_h=(0, 10),
        high_h=(170, 180),
    )
    mask = _apply_antiglare_rescue_mask(
        mask,
        image,
        hsv_image,
        antiglare_min_x_px=antiglare_min_x_px,
        antiglare_max_x_px=antiglare_max_x_px,
        antiglare_min_y_px=antiglare_min_y_px,
        antiglare_max_y_px=antiglare_max_y_px,
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL_3X3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL_3X3, iterations=1)

    center = _select_component_centroid(
        mask,
        puck_history=puck_history,
        center_offset_constant=center_offset_constant,
    )
    if center is None:
        return _fallback_puck(puck_history)

    x, y = center
    robot_x, robot_y = _pixel_to_robot_xy(x, y)
    return _validated_detection(
        robot_x, robot_y, puck_history,
        center_offset_constant=center_offset_constant,
        **_ignored_kwargs,
    )