import time
from collections import Counter
import os

import cv2
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from airhockey.sims.real.robot_control import apply_negative_z_force
import imageio

TEMP_CALIB_DIR = "temp/calibration_collect"
LATEST_POSE_FILE = "temp/calibration_collect/20260218_125147/robot_poses.npz"
# If latest pose record has no stored puck image path, this optional fallback is used.
# Example: "temp/calibration_collect/final_state/puck_capture_raw.png"
DEFAULT_REUSE_IMAGE_PATH = "temp/calibration_collect/20260218_125147/puck_capture_raw.png"


def _ensure_temp_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_robot_pose_record(path, data):
    np.savez(path, **data)


def _load_robot_pose_record(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as record:
        return {key: record[key] for key in record.files}


def _save_puck_capture_artifacts(session_dir, frame_bgr, mask, points_row_col):
    _ensure_temp_dir(session_dir)
    if frame_bgr is None:
        return {}

    raw_path = os.path.join(session_dir, "puck_capture_raw.png")
    cv2.imwrite(raw_path, frame_bgr)

    overlay = frame_bgr.copy()
    if points_row_col is not None:
        _draw_indexed_points(overlay, points_row_col, color=(0, 255, 0))
    overlay_path = os.path.join(session_dir, "puck_capture_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    if mask is not None:
        mask_path = os.path.join(session_dir, "puck_capture_mask.png")
        cv2.imwrite(mask_path, mask)
    else:
        mask_path = ""

    return {
        "raw_path": raw_path,
        "overlay_path": overlay_path,
        "mask_path": mask_path,
    }


def _profile_path_value(profile, key):
    if profile is None or key not in profile:
        return ""
    value = profile[key]
    if np.ndim(value) == 0:
        return str(value.item())
    if len(value) == 0:
        return ""
    return str(np.ravel(value)[0])


def _resolve_reuse_image_path(profile):
    candidate = _profile_path_value(profile, "puck_capture_image_path")
    if candidate and os.path.exists(candidate):
        return candidate
    if DEFAULT_REUSE_IMAGE_PATH and os.path.exists(DEFAULT_REUSE_IMAGE_PATH):
        return DEFAULT_REUSE_IMAGE_PATH
    return ""

def find_robo_pixel(cap, offset):
    pixels = list()
    _ensure_temp_dir("temp/ar_frames")
    for i in range(100):
        ret, image = cap.read()
        imageio.imsave("temp/ar_frames/frame_" + str(i) +".png", image)
        px = find_red_dot(image, offset)
        if px is not None: pixels.append(px)
    return np.mean(np.array(pixels), axis=0)

def find_red_dot(image, offset):
    # Load the image
    # image = cv2.imread(image_path)
    image = cv2.rotate(image, cv2.ROTATE_180)

    # Convert to HSV color space
    image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])), 
                    interpolation = cv2.INTER_LINEAR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:int(300)] = 0
    hsv_image[int(350):,:] = 0

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
    # cv2.imshow('mask',mask)
    # cv2.waitKey(10)
    vals = np.where(mask > 0)
    if len(vals[0]) < 10:
        return None
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))

    # # Draw detected blobs as red circles
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # print(image.shape)
    
    # image_with_keypoints = cv2.drawKeypoints(image, [(x,y)], np.array([]), (0, 0, 255),
    #                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # py = int(keypoints[0].pt[0])
    # px = int(keypoints[0].pt[1])
    # width=100
    # x,y = x + 30, y + 40 # top far
    # x,y = x - 20, y + 40 # bot far
    # x,y = x - 15, y + 10 # bot far
    # x,y = x + 40, y + 12 # bot far
    x, y = x + offset[0], y + offset[1]
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, h - 1))
    y = int(np.clip(y, 0, w - 1))

    # Draw a clear 2x2 marker at the detected robot pixel.
    top = max(0, x - 1)
    left = max(0, y - 1)
    bottom = min(h, top + 2)
    right = min(w, left + 2)
    top = max(0, bottom - 2)
    left = max(0, right - 2)
    image[top:bottom, left:right, :] = (0, 255, 0)

    # Add a thin white outline so the marker is visible on bright backgrounds.
    cv2.rectangle(
        image,
        (max(0, left - 2), max(0, top - 2)),
        (min(w - 1, right + 1), min(h - 1, bottom + 1)),
        (255, 255, 255),
        1,
    )
    cv2.imshow('id-ed',image)
    cv2.waitKey(10)

    return x,y


def find_red_pucks(image, min_area=60.0, max_area=6000.0):
    """Return red blob centroids as (row, col, area) tuples."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 120, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv_image, lower_red1, upper_red1) | cv2.inRange(
        hsv_image, lower_red2, upper_red2
    )

    # Simple cleanup to avoid small speckles and fragmented blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        col = float(moments["m10"] / moments["m00"])
        row = float(moments["m01"] / moments["m00"])
        blobs.append((row, col, area))

    centroids = _filter_size_consistent_blobs(blobs, target_count=4)

    return centroids, mask


def _filter_size_consistent_blobs(blobs, target_count=4):
    """
    Keep blobs whose areas are dynamically consistent with the dominant puck size.
    Blobs with significantly different sizes are removed automatically.
    """
    if len(blobs) <= 1:
        return blobs

    areas = np.array([blob[2] for blob in blobs], dtype=np.float32)
    median_area = float(np.median(areas))
    if median_area <= 1e-6:
        return blobs

    rel_dev = np.abs(areas - median_area) / median_area
    mad_rel = float(np.median(rel_dev))

    # Dynamic tolerance: tighter when blobs are consistent, looser when noisy.
    allowed_rel_dev = float(np.clip(2.5 * mad_rel + 0.18, 0.12, 0.45))

    keep_indices = np.where(rel_dev <= allowed_rel_dev)[0]

    # If strict filtering removes too many, keep the closest-to-median candidates.
    if len(keep_indices) < target_count and len(blobs) >= target_count:
        keep_indices = np.argsort(rel_dev)[:target_count]

    # If many blobs remain size-consistent, keep the most area-consistent set.
    if len(keep_indices) > target_count:
        ordered = keep_indices[np.argsort(rel_dev[keep_indices])]
        keep_indices = ordered[:target_count]

    return [blobs[int(i)] for i in keep_indices]


def _ordered_indices(points_xy):
    """Return clockwise ordering with the first point near top-left."""
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if len(points_xy) == 0:
        return np.array([], dtype=np.int32)

    center = np.mean(points_xy, axis=0)
    angles = np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0])
    angle_order = np.argsort(angles)
    ordered = points_xy[angle_order]
    top_left_idx = int(np.argmin(np.sum(ordered, axis=1)))
    return np.roll(angle_order, -top_left_idx)


def _order_row_col_points(points_row_col):
    points_row_col = np.asarray(points_row_col, dtype=np.float32)
    points_xy = np.stack((points_row_col[:, 1], points_row_col[:, 0]), axis=1)
    return points_row_col[_ordered_indices(points_xy)]


def _draw_indexed_points(image, points_row_col, color=(0, 255, 0)):
    for idx, (row, col) in enumerate(points_row_col):
        row_i = int(np.round(row))
        col_i = int(np.round(col))
        cv2.circle(image, (col_i, row_i), 8, color, 2)
        cv2.putText(
            image,
            str(idx),
            (col_i + 6, row_i - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def detect_four_red_pucks(cap, sample_frames=60, min_valid_frames=10):
    """Detect and average 4 red pucks over multiple frames."""
    valid_points = []
    count_history = []
    last_frame = None
    last_mask = None

    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        last_frame = frame

        centroids, mask = find_red_pucks(frame)
        last_mask = mask
        count_history.append(len(centroids))

        preview = frame.copy()
        points_rc = np.array([[row, col] for row, col, _ in centroids], dtype=np.float32)
        if len(points_rc) > 0:
            _draw_indexed_points(preview, points_rc, color=(0, 255, 255))

        if len(points_rc) == 4:
            ordered = _order_row_col_points(points_rc)
            valid_points.append(ordered)
            accepted_preview = frame.copy()
            _draw_indexed_points(accepted_preview, ordered, color=(0, 255, 0))
            cv2.imshow("puck-detect", accepted_preview)
        else:
            cv2.imshow("puck-detect", preview)
        cv2.imshow("puck-mask", mask)
        cv2.waitKey(1)

    dominant_count = Counter(count_history).most_common(1)[0][0] if count_history else 0
    if len(valid_points) < min_valid_frames:
        return None, dominant_count, last_frame, last_mask

    averaged = np.mean(np.stack(valid_points, axis=0), axis=0)
    averaged = _order_row_col_points(averaged)
    return averaged, 4, last_frame, last_mask

def calibrate_homography(camera_id, save_homographies):
    rtde_frequency = 500.0
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")
    # moves the robot to fixed positions, then aligns the homography pixels so that they match those of the robot
    cap = cv2.VideoCapture(camera_id)

    ret, image = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read from camera_id={camera_id}")
    image = cv2.rotate(image, cv2.ROTATE_180)
    upscale_constant = 3
    visual_downscale_constant = 2
    image = cv2.resize(
        image,
        (int(640 * upscale_constant), int(480 * upscale_constant)),
        interpolation=cv2.INTER_LINEAR,
    )
    # cv2.imshow("image", image)
    # cv2.waitKey(1)

    original_size = np.array([640, 480])
    offset_constants = np.array((2250, 500), dtype=np.float32)
    robot_points_mm = np.float32([[-820, 330], [-820, -330], [-475, -330], [-475, 330]])
    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(TEMP_CALIB_DIR, session_stamp)
    _ensure_temp_dir(session_dir)

    vel = 0.3  # velocity limit
    acc = 0.3  # acceleration limit
    angle = [-0.00153677648744038, -3.0647520618606172, 0.0]

    reuse_saved_calibration = False
    saved_points_row_col = None
    loaded_profile = _load_robot_pose_record(LATEST_POSE_FILE)
    reuse_image_path = _resolve_reuse_image_path(loaded_profile)
    if loaded_profile is not None:
        use_saved = input(
            "Found saved robot pose record. Type 'reuse' to start from saved calibration poses, or press Enter for rollout default: "
        ).strip().lower()
        if use_saved in {"reuse", "r", "yes", "y"}:
            if "rollout_start_pose" in loaded_profile and loaded_profile["rollout_start_pose"].shape[0] >= 6:
                rollout_start_pose = loaded_profile["rollout_start_pose"].astype(float).tolist()
            else:
                rollout_start_pose = [-0.68, 0.0, 0.33] + angle

            if "robot_points_mm" in loaded_profile and loaded_profile["robot_points_mm"].shape == (4, 2):
                robot_points_mm = loaded_profile["robot_points_mm"].astype(np.float32)
                print("Using saved robot_points_mm from latest robot pose record.")
            else:
                print("Saved profile has no valid robot_points_mm; using script defaults.")

            if (
                "detected_puck_points_row_col" in loaded_profile
                and loaded_profile["detected_puck_points_row_col"].shape == (4, 2)
            ):
                saved_points_row_col = loaded_profile["detected_puck_points_row_col"].astype(np.float32)
                reuse_saved_calibration = True
                print("Using saved detected_puck_points_row_col for final calibration.")
            elif (
                "detected_puck_points_xy" in loaded_profile
                and loaded_profile["detected_puck_points_xy"].shape == (4, 2)
            ):
                saved_xy = loaded_profile["detected_puck_points_xy"].astype(np.float32)
                saved_points_row_col = saved_xy[:, [1, 0]]
                reuse_saved_calibration = True
                print("Using saved detected_puck_points_xy for final calibration.")
            else:
                print("Reuse not possible: no saved 4-point puck detections found in latest_robot_poses.npz.")
                print("Falling back to fresh calibration data collection.")

            print("Using saved rollout_start_pose from latest robot pose record.")
        else:
            if use_saved not in {"", "default", "fresh", "new"}:
                print(
                    f"Reuse not enabled: unrecognized response '{use_saved}'. "
                    "Type 'reuse' to reuse saved state, otherwise press Enter for fresh collection."
                )
            else:
                print("Reuse not requested; running fresh calibration data collection.")
            rollout_start_pose = [-0.68, 0.0, 0.33] + angle
    else:
        print(
            "Reuse not possible: latest robot pose record not found at "
            f"{LATEST_POSE_FILE}. Running fresh calibration data collection."
        )
        rollout_start_pose = [-0.68, 0.0, 0.33] + angle

    # Match rollout default reset pose (AirHockeyReal reset_pos_setting="hitting").
    print("Moving to rollout initial pose before calibration...")
    start_success = ctrl.moveL(rollout_start_pose, vel, acc, False)
    print("move_to_rollout_initial_success:", start_success)

    def wait_for_recorded_pose(target_pose, timeout_s=6.0, pos_tol_m=0.004, rot_tol_rad=0.06):
        deadline = time.time() + timeout_s
        last_pose = None
        while time.time() < deadline:
            pose = rcv.getActualTCPPose()
            if pose is not None and len(pose) >= 6:
                last_pose = list(pose[:6])
                pos_err = float(
                    np.linalg.norm(np.array(last_pose[:3], dtype=np.float32) - np.array(target_pose[:3], dtype=np.float32))
                )
                rot_err = float(
                    np.linalg.norm(np.array(last_pose[3:6], dtype=np.float32) - np.array(target_pose[3:6], dtype=np.float32))
                )
                if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
                    return last_pose, True
            time.sleep(0.05)
        return last_pose, False

    initial_pose, pose_recorded = wait_for_recorded_pose(rollout_start_pose)
    if initial_pose is None:
        initial_pose = list(rollout_start_pose)
        print("Warning: TCP pose not recorded from receiver; using rollout target pose for return.")
    elif not pose_recorded:
        print("Warning: timed out waiting for rollout pose settle; using latest recorded TCP pose.")
    print("initial_pose_for_return:", initial_pose)
    mark_pose_targets = []
    mark_pose_actual = []

    def return_to_initial(tag):
        print(f"Returning robot to initial pose ({tag})...")
        success = ctrl.moveL(initial_pose, vel, acc, False)
        print(f"{tag}_return_to_initial_success:", success)
        time.sleep(1.0)
        return success

    def persist_pose_record(extra_data=None, update_latest=False):
        pose_data = {
            "timestamp_epoch_s": np.array([time.time()], dtype=np.float64),
            "rollout_start_pose": np.array(rollout_start_pose, dtype=np.float32),
            "initial_pose_for_return": np.array(initial_pose, dtype=np.float32),
            "robot_points_mm": np.array(robot_points_mm, dtype=np.float32),
            "target_mark_poses": np.array(mark_pose_targets, dtype=np.float32),
            "actual_mark_poses": np.array(mark_pose_actual, dtype=np.float32),
        }
        if extra_data is not None:
            for key, val in extra_data.items():
                pose_data[key] = val

        session_pose_file = os.path.join(session_dir, "robot_poses.npz")
        _save_robot_pose_record(session_pose_file, pose_data)
        if update_latest:
            _save_robot_pose_record(LATEST_POSE_FILE, pose_data)
        return session_pose_file

    accepted_points = None
    accepted_frame = None
    detect_mask = None

    if reuse_saved_calibration and saved_points_row_col is not None:
        print("Reuse mode enabled: skipping robot mark collection and puck re-detection.")
        accepted_points = np.array(saved_points_row_col, dtype=np.float32)
    else:
        apply_negative_z_force(ctrl)
        print("Moving robot through 4 calibration positions...")
        for idx, robo_pt in enumerate(robot_points_mm):
            mark_pose = [robo_pt[0] * 0.001, robo_pt[1] * 0.001, 0.33] + angle
            mark_pose_targets.append(mark_pose)
            move_success = ctrl.moveL(mark_pose, vel, acc, False)
            print(f"[{idx}] moved to robot point {robo_pt.tolist()} success={move_success}")
            time.sleep(15.0)
            pose_now = rcv.getActualTCPPose()
            if pose_now is not None and len(pose_now) >= 6:
                mark_pose_actual.append(list(pose_now[:6]))
            else:
                mark_pose_actual.append([np.nan] * 6)
        persist_pose_record(update_latest=False)

        return_to_initial("post_marking")
        time.sleep(1.0)

        while True:
            ready = input(
                "Place 4 red pucks at the marked positions, then type 'done' and press Enter (or 'q' to quit): "
            ).strip().lower()
            if ready in {"done", "d", ""}:
                break
            if ready in {"q", "quit", "exit"}:
                print("Calibration canceled by user.")
                pose_file = persist_pose_record({"aborted": np.array([1], dtype=np.int32)}, update_latest=False)
                print(f"Saved robot pose record: {pose_file}")
                return_to_initial("cancel")
                cap.release()
                cv2.destroyAllWindows()
                return

        while True:
            print("Detecting 4 red pucks...")
            detected_points, detected_count, detect_frame, detect_mask = detect_four_red_pucks(cap)
            if detect_frame is not None:
                preview = detect_frame.copy()
                if detected_points is not None:
                    _draw_indexed_points(preview, detected_points, color=(0, 255, 0))
                cv2.imshow("puck-detect-final", preview)
                if detect_mask is not None:
                    cv2.imshow("puck-mask-final", detect_mask)
                cv2.waitKey(1)

            if detected_points is None:
                retry = input(
                    f"Detected {detected_count} red blobs (need exactly 4). Adjust and press Enter to retry, or type 'q' to quit: "
                ).strip().lower()
                if retry in {"q", "quit", "exit"}:
                    print("Calibration canceled by user.")
                    pose_file = persist_pose_record({"aborted": np.array([1], dtype=np.int32)}, update_latest=False)
                    print(f"Saved robot pose record: {pose_file}")
                    return_to_initial("cancel")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                continue

            confirm = input(
                "Detected 4 pucks. Type 'done' to accept these positions, or press Enter to recapture: "
            ).strip().lower()
            if confirm in {"done", "d"}:
                accepted_points = detected_points
                accepted_frame = detect_frame
                break

    print("Detected puck points (row, col):", accepted_points.tolist())
    capture_artifacts = _save_puck_capture_artifacts(session_dir, accepted_frame, detect_mask, accepted_points)

    robot_reference_xy = robot_points_mm + offset_constants
    robot_order = _ordered_indices(robot_reference_xy)
    robot_points_ordered = robot_points_mm[robot_order]
    robot_reference_ordered = robot_reference_xy[robot_order]

    # Convert detector output from (row, col) to OpenCV point order (x, y).
    accepted_points_row_col = np.float32(accepted_points)
    accepted_points_xy = accepted_points_row_col[:, [1, 0]]

    # add some optional tuning (this is just magic numbers to get the calibration to work)
    tuning_offsets = np.float32([[-1, -1], [0, 0], [0, 1], [-1, 1]])
    accepted_points_xy = accepted_points_xy + tuning_offsets
    print("Using puck points for calibration as (x, y):", accepted_points_xy.tolist())

    

    # final calibration
    pts1 = accepted_points_xy
    pts1 *= upscale_constant
    Mrob = cv2.getPerspectiveTransform(pts1, robot_points_ordered)

    print("Final correspondences used for calibration:")
    for idx, (pixel_pt, robo_pt) in enumerate(zip(accepted_points_xy, robot_points_ordered)):
        print(f"  idx={idx} pixel(x,y)={pixel_pt.tolist()} -> robot(mm)={robo_pt.tolist()}")

    # Colors for each point, in order:
    # 0: Green (0,255,0)
    # 1: Red (0,0,255)
    # 2: Blue (255,0,0)
    # 3: Yellow (0,255,255)
    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255)]
    if accepted_frame is not None:
        image = accepted_frame
    else:
        if reuse_image_path:
            loaded_image = cv2.imread(reuse_image_path)
            if loaded_image is not None:
                image = loaded_image
                print(f"Using reuse image from: {reuse_image_path}")
            else:
                print(f"Warning: failed to load reuse image at {reuse_image_path}. Using startup frame.")
        else:
            print(
                "No stored reuse image found in latest pose profile and no fallback image set; "
                "using startup frame for transformed preview."
            )
    image = cv2.resize(
        image,
        (int(640 * upscale_constant), int(480 * upscale_constant)),
        interpolation=cv2.INTER_LINEAR,
    )
    for idx, val in enumerate(pts1.astype(np.int32)):
        color = colors[idx % len(colors)]
        cv2.circle(image, (int(val[0]), int(val[1])), 5, color, -1)
    cv2.imshow("image", image)
    cv2.waitKey(5000)

    Mimg = cv2.getPerspectiveTransform(pts1, robot_reference_ordered)

    dst = cv2.warpPerspective(image, Mimg, original_size * upscale_constant)

    image_preview = cv2.resize(
        image,
        (
            int(640 * upscale_constant / visual_downscale_constant),
            int(480 * upscale_constant / visual_downscale_constant),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    dst_preview = cv2.resize(
        dst,
        (
            int(640 * upscale_constant / visual_downscale_constant),
            int(480 * upscale_constant / visual_downscale_constant),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imshow("image", image_preview)
    cv2.imshow("transformed", dst_preview)
    cv2.waitKey(5000)
    input("Transformed view shown. Press Enter to finish calibration and exit... ")

    # Save calibration data
    if save_homographies:
        np.save("Mimg.npy", Mimg)
        np.save("Mrob.npy", Mrob)

    pose_file = persist_pose_record(
        {
            "aborted": np.array([0], dtype=np.int32),
            "detected_puck_points_row_col": np.array(accepted_points_row_col, dtype=np.float32),
            "detected_puck_points_xy": np.array(accepted_points_xy, dtype=np.float32),
            "robot_points_ordered_mm": np.array(robot_points_ordered, dtype=np.float32),
            "puck_capture_image_path": np.array(
                capture_artifacts.get("raw_path", reuse_image_path) if capture_artifacts else reuse_image_path,
                dtype=np.str_,
            ),
        },
        update_latest=True,
    )
    np.save(os.path.join(session_dir, "detected_puck_points_row_col.npy"), np.array(accepted_points_row_col, dtype=np.float32))
    np.save(os.path.join(session_dir, "detected_puck_points_xy.npy"), np.array(accepted_points_xy, dtype=np.float32))
    print(f"Saved puck capture artifacts in: {session_dir}")
    print(f"Saved robot pose record: {pose_file}")

    # End at startup pose so post-calibration setup is convenient.
    return_to_initial("final")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # read in argument whether to save the homography
    import sys

    save_homographies = False
    if "--save-homographies" in sys.argv or "-s" in sys.argv:
        save_homographies = True
    calibrate_homography(1, save_homographies)