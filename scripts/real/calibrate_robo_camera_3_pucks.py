"""Robot-camera homography calibration with only 3 red pucks.

Same procedure as ``calibrate_robo_camera.py``, but the 4 puck detections are
collected in two passes:

  1. The robot marks robot points 0, 1, 2. You place 3 red pucks there and the
     script records their image positions.
  2. You remove the 3 pucks. The robot marks robot point 3. You place a single
     red puck there and the script records its image position.

The 4 image points are then merged and Mimg / Mrob are computed exactly as in
``calibrate_robo_camera.py`` (same ordering, tuning offsets, and scaling).

    python scripts/real/calibrate_robo_camera_3_puck.py
    python scripts/real/calibrate_robo_camera_3_puck.py --save-homographies
    python scripts/real/calibrate_robo_camera_3_puck.py --camera-index 1
    python scripts/real/calibrate_robo_camera_3_puck.py --start-from-third

``--start-from-third`` is for when positions 0 and 1 are already marked on the
table (e.g. a previous run where the position-2 mark was lost). The robot does
not revisit 0 and 1: you place 2 pucks on those marks, their image positions
are saved to ``temp/calibration_collect/first_two_pucks.npz`` (reused on the
next ``--start-from-third`` run if you ask), then the robot marks positions 2
and 3 and you place 2 pucks there.

Without ``--camera-index`` the first camera that returns a frame is used.
"""

import glob
import os
import re
import sys
import termios
import time
import tty
from collections import Counter

import cv2
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from airhockey.sims.real.robot_control import apply_negative_z_force

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_robo_camera import (  # noqa: E402
    LATEST_POSE_FILE,
    TEMP_CALIB_DIR,
    _draw_indexed_points,
    _ensure_temp_dir,
    _load_robot_pose_record,
    _order_row_col_points,
    _ordered_indices,
    _save_puck_capture_artifacts,
    _save_robot_pose_record,
    find_red_pucks,
)

# Robot point indices marked in each pass (indices into robot_points_mm).
FIRST_PASS_INDICES = (0, 1, 2)
SECOND_PASS_INDICES = (3,)
# Minimum pixel distance between the single puck and the 3 earlier pucks; closer
# than this almost certainly means the same spot was detected twice.
MIN_POINT_SEPARATION_PX = 20.0
# --start-from-third: image positions of the pucks on marks 0 and 1, saved so a
# later run can skip re-detecting them.
FIRST_TWO_INDICES = (0, 1)
LAST_TWO_INDICES = (2, 3)
FIRST_TWO_FILE = os.path.join(TEMP_CALIB_DIR, "first_two_pucks.npz")


def banner(text):
    """Print a step header so it stands out from the robot / detection logs."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def wait_for_space(prompt):
    """Block until SPACE is pressed in the terminal (no Enter needed)."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        input()
        print()
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while sys.stdin.read(1) != " ":
            pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print()


def open_camera(camera_id=None):
    """Open ``camera_id``, or the first available camera that returns a frame."""
    if camera_id is not None:
        candidates = [camera_id]
    else:
        # /dev/videoN on Linux; the numbers are not contiguous when a camera is unplugged.
        candidates = sorted(
            int(m.group(1))
            for m in (re.match(r"/dev/video(\d+)$", path) for path in glob.glob("/dev/video*"))
            if m
        ) or list(range(10))

    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Using camera index {idx}")
                return cap, idx
        cap.release()
    raise RuntimeError(f"No working camera found (tried indices {candidates})")


def detect_n_red_pucks(cap, n_pucks, sample_frames=60, min_valid_frames=10):
    """Detect and average exactly ``n_pucks`` red pucks over multiple frames."""
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

        centroids, mask = find_red_pucks(frame, target_count=n_pucks)
        last_mask = mask
        count_history.append(len(centroids))

        preview = frame.copy()
        points_rc = np.array([[row, col] for row, col, _ in centroids], dtype=np.float32)
        if len(points_rc) > 0:
            _draw_indexed_points(preview, points_rc, color=(0, 255, 255))

        if len(points_rc) == n_pucks:
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
    return averaged, n_pucks, last_frame, last_mask


def calibrate_homography(camera_id, save_homographies, start_from_third=False):
    # Open the camera before connecting to the robot so a missing camera fails fast.
    cap, camera_id = open_camera(camera_id)
    rtde_frequency = 500.0
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")

    ret, image = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read from camera_id={camera_id}")
    image = cv2.rotate(image, cv2.ROTATE_180)

    upscale_constant = 3
    visual_downscale_constant = 2
    original_size = np.array([640, 480])
    offset_constants = np.array((2250, 500), dtype=np.float32)
    #                               pos 0 (X,Y)        pos 1         pos 2        pos 3
    # robot_points_mm = np.float32([[-820, 330], [-820, -330], [-475, -330], [-475, 330]])  # originalk
    robot_points_mm = np.float32([[-820, 330], [-820, -330], [-600, -330], [-600, 330]])    # shifted up toward the top of the table because I think it's going to hit the edge

    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(TEMP_CALIB_DIR, session_stamp + "_3puck")
    _ensure_temp_dir(session_dir)

    vel = 0.3  # velocity limit
    acc = 0.3  # acceleration limit
    angle = [-0.00153677648744038, -3.0647520618606172, 0.0]
    rollout_start_pose = [-0.68, 0.0, 0.33] + angle

    # Match rollout default reset pose (AirHockeyReal reset_pos_setting="hitting").
    banner(
        "3-PUCK CALIBRATION\n"
        "  Pass 1: robot visits positions 0, 1, 2 -> you mark them, then place 3 pucks.\n"
        "  Pass 2: remove the pucks, robot visits position 3 -> you mark it, then place 1 puck.\n"
        "  Clear the table of red objects before starting."
    )
    print(f"Moving robot to the initial pose {rollout_start_pose[:3]} (center of its workspace)...")
    start_success = ctrl.moveL(rollout_start_pose, vel, acc, False)
    print("move_to_rollout_initial_success:", start_success)
    if not start_success:
        print("WARNING: move to initial pose failed. Check the teach pendant for a protective stop.")

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

    # Indexed by robot point index so the record stays in robot_points_mm order.
    mark_pose_targets = [[np.nan] * 6 for _ in range(len(robot_points_mm))]
    mark_pose_actual = [[np.nan] * 6 for _ in range(len(robot_points_mm))]

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

    def cancel():
        print("Calibration canceled by user.")
        pose_file = persist_pose_record({"aborted": np.array([1], dtype=np.int32)}, update_latest=False)
        print(f"Saved robot pose record: {pose_file}")
        return_to_initial("cancel")
        cap.release()
        cv2.destroyAllWindows()

    def mark_robot_points(indices):
        apply_negative_z_force(ctrl)
        print(f"Robot will visit calibration position(s) {list(indices)} one at a time.")
        for step, idx in enumerate(indices, start=1):
            robo_pt = robot_points_mm[idx]
            mark_pose = [robo_pt[0] * 0.001, robo_pt[1] * 0.001, 0.33] + angle
            mark_pose_targets[idx] = mark_pose
            print(f"\n[{step}/{len(indices)}] Moving robot to position {idx} at robot (x, y) = {robo_pt.tolist()} mm...")
            move_success = ctrl.moveL(mark_pose, vel, acc, False)
            print(f"[{step}/{len(indices)}] Arrived at position {idx}: success={move_success}")
            if not move_success:
                print("WARNING: the move failed, so the robot is NOT at this position. Do not mark it.")

            wait_for_space(f"Mark the spot under the paddle for position {idx}, then press SPACE to continue... ")

            pose_now = rcv.getActualTCPPose()
            if pose_now is not None and len(pose_now) >= 6:
                mark_pose_actual[idx] = list(pose_now[:6])
        persist_pose_record(update_latest=False)
        print(f"\nDone marking position(s) {list(indices)}. Moving robot back to the initial pose to clear the table.")
        return_to_initial(f"post_marking_{'_'.join(str(i) for i in indices)}")
        time.sleep(1.0)

    def wait_for_placement(n_pucks):
        """Returns False if the user quits."""
        while True:
            ready = input(
                f"Place {n_pucks} red puck(s) at the marked position(s) (and nothing else red on the table), "
                "then type 'done' and press Enter (or 'q' to quit): "
            ).strip().lower()
            if ready in {"done", "d", ""}:
                return True
            if ready in {"q", "quit", "exit"}:
                return False

    def collect_pucks(n_pucks, window_suffix):
        """Detect n_pucks until the user accepts. Returns (points_row_col, frame, mask) or None on quit."""
        while True:
            print(f"Detecting {n_pucks} red puck(s)...")
            detected_points, detected_count, detect_frame, detect_mask = detect_n_red_pucks(cap, n_pucks)
            if detect_frame is not None:
                preview = detect_frame.copy()
                if detected_points is not None:
                    _draw_indexed_points(preview, detected_points, color=(0, 255, 0))
                cv2.imshow(f"puck-detect-{window_suffix}", preview)
                if detect_mask is not None:
                    cv2.imshow(f"puck-mask-{window_suffix}", detect_mask)
                cv2.waitKey(1)

            if detected_points is None:
                retry = input(
                    f"Detected {detected_count} red blobs (need exactly {n_pucks}). "
                    "Adjust and press Enter to retry, or type 'q' to quit: "
                ).strip().lower()
                if retry in {"q", "quit", "exit"}:
                    return None
                continue

            confirm = input(
                f"Detected {n_pucks} puck(s). Type 'done' to accept these positions, or press Enter to recapture: "
            ).strip().lower()
            if confirm in {"done", "d"}:
                return detected_points, detect_frame, detect_mask

    def wait_for_removal(n_pucks):
        """Returns False if the user quits."""
        while True:
            ready = input(
                f"Remove all {n_pucks} pucks from the table, then type 'done' and press Enter (or 'q' to quit): "
            ).strip().lower()
            if ready in {"done", "d", ""}:
                return True
            if ready in {"q", "quit", "exit"}:
                return False

    def collect_pass(indices, window_suffix, previous_points=None):
        """Place + detect pucks on the marks for ``indices``, rejecting detections that
        overlap ``previous_points``. Returns (points_row_col, frame, mask) or None on quit."""
        while True:
            if not wait_for_placement(len(indices)):
                return None
            result = collect_pucks(len(indices), window_suffix)
            if result is None:
                return None
            if previous_points is None:
                return result
            min_dist = float(
                np.min(np.linalg.norm(previous_points[:, None, :] - result[0][None, :, :], axis=-1))
            )
            if min_dist >= MIN_POINT_SEPARATION_PX:
                return result
            print(
                f"A detected puck is only {min_dist:.1f} px from a pass-1 puck; it looks like the same spot. "
                "Check the placement and try again."
            )

    def load_or_detect_first_two():
        """--start-from-third pass 1: saved or freshly detected pucks on marks 0 and 1."""
        saved = _load_robot_pose_record(FIRST_TWO_FILE)
        if saved is not None:
            saved_robot_pts = saved["robot_points_mm"]
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(saved["timestamp_epoch_s"][0])))
            print(f"Found saved positions of the pucks on marks {list(FIRST_TWO_INDICES)} ({saved_time}, {FIRST_TWO_FILE}):")
            print("  image points (row, col):", saved["puck_points_row_col"].tolist())
            print("  robot points (mm):      ", saved_robot_pts.tolist())
            if not np.allclose(saved_robot_pts, robot_points_mm[list(FIRST_TWO_INDICES)]):
                print(
                    "WARNING: these were saved for different robot points than the current robot_points_mm "
                    f"{robot_points_mm[list(FIRST_TWO_INDICES)].tolist()}; re-detect instead of reusing."
                )
            answer = input("Type 'reuse' to use them, or press Enter to re-detect the 2 pucks: ").strip().lower()
            if answer in {"reuse", "r", "yes", "y"}:
                frame = cv2.imread(str(saved["frame_path"]))
                if frame is None:
                    print(f"Note: saved frame {saved['frame_path']} not found; the pass-2 frame will be used for previews.")
                print("Reusing the saved positions for marks 0 and 1.")
                return saved["puck_points_row_col"].astype(np.float32), frame, None

        banner(
            f"PASS 1 (--start-from-third): detect the pucks on marks {list(FIRST_TWO_INDICES)}\n"
            f"  The robot stays at the initial pose; it does NOT revisit positions {list(FIRST_TWO_INDICES)}.\n"
            f"  Place 2 red pucks on your existing marks for positions {list(FIRST_TWO_INDICES)}:\n"
            f"    position 0 = robot (x, y) {robot_points_mm[0].tolist()} mm\n"
            f"    position 1 = robot (x, y) {robot_points_mm[1].tolist()} mm"
        )
        result = collect_pass(FIRST_TWO_INDICES, "2pucks_first")
        if result is None:
            return None
        points, frame, mask = result
        pass_dir = os.path.join(session_dir, "pass1_2pucks")
        artifacts = _save_puck_capture_artifacts(pass_dir, frame, mask, points)
        _save_robot_pose_record(
            FIRST_TWO_FILE,
            {
                "timestamp_epoch_s": np.array([time.time()], dtype=np.float64),
                "puck_points_row_col": np.array(points, dtype=np.float32),
                "robot_points_mm": np.array(robot_points_mm[list(FIRST_TWO_INDICES)], dtype=np.float32),
                "frame_path": np.array(artifacts.get("raw_path", ""), dtype=np.str_),
            },
        )
        print(f"Saved the positions of the pucks on marks {list(FIRST_TWO_INDICES)} to {FIRST_TWO_FILE}")
        return points, frame, mask

    if start_from_third:
        first_indices, second_indices = FIRST_TWO_INDICES, LAST_TWO_INDICES
        result = load_or_detect_first_two()
        if result is None:
            cancel()
            return
        first_points, first_frame, first_mask = result
    else:
        # ---- Pass 1: robot marks 3 positions, detect 3 pucks ----
        first_indices, second_indices = FIRST_PASS_INDICES, SECOND_PASS_INDICES
        banner(f"PASS 1: marking positions {list(first_indices)}")
        mark_robot_points(first_indices)
        banner(
            f"PASS 1: place the pucks\n"
            f"  Since we only have 3 pucks, the robot is now back at the initial pose.\n"
            f"  Place the 3 red pucks on the marks for positions {list(first_indices)}."
        )
        result = collect_pass(first_indices, "3pucks")
        if result is None:
            cancel()
            return
        first_points, first_frame, first_mask = result
        _save_puck_capture_artifacts(os.path.join(session_dir, "pass1_3pucks"), first_frame, first_mask, first_points)
    print("Pass 1 puck points (row, col):", first_points.tolist())
    print(f"Pass 1 complete: {len(first_indices)} puck positions recorded.")

    # ---- Pass 2: remove pucks, robot marks the remaining position(s), detect them ----
    n_second = len(second_indices)
    banner(
        f"PASS 2: marking position(s) {list(second_indices)}\n"
        f"  The robot needs a clear table, so remove the {len(first_indices)} pucks first."
    )
    if not wait_for_removal(len(first_indices)):
        cancel()
        return

    mark_robot_points(second_indices)
    banner(
        f"PASS 2: place the puck(s)\n"
        f"  The robot is back at the initial pose.\n"
        f"  Place {n_second} red puck(s) on the mark(s) for position(s) {list(second_indices)} "
        f"(leave the marks for {list(first_indices)} empty)."
    )
    result = collect_pass(second_indices, f"{n_second}puck", previous_points=first_points)
    if result is None:
        cancel()
        return
    second_points, second_frame, second_mask = result
    print("Pass 2 puck points (row, col):", second_points.tolist())
    banner("All 4 puck positions recorded. Computing the homography...")
    _save_puck_capture_artifacts(
        os.path.join(session_dir, f"pass2_{n_second}puck"), second_frame, second_mask, second_points
    )
    if first_frame is None:
        first_frame = second_frame

    # ---- Merge into the 4 points and continue exactly as calibrate_robo_camera.py ----
    accepted_points = _order_row_col_points(np.concatenate([first_points, second_points], axis=0))
    accepted_frame = first_frame
    print("Detected puck points (row, col):", accepted_points.tolist())

    # Combined overlay: pass-1 frame with all 4 merged points drawn on it.
    combined_overlay = accepted_frame.copy()
    _draw_indexed_points(combined_overlay, accepted_points, color=(0, 255, 0))
    combined_overlay_path = os.path.join(session_dir, "puck_capture_combined_overlay.png")
    cv2.imwrite(combined_overlay_path, combined_overlay)
    capture_artifacts = _save_puck_capture_artifacts(session_dir, accepted_frame, None, accepted_points)

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
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    image = cv2.resize(
        accepted_frame,
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
        print("Saving homographies to Mimg.npy and Mrob.npy")
        np.save("Mimg.npy", Mimg)
        np.save("Mrob.npy", Mrob)

    pose_file = persist_pose_record(
        {
            "aborted": np.array([0], dtype=np.int32),
            "detected_puck_points_row_col": np.array(accepted_points_row_col, dtype=np.float32),
            "detected_puck_points_xy": np.array(accepted_points_xy, dtype=np.float32),
            "robot_points_ordered_mm": np.array(robot_points_ordered, dtype=np.float32),
            "pass1_puck_points_row_col": np.array(first_points, dtype=np.float32),
            "pass2_puck_points_row_col": np.array(second_points, dtype=np.float32),
            "puck_capture_image_path": np.array(capture_artifacts.get("raw_path", ""), dtype=np.str_),
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
    save_homographies = "--save-homographies" in sys.argv or "-s" in sys.argv

    # --camera-index N forces a specific camera; otherwise the first working one is used.
    camera_id = None
    if "--camera-index" in sys.argv:
        camera_idx = sys.argv.index("--camera-index") + 1
        if camera_idx >= len(sys.argv):
            raise SystemExit("--camera-index requires an integer camera id.")
        camera_id = int(sys.argv[camera_idx])

    # --start-from-third: marks 0 and 1 already exist; detect them, then mark 2 and 3.
    start_from_third = "--start-from-third" in sys.argv

    calibrate_homography(camera_id, save_homographies, start_from_third=start_from_third)
