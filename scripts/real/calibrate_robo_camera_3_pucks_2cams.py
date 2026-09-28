"""Two-camera robot-camera homography calibration with only 3 red pucks.

Same robot / puck procedure as ``calibrate_robo_camera_3_pucks.py``, but every
puck detection is made in BOTH cameras at once, so each camera gets its own
homography into the shared robot (table-plane) frame:

  1. The robot marks robot points 0, 1, 2. You place 3 red pucks there and both
     cameras record their image positions.
  2. You remove the 3 pucks. The robot marks robot point 3. You place a single
     red puck there and both cameras record its image position.

Per camera k, Mimg_cam{k} / Mrob_cam{k} are computed exactly as in
``calibrate_robo_camera.py`` (same ordering, tuning offsets, and scaling).
Because both homographies land in the same robot frame, the script also gives

  * H_cam1_to_cam0 = inv(Mrob_cam0) @ Mrob_cam1: camera-1 pixel -> camera-0 pixel
    (both in the x3 upscaled pixel frame, like Mrob), valid for points on the table;
  * a fused top-down view (both cameras warped with their Mimg and blended);
  * a live check: place a puck anywhere and both cameras report its robot (x, y);
    the difference between them is the calibration error off the 4 marks.

Both cameras must see all 4 calibration marks.

    python scripts/real/calibrate_robo_camera_3_pucks_2cams.py
    python scripts/real/calibrate_robo_camera_3_pucks_2cams.py --camera-indices 0 2 -s
    python scripts/real/calibrate_robo_camera_3_pucks_2cams.py --start-from-third
    python scripts/real/calibrate_robo_camera_3_pucks_2cams.py --rotations 180,0
    python scripts/real/calibrate_robo_camera_3_pucks_2cams.py --last-two-placed -s

``--camera-indices A B``  cameras to use as cam0 / cam1 (default: the first two
                          /dev/video* devices that return a frame).
``--rotations R0,R1``     rotation applied to each camera's frames, 0 or 180
                          (default 180,180, matching the single-camera scripts).
``--start-from-third``    marks 0 and 1 already exist on the table: the robot does
                          not revisit them, both cameras detect 2 pucks there (saved
                          to ``temp/calibration_collect/first_two_pucks_2cams.npz``
                          and reusable on the next run), then the robot marks 2 and 3.
``--last-two-placed``     2 pucks are already sitting on positions 2 and 3: both cameras
                          record them (saved to ``temp/calibration_collect/last_two_pucks_2cams.npz``
                          and reusable on the next run), you remove them, the robot
                          visits position 0 (SPACE), then position 1 while you put a
                          puck where it was at 0 (SPACE), then returns to the initial
                          pose while you put a puck on 1; both cameras record 0 and 1.
``-s / --save-homographies``  write Mimg_cam{0,1}.npy, Mrob_cam{0,1}.npy and
                          H_cam1_to_cam0.npy to the current directory. The existing
                          single-camera Mimg.npy / Mrob.npy are not touched.
"""

import glob
import os
import re
import sys
import time
from collections import Counter

import cv2
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from airhockey.sims.real.robot_control import apply_negative_z_force

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_robo_camera import (  # noqa: E402
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
from calibrate_robo_camera_3_pucks import (  # noqa: E402
    FIRST_PASS_INDICES,
    FIRST_TWO_INDICES,
    LAST_TWO_INDICES,
    MIN_POINT_SEPARATION_PX,
    SECOND_PASS_INDICES,
    banner,
    wait_for_space,
)

N_CAMERAS = 2
FIRST_TWO_FILE_2CAMS = os.path.join(TEMP_CALIB_DIR, "first_two_pucks_2cams.npz")
LAST_TWO_FILE_2CAMS = os.path.join(TEMP_CALIB_DIR, "last_two_pucks_2cams.npz")
LATEST_POSE_FILE_2CAMS = os.path.join(TEMP_CALIB_DIR, "latest_robot_poses_2cams.npz")
ROTATION_CODES = {0: None, 180: cv2.ROTATE_180}


def list_camera_indices():
    """/dev/videoN indices on Linux; the numbers are not contiguous when a camera is unplugged."""
    return sorted(
        int(m.group(1))
        for m in (re.match(r"/dev/video(\d+)$", path) for path in glob.glob("/dev/video*"))
        if m
    ) or list(range(10))


def open_cameras(camera_ids=None):
    """Open the given cameras, or the first N_CAMERAS that return a frame.

    Many webcams expose a second /dev/video node for metadata; it opens but never
    returns a frame, so it is skipped automatically.
    """
    candidates = list(camera_ids) if camera_ids is not None else list_camera_indices()
    caps, ids = [], []
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"cam{len(caps)}: using camera index {idx}")
                caps.append(cap)
                ids.append(idx)
                if len(caps) == N_CAMERAS:
                    return caps, ids
                continue
        cap.release()
        if camera_ids is not None:
            break
    for cap in caps:
        cap.release()
    raise RuntimeError(
        f"Need {N_CAMERAS} working cameras, found {len(ids)} ({ids}) among indices {candidates}. "
        "Pass --camera-indices A B to choose them."
    )


def read_frame(cap, rotation):
    ret, frame = cap.read()
    if not ret:
        return None
    code = ROTATION_CODES[rotation]
    return frame if code is None else cv2.rotate(frame, code)


def detect_n_red_pucks(cap, rotation, n_pucks, window_prefix, sample_frames=60, min_valid_frames=10):
    """Detect and average exactly ``n_pucks`` red pucks over multiple frames of one camera."""
    valid_points = []
    count_history = []
    last_frame = None
    last_mask = None

    for _ in range(sample_frames):
        frame = read_frame(cap, rotation)
        if frame is None:
            continue
        last_frame = frame

        centroids, mask = find_red_pucks(frame, target_count=n_pucks)
        last_mask = mask
        count_history.append(len(centroids))

        preview = frame.copy()
        points_rc = np.array([[row, col] for row, col, _ in centroids], dtype=np.float32)
        if len(points_rc) == n_pucks:
            ordered = _order_row_col_points(points_rc)
            valid_points.append(ordered)
            _draw_indexed_points(preview, ordered, color=(0, 255, 0))
        elif len(points_rc) > 0:
            _draw_indexed_points(preview, points_rc, color=(0, 255, 255))
        cv2.imshow(f"{window_prefix}-puck-detect", preview)
        cv2.imshow(f"{window_prefix}-puck-mask", mask)
        cv2.waitKey(1)

    dominant_count = Counter(count_history).most_common(1)[0][0] if count_history else 0
    if len(valid_points) < min_valid_frames:
        return None, dominant_count, last_frame, last_mask

    averaged = np.mean(np.stack(valid_points, axis=0), axis=0)
    averaged = _order_row_col_points(averaged)
    return averaged, n_pucks, last_frame, last_mask


def pixels_to_robot_mm(points_xy, Mrob, upscale_constant):
    """Map camera pixels (x, y), in the raw 640x480 frame, to robot (x, y) mm."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2) * upscale_constant
    return cv2.perspectiveTransform(pts, Mrob).reshape(-1, 2)


def calibrate_homography(camera_ids, rotations, save_homographies, start_from_third=False, last_two_placed=False):
    # Open the cameras before connecting to the robot so a missing camera fails fast.
    caps, camera_ids = open_cameras(camera_ids)
    cam_names = [f"cam{k}" for k in range(N_CAMERAS)]
    rtde_frequency = 500.0
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")

    for cap, name, rotation in zip(caps, cam_names, rotations):
        frame = read_frame(cap, rotation)
        if frame is None:
            raise RuntimeError(f"Failed to read from {name}")
        if frame.shape[:2] != (480, 640):
            print(f"WARNING: {name} frames are {frame.shape[1]}x{frame.shape[0]}, but the homography scaling assumes 640x480.")

    upscale_constant = 3
    visual_downscale_constant = 2
    original_size = np.array([640, 480])
    offset_constants = np.array((2250, 500), dtype=np.float32)
    #                               pos 0 (X,Y)        pos 1         pos 2        pos 3
    # robot_points_mm = np.float32([[-820, 330], [-820, -330], [-475, -330], [-475, 330]])  # original
    robot_points_mm = np.float32([[-820, 330], [-820, -330], [-600, -330], [-600, 330]])    # shifted up toward the top of the table because I think it's going to hit the edge

    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(TEMP_CALIB_DIR, session_stamp + "_3puck_2cams")
    _ensure_temp_dir(session_dir)

    vel = 0.3  # velocity limit
    acc = 0.3  # acceleration limit
    angle = [-0.00153677648744038, -3.0647520618606172, 0.0]
    rollout_start_pose = [-0.68, 0.0, 0.33] + angle

    # Match rollout default reset pose (AirHockeyReal reset_pos_setting="hitting").
    banner(
        "3-PUCK CALIBRATION, 2 CAMERAS\n"
        f"  cam0 = camera index {camera_ids[0]} (rotation {rotations[0]}), "
        f"cam1 = camera index {camera_ids[1]} (rotation {rotations[1]})\n"
        + (
            "  Pass 1: both cameras record the pucks already on positions 2 and 3.\n"
            "  Pass 2: remove them, robot visits 0 then 1 -> you put a puck on 0 while it is at 1,\n"
            "          then on 1 once it is back at the initial pose; both cameras record 0 and 1.\n"
            if last_two_placed
            else "  Pass 1: robot visits positions 0, 1, 2 -> you mark them, then place 3 pucks.\n"
            "  Pass 2: remove the pucks, robot visits position 3 -> you mark it, then place 1 puck.\n"
        )
        + "  Every detection runs in both cameras; both must see all 4 marks.\n"
        + (
            "  Leave only the 2 pucks on positions 2 and 3; clear any other red objects."
            if last_two_placed
            else "  Clear the table of red objects before starting."
        )
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

    def release_all():
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    def persist_pose_record(extra_data=None, update_latest=False):
        pose_data = {
            "timestamp_epoch_s": np.array([time.time()], dtype=np.float64),
            "camera_indices": np.array(camera_ids, dtype=np.int32),
            "camera_rotations": np.array(rotations, dtype=np.int32),
            "rollout_start_pose": np.array(rollout_start_pose, dtype=np.float32),
            "initial_pose_for_return": np.array(initial_pose, dtype=np.float32),
            "robot_points_mm": np.array(robot_points_mm, dtype=np.float32),
            "target_mark_poses": np.array(mark_pose_targets, dtype=np.float32),
            "actual_mark_poses": np.array(mark_pose_actual, dtype=np.float32),
        }
        if extra_data is not None:
            pose_data.update(extra_data)

        session_pose_file = os.path.join(session_dir, "robot_poses.npz")
        _save_robot_pose_record(session_pose_file, pose_data)
        if update_latest:
            _save_robot_pose_record(LATEST_POSE_FILE_2CAMS, pose_data)
        return session_pose_file

    def cancel():
        print("Calibration canceled by user.")
        pose_file = persist_pose_record({"aborted": np.array([1], dtype=np.int32)}, update_latest=False)
        print(f"Saved robot pose record: {pose_file}")
        return_to_initial("cancel")
        release_all()

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

    def collect_pucks(n_pucks, window_suffix):
        """Detect n_pucks in every camera until the user accepts.

        Returns a per-camera list of (points_row_col, frame, mask), or None on quit.
        """
        while True:
            print(f"Detecting {n_pucks} red puck(s) in both cameras...")
            results = []
            for cap, name, rotation in zip(caps, cam_names, rotations):
                points, count, frame, mask = detect_n_red_pucks(cap, rotation, n_pucks, name)
                if frame is not None:
                    preview = frame.copy()
                    if points is not None:
                        _draw_indexed_points(preview, points, color=(0, 255, 0))
                    cv2.imshow(f"{name}-puck-detect-{window_suffix}", preview)
                    cv2.waitKey(1)
                status = "OK" if points is not None else f"FAILED (saw {count} red blobs)"
                print(f"  {name}: {status}")
                results.append((points, count, frame, mask))

            if any(points is None for points, _, _, _ in results):
                retry = input(
                    f"Not every camera detected exactly {n_pucks} puck(s). Check that both cameras see every mark, "
                    "adjust, and press Enter to retry, or type 'q' to quit: "
                ).strip().lower()
                if retry in {"q", "quit", "exit"}:
                    return None
                continue

            confirm = input(
                f"Both cameras detected {n_pucks} puck(s). Type 'done' to accept, or press Enter to recapture: "
            ).strip().lower()
            if confirm in {"done", "d"}:
                return [(points, frame, mask) for points, _, frame, mask in results]

    def collect_pass(indices, window_suffix, previous_points=None):
        """Place + detect pucks on the marks for ``indices`` in both cameras, rejecting detections
        that overlap ``previous_points`` (per-camera list). Returns the per-camera list or None on quit."""
        while True:
            if not wait_for_placement(len(indices)):
                return None
            results = collect_pucks(len(indices), window_suffix)
            if results is None:
                return None
            if previous_points is None:
                return results
            too_close = []
            for name, prev, (points, _, _) in zip(cam_names, previous_points, results):
                min_dist = float(np.min(np.linalg.norm(prev[:, None, :] - points[None, :, :], axis=-1)))
                if min_dist < MIN_POINT_SEPARATION_PX:
                    too_close.append(f"{name} ({min_dist:.1f} px)")
            if not too_close:
                return results
            print(
                f"A detected puck is too close to a pass-1 puck in {', '.join(too_close)}; it looks like the "
                "same spot. Check the placement and try again."
            )

    def load_or_detect_saved_pair(indices, saved_file, pass_banner):
        """Pass 1 of --start-from-third / --last-two-placed: saved or freshly detected pucks on
        the existing marks for ``indices``, per camera. Fresh detections are saved to ``saved_file``."""
        indices = list(indices)
        saved = _load_robot_pose_record(saved_file)
        if saved is not None:
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(saved["timestamp_epoch_s"][0])))
            print(f"Found saved positions of the pucks on marks {indices} ({saved_time}, {saved_file}):")
            for k, name in enumerate(cam_names):
                print(f"  {name} image points (row, col):", saved[f"puck_points_row_col_cam{k}"].tolist())
            print("  robot points (mm):", saved["robot_points_mm"].tolist())
            if not np.allclose(saved["robot_points_mm"], robot_points_mm[indices]):
                print(
                    "WARNING: these were saved for different robot points than the current robot_points_mm "
                    f"{robot_points_mm[indices].tolist()}; re-detect instead of reusing."
                )
            if saved["camera_indices"].tolist() != list(camera_ids) or saved["camera_rotations"].tolist() != list(rotations):
                print(
                    f"WARNING: these were saved with cameras {saved['camera_indices'].tolist()} / rotations "
                    f"{saved['camera_rotations'].tolist()}, not {list(camera_ids)} / {list(rotations)}; re-detect instead of reusing."
                )
            answer = input("Type 'reuse' to use them, or press Enter to re-detect the 2 pucks: ").strip().lower()
            if answer in {"reuse", "r", "yes", "y"}:
                results = []
                for k, name in enumerate(cam_names):
                    frame = cv2.imread(str(saved[f"frame_path_cam{k}"]))
                    if frame is None:
                        print(f"Note: saved {name} frame not found; the pass-2 frame will be used for previews.")
                    results.append((saved[f"puck_points_row_col_cam{k}"].astype(np.float32), frame, None))
                print(f"Reusing the saved positions for marks {indices}.")
                return results

        banner(pass_banner)
        results = collect_pass(indices, "2pucks_first")
        if results is None:
            return None
        record = {
            "timestamp_epoch_s": np.array([time.time()], dtype=np.float64),
            "robot_points_mm": np.array(robot_points_mm[indices], dtype=np.float32),
            "camera_indices": np.array(camera_ids, dtype=np.int32),
            "camera_rotations": np.array(rotations, dtype=np.int32),
        }
        for k, (name, (points, frame, mask)) in enumerate(zip(cam_names, results)):
            artifacts = _save_puck_capture_artifacts(os.path.join(session_dir, name, "pass1_2pucks"), frame, mask, points)
            record[f"puck_points_row_col_cam{k}"] = np.array(points, dtype=np.float32)
            record[f"frame_path_cam{k}"] = np.array(artifacts.get("raw_path", ""), dtype=np.str_)
        _save_robot_pose_record(saved_file, record)
        print(f"Saved the positions of the pucks on marks {indices} to {saved_file}")
        return results

    def mark_first_two_placing_pucks():
        """--last-two-placed pass 2: visit position 0, then 1 (puck goes on 0), then return (puck goes on 1)."""
        idx0, idx1 = FIRST_TWO_INDICES
        apply_negative_z_force(ctrl)
        for idx in (idx0, idx1):
            robo_pt = robot_points_mm[idx]
            mark_pose = [robo_pt[0] * 0.001, robo_pt[1] * 0.001, 0.33] + angle
            mark_pose_targets[idx] = mark_pose
            print(f"\nMoving robot to position {idx} at robot (x, y) = {robo_pt.tolist()} mm...")
            move_success = ctrl.moveL(mark_pose, vel, acc, False)
            print(f"Arrived at position {idx}: success={move_success}")
            if not move_success:
                print("WARNING: the move failed, so the robot is NOT at this position. Do not place a puck for it.")
            pose_now = rcv.getActualTCPPose()
            if pose_now is not None and len(pose_now) >= 6:
                mark_pose_actual[idx] = list(pose_now[:6])

            if idx == idx0:
                wait_for_space(
                    f"The paddle is on position {idx0}. Note exactly where it sits, then press SPACE: "
                    f"the robot moves to position {idx1} and you put a puck on position {idx0}... "
                )
            else:
                print(f"The paddle is now on position {idx1}. Put a red puck exactly where the paddle was on position {idx0}.")
                wait_for_space(
                    f"Once the puck is on position {idx0}, note exactly where the paddle sits on position {idx1}, "
                    "then press SPACE: the robot returns to the initial pose... "
                )
        persist_pose_record(update_latest=False)
        print(f"\nDone with positions {list(FIRST_TWO_INDICES)}. Moving robot back to the initial pose.")
        return_to_initial("post_marking_0_1")
        time.sleep(1.0)

    if last_two_placed:
        first_indices, second_indices = LAST_TWO_INDICES, FIRST_TWO_INDICES
        first_results = load_or_detect_saved_pair(
            LAST_TWO_INDICES,
            LAST_TWO_FILE_2CAMS,
            f"PASS 1 (--last-two-placed): record the pucks already on positions {list(LAST_TWO_INDICES)}\n"
            f"  The robot stays at the initial pose. Leave the 2 pucks where they are:\n"
            f"    position {LAST_TWO_INDICES[0]} = robot (x, y) {robot_points_mm[LAST_TWO_INDICES[0]].tolist()} mm\n"
            f"    position {LAST_TWO_INDICES[1]} = robot (x, y) {robot_points_mm[LAST_TWO_INDICES[1]].tolist()} mm\n"
            "  Type 'done' below and both cameras will take the picture.",
        )
        if first_results is None:
            cancel()
            return
    elif start_from_third:
        first_indices, second_indices = FIRST_TWO_INDICES, LAST_TWO_INDICES
        first_results = load_or_detect_saved_pair(
            FIRST_TWO_INDICES,
            FIRST_TWO_FILE_2CAMS,
            f"PASS 1 (--start-from-third): detect the pucks on marks {list(FIRST_TWO_INDICES)}\n"
            f"  The robot stays at the initial pose; it does NOT revisit positions {list(FIRST_TWO_INDICES)}.\n"
            f"  Place 2 red pucks on your existing marks for positions {list(FIRST_TWO_INDICES)}:\n"
            f"    position 0 = robot (x, y) {robot_points_mm[0].tolist()} mm\n"
            f"    position 1 = robot (x, y) {robot_points_mm[1].tolist()} mm",
        )
        if first_results is None:
            cancel()
            return
    else:
        # ---- Pass 1: robot marks 3 positions, detect 3 pucks in both cameras ----
        first_indices, second_indices = FIRST_PASS_INDICES, SECOND_PASS_INDICES
        banner(f"PASS 1: marking positions {list(first_indices)}")
        mark_robot_points(first_indices)
        banner(
            f"PASS 1: place the pucks\n"
            f"  Since we only have 3 pucks, the robot is now back at the initial pose.\n"
            f"  Place the 3 red pucks on the marks for positions {list(first_indices)}."
        )
        first_results = collect_pass(first_indices, "3pucks")
        if first_results is None:
            cancel()
            return
        for name, (points, frame, mask) in zip(cam_names, first_results):
            _save_puck_capture_artifacts(os.path.join(session_dir, name, "pass1_3pucks"), frame, mask, points)
    for name, (points, _, _) in zip(cam_names, first_results):
        print(f"Pass 1 {name} puck points (row, col):", points.tolist())
    print(f"Pass 1 complete: {len(first_indices)} puck positions recorded in both cameras.")

    # ---- Pass 2: remove pucks, robot marks the remaining position(s), detect them ----
    n_second = len(second_indices)
    banner(
        f"PASS 2: marking position(s) {list(second_indices)}\n"
        f"  The robot needs a clear table, so remove the {len(first_indices)} pucks first."
    )
    if not wait_for_removal(len(first_indices)):
        cancel()
        return

    if last_two_placed:
        mark_first_two_placing_pucks()
        banner(
            f"PASS 2: place the second puck\n"
            f"  The robot is back at the initial pose. Position {FIRST_TWO_INDICES[0]} already has its puck.\n"
            f"  Put a red puck exactly where the paddle was on position {FIRST_TWO_INDICES[1]} "
            f"(leave positions {list(first_indices)} empty).\n"
            "  Type 'done' below and both cameras will take the picture."
        )
    else:
        mark_robot_points(second_indices)
        banner(
            f"PASS 2: place the puck(s)\n"
            f"  The robot is back at the initial pose.\n"
            f"  Place {n_second} red puck(s) on the mark(s) for position(s) {list(second_indices)} "
            f"(leave the marks for {list(first_indices)} empty)."
        )
    second_results = collect_pass(
        second_indices, f"{n_second}puck", previous_points=[points for points, _, _ in first_results]
    )
    if second_results is None:
        cancel()
        return
    for name, (points, frame, mask) in zip(cam_names, second_results):
        print(f"Pass 2 {name} puck points (row, col):", points.tolist())
        _save_puck_capture_artifacts(os.path.join(session_dir, name, f"pass2_{n_second}puck"), frame, mask, points)
    banner("All 4 puck positions recorded in both cameras. Computing the homographies...")

    # ---- Per camera: merge into the 4 points and continue exactly as calibrate_robo_camera.py ----
    robot_reference_xy = robot_points_mm + offset_constants
    robot_order = _ordered_indices(robot_reference_xy)
    robot_points_ordered = robot_points_mm[robot_order]
    robot_reference_ordered = robot_reference_xy[robot_order]

    # add some optional tuning (this is just magic numbers to get the calibration to work)
    tuning_offsets = np.float32([[-1, -1], [0, 0], [0, 1], [-1, 1]])
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    canvas_size = tuple(int(v) for v in original_size * upscale_constant)
    preview_size = (
        int(640 * upscale_constant / visual_downscale_constant),
        int(480 * upscale_constant / visual_downscale_constant),
    )

    cams = []
    for k, name in enumerate(cam_names):
        first_points, first_frame, _ = first_results[k]
        second_points, second_frame, _ = second_results[k]
        accepted_frame = first_frame if first_frame is not None else second_frame

        accepted_points = _order_row_col_points(np.concatenate([first_points, second_points], axis=0))
        accepted_points_row_col = np.float32(accepted_points)
        # Convert detector output from (row, col) to OpenCV point order (x, y).
        accepted_points_xy = accepted_points_row_col[:, [1, 0]] + tuning_offsets
        pts1 = accepted_points_xy * upscale_constant

        Mrob = cv2.getPerspectiveTransform(pts1, robot_points_ordered)
        Mimg = cv2.getPerspectiveTransform(pts1, robot_reference_ordered)

        print(f"\n{name} (camera index {camera_ids[k]}) correspondences used for calibration:")
        for idx, (pixel_pt, robo_pt) in enumerate(zip(accepted_points_xy, robot_points_ordered)):
            print(f"  idx={idx} pixel(x,y)={pixel_pt.tolist()} -> robot(mm)={robo_pt.tolist()}")

        cam_dir = os.path.join(session_dir, name)
        _ensure_temp_dir(cam_dir)
        overlay = accepted_frame.copy()
        _draw_indexed_points(overlay, accepted_points, color=(0, 255, 0))
        cv2.imwrite(os.path.join(cam_dir, "puck_capture_combined_overlay.png"), overlay)
        capture_artifacts = _save_puck_capture_artifacts(cam_dir, accepted_frame, None, accepted_points)

        image = cv2.resize(accepted_frame, canvas_size, interpolation=cv2.INTER_LINEAR)
        for idx, val in enumerate(pts1.astype(np.int32)):
            cv2.circle(image, (int(val[0]), int(val[1])), 5, colors[idx % len(colors)], -1)
        warped = cv2.warpPerspective(image, Mimg, canvas_size)
        coverage = cv2.warpPerspective(np.ones(image.shape[:2], dtype=np.float32), Mimg, canvas_size)

        cams.append(
            {
                "name": name,
                "Mrob": Mrob,
                "Mimg": Mimg,
                "points_row_col": accepted_points_row_col,
                "points_xy": accepted_points_xy,
                "first_points": first_points,
                "second_points": second_points,
                "image": image,
                "warped": warped,
                "coverage": coverage,
                "raw_path": capture_artifacts.get("raw_path", ""),
            }
        )

    # ---- Multi-camera: pixel->pixel homography between the cameras and a fused top-down view ----
    # Both Mrob map (x3 upscaled) pixels into the same robot frame, so chaining one with the
    # inverse of the other maps camera-1 table pixels onto camera-0 table pixels.
    H_cam1_to_cam0 = np.linalg.inv(cams[0]["Mrob"]) @ cams[1]["Mrob"]
    H_cam1_to_cam0 /= H_cam1_to_cam0[2, 2]
    print("\nH_cam1_to_cam0 (cam1 upscaled pixel -> cam0 upscaled pixel, table plane):")
    print(np.array2string(H_cam1_to_cam0, precision=5, suppress_small=True))

    # Fused view: average both warped cameras where both see the table, else take whichever does.
    weight_sum = np.zeros(canvas_size[::-1], dtype=np.float32)
    fused = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    for cam in cams:
        weight = (cam["coverage"] > 0.5).astype(np.float32)
        fused += cam["warped"].astype(np.float32) * weight[..., None]
        weight_sum += weight
    fused = (fused / np.maximum(weight_sum, 1.0)[..., None]).astype(np.uint8)
    overlap_fraction = float(np.mean(weight_sum >= 2)) / max(float(np.mean(weight_sum >= 1)), 1e-6)
    print(f"Fused view: the cameras overlap on {100 * overlap_fraction:.0f}% of the covered top-down area.")

    for cam in cams:
        cv2.imshow(f"{cam['name']}-image", cv2.resize(cam["image"], preview_size, interpolation=cv2.INTER_LINEAR))
        cv2.imshow(f"{cam['name']}-transformed", cv2.resize(cam["warped"], preview_size, interpolation=cv2.INTER_LINEAR))
    cv2.imshow("fused-transformed", cv2.resize(fused, preview_size, interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(5000)
    cv2.imwrite(os.path.join(session_dir, "fused_transformed.png"), fused)

    # ---- Live cross-camera check: the 4 marks fit exactly by construction, so test elsewhere ----
    banner(
        "CROSS-CAMERA CHECK\n"
        "  Remove the calibration pucks and place ONE red puck anywhere both cameras can see it.\n"
        "  Both cameras map it to robot (x, y); the difference is the calibration error away from the marks.\n"
        "  Try a few spots, especially outside the rectangle of marks."
    )
    check_errors_mm = []
    while True:
        answer = input("Place 1 puck and press Enter to check, or type 'q' to finish checking: ").strip().lower()
        if answer in {"q", "quit", "exit", "done", "d"}:
            break
        robot_xy = []
        for cam, cap, rotation in zip(cams, caps, rotations):
            points, count, _, _ = detect_n_red_pucks(cap, rotation, 1, cam["name"])
            if points is None:
                print(f"  {cam['name']}: could not see exactly 1 puck (saw {count} red blobs).")
                robot_xy.append(None)
                continue
            xy = pixels_to_robot_mm(points[:, [1, 0]], cam["Mrob"], upscale_constant)[0]
            print(f"  {cam['name']}: robot (x, y) = ({xy[0]:.1f}, {xy[1]:.1f}) mm")
            robot_xy.append(xy)
        if all(xy is not None for xy in robot_xy):
            err = float(np.linalg.norm(robot_xy[0] - robot_xy[1]))
            check_errors_mm.append(err)
            print(f"  cam0 vs cam1 disagreement: {err:.1f} mm")
    if check_errors_mm:
        print(
            f"Cross-camera disagreement over {len(check_errors_mm)} check(s): "
            f"mean {np.mean(check_errors_mm):.1f} mm, max {np.max(check_errors_mm):.1f} mm"
        )

    input("Press Enter to finish calibration and exit... ")

    # Save calibration data
    if save_homographies:
        for k, cam in enumerate(cams):
            np.save(f"Mimg_cam{k}.npy", cam["Mimg"])
            np.save(f"Mrob_cam{k}.npy", cam["Mrob"])
        np.save("H_cam1_to_cam0.npy", H_cam1_to_cam0)
        print(
            "Saved Mimg_cam0.npy, Mrob_cam0.npy, Mimg_cam1.npy, Mrob_cam1.npy and H_cam1_to_cam0.npy "
            f"to {os.getcwd()}"
        )

    extra = {
        "aborted": np.array([0], dtype=np.int32),
        "robot_points_ordered_mm": np.array(robot_points_ordered, dtype=np.float32),
        "H_cam1_to_cam0": np.array(H_cam1_to_cam0, dtype=np.float64),
        "cross_camera_check_errors_mm": np.array(check_errors_mm, dtype=np.float32),
    }
    for k, cam in enumerate(cams):
        extra[f"Mimg_cam{k}"] = np.array(cam["Mimg"], dtype=np.float64)
        extra[f"Mrob_cam{k}"] = np.array(cam["Mrob"], dtype=np.float64)
        extra[f"detected_puck_points_row_col_cam{k}"] = cam["points_row_col"]
        extra[f"detected_puck_points_xy_cam{k}"] = np.array(cam["points_xy"], dtype=np.float32)
        extra[f"pass1_puck_points_row_col_cam{k}"] = np.array(cam["first_points"], dtype=np.float32)
        extra[f"pass2_puck_points_row_col_cam{k}"] = np.array(cam["second_points"], dtype=np.float32)
        extra[f"puck_capture_image_path_cam{k}"] = np.array(cam["raw_path"], dtype=np.str_)
    pose_file = persist_pose_record(extra, update_latest=True)
    print(f"Saved puck capture artifacts in: {session_dir}")
    print(f"Saved robot pose record: {pose_file}")

    # End at startup pose so post-calibration setup is convenient.
    return_to_initial("final")
    release_all()


def _parse_args(argv):
    save_homographies = "--save-homographies" in argv or "-s" in argv

    # --camera-indices A B forces the cameras; otherwise the first two working ones are used.
    camera_ids = None
    if "--camera-indices" in argv:
        i = argv.index("--camera-indices")
        try:
            camera_ids = [int(argv[i + 1]), int(argv[i + 2])]
        except (IndexError, ValueError):
            raise SystemExit("--camera-indices requires two integer camera ids, e.g. --camera-indices 0 2")

    rotations = [180] * N_CAMERAS
    if "--rotations" in argv:
        i = argv.index("--rotations")
        try:
            rotations = [int(v) for v in argv[i + 1].split(",")]
        except (IndexError, ValueError):
            raise SystemExit("--rotations requires a comma-separated list, e.g. --rotations 180,0")
        if len(rotations) != N_CAMERAS or any(r not in ROTATION_CODES for r in rotations):
            raise SystemExit(f"--rotations needs {N_CAMERAS} values, each one of {sorted(ROTATION_CODES)}")

    # --start-from-third: marks 0 and 1 already exist; detect them, then mark 2 and 3.
    start_from_third = "--start-from-third" in argv
    # --last-two-placed: pucks already sit on 2 and 3; record them, then visit 0 and 1.
    last_two_placed = "--last-two-placed" in argv
    if start_from_third and last_two_placed:
        raise SystemExit("--start-from-third and --last-two-placed cannot be combined.")
    return camera_ids, rotations, save_homographies, start_from_third, last_two_placed


if __name__ == "__main__":
    camera_ids, rotations, save_homographies, start_from_third, last_two_placed = _parse_args(sys.argv)
    calibrate_homography(
        camera_ids,
        rotations,
        save_homographies,
        start_from_third=start_from_third,
        last_two_placed=last_two_placed,
    )
