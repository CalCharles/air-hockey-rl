#!/usr/bin/env python3
"""Move slowly through the four calibration corners.

Same pose recipe as ``calibrate_robo_camera.py``: each row of
``robot_points_mm`` is x, y in millimeters, z is fixed at 0.33 m, and the
wrist is the fixed down-facing ``angle``. On the UR5e tablet those land as
x, y in millimeters and z near 330 mm.

The arm holds at each corner until you press space. Ctrl+C stops a move
and leaves the arm where it is.

    python scripts/real/sanity_check_TCP_pos_over_april_tags.py
"""

from __future__ import annotations

import select
import signal
import sys
import termios
import time
import tty

import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

ROBOT_HOST = "172.22.22.2"
RTDE_FREQUENCY = 500.0

# Same corners as calibrate_robo_camera.py. x, y in millimeters.
ROBOT_POINTS_MM = np.float32([[-820, 330], [-820, -330], [-475, -330], [-475, 330]])
# Same height and wrist as mark_pose in that script.
Z_M = 0.33
ANGLE = [-0.00153677648744038, -3.0647520618606172, 0.0]

VEL = 0.02  # m/s
ACC = 0.05  # m/s^2
STOP_DECEL = 0.4  # m/s^2, used by stopL so the arm halts in place
POSE_TOL_M = 0.003
POLL_S = 0.02


class StopRequested(Exception):
    """Space or Ctrl+C asked the current move to halt."""


def _halt(ctrl):
    """Stop the running linear move and leave the arm where it is."""
    try:
        ctrl.stopL(STOP_DECEL)
    except Exception as exc:
        print(f"stopL failed ({exc}); stopping the control script.")
        try:
            ctrl.stopScript()
        except Exception as stop_exc:
            print(f"stopScript failed ({stop_exc}).")


def _install_sigint(ctrl):
    def _on_sigint(_signum, _frame):
        _halt(ctrl)
        raise StopRequested

    signal.signal(signal.SIGINT, _on_sigint)


def _poll_stop_key(ctrl):
    """Ctrl+C stops the move. Space pressed during a move is ignored."""
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return
    ch = sys.stdin.read(1)
    if ch == "\x03":
        _halt(ctrl)
        raise StopRequested


def _wait_for_space(ctrl):
    """Hold here until space. Ctrl+C still aborts."""
    print("Holding. Press space for the next corner, or Ctrl+C to stop.")
    while True:
        readable, _, _ = select.select([sys.stdin], [], [], POLL_S)
        if not readable:
            continue
        ch = sys.stdin.read(1)
        if ch == " ":
            return
        if ch == "\x03":
            _halt(ctrl)
            raise StopRequested


def _wait_for_move(ctrl):
    """Block until the async moveL finishes, or Ctrl+C stops it.

    ``getAsyncOperationProgress()`` is -1 both before the move has started and
    after it has finished. Returning on that value immediately cancels the
    path, because the next ``moveL`` (and the final ``stopScript``) lands
    before the arm has moved.
    """
    started = time.time()
    while ctrl.getAsyncOperationProgress() < 0:
        _poll_stop_key(ctrl)
        if time.time() - started > 2.0:
            raise RuntimeError("moveL was accepted but never started")
        time.sleep(POLL_S)
    while ctrl.getAsyncOperationProgress() >= 0:
        _poll_stop_key(ctrl)
        time.sleep(POLL_S)


def _move_l(ctrl, rcv, pose, label):
    print(
        f"{label}: x={pose[0] * 1000:.2f} y={pose[1] * 1000:.2f} "
        f"z={pose[2] * 1000:.2f} mm"
    )
    accepted = ctrl.moveL(list(pose), VEL, ACC, True)
    if not accepted:
        raise RuntimeError(f"moveL rejected for {label}")
    _wait_for_move(ctrl)
    actual = rcv.getActualTCPPose()
    if actual is None or len(actual) < 3:
        print(f"{label}: move finished, TCP pose unread")
        return
    actual_xyz = np.asarray(actual[:3], dtype=float)
    err = float(np.linalg.norm(actual_xyz - np.asarray(pose[:3])))
    print(
        f"{label}: tablet should read x={actual_xyz[0] * 1000:.1f} "
        f"y={actual_xyz[1] * 1000:.1f} z={actual_xyz[2] * 1000:.1f} mm "
        f"(error {err * 1000:.1f} mm)"
    )
    if err > POSE_TOL_M * 4:
        print(f"{label}: warning, still {err * 1000:.1f} mm from the target")


def main():
    print("Calibration-corner sanity check")
    print(f"z: {Z_M * 1000:.1f} mm, wrist ry: {ANGLE[1]:.4f} rad")
    print(f"speed: {VEL:.3f} m/s, acceleration: {ACC:.3f} m/s^2")
    print("Tablet x, y should match these millimeters when each move stops.")
    print("The arm holds at each corner until you press space.")
    print("Ctrl+C stops a move and leaves the arm in place.\n")
    for idx, (x_mm, y_mm) in enumerate(ROBOT_POINTS_MM):
        print(f"  [{idx}] x={float(x_mm):.1f} y={float(y_mm):.1f} z={Z_M * 1000:.1f} mm")
    input("\nPress Enter to start, or Ctrl+C to quit. ")

    ctrl = RTDEControl(ROBOT_HOST, RTDE_FREQUENCY, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive(ROBOT_HOST)
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    try:
        _install_sigint(ctrl)
        tty.setcbreak(fd)
        for idx, (x_mm, y_mm) in enumerate(ROBOT_POINTS_MM):
            pose = [float(x_mm) * 0.001, float(y_mm) * 0.001, Z_M] + ANGLE
            print(f"\nCorner {idx}")
            _move_l(ctrl, rcv, pose, f"corner {idx}")
            _wait_for_space(ctrl)
    except StopRequested:
        print("\nStopped. The arm stays at its current pose.")
    except KeyboardInterrupt:
        _halt(ctrl)
        print("\nStopped. The arm stays at its current pose.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        try:
            ctrl.stopScript()
        except Exception:
            pass
        try:
            ctrl.disconnect()
        except Exception:
            pass
        try:
            rcv.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
