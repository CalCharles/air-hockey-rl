import numpy as np
import time
from .coordinate_transform import clip_limits

def apply_negative_z_force(ctrl, rcv=None, wrench_z=None):
    # Keep a constant Z-axis force mode active for table contact bias.
    # This helper intentionally preserves existing frame/sign behavior.
    using_rcv_frame = rcv is not None
    frame_mode = "target_tcp_frame" if using_rcv_frame else "world_origin_frame"
    if rcv is None:
        force_frame = [0, 0, 0, 0, 0, 0]
        default_wrench_z = -5.0
    else:
        force_frame = rcv.getTargetTCPPose()
        default_wrench_z = 5.0  # why was this 5?
    sign = -1.0 if float(default_wrench_z) < 0.0 else 1.0
    magnitude = abs(float(default_wrench_z)) if wrench_z is None else abs(float(wrench_z))
    applied_wrench_z = sign * magnitude
    z_axis_wrench = [0.0, 0.0, applied_wrench_z, 0.0, 0.0, 0.0]

    # TODO: Verify and unify wrench sign convention across force-frame choices.
    # Current behavior uses opposite Z signs depending on frame source.
    constrained_axes = [0, 0, 1, 0, 0, 0]
    ctrl_type = 2  # Keep current force frame interpretation.
    force_limits = [2.0, 2.0, 1.5, 1.0, 1.0, 1.0]
    # TODO: Modify control type 

    # Diagnostics: log immediately when frame source changes, and periodically otherwise.
    now_s = time.time()
    prev_mode = getattr(apply_negative_z_force, "_last_frame_mode", None)
    call_count = int(getattr(apply_negative_z_force, "_call_count", 0)) + 1
    should_log = prev_mode != frame_mode or call_count % 200 == 0
    if should_log:
        frame_z = float(force_frame[2]) if using_rcv_frame else 0.0
        # print(
        #     "[force_diag] "
        #     f"apply_negative_z_force call={call_count} mode={frame_mode} "
        #     f"rcv_is_none={rcv is None} wrench_z={float(z_axis_wrench[2]):.3f} "
        #     f"frame_z={frame_z:.4f} ctrl_type={ctrl_type} "
        #     f"selection={constrained_axes} limits={force_limits}"
        # )
    apply_negative_z_force._last_frame_mode = frame_mode
    apply_negative_z_force._call_count = call_count
    apply_negative_z_force._last_log_time_s = now_s

    ctrl.forceMode(force_frame, constrained_axes, z_axis_wrench, ctrl_type, force_limits)


def filter_update(vel, pose_hist, dpose_hist):
    pose_vel = np.array(dpose_hist[-1]) - np.array(pose_hist[-1])
    transform_vel = pose_vel
    if len(dpose_hist) > 1:
        pose_vels = [np.array(dpose_hist[i]) - np.array(pose_hist[i]) for i in range(len(dpose_hist))]

        # last_pose_vel = np.array(dpose_hist[-2]) - np.array(pose_hist[-2])
        transform_vel = np.mean(pose_vels, axis=0)
    desired_pose = transform_vel + pose_hist[-1]
    return desired_pose

class MotionPrimitive:
    def __init__(self):
        self.is_strike = False
        self.return_count = 15
        self.return_counter = 0
        self.strike_speed = 0.95
        self.slow_strike_speed = 0.70
        self.is_fast = False

    def compute_primitive(self, val, true_pose, lims, move_lims, edge_lims):
        # takes an action based on the current position, the keyboard input and whether we are currently taking an action
        delta = np.zeros((2,))
        if val == 'a':
            delta[1] = -0.04
        elif val == 'd':
            delta[1] = 0.04
        if self.is_strike: # if we are striking, ignore strike commands and execute striking behavior
            if self.is_fast:
                ss = self.strike_speed
            else:
                ss = self.slow_strike_speed
            if self.return_counter > 0: #  in the return phase
                self.return_counter += 1
                if self.return_counter == self.return_count:
                    self.is_strike = False
                    self.return_counter = 0
                if self.is_strike and (0 < self.return_counter <= 10):
                    delta[0] = 0.0 # wait at the top
                if self.is_strike and self.return_counter > 10:
                    delta[0] = move_lims[0] * 0.7 # return at 0.8 times the speed
            else:
                if true_pose[0] < -0.78: 
                    self.return_counter += 1
                else:
                    delta[0] = -move_lims[0] * ss # strike at basically maximum speed
        else:
            if val == 'w': # strike
                self.is_strike = True
                delta[0] = -move_lims[0] * self.strike_speed # strike at basically maximum speed
            
            elif val == 'e': # slow_strike
                self.is_strike = True
                delta[0] = -move_lims[0] * self.slow_strike_speed # strike at basically maximum speed
            else: delta[0] = 0.05
        x, y = true_pose[0] + delta[0], true_pose[1] + delta[1]
        x,y = clip_limits(x,y,lims, edge_lims)
        return x,y
                
