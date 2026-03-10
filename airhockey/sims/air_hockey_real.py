import time
from collections import deque
import numpy as np
from multiprocessing import shared_memory
from .real.multiprocessing import ProtectedArray, NonBlockingConsole
from .real.control_parameters import (
    camera_callback,
    save_callback,
    mimic_control,
    save_collect,
    observe_collect,
    visual_downscale_constant,
)
from .real.trajectory_merging import merge_trajectory, clear_images, write_trajectory, get_trajectory_idx
from .real.robot_control import MotionPrimitive, apply_negative_z_force, filter_update
from .real.coordinate_transform import compute_rect, compute_pol, clip_limits
from .real.proprioceptive_state import get_state_array
from .real.image_detection import find_red_hockey_puck, find_red_hockey_puck_antiglare
from .real.overlay_utils import draw_target_marker, draw_puck_marker_from_state, draw_paddle_marker
import multiprocessing
import cv2
import copy
from ..utils import dict_to_namespace

puck_detectors = {
    "red_puck": find_red_hockey_puck,
    "red_puck_antiglare": find_red_hockey_puck_antiglare,
    # TODO: other puck detectors here
}

reset_positions = {
    "hitting": [-0.68, 0., 0.33],
    "stationary": [-0.38, 0., 0.33],
    "forward_hitting": [-0.78, 0., 0.33],
    "negative_regions": [-0.38, -0.345, 0.33]
}


_ASYNC_RENDER_METADATA_WIDTH = 7


def _async_render_worker(
    frame_shm_name,
    frame_shape,
    metadata,
    metadata_lock,
    frame_seq,
    frame_epoch,
    stop_event,
    window_name,
    puck_radius,
    paddle_radius,
    center_offset_constant,
    offset_constants,
    visual_downscale_constant,
    poll_sleep_s,
    debug,
):
    frame_shm = None
    try:
        frame_shm = shared_memory.SharedMemory(name=frame_shm_name)
        shared_frame = np.ndarray(tuple(frame_shape), dtype=np.uint8, buffer=frame_shm.buf)
        poll_sleep_s = max(0.0005, float(poll_sleep_s))
        last_seq = -1
        last_epoch = int(frame_epoch.value)
        while not stop_event.is_set():
            current_epoch = int(frame_epoch.value)
            current_seq = int(frame_seq.value)
            if current_epoch != last_epoch:
                last_epoch = current_epoch
                last_seq = -1
            if current_seq <= last_seq:
                time.sleep(poll_sleep_s)
                continue

            with metadata_lock:
                frame = np.array(shared_frame, copy=True)
                data = np.array(metadata[:], dtype=float)
                current_seq = int(frame_seq.value)

            target_xy = (float(data[0]), float(data[1]))
            puck_state = np.array((data[2], data[3], data[4]), dtype=float)
            paddle_xy = (float(data[5]), float(data[6]))
            draw_target_marker(
                frame,
                target_xy,
                offset_constants=offset_constants,
                visual_downscale_constant=visual_downscale_constant,
            )
            draw_puck_marker_from_state(
                frame,
                puck_state,
                puck_radius,
                x_offset_for_state=center_offset_constant,
                offset_constants=offset_constants,
                visual_downscale_constant=visual_downscale_constant,
                color=(0, 255, 0),
                require_visible=True,
            )
            draw_paddle_marker(
                frame,
                paddle_xy,
                paddle_radius,
                offset_constants=offset_constants,
                visual_downscale_constant=visual_downscale_constant,
                color=(255, 0, 0),
            )
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
            last_seq = current_seq
    except Exception as exc:
        if debug:
            print(f"[async_render] worker disabled after exception: {exc}")
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        if frame_shm is not None:
            frame_shm.close()


class AirHockeyReal:
    def __init__(self, **kwargs):
        defaults = {
            "force_scaling": 1, 
            "paddle_damping": 1000, 
            'puck_damping': 1000,
            'render_size': 360,
            'seed': 42,
            'action_x_scaling': 1.0,
            'action_y_scaling': 1.0,
            'render_masks': False,
            'gravity': -5,
            'paddle_density': 1000,
            'puck_density': 250,
            'block_density': 1000,
            'max_paddle_vel': 2,
            'time_frequency': 20,
            'paddle_bounds': [-0.8, -0.33, -0.3582, 0.350],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2,
            'puck_restitution': 1.0,

            "control_mode": 'mouse',
            "control_type": "rect",
            "puck_history_len": 5,
            "paddle_history_len": 5,
            "puck_detector": "red_puck_antiglare",
            # Antiglare bounds are interpreted in the raw detector input image
            # (the image passed into the puck detector before preprocess).
            "antiglare_bounds_in_raw_image": True,
            "antiglare_min_x_px": 290,
            "antiglare_max_x_px": 451,
            "antiglare_min_y_px": 186,
            "antiglare_max_y_px": 465,
            "image_path": "./temp/images/",
            "save_path": "./data/rollout/temp_saving",
            "vel_lim": 0.8,
            "acc_lim": 0.8,
            "rmax_x": 0.26,
            "rmax_y": 0.12,
            "teleoperation_noise": 0.0,
            "block_time": 0.049,
            "runtime": 0.0,
            "lookahead": 0.2,
            "gain": 700,
            "angle": [-0.00153677648744038, -3.0647520618606172, 0.],
            "zslope": 0.02577,
            "x_offset": 1.2,
            "y_offset": 0.0,
            "paddle_additional_x_offset": -0.075,
            "paddle_additional_y_offset": -0.03,
            "bot_abs": 0.1,
            "top_abs": 0.8,
            "max_bias_p": -0.15,
            "max_bias_m": -0.15,
            "reset_pos_setting": "hitting",
            "xv_min": -0.5,
            "xv_max": 0.5,
            "yv_min": -0.3,
            "yv_max": 0.3,
            "hist_len": 2,
            "camera_index": 0,
            "wait_for_space_to_start": True,
            "debug_control": False,
            "debug_control_every": 1,
            "transition_hold_steps_on_estop_enter": 0,
            "transition_hold_steps_on_estop_clear": 8,
            "transition_hold_steps_on_safety_rearm": 3,
            "transition_hold_debug": False,

            # The current state prediction algorithm uses true current position
            # and adds a predictive horizon on top
            "use_actual_tcp_for_state": True,
            # "state_prediction_horizon_s": 0.05,
            "state_prediction_horizon_s": 0.05,
            "state_prediction_blend": 0.5, # run regression over a trajectory and see this
            "state_prediction_opposite_dir_brake": 1.5,
            "disable_prediction_on_estop": True,
            "async_render_enabled": False,
            "async_render_debug": False,
            "async_render_poll_sleep_s": 0.001,
            "async_render_window_name": "showdst",
            "async_render_frame_width": 960,
            "async_render_frame_height": 720,
        }
        kwargs = {**defaults, **kwargs}
        config = dict_to_namespace(kwargs)

        # physics / world params
        # TODO: special config for real
        self.length, self.width = config.length, config.width
        self.paddle_radius = config.paddle_radius
        self.puck_radius = config.puck_radius
        self.block_width = config.block_width
        self.max_force_timestep = config.max_force_timestep
        self.time_frequency = config.time_frequency
        self.time_per_step = 1 / self.time_frequency
        self.force_scaling = config.force_scaling
        self.absorb_target = config.absorb_target
        self.paddle_damping = config.paddle_damping
        self.puck_damping = config.puck_damping
        self.gravity = config.gravity
        self.puck_min_height = (-config.length / 2) + (config.length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = config.width
        self.min_speed_start = -config.width
        self.paddle_density = config.paddle_density
        self.puck_density = config.puck_density
        self.block_density = config.block_density
        # these assume 2d, in 3d since we have height it would be higher mass
        self.paddle_mass = self.paddle_density * np.pi * self.paddle_radius ** 2
        self.puck_mass = self.puck_density * np.pi * self.puck_radius ** 2
        self.center_offset_constant = config.center_offset_constant

        # these 2 will depend on the other parameters
        self.max_paddle_vel = config.max_paddle_vel # m/s. This will be dependent on the robot arm
        # compute maximum force based on max paddle velocity
        max_a = self.max_paddle_vel / self.time_per_step
        max_f = self.paddle_mass * max_a
        # assume maximum force transfer
        puck_max_a = max_f / self.puck_mass
        self.max_puck_vel = puck_max_a * self.time_per_step

        # box2d visualization params (but the visualization is done in the Render file)
        self.ppm = config.render_size / self.width
        self.render_width = int(config.render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = config.render_masks

        self.table_x_min = -self.width / 2
        self.table_x_max = self.width / 2
        self.table_y_min = -self.length / 2
        self.table_y_max = self.length / 2

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

        self.metadata = {}

        self.transition_start = time.time()
        rtde_frequency = 500.0
        self.control_mode = config.control_mode # mouse, mimic, keyboard, RL, BC, IQL, rnet, reach, observe
        self.control_type = config.control_type # rect, pol or prim
        # input modes: state force_acc puck_vals goal goal_vel
        # algo options: iql, ppo
        self.additional_args = {"image_input": False, "frame_stack": 1, "algo": "iql", "goal_type": "goal_vel", "input_mode": "puck_vals",
                        "normalize": True} # Goal conditoned args

        from rtde_control import RTDEControlInterface as RTDEControl
        from rtde_receive import RTDEReceiveInterface as RTDEReceive

        self.ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
        self.rcv = RTDEReceive("172.22.22.2")

        teleoperation_modes = ['mouse', 'mimic', 'keyboard']
        autonomous_modes = ['BC', 'RL', 'IQL', 'rnet', 'reach', 'rand']
        autonomous_model = None
        # if control_mode in autonomous_modes:
        #     autonomous_model = initialize_agent(control_mode, load_path, additional_args=additional_args)
        # control_mode = 'mouse' # 'mimic'
        # control_mode = 'mimic'

        # TODO: we should have these come in as parameters
        self.puck_history_len = 5
        self.paddle_history_len = 5
        self.puck_detector = puck_detectors[config.puck_detector]
        self.puck_detector_kwargs = {
            "antiglare_bounds_in_raw_image": config.antiglare_bounds_in_raw_image,
            "antiglare_min_x_px": config.antiglare_min_x_px,
            "antiglare_max_x_px": config.antiglare_max_x_px,
            "antiglare_min_y_px": config.antiglare_min_y_px,
            "antiglare_max_y_px": config.antiglare_max_y_px,
            "center_offset_constant": self.center_offset_constant,
        }
        self.image_path = config.image_path
        self.save_path = config.save_path
        self.tidx = get_trajectory_idx(self.save_path)


        shared_mouse_pos = multiprocessing.Array("f", 3)
        shared_puck_pos = multiprocessing.Array("f", 3)
        shared_paddle_pos = multiprocessing.Array("f", 3)
        shared_target_pos = multiprocessing.Array("f", 3)
        shared_image_check = multiprocessing.Array("f", 1)
        shared_mouse_pos[0] = 0
        shared_mouse_pos[1] = 0
        shared_mouse_pos[2] = 1
        shared_target_pos[2] = 0
        shared_image_check[0] = 0
        self.protected_mouse_pos = ProtectedArray(shared_mouse_pos)
        self.protected_puck_pos = ProtectedArray(shared_puck_pos)
        self.protected_img_check = ProtectedArray(shared_image_check)
        self.protected_paddle_pos = ProtectedArray(shared_paddle_pos)
        self.protected_target_pos = ProtectedArray(shared_target_pos)
        self.cap, self.camera_process, self.mimic_process = None, None, None
        self.async_render_enabled = bool(config.async_render_enabled)
        self.async_render_debug = bool(config.async_render_debug)
        self.async_render_poll_sleep_s = max(0.0005, float(config.async_render_poll_sleep_s))
        self.async_render_window_name = str(config.async_render_window_name)
        self._async_render_runtime_enabled = bool(self.async_render_enabled)
        self._async_render_default_frame_shape = (
            int(config.async_render_frame_height),
            int(config.async_render_frame_width),
            3,
        )
        self._render_shared_mem = None
        self._render_frame_shape = None
        self._render_metadata = None
        self._render_metadata_lock = None
        self._render_seq = None
        self._render_epoch = None
        self._render_stop_event = None
        self._render_process = None
        if self.control_type == "prim":
            self.motion_primitive = MotionPrimitive()

        self.images = list() # image data of the trajectory
        self.vals = list() # proprioceptive data of the trajectory
        # self.num_trajectories = num_trajectories
        self.vel = 0.8 # velocity limit
        self.acc = 0.8 # acceleration limit 
        self.x_convert_offset = config.center_offset_constant # offset to convert positions to centered coordinate frame

        # rmax_x = 0.23
        # rmax_y = 0.12
        # fast limits
        self.rmax_x = config.rmax_x
        self.rmax_y = config.rmax_y
        # self.teleoperation_noise = 0.20 # adds noise to the robot # TODO: make this an input parameter
        self.teleoperation_noise = config.teleoperation_noise # adds noise to the robot # TODO: make this an input parameter

        # safe limits 
        # rmax_x = 0.1
        # rmax_y = 0.05

        # servol control parameters and general frame rate (20Hz)
        self.block_time = config.block_time # time for the robot to reach a position (blocking)
        self.runtime = config.runtime
        if self.control_mode == "mimic":
            self.compute_time = 0.004
        elif self.control_mode == "mouse":
            self.compute_time = 0.002
        elif self.control_mode == "keyboard":
            self.compute_time = 0.025
        # compute_time = 0.004 if control_mode == 'mimic' else 0.002 # TODO: figure out the numbers for learned policies
        self.lookahead = config.lookahead # smooths more with larger values (0.03-0.2)
        self.gain = config.gain # 100-2000
        
        # may need to calibrate angle of end effector
        # angle = [-0.05153677648744038, -2.9847520618606172, 0.]
        self.angle = config.angle

        # if z is used to compute angles
        self.zslope = config.zslope
        self.computez = lambda x: self.zslope * (x + 0.310) - 0.310

        # homography offsets
        self.offset_constants = np.array((2250, 500))
        self.visual_downscale_constant = visual_downscale_constant
        
        # max workspace limits
        self.x_offset = config.x_offset
        self.paddle_additional_x_offset = config.paddle_additional_x_offset
        self.paddle_additional_y_offset = config.paddle_additional_y_offset
        

        # self.x_min_lim = -0.8
        # self.x_max_lim = -0.26

        # y_min = -0.3382
        # y_max = 0.388
        # y_min = -0.3782
        # y_max = 0.360


        # self.y_min = -0.3582
        
        # self.y_min = -0.42
        # self.y_max = 0.42

        # magic numbers representing the boundary
        self.x_min_lim = -0.79
        self.x_max_lim = -0.375
        self.y_min = -0.370 # temporary for right now
        self.y_max = 0.350 

        self.bot_abs = config.bot_abs
        self.top_abs = config.top_abs
        self.max_bias_p = config.max_bias_p
        self.max_bias_m = config.max_bias_m
        self.edge_lims = [self.top_abs, self.bot_abs, self.max_bias_p, self.max_bias_m]

        # y_min = -0.3482
        # y_max = 0.350

        # x_min = -1.5
        # x_max = -0.1
        # y_min = -5
        # y_max = 5

        # x_min = -0.8
        # x_max = -0.4
        # y_min = -0.30
        # y_max = 0.30

        # velocity limits
        self.xv_min = config.xv_min
        self.xv_max = config.xv_max
        # y_min = -0.3382
        # y_max = 0.388
        self.yv_min = config.yv_min
        self.yv_max = config.yv_max


        # robot reset pose
        self.reset_pose_list = np.array([
            [-.340,.050],
            [-.388, -.348],
            [-.692,-.307],
            [-.564,-.196],
            [-.414,-.078],
            [-.330,-.170],
            [-.698,.028],
            [-.572,.176],
            [-.368,.290],
            [-.688,.315]
        ])
        # reset_pose = ([-0.68, 0., 0.34] + angle, vel,acc)
        # TODO: make the reset pose not hardcoded but from the high level environment
        self.reset_pose = (reset_positions[config.reset_pos_setting] + self.angle, self.vel, self.acc)
        # self.reset_pose = ([-0.68, 0., 0.33] + self.angle, self.vel,self.acc) # hitting reset pose
        # self.reset_pose = ([-0.38, 0., 0.33] + self.angle, self.vel,self.acc) # stationary hitting reset pose
        # self.reset_pose = ([-0.78, 0., 0.33] + self.angle, self.vel,self.acc) # hitting reset ahead pose
        # self.reset_pose = ([-0.38, -0.345, 0.33] + self.angle, self.vel,self.acc) # negative regions reset
        self.high_reset_val = 0.38
        self.very_high_reset_val = 0.42
        self.high_reset = False # negative regions
        # self.reset_pose = ([-0.68, 0., self.high_reset_val] + self.angle, self.vel,self.acc)
        self.random_reset = False # negative regions data collection only 
        self.preset_reset = False # negative regions evaluation only
        self.above_table = False # negative regions
        self.reset_idx = 0
        self.control_off = self.control_mode in ["observe"]
        self.lims = (self.x_min_lim, self.x_max_lim, self.y_min, self.y_max)
        self.move_lims = (self.rmax_x, self.rmax_y)

        # smooth_history
        self.hist_len = config.hist_len
        self.camera_index = config.camera_index
        self.wait_for_space_to_start = config.wait_for_space_to_start
        self.debug_control = bool(config.debug_control)
        self.debug_control_every = max(1, int(config.debug_control_every))
        self.transition_hold_steps_on_estop_enter = max(0, int(config.transition_hold_steps_on_estop_enter))
        self.transition_hold_steps_on_estop_clear = max(0, int(config.transition_hold_steps_on_estop_clear))
        self.transition_hold_steps_on_safety_rearm = max(0, int(config.transition_hold_steps_on_safety_rearm))
        self.transition_hold_debug = bool(config.transition_hold_debug)
        self.use_actual_tcp_for_state = bool(config.use_actual_tcp_for_state)
        self.state_prediction_horizon_s = float(config.state_prediction_horizon_s)
        self.state_prediction_blend = float(np.clip(config.state_prediction_blend, 0.0, 1.0))
        self.state_prediction_opposite_dir_brake = max(1.0, float(config.state_prediction_opposite_dir_brake))
        self.disable_prediction_on_estop = bool(config.disable_prediction_on_estop)
        self._actual_tcp_fallback_warned = False
        self._last_observed_xy = None
        self._last_step_timing = {}
        self._protective_stop_prev = False
        self._hold_current_target_after_estop = False
        self._transition_hold_steps_remaining = 0
        self._transition_hold_reason = "none"
        self._command_blocked_prev = False
        self._command_rearm_event = False
        self._rearm_pending = False
        self._rearm_pending_reason = "none"
        self._last_readiness_signature = None


        # creating the ground -- need to only call once! otherwise it can be laggy
        # self.reset(seed)
    
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyReal(**state_dict)

    def _should_debug_control(self):
        timestep = getattr(self, "timestep", 0)
        return self.debug_control and timestep % self.debug_control_every == 0

    @staticmethod
    def _vector_or_default(values, width, default_value=0.0):
        vec = np.asarray(values, dtype=float).reshape(-1)
        out = np.full((int(width),), float(default_value), dtype=float)
        copy_width = min(int(width), int(vec.shape[0]))
        if copy_width > 0:
            out[:copy_width] = vec[:copy_width]
        return out

    def _safe_target_pose_speed(self):
        fallback_pose = self._vector_or_default(getattr(self, "pose", self.reset_pose[0]), 6, default_value=0.0)
        fallback_speed = self._vector_or_default(getattr(self, "speed", np.zeros(6, dtype=float)), 6, default_value=0.0)
        try:
            tcp_target_pose = self._vector_or_default(self.rcv.getTargetTCPPose(), 6, default_value=0.0)
            tcp_target_speed = self._vector_or_default(self.rcv.getTargetTCPSpeed(), 6, default_value=0.0)
            return tcp_target_pose, tcp_target_speed
        except Exception:
            return fallback_pose, fallback_speed

    def robot_command_readiness(self):
        """Return current robot command/step readiness for safety gating."""
        ctrl_connected = True
        rcv_connected = True
        control_program_running = True
        control_program_running_read_ok = True

        ctrl_is_connected = getattr(self.ctrl, "isConnected", None)
        if callable(ctrl_is_connected):
            try:
                ctrl_connected = bool(ctrl_is_connected())
            except Exception:
                ctrl_connected = False

        rcv_is_connected = getattr(self.rcv, "isConnected", None)
        if callable(rcv_is_connected):
            try:
                rcv_connected = bool(rcv_is_connected())
            except Exception:
                rcv_connected = False

        controller_connected = bool(ctrl_connected and rcv_connected)
        ctrl_is_program_running = getattr(self.ctrl, "isProgramRunning", None)
        if controller_connected and callable(ctrl_is_program_running):
            try:
                control_program_running = bool(ctrl_is_program_running())
            except Exception:
                control_program_running = False
                control_program_running_read_ok = False
                controller_connected = False
        elif not controller_connected:
            control_program_running = False

        protective_stop = False
        protective_stop_read_ok = True
        try:
            protective_stop = bool(self.rcv.isProtectiveStopped())
        except Exception:
            protective_stop_read_ok = False
            controller_connected = False

        transition_hold_active = bool(
            self._transition_hold_steps_remaining > 0 or self._hold_current_target_after_estop
        )
        step_ready = bool(
            controller_connected
            and control_program_running
            and protective_stop_read_ok
            and (not protective_stop)
        )
        command_ready = bool(step_ready and (not transition_hold_active) and (not self.control_off))

        if not controller_connected:
            reason = "controller_disconnected"
        elif not control_program_running_read_ok:
            reason = "rtde_program_state_unavailable"
        elif not control_program_running:
            reason = "rtde_program_not_running"
        elif not protective_stop_read_ok:
            reason = "protective_stop_unavailable"
        elif protective_stop:
            reason = "protective_stop"
        elif self.control_off:
            reason = "observe_mode"
        elif transition_hold_active:
            reason = f"transition_hold:{self._transition_hold_reason}"
        else:
            reason = "ready"

        signature = (step_ready, command_ready, reason)
        if signature != self._last_readiness_signature and self._should_debug_control():
            print(
                "[control_gate] "
                f"step_ready={step_ready} command_ready={command_ready} "
                f"controller_connected={controller_connected} "
                f"control_program_running={control_program_running} "
                f"protective_stop={protective_stop} "
                f"transition_hold_active={transition_hold_active} reason={reason}"
            )
        self._last_readiness_signature = signature
        return {
            "controller_connected": bool(controller_connected),
            "control_program_running": bool(control_program_running),
            "control_program_running_read_ok": bool(control_program_running_read_ok),
            "protective_stop": bool(protective_stop),
            "protective_stop_read_ok": bool(protective_stop_read_ok),
            "transition_hold_active": bool(transition_hold_active),
            "step_ready": bool(step_ready),
            "command_ready": bool(command_ready),
            "reason": str(reason),
        }

    def _wait_until_robot_step_ready(self, context: str, poll_s: float = 0.25):
        wait_logged = False
        while True:
            readiness = self.robot_command_readiness()
            if bool(readiness["step_ready"]):
                if wait_logged:
                    print(f"[control_gate] {context}: robot ready; resuming.")
                return readiness
            if (not wait_logged) or self._should_debug_control():
                print(f"[control_gate] {context}: waiting for robot readiness ({readiness['reason']})")
                wait_logged = True
            time.sleep(float(max(0.01, poll_s)))

    def _paddle_display_xy_from_pose(self, pose_xy):
        """Convert robot TCP XY to paddle-center XY for rendering."""
        return np.array(
            (
                float(pose_xy[0]) + self.paddle_additional_x_offset,
                float(pose_xy[1]) + self.paddle_additional_y_offset,
            ),
            dtype=float,
        )

    def _paddle_observation_xy_from_pose(self, pose_xy):
        """Convert robot TCP XY to observation-frame paddle XY."""
        paddle_xy = self._paddle_display_xy_from_pose(pose_xy)
        paddle_xy[0] += self.x_offset
        return paddle_xy

    def _resolve_state_pose_speed(self, tcp_target_pose, tcp_target_speed):
        """Return pose/speed source for observation state."""
        if not self.use_actual_tcp_for_state:
            return np.array(tcp_target_pose, dtype=float), np.array(tcp_target_speed, dtype=float), "tcp_target"
        try:
            actual_tcp_pose = np.array(self.rcv.getActualTCPPose(), dtype=float)
            actual_tcp_speed = np.array(self.rcv.getActualTCPSpeed(), dtype=float)
            return actual_tcp_pose, actual_tcp_speed, "actual_tcp"
        except Exception:
            if not self._actual_tcp_fallback_warned:
                print("[control_debug] Falling back to target TCP pose/speed for state observations.")
                self._actual_tcp_fallback_warned = True
            return np.array(tcp_target_pose, dtype=float), np.array(tcp_target_speed, dtype=float), "tcp_target_fallback"

    def _predict_next_pose_xy(self, pose_for_state, speed_for_state, cmd_pose, dt):
        """Bounded one-step XY prediction toward filtered command target."""
        dt = max(1e-4, float(dt))
        current_xy = np.array(pose_for_state[:2], dtype=float)
        current_vxy = np.array(speed_for_state[:2], dtype=float)
        command_xy = np.array(cmd_pose[:2], dtype=float)
        delta = command_xy - current_xy
        dist = np.linalg.norm(delta)
        if dist < 1e-8:
            predicted_xy = np.array(clip_limits(current_xy[0], current_xy[1], self.lims, self.edge_lims), dtype=float)
            return predicted_xy

        desired_vxy = delta / dt
        vel_cap = max(1e-4, float(self.vel))
        desired_v_norm = np.linalg.norm(desired_vxy)
        if desired_v_norm > vel_cap:
            desired_vxy = desired_vxy * (vel_cap / desired_v_norm)

        dv = desired_vxy - current_vxy
        max_dv = max(1e-4, float(self.acc)) * dt
        if np.dot(current_vxy, delta) < 0.0:
            max_dv *= self.state_prediction_opposite_dir_brake
        dv_norm = np.linalg.norm(dv)
        if dv_norm > max_dv:
            dv = dv * (max_dv / dv_norm)
        predicted_vxy = current_vxy + dv
        predicted_v_norm = np.linalg.norm(predicted_vxy)
        if predicted_v_norm > vel_cap:
            predicted_vxy = predicted_vxy * (vel_cap / predicted_v_norm)

        predicted_xy = current_xy + predicted_vxy * dt
        direction = delta / (dist + 1e-8)
        projected_step = np.dot(predicted_xy - current_xy, direction)
        if projected_step > dist:
            predicted_xy = current_xy + direction * dist
        predicted_xy = np.array(clip_limits(predicted_xy[0], predicted_xy[1], self.lims, self.edge_lims), dtype=float)
        return predicted_xy

    def _anchor_command_target_to_pose(self, anchor_pose):
        """Reset command/filter history so desired TCP target matches current TCP pose."""
        anchor_pose = np.array(anchor_pose, dtype=float).reshape(-1)
        if anchor_pose.shape[0] < 6:
            padded_pose = np.zeros(6, dtype=float)
            padded_pose[: anchor_pose.shape[0]] = anchor_pose
            anchor_pose = padded_pose
        hold_z = self.high_reset_val if self.above_table else 0.30
        hold_cmd_pose = np.array(
            [float(anchor_pose[0]), float(anchor_pose[1]), hold_z] + self.angle,
            dtype=float,
        )
        self.pose_hist = deque(maxlen=self.hist_len)
        self.dpose_hist = deque(maxlen=self.hist_len)
        for _ in range(max(1, int(self.hist_len))):
            self.pose_hist.append(anchor_pose.copy())
            self.dpose_hist.append(hold_cmd_pose.copy())
        self.protected_target_pos[0] = hold_cmd_pose[0]
        self.protected_target_pos[1] = hold_cmd_pose[1]
        self.protected_target_pos[2] = 1
        return hold_cmd_pose

    def begin_transition_hold(self, steps: int, reason: str = "external_transition"):
        """Force temporary no-motion command behavior during control transitions."""
        steps_i = max(0, int(steps))
        if steps_i <= 0:
            return
        self._transition_hold_steps_remaining = max(int(self._transition_hold_steps_remaining), steps_i)
        self._transition_hold_reason = str(reason)
        self._hold_current_target_after_estop = True
        try:
            tcp_target_pose = self.rcv.getTargetTCPPose()
            tcp_target_speed = self.rcv.getTargetTCPSpeed()
            state_pose, _, _ = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
            self._anchor_command_target_to_pose(state_pose)
        except Exception:
            pass
        if self.transition_hold_debug:
            print(
                "[control_transition] "
                f"begin_transition_hold reason={self._transition_hold_reason} "
                f"steps={self._transition_hold_steps_remaining}"
            )

    def start_callbacks(self, **kwargs):
        self.region_info = kwargs["region_info"] if "region_info" in kwargs else None
        self.goal_info = kwargs["goal_info"] if "goal_info" in kwargs else None
        if self.control_mode == 'mouse':
            self.camera_process = multiprocessing.Process(
                target=camera_callback,
                args=(
                    self.protected_mouse_pos,
                    self.protected_img_check,
                    self.protected_puck_pos,
                    self.protected_paddle_pos,
                    self.protected_target_pos,
                    self.region_info,
                    self.goal_info,
                    self.lims,
                    self.edge_lims,
                    self.puck_detector,
                    self.puck_detector_kwargs,
                    self.puck_radius,
                    self.x_offset,
                ),
            )
            self.camera_process.start()
        elif self.control_mode == 'mimic':
            self.mimic_process = multiprocessing.Process(target=mimic_control, args=(self.protected_mouse_pos,))
            self.mimic_process.start()
            self.camera_process = multiprocessing.Process(target=save_callback, args=(self.protected_img_check,))
            self.camera_process.start()
        else:
            self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._async_render_runtime_enabled and self.control_mode not in ["observe"]:
                self._start_async_renderer(self._async_render_default_frame_shape)

    def _render_overlay_inline(self, image, target_xy, puck_state, paddle_xy):
        if image is None:
            return
        draw_target_marker(
            image,
            target_xy,
            offset_constants=self.offset_constants,
            visual_downscale_constant=self.visual_downscale_constant,
        )
        draw_puck_marker_from_state(
            image,
            puck_state,
            self.puck_radius,
            x_offset_for_state=self.center_offset_constant,
            offset_constants=self.offset_constants,
            visual_downscale_constant=self.visual_downscale_constant,
            color=(0, 255, 0),
            require_visible=True,
        )
        draw_paddle_marker(
            image,
            paddle_xy,
            self.paddle_radius,
            offset_constants=self.offset_constants,
            visual_downscale_constant=self.visual_downscale_constant,
            color=(255, 0, 0),
        )
        cv2.imshow(self.async_render_window_name, image)
        cv2.waitKey(1)

    def _stop_async_renderer(self):
        if self._render_stop_event is not None:
            self._render_stop_event.set()
        if self._render_process is not None:
            try:
                self._render_process.join(timeout=1.0)
                if self._render_process.is_alive():
                    self._render_process.terminate()
                    self._render_process.join(timeout=0.5)
            except Exception:
                pass
        self._render_process = None
        self._render_stop_event = None
        self._render_metadata = None
        self._render_metadata_lock = None
        self._render_seq = None
        self._render_epoch = None
        self._render_frame_shape = None
        if self._render_shared_mem is not None:
            try:
                self._render_shared_mem.close()
            except Exception:
                pass
            try:
                self._render_shared_mem.unlink()
            except Exception:
                pass
            self._render_shared_mem = None

    def _start_async_renderer(self, frame_shape):
        if (not self._async_render_runtime_enabled) or self.control_mode in ["observe"]:
            return False
        frame_shape = tuple(int(v) for v in frame_shape)
        if len(frame_shape) != 3 or any(v <= 0 for v in frame_shape):
            return False

        if (
            self._render_process is not None
            and self._render_process.is_alive()
            and self._render_shared_mem is not None
            and self._render_frame_shape == frame_shape
        ):
            return True

        self._stop_async_renderer()
        frame_nbytes = int(np.prod(np.array(frame_shape, dtype=np.int64)) * np.dtype(np.uint8).itemsize)
        try:
            self._render_shared_mem = shared_memory.SharedMemory(create=True, size=frame_nbytes)
            self._render_frame_shape = frame_shape
            self._render_metadata = multiprocessing.Array("d", _ASYNC_RENDER_METADATA_WIDTH, lock=False)
            self._render_metadata_lock = multiprocessing.Lock()
            self._render_seq = multiprocessing.Value("q", -1, lock=False)
            self._render_epoch = multiprocessing.Value("i", 0, lock=False)
            self._render_stop_event = multiprocessing.Event()
            self._render_process = multiprocessing.Process(
                target=_async_render_worker,
                args=(
                    self._render_shared_mem.name,
                    self._render_frame_shape,
                    self._render_metadata,
                    self._render_metadata_lock,
                    self._render_seq,
                    self._render_epoch,
                    self._render_stop_event,
                    self.async_render_window_name,
                    self.puck_radius,
                    self.paddle_radius,
                    self.center_offset_constant,
                    self.offset_constants,
                    self.visual_downscale_constant,
                    self.async_render_poll_sleep_s,
                    self.async_render_debug,
                ),
                daemon=True,
            )
            self._render_process.start()
            return True
        except Exception as exc:
            if self.async_render_debug:
                print(f"[async_render] failed to start worker: {exc}")
            self._stop_async_renderer()
            self._async_render_runtime_enabled = False
            return False

    def _mark_async_render_reset(self):
        if self._render_epoch is None:
            return
        try:
            self._render_epoch.value = int(self._render_epoch.value) + 1
        except Exception:
            pass

    def _publish_async_render_frame(self, image, target_xy, puck_state, paddle_xy):
        if (not self._async_render_runtime_enabled) or image is None:
            return False
        frame = np.asarray(image)
        if frame.ndim != 3:
            return False
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_shape = tuple(int(v) for v in frame.shape)
        if not self._start_async_renderer(frame_shape):
            return False
        if (
            self._render_process is None
            or (not self._render_process.is_alive())
            or self._render_shared_mem is None
            or self._render_metadata is None
            or self._render_metadata_lock is None
            or self._render_seq is None
        ):
            self._async_render_runtime_enabled = False
            return False
        try:
            with self._render_metadata_lock:
                shared_frame = np.ndarray(self._render_frame_shape, dtype=np.uint8, buffer=self._render_shared_mem.buf)
                np.copyto(shared_frame, frame, casting="unsafe")
                self._render_metadata[0] = float(target_xy[0])
                self._render_metadata[1] = float(target_xy[1])
                self._render_metadata[2] = float(puck_state[0])
                self._render_metadata[3] = float(puck_state[1])
                self._render_metadata[4] = float(puck_state[2]) if len(puck_state) > 2 else 1.0
                self._render_metadata[5] = float(paddle_xy[0])
                self._render_metadata[6] = float(paddle_xy[1])
                self._render_seq.value = int(self._render_seq.value) + 1
            return True
        except Exception as exc:
            if self.async_render_debug:
                print(f"[async_render] publish failed: {exc}")
            self._stop_async_renderer()
            self._async_render_runtime_enabled = False
            return False

    def _compute_state(self, pose, speed, i, puck_history):
        # This should be the only place where it is necessary to correct detection by the offsets
        puck = np.array(puck_history[i])[:2] # the i-th most recent position
        # puck[0] += self.x_offset
        self.puck = puck
        self.pose = pose
        self.speed = speed

        state_info = self.get_current_state()

        return state_info

    def get_current_state(self):
        state_info = dict()
        state_info['paddles'] = dict()
        state_info['paddles']['paddle_ego'] = dict()
        state_info['paddles']['paddle_ego']['position'] = self._paddle_observation_xy_from_pose(self.pose[:2])
        state_info['paddles']['paddle_ego']['velocity'] = copy.deepcopy(self.speed[:2])
        state_info['paddles']['paddle_ego']['history'] = self.paddle_history[- self.paddle_history_len :]
        state_info["pucks"] = list()
        state_info["pucks"].append({"history": self.puck_history[- self.puck_history_len:], 
                                    "position": copy.deepcopy(self.puck), 
                                    "velocity": np.array(self.puck_history[-1])[:2] - np.array(self.puck_history[-2])[:2], 
                                    "occluded": np.array(self.puck_history[-1])[-1:]})
        # print("state_info", state_info)
        return state_info


    def take_action(self, action, pose, speed, force, acc, estop, image, images, puck_history, lims, move_lims):
        # converts an action from the agent to an action in the robot space
        if self.puck_detector is not None: 
            puck = self.puck_detector(
                image,
                puck_history,
                rotate=False,
                **self.puck_detector_kwargs,
            )
            puck = np.array(puck)
            # Detector hit (occluded==0) is detector-frame x and needs +center offset.
            # Occlusion fallback (occluded==1) already comes from puck_history/state frame.
            if int(puck[2]) == 0:
                puck[0] += self.center_offset_constant
        else: puck = (puck_history[-1][0],puck_history[-1][1],0)
        puck_vals = np.concatenate( [np.array(puck_history[self.puck_history_len-i]) for i in range(1,self.puck_history_len)] + [np.array(puck)])
        puck_vel = (np.array(puck)[:2] - np.array(puck_history[-self.puck_history_len])[:2])
        paddle_puck_rel = np.array((pose[0] - self.center_offset_constant, pose[1])) - np.array(puck[:2])
        delta_x, delta_y = action
        move_vector = np.array((delta_x,delta_y)) * np.array(move_lims)
        x, y = move_vector + pose[:2]
        
        # x, y = action
        
        # x, y = clip_limits(delta_vector[0], delta_vector[1],lims)
        # print(action, move_vector, delta_x, delta_y, pose[:2],  x,y)
        return x, y, puck
    
    def set_object_links(self):
        # doesn't do anything because naming isn't supported
        return None




    def reset(self, seed, **kwargs):
        # TODO: Consolidate reset motion + force-application into one reusable sequence,
        # and centralize success / protective-stop handling in one place.
        if not self.control_off:
            self._wait_until_robot_step_ready(context="reset:start")
            try:
                print("[control_gate] reset:start servoStop begin")
                self.ctrl.servoStop(6)
                print("[control_gate] reset:start servoStop done")
            except Exception as exc:
                print(f"[control_gate] reset:start servoStop skipped: {exc}")
            try:
                print("[control_gate] reset:start forceModeStop begin")
                self.ctrl.forceModeStop()
                print("[control_gate] reset:start forceModeStop done")
            except Exception as exc:
                print(f"[control_gate] reset:start forceModeStop skipped: {exc}")

        # ---- Trajectory finalization for previous rollout ----
        should_write_traj = "write_traj" in kwargs and kwargs["write_traj"]
        imgs, vals = None, None
        if should_write_traj:
            imgs, vals = merge_trajectory(self.image_path, self.images, self.vals)
        clear_images(folder=self.image_path)
        if should_write_traj and imgs is not None:
            write_trajectory(self.save_path, self.tidx, imgs, vals) # TODO: not necessarily the best place to do writing
            self.tidx += 1

        # ---- Episode-local buffers and state ----
        self.images = list()
        self.vals = list()
        self.timestep = 0
        self.pose_hist, self.dpose_hist = deque(maxlen=self.hist_len), deque(maxlen=self.hist_len)
        self.puck_history = [(-2 + self.center_offset_constant,0,1) for i in range(5)] # pretend that the puck starts at the other end of the table, but is occluded, for 5 frames
        self.paddle_history = [
            (
                -2 + self.center_offset_constant + self.paddle_additional_x_offset,
                self.paddle_additional_y_offset,
                1,
            )
            for i in range(5)
        ]
        self.total = time.time()
        self.runtime = 0.0
        self._last_observed_xy = None
        self._last_step_timing = {}
        self._mark_async_render_reset()

        # TODO: set these with desired values, not yet finished
        self.paddles = dict()
        self.pucks = dict()
        self.blocks = dict()
        self.block_initial_positions = dict()
        self.obstacles = dict()
        self.targets = dict()
        
        self.multiagent = False

        self.paddle_attrs = None
        self.target_attrs = None

        self.object_dict = {}

        def _maybe_assign_random_reset_xy():
            if self.random_reset:
                self.reset_pose[0][0], self.reset_pose[0][1] = (
                    np.random.rand(2) * np.array([self.x_max_lim - self.x_min_lim, self.y_max - self.y_min])
                    + np.array([self.x_min_lim, self.y_min])
                )

        # ---- Optional high-reset pre-stage ----
        if self.high_reset and not self.control_off:
            self._wait_until_robot_step_ready(context="reset:high_pre_stage")
            tcp_target_pose, _ = self._safe_target_pose_speed()
            tcp_target_pose[2] = self.very_high_reset_val
            try:
                high_reset_success = self.ctrl.moveL(
                    tcp_target_pose.tolist(),
                    self.reset_pose[1],
                    self.reset_pose[2],
                    False,
                )
            except Exception as exc:
                high_reset_success = False
                print(f"[control_gate] reset:high_pre_stage moveL skipped: {exc}")
            if self.preset_reset:
                self.reset_pose[0][0], self.reset_pose[0][1] = self.reset_pose_list[self.reset_idx % len(self.reset_pose_list)]
                self.reset_pose[0][2] = self.very_high_reset_val
                self._wait_until_robot_step_ready(context="reset:high_pre_stage_preset")
                try:
                    high_reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
                except Exception as exc:
                    high_reset_success = False
                    print(f"[control_gate] reset:high_pre_stage_preset moveL skipped: {exc}")

        # ---- Main reset move and start gate ----
        with NonBlockingConsole() as nbc:

            # Setting a reset pose for the robot
            if not self.high_reset and not self.control_off:
                _maybe_assign_random_reset_xy()
                self._wait_until_robot_step_ready(context="reset:main_stage")
                try:
                    print("[control_gate] reset:main_stage moveL begin")
                    reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
                    print(f"[control_gate] reset:main_stage moveL done success={reset_success}")
                except Exception as exc:
                    reset_success = False
                    print(f"[control_gate] reset:main_stage moveL skipped: {exc}")
                # Keep force mode engaged so the tool remains biased toward table contact.
                readiness = self.robot_command_readiness()
                if bool(readiness["command_ready"]):
                    try:
                        apply_negative_z_force(self.ctrl, self.rcv)
                    except Exception as exc:
                        print(f"[control_gate] reset:main_stage forceMode skipped: {exc}")
                print("reset to initial pose:", reset_success)
            count = 0
            time.sleep(0.7)
            # wait to start moving unless disabled by config
            if self.wait_for_space_to_start:
                print("Press space to start")
                for j in range(10000):
                    time.sleep(0.01)  # To prevent high CPU usage
                    if nbc.get_data() == ' ':  # x1b is ESC
                        break

        # ---- Final reset target setup and move ----
        self.protected_img_check[0] = 1 and bool(self.save_path)
        _maybe_assign_random_reset_xy()
        if self.preset_reset:
            self.reset_pose[0][0], self.reset_pose[0][1] = self.reset_pose_list[self.reset_idx % len(self.reset_pose_list)]
            print(self.reset_idx, self.reset_pose_list[self.reset_idx % len(self.reset_pose_list)])
            self.reset_idx += 1
        if not self.control_off: 
            self._wait_until_robot_step_ready(context="reset:final_stage")
            try:
                print("[control_gate] reset:final_stage moveL begin")
                reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
                print(f"[control_gate] reset:final_stage moveL done success={reset_success}")
            except Exception as exc:
                reset_success = False
                print(f"[control_gate] reset:final_stage moveL skipped: {exc}")
            print("reset to initial pose:", reset_success)
        time.sleep(0.2)
        if self.high_reset and not self.above_table and not self.control_off:
            readiness = self.robot_command_readiness()
            if bool(readiness["command_ready"]):
                try:
                    apply_negative_z_force(self.ctrl, self.rcv)
                except Exception as exc:
                    print(f"[control_gate] reset:post_stage forceMode skipped: {exc}")
        count = 0
        time.sleep(0.7)

        # TODO: Add explicit post-reset verification (actual TCP z / contact checks + retry policy)
        # in a future behavior-changing pass.
        tcp_target_pose, tcp_target_speed = self._safe_target_pose_speed()
        state_pose, state_speed, _ = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
        self._anchor_command_target_to_pose(state_pose)
        readiness = self.robot_command_readiness()
        self._protective_stop_prev = bool(readiness["protective_stop"])
        self._hold_current_target_after_estop = False
        self._transition_hold_steps_remaining = 0
        self._transition_hold_reason = "none"
        self._command_blocked_prev = False
        self._command_rearm_event = False
        self._rearm_pending = False
        self._rearm_pending_reason = "none"
        state_info = self._compute_state(state_pose, state_speed, -1, self.puck_history)

        print("To exit press 'q'") # TODO: make this actually usable

        return state_info

    def soft_reset(self):
        """Reset episode-local buffers without physical robot movement."""
        self.images = list()
        self.vals = list()
        self.timestep = 0
        self.pose_hist, self.dpose_hist = deque(maxlen=self.hist_len), deque(maxlen=self.hist_len)
        self.puck_history = [(-2 + self.center_offset_constant, 0, 1) for i in range(5)]
        self.paddle_history = [
            (
                -2 + self.center_offset_constant + self.paddle_additional_x_offset,
                self.paddle_additional_y_offset,
                1,
            )
            for i in range(5)
        ]
        self.total = time.time()
        self.runtime = 0.0
        self._last_observed_xy = None
        self._last_step_timing = {}
        self._mark_async_render_reset()

        tcp_target_pose, tcp_target_speed = self._safe_target_pose_speed()
        state_pose, state_speed, _ = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
        self._anchor_command_target_to_pose(state_pose)
        readiness = self.robot_command_readiness()
        self._protective_stop_prev = bool(readiness["protective_stop"])
        self._hold_current_target_after_estop = False
        self._transition_hold_steps_remaining = 0
        self._transition_hold_reason = "none"
        self._command_blocked_prev = False
        self._command_rearm_event = False
        self._rearm_pending = False
        self._rearm_pending_reason = "none"
        state_info = self._compute_state(state_pose, state_speed, -1, self.puck_history)
        return state_info

    def instantiate_objects(self):
        # TODO: put telling the human where to reset physical objects
        # Do this here. Also have option for running automatic recovery
        pass
    
    def get_transition(self, action):
        # TODO: change self.block_time if additional computation happens outside of get_transition
        runtime = time.time() - self.transition_start
        sleep_time = max(0, self.block_time - runtime)
        time.sleep(sleep_time)
        # print("runtime", time.time() - self.total, runtime)
        self.total = time.time()
        self.transition_start = time.time()
        step_start_s = self.transition_start

        # ret, image = cap.read()
        # cv2.imshow('image',image)
        # cv2.setMouseCallback('image', move_event)
        # cv2.waitKey(1)
        pixel_coord = np.array([0, 0])
        if self.control_mode == "mouse":
            pixel_coord[0] = self.protected_mouse_pos[0]
            pixel_coord[1] = self.protected_mouse_pos[1]
        # pixel_coord[2] = protected_mouse_pos[2]
        # print("Consumer Side Pixel Coord: ", pixel_coord)

        readiness = self.robot_command_readiness()
        controller_connected = bool(readiness["controller_connected"])
        control_program_running = bool(readiness.get("control_program_running", True))
        protective_stop = bool(readiness["protective_stop"])

        # force control, need it to keep it on the table
        if not self.above_table and not self.control_off and bool(readiness["command_ready"]):
            try:
                apply_negative_z_force(self.ctrl, self.rcv)
            except Exception as exc:
                controller_connected = False
                print(f"[control_gate] forceMode skipped in get_transition: {exc}")

        # acquire useful statistics
        tcp_target_pose, tcp_target_speed = self._safe_target_pose_speed()
        actual_tcp_force = np.zeros((6,), dtype=float)
        measured_acc = np.zeros((3,), dtype=float)
        if controller_connected:
            try:
                actual_tcp_force = self._vector_or_default(self.rcv.getActualTCPForce(), 6, default_value=0.0)
            except Exception:
                controller_connected = False
            try:
                measured_acc = self._vector_or_default(self.rcv.getActualToolAccelerometer(), 3, default_value=0.0)
            except Exception:
                controller_connected = False
        state_pose, state_speed, state_pose_source = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
        entered_protective_stop = bool(protective_stop and not self._protective_stop_prev)
        cleared_protective_stop = bool((not protective_stop) and self._protective_stop_prev)
        if entered_protective_stop:
            self._hold_current_target_after_estop = True
            self.begin_transition_hold(
                self.transition_hold_steps_on_estop_enter,
                reason="estop_enter",
            )
        if cleared_protective_stop:
            self.begin_transition_hold(
                self.transition_hold_steps_on_estop_clear,
                reason="estop_clear",
            )
        transition_hold_active = bool(self._transition_hold_steps_remaining > 0)
        hold_target_to_current = bool(
            (not controller_connected)
            or (not control_program_running)
            or protective_stop
            or self._hold_current_target_after_estop
            or transition_hold_active
        )
        telemetry_read_s = time.time()
        paddle_display_xy = self._paddle_display_xy_from_pose(state_pose[:2])
        self.protected_paddle_pos[0] = paddle_display_xy[0]
        self.protected_paddle_pos[1] = paddle_display_xy[1]
        self.protected_paddle_pos[2] = self.paddle_radius

        image = None
        camera_frame_received_s = np.nan
        render_publish_started_s = np.nan
        render_publish_done_s = np.nan
        render_publish_ms = 0.0
        render_used_async = False
        # get image data
        if self.cap is not None:
            # Detection runs on this frame immediately after capture; skip optional
            # overlay drawing here to reduce per-step CPU overhead.
            image, save_img, camera_frame_received_s = save_collect(
                self.cap,
                [paddle_display_xy[0], paddle_display_xy[1], self.paddle_radius],
                None,
                None,
                show=False,
                lims=None,
                edge_lims=None,
                region_x_offset=self.x_offset,
            )
            self.images.append(save_img)

        
        if self.control_mode in ["mouse", "mimic"]:
            x, y = (pixel_coord - self.offset_constants) * 0.001
            y= -y
            if self.teleoperation_noise > 0: # add some random normal noise
                noise = np.random.normal(0.0, self.teleoperation_noise, 2)
                x = x + noise[0] * self.rmax_x
                y = y + noise[1] * self.rmax_y
            puck = np.zeros(3)
            puck[0] = self.protected_puck_pos[0] + self.center_offset_constant
            puck[1] = self.protected_puck_pos[1]
            puck[2] = self.protected_puck_pos[2]
            if self.protected_puck_pos[2] == 1: 
                puck[0] = self.puck_history[-1][0]
                puck[1] = self.puck_history[-1][1]
                puck[2] = 1
            # print("puck", puck, self.protected_puck_pos)
            self.puck_history.append(puck)
        elif self.control_mode in ["observe"]:
            x,y, occluded = observe_collect(
                image,
                [paddle_display_xy[0], paddle_display_xy[1], self.paddle_radius],
                self.region_info,
                self.goal_info,
                save_image=True,
            )
            puck = np.array([x,y, occluded])
            state_pose[0] = x
            state_pose[1] = y
            state_pose[2] = occluded
            self.puck_history.append(puck)
        else:
            x,y, puck = self.take_action(action, tcp_target_pose, tcp_target_speed, actual_tcp_force, measured_acc, protective_stop, image, self.images, self.puck_history, self.lims, self.move_lims) # TODO: add image handling
            puck = np.array(puck)
            # print("puck", puck)
            self.puck_history.append(puck)
            srvpose = [[x, y, 0.30] + self.angle, self.vel,self.acc]
        puck_detection_done_s = time.time()
        ###### servoL #####
        requested_target_xy = (x, y)

        if self.control_type == "pol":
            polx, poly = compute_pol(x, y, tcp_target_pose, self.lims, self.move_lims, self.edge_lims)
            srvpose = [[polx, poly, 0.30] + self.angle, self.vel,self.acc]
        elif self.control_type == "rect":
            # x,y = tcp_target_pose[:2] + (np.random.rand(2) * ((np.random.randint(2) - 0.5) * 2)) # uncomment to test random actions
            recx, recy = compute_rect(x, y, tcp_target_pose, self.lims, self.move_lims, self.edge_lims)
            # print(recx - tcp_target_pose[0], recy -tcp_target_pose[1], tcp_target_pose[:2],recx, recy,  x,y)
            if self.above_table :srvpose = [[recx, recy, self.high_reset_val] + self.angle, self.vel,self.acc]
            else: srvpose = [[recx, recy, 0.30] + self.angle, self.vel,self.acc]
        elif self.control_type == "prim":
            x, y = self.motion_primitive.compute_primitive(action, tcp_target_pose, self.lims, self.move_lims, self.edge_lims)
            srvpose = [[x, y, 0.30] + self.angle, self.vel,self.acc]
        
        # TODO: change of direction is currently very sudden, we need to tune that
        # print("servl", srvpose[0][1], tcp_target_speed, actual_tcp_force, measured_acc, ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain))
        
        pre_filter_srvpose = copy.deepcopy(srvpose[0])
        if hold_target_to_current:
            hold_cmd_pose = self._anchor_command_target_to_pose(state_pose)
            requested_target_xy = (float(hold_cmd_pose[0]), float(hold_cmd_pose[1]))
            pre_filter_srvpose = hold_cmd_pose.tolist()
            srvpose[0] = hold_cmd_pose.tolist()
            if (not protective_stop) and self._hold_current_target_after_estop and (not transition_hold_active):
                self._hold_current_target_after_estop = False
            if self._should_debug_control():
                print(
                    "[control_debug] "
                    f"step={self.timestep} estop_target_anchor=True "
                    f"entered_protective_stop={entered_protective_stop} "
                    f"protective_stop={protective_stop} "
                    f"transition_hold_active={transition_hold_active} "
                    f"transition_hold_reason={self._transition_hold_reason} "
                    f"anchor_xy=({hold_cmd_pose[0]:.4f},{hold_cmd_pose[1]:.4f})"
                )
        else:
            self.pose_hist.append(tcp_target_pose)
            self.dpose_hist.append(srvpose[0])
            srvpose[0] = filter_update(tcp_target_speed, self.pose_hist, self.dpose_hist)
            self.protected_target_pos[0] = srvpose[0][0]
            self.protected_target_pos[1] = srvpose[0][1]
            self.protected_target_pos[2] = 1
        if self.cap is not None and self.control_mode not in ["observe"]:
            render_publish_started_s = time.time()
            render_used_async = self._publish_async_render_frame(
                image=image,
                target_xy=srvpose[0][:2],
                puck_state=puck,
                paddle_xy=paddle_display_xy,
            )
            if not render_used_async:
                self._render_overlay_inline(
                    image=image,
                    target_xy=srvpose[0][:2],
                    puck_state=puck,
                    paddle_xy=paddle_display_xy,
                )
            render_publish_done_s = time.time()
            render_publish_ms = max(0.0, (render_publish_done_s - render_publish_started_s) * 1000.0)
        safety_check = False
        if controller_connected and control_program_running:
            try:
                safety_check = bool(self.ctrl.isPoseWithinSafetyLimits(srvpose[0]))
            except Exception:
                controller_connected = False
                safety_check = False
        if self._should_debug_control():
            if self.control_mode in ["mouse", "mimic"]:
                action_repr = "mouse/mimic_absolute_target"
            else:
                action_repr = np.array(action).tolist() if action is not None else None
            print(
                "[control_debug] "
                f"step={self.timestep} mode={self.control_mode} type={self.control_type} "
                f"state_pose_source={state_pose_source} "
                f"tcp_target_pose_xy=({tcp_target_pose[0]:.4f},{tcp_target_pose[1]:.4f}) "
                f"actual_state_pose_xy=({state_pose[0]:.4f},{state_pose[1]:.4f}) "
                f"requested_target_xy=({requested_target_xy[0]:.4f},{requested_target_xy[1]:.4f}) "
                f"pre_filter_target_xy=({pre_filter_srvpose[0]:.4f},{pre_filter_srvpose[1]:.4f}) "
                f"post_filter_target_xy=({srvpose[0][0]:.4f},{srvpose[0][1]:.4f}) "
                f"action={action_repr} "
                f"safety_check={safety_check} protective_stop={protective_stop} "
                f"controller_connected={controller_connected} "
                f"control_program_running={control_program_running} "
                f"transition_hold_active={transition_hold_active}"
            )
        estop_or_disconnect = bool(
            protective_stop or (not controller_connected) or (not control_program_running)
        )
        safety_for_log = bool(
            safety_check
            and controller_connected
            and control_program_running
            and (not protective_stop)
        )
        values = get_state_array(
            time.time(),
            self.tidx,
            self.timestep,
            state_pose,
            state_speed,
            actual_tcp_force,
            measured_acc,
            srvpose,
            estop_or_disconnect,
            safety_for_log,
            puck,
        )
        self.vals.append(values), #frames.append(np.array(protected_img[:]).reshape(640,480,3))

        # print("servl", tcp_target_speed[:2], srvpose[0][:2], x,y, safety_check)# srvpose[0][:2], x,y, tcp_target_pose[:2], rcv.isProtectiveStopped())# , tcp_target_speed, actual_tcp_force, measured_acc, )
        # print("desired_pose", srvpose[0][:2])
        # print("delta desired", np.array(srvpose[0][:2]) - tcp_target_pose[:2])
        # print("unnorm_delta", x- tcp_target_pose[0],y - tcp_target_pose[1], safety_check, self.rcv.isProtectiveStopped())# srvpose[0][:2], x,y, tcp_target_pose[:2], rcv.isProtectiveStopped())# , tcp_target_speed, actual_tcp_force, measured_acc, )
        self._command_rearm_event = False
        if self.control_mode in ["observe"]:
            command_block_reason = "observe_mode"
        elif not controller_connected:
            command_block_reason = "controller_disconnected"
        elif not control_program_running:
            command_block_reason = "rtde_program_not_running"
        elif protective_stop:
            command_block_reason = "protective_stop_active"
        elif transition_hold_active:
            command_block_reason = f"transition_hold:{self._transition_hold_reason}"
        elif not safety_check:
            command_block_reason = "safety_check_failed"
        else:
            command_block_reason = "none"
        command_blocked_now = bool(command_block_reason != "none")
        recovery_block_reason = command_block_reason
        if not command_blocked_now:
            try:
                self.ctrl.servoL(srvpose[0], self.vel, self.acc, self.block_time, self.lookahead, self.gain)
                command_sent_s = time.time()
            except Exception as exc:
                command_blocked_now = True
                command_block_reason = f"servol_exception:{exc.__class__.__name__}"
                recovery_block_reason = command_block_reason
                command_sent_s = np.nan
                controller_connected = False
            if self._should_debug_control():
                print(
                    "[control_debug] "
                    f"step={self.timestep} servoL_sent={not command_blocked_now} vel={self.vel:.3f} "
                    f"acc={self.acc:.3f} block_time={self.block_time:.4f} "
                    f"lookahead={self.lookahead:.3f} gain={self.gain} reason={command_block_reason}"
                )
        else:
            command_sent_s = np.nan
            if self._should_debug_control():
                print(
                    "[control_debug] "
                    f"step={self.timestep} servoL_sent=False "
                    f"reason={command_block_reason}"
                )
        if command_blocked_now and (
            recovery_block_reason in {
                "controller_disconnected",
                "rtde_program_not_running",
                "protective_stop_active",
                "safety_check_failed",
            }
            or str(recovery_block_reason).startswith("servol_exception:")
        ):
            # Only arm safety rearm after genuine command-path recovery events.
            # Internal transition holds already provide their own smoothing and
            # must not recursively qualify as a new recovery.
            self._rearm_pending = True
            self._rearm_pending_reason = str(recovery_block_reason)
        if (not command_blocked_now) and self._rearm_pending:
            self._command_rearm_event = True
            rearm_trigger_reason = self._rearm_pending_reason
            self._rearm_pending = False
            self._rearm_pending_reason = "none"
            self.begin_transition_hold(
                self.transition_hold_steps_on_safety_rearm,
                reason="safety_rearm",
            )
            transition_hold_active = bool(self._transition_hold_steps_remaining > 0)
            if self.transition_hold_debug:
                print(
                    "[control_transition] "
                    f"safety_rearm hold_steps={self._transition_hold_steps_remaining} "
                    f"triggered_by={rearm_trigger_reason}"
                )
        self._command_blocked_prev = command_blocked_now
        # Use telemetry-derived paddle state directly for policy/logging state.
        state_pose_for_observation = np.array(state_pose, dtype=float)
        state_speed_for_observation = np.array(state_speed, dtype=float)
        self._last_observed_xy = np.array(state_pose_for_observation[:2], dtype=float)

        step_end_s = time.time()
        self._last_step_timing = {
            "step_start_s": float(step_start_s),
            "telemetry_read_s": float(telemetry_read_s),
            "puck_detection_done_s": float(puck_detection_done_s),
            "camera_frame_received_s": float(camera_frame_received_s) if np.isfinite(camera_frame_received_s) else float("nan"),
            "render_publish_started_s": float(render_publish_started_s) if np.isfinite(render_publish_started_s) else float("nan"),
            "render_publish_done_s": float(render_publish_done_s) if np.isfinite(render_publish_done_s) else float("nan"),
            "render_publish_ms": float(render_publish_ms),
            "render_used_async": bool(render_used_async),
            "render_async_enabled": bool(self._async_render_runtime_enabled),
            "command_sent_s": float(command_sent_s) if np.isfinite(command_sent_s) else float("nan"),
            "step_end_s": float(step_end_s),
            "sleep_before_step_s": float(sleep_time),
            "loop_runtime_before_sleep_s": float(runtime),
        }
        if (
            self._transition_hold_steps_remaining > 0
            and (not protective_stop)
            and controller_connected
            and control_program_running
        ):
            self._transition_hold_steps_remaining -= 1
            if self._transition_hold_steps_remaining <= 0:
                self._transition_hold_steps_remaining = 0
                self._transition_hold_reason = "none"
                self._hold_current_target_after_estop = False

        puck_occluded = bool(np.asarray(puck).reshape(-1)[2] > 0.5) if np.asarray(puck).size >= 3 else False
        puck_used_fallback = bool(puck_occluded)
        robot_step_ready = bool(controller_connected and control_program_running and (not protective_stop))
        robot_command_ready = bool(
            robot_step_ready and (not transition_hold_active) and (not self.control_mode in ["observe"]) and safety_check
        )

        if protective_stop or (not controller_connected) or (not control_program_running):
            next_state = self._compute_state(state_pose_for_observation, state_speed_for_observation, -1, self.puck_history)
            next_state["timing"] = copy.deepcopy(self._last_step_timing)
            next_state["protective_stop"] = bool(protective_stop)
            next_state["paddle_actual_pose"] = np.array(state_pose, dtype=float)
            next_state["paddle_actual_speed"] = np.array(state_speed, dtype=float)
            next_state["paddle_target_pose_pre_filter"] = np.array(pre_filter_srvpose, dtype=float)
            next_state["paddle_target_pose_post_filter"] = np.array(srvpose[0], dtype=float)
            next_state["puck_detector_used_fallback"] = puck_used_fallback
            next_state["transition_hold_active"] = bool(self._transition_hold_steps_remaining > 0)
            next_state["transition_hold_reason"] = str(self._transition_hold_reason)
            next_state["transition_hold_steps_remaining"] = int(self._transition_hold_steps_remaining)
            next_state["command_rearm_event"] = bool(self._command_rearm_event)
            next_state["controller_connected"] = bool(controller_connected)
            next_state["control_program_running"] = bool(control_program_running)
            next_state["robot_step_ready"] = bool(robot_step_ready)
            next_state["robot_command_ready"] = bool(robot_command_ready)
            next_state["command_block_reason"] = str(command_block_reason)
            paddle_position = next_state["paddles"]["paddle_ego"]["position"]
            self.paddle_history.append(list(paddle_position) + [0])
            if self._should_debug_control():
                print(
                    "[control_debug] "
                    f"step={self.timestep} returned_state_xy=({paddle_position[0]:.4f},{paddle_position[1]:.4f}) "
                    f"state_source=actual_telemetry protective_stop={protective_stop} "
                    f"controller_connected={controller_connected} "
                    f"control_program_running={control_program_running}"
                )
            self._protective_stop_prev = bool(protective_stop)
            return next_state

        # print("servl", np.abs(polx - tcp_target_pose[0]), np.abs(poly - tcp_target_pose[1]), pixel_coord, srvpose[0], rcv.isProtectiveStopped())# , tcp_target_speed, actual_tcp_force, measured_acc, )
        # print("time", time.time() - start)
        self.timestep += 1
        self.runtime = time.time() - self.transition_start
        next_state = self._compute_state(state_pose_for_observation, state_speed_for_observation, -1, self.puck_history)
        next_state["timing"] = copy.deepcopy(self._last_step_timing)
        next_state["protective_stop"] = bool(protective_stop)
        next_state["paddle_actual_pose"] = np.array(state_pose, dtype=float)
        next_state["paddle_actual_speed"] = np.array(state_speed, dtype=float)
        next_state["paddle_target_pose_pre_filter"] = np.array(pre_filter_srvpose, dtype=float)
        next_state["paddle_target_pose_post_filter"] = np.array(srvpose[0], dtype=float)
        next_state["puck_detector_used_fallback"] = puck_used_fallback
        next_state["transition_hold_active"] = bool(self._transition_hold_steps_remaining > 0)
        next_state["transition_hold_reason"] = str(self._transition_hold_reason)
        next_state["transition_hold_steps_remaining"] = int(self._transition_hold_steps_remaining)
        next_state["command_rearm_event"] = bool(self._command_rearm_event)
        next_state["controller_connected"] = bool(controller_connected)
        next_state["control_program_running"] = bool(control_program_running)
        next_state["robot_step_ready"] = bool(robot_step_ready)
        next_state["robot_command_ready"] = bool(robot_command_ready)
        next_state["command_block_reason"] = str(command_block_reason)
        paddle_position = next_state["paddles"]["paddle_ego"]["position"]
        self.paddle_history.append(list(paddle_position) + [0])
        if self._should_debug_control():
            print(
                "[control_debug] "
                f"step={self.timestep} returned_state_xy=({paddle_position[0]:.4f},{paddle_position[1]:.4f}) "
                f"state_source=actual_telemetry protective_stop=False"
            )
        self._protective_stop_prev = bool(protective_stop)
        return next_state

    def close(self):
        self._stop_async_renderer()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        for proc_name in ("camera_process", "mimic_process"):
            proc = getattr(self, proc_name, None)
            if proc is None:
                continue
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)
            except Exception:
                pass
            setattr(self, proc_name, None)
        try:
            cv2.destroyWindow(self.async_render_window_name)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def spawn_puck(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass
    