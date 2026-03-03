import time
from collections import deque
import numpy as np
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

            # The current state prediction algorithm uses true current position
            # and adds a predictive horizon on top
            "use_actual_tcp_for_state": True,
            # "state_prediction_horizon_s": 0.05,
            "state_prediction_horizon_s": 0.05,
            "state_prediction_blend": 0.5, # run regression over a trajectory and see this
            "state_prediction_opposite_dir_brake": 1.5,
            "disable_prediction_on_estop": True,
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
        self.y_min = -0.360 # temporary for right now
        self.y_max = 0.360

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
        self.use_actual_tcp_for_state = bool(config.use_actual_tcp_for_state)
        self.state_prediction_horizon_s = float(config.state_prediction_horizon_s)
        self.state_prediction_blend = float(np.clip(config.state_prediction_blend, 0.0, 1.0))
        self.state_prediction_opposite_dir_brake = max(1.0, float(config.state_prediction_opposite_dir_brake))
        self.disable_prediction_on_estop = bool(config.disable_prediction_on_estop)
        self._actual_tcp_fallback_warned = False
        self._last_observed_xy = None


        # creating the ground -- need to only call once! otherwise it can be laggy
        # self.reset(seed)
    
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyReal(**state_dict)

    def _should_debug_control(self):
        timestep = getattr(self, "timestep", 0)
        return self.debug_control and timestep % self.debug_control_every == 0

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
            self.cap = cv2.VideoCapture(1)

    def _compute_state(self, pose, speed, i, puck_history):
        # This should be the only place where it is necessary to correct detection by the offsets
        puck = np.array(puck_history[i])[:2]
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
        self.ctrl.servoStop(6)
        self.ctrl.forceModeStop()

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
            tcp_target_pose = self.rcv.getTargetTCPPose()
            tcp_target_pose[2] = self.very_high_reset_val
            high_reset_success = self.ctrl.moveL(tcp_target_pose, self.reset_pose[1], self.reset_pose[2], False)
            if self.preset_reset:
                self.reset_pose[0][0], self.reset_pose[0][1] = self.reset_pose_list[self.reset_idx % len(self.reset_pose_list)]
                self.reset_pose[0][2] = self.very_high_reset_val
                high_reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)

        # ---- Main reset move and start gate ----
        with NonBlockingConsole() as nbc:

            # Setting a reset pose for the robot
            if not self.high_reset and not self.control_off:
                _maybe_assign_random_reset_xy()
                reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
                # Keep force mode engaged so the tool remains biased toward table contact.
                apply_negative_z_force(self.ctrl, self.rcv)
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
            reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
            print("reset to initial pose:", reset_success)
        time.sleep(0.2)
        if self.high_reset and not self.above_table and not self.control_off: apply_negative_z_force(self.ctrl, self.rcv)
        count = 0
        time.sleep(0.7)

        # TODO: Add explicit post-reset verification (actual TCP z / contact checks + retry policy)
        # in a future behavior-changing pass.
        tcp_target_pose = self.rcv.getTargetTCPPose()
        tcp_target_speed = self.rcv.getTargetTCPSpeed()
        state_pose, state_speed, _ = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
        state_info = self._compute_state(state_pose, state_speed, 0, self.puck_history) # TODO: not sure if i=0 is correct

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

        tcp_target_pose = self.rcv.getTargetTCPPose()
        tcp_target_speed = self.rcv.getTargetTCPSpeed()
        state_pose, state_speed, _ = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
        state_info = self._compute_state(state_pose, state_speed, 0, self.puck_history)
        return state_info

    def instantiate_objects(self):
        # TODO: put telling the human where to reset physical objects
        # Do this here. Also have option for running automatic recovery
        pass
    
    def get_transition(self, action):
        # TODO: change self.block_time if additional computation happens outside of get_transition
        runtime = time.time() - self.transition_start 
        time.sleep(max(0,self.block_time - runtime))
        # print("runtime", time.time() - self.total, runtime)
        self.total = time.time()
        self.transition_start = time.time()

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

        # force control, need it to keep it on the table
        if not self.above_table and not self.control_off: apply_negative_z_force(self.ctrl, self.rcv)

        
        # acquire useful statistics
        tcp_target_pose = self.rcv.getTargetTCPPose()
        tcp_target_speed = self.rcv.getTargetTCPSpeed()
        actual_tcp_force = self.rcv.getActualTCPForce()
        measured_acc = self.rcv.getActualToolAccelerometer()
        protective_stop = self.rcv.isProtectiveStopped()
        state_pose, state_speed, state_pose_source = self._resolve_state_pose_speed(tcp_target_pose, tcp_target_speed)
        paddle_display_xy = self._paddle_display_xy_from_pose(state_pose[:2])
        self.protected_paddle_pos[0] = paddle_display_xy[0]
        self.protected_paddle_pos[1] = paddle_display_xy[1]
        self.protected_paddle_pos[2] = self.paddle_radius

        image = None
        # get image data
        if self.cap is not None:
            image, save_img = save_collect(
                self.cap,
                [paddle_display_xy[0], paddle_display_xy[1], self.paddle_radius],
                self.region_info if not self.control_mode in ["observe"] else None,
                self.goal_info,
                show=False,
                lims=self.lims,
                edge_lims=self.edge_lims,
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
        self.pose_hist.append(tcp_target_pose)
        self.dpose_hist.append(srvpose[0])
        srvpose[0] = filter_update(tcp_target_speed, self.pose_hist, self.dpose_hist)
        self.protected_target_pos[0] = srvpose[0][0]
        self.protected_target_pos[1] = srvpose[0][1]
        self.protected_target_pos[2] = 1
        if self.cap is not None and self.control_mode not in ["observe"]:
            draw_target_marker(
                image,
                srvpose[0][:2],
                offset_constants=self.offset_constants,
                visual_downscale_constant=self.visual_downscale_constant,
            )
            draw_puck_marker_from_state(
                image,
                puck,
                self.puck_radius,
                x_offset_for_state=self.center_offset_constant,
                offset_constants=self.offset_constants,
                visual_downscale_constant=self.visual_downscale_constant,
                color=(0, 255, 0),
                require_visible=True,
            )
            draw_paddle_marker(
                image,
                paddle_display_xy,
                self.paddle_radius,
                offset_constants=self.offset_constants,
                visual_downscale_constant=self.visual_downscale_constant,
                color=(255, 0, 0),
            )
            cv2.imshow("showdst", image)
            cv2.waitKey(1)
        safety_check = self.ctrl.isPoseWithinSafetyLimits(srvpose[0])
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
                f"safety_check={safety_check} protective_stop={protective_stop}"
            )
        values = get_state_array(time.time(), self.tidx, self.timestep, state_pose, state_speed, actual_tcp_force, measured_acc, srvpose, protective_stop, safety_check, puck)
        self.vals.append(values), #frames.append(np.array(protected_img[:]).reshape(640,480,3))

        # print("servl", tcp_target_speed[:2], srvpose[0][:2], x,y, safety_check)# srvpose[0][:2], x,y, tcp_target_pose[:2], rcv.isProtectiveStopped())# , tcp_target_speed, actual_tcp_force, measured_acc, )
        # print("desired_pose", srvpose[0][:2])
        # print("delta desired", np.array(srvpose[0][:2]) - tcp_target_pose[:2])
        # print("unnorm_delta", x- tcp_target_pose[0],y - tcp_target_pose[1], safety_check, self.rcv.isProtectiveStopped())# srvpose[0][:2], x,y, tcp_target_pose[:2], rcv.isProtectiveStopped())# , tcp_target_speed, actual_tcp_force, measured_acc, )
        if safety_check and self.control_mode not in ["observe"]:
            self.ctrl.servoL(srvpose[0], self.vel, self.acc, self.block_time, self.lookahead, self.gain)
            if self._should_debug_control():
                print(
                    "[control_debug] "
                    f"step={self.timestep} servoL_sent=True vel={self.vel:.3f} acc={self.acc:.3f} "
                    f"block_time={self.block_time:.4f} lookahead={self.lookahead:.3f} gain={self.gain}"
                )
        elif self._should_debug_control():
            print(
                "[control_debug] "
                f"step={self.timestep} servoL_sent=False reason={'safety_check_failed' if not safety_check else 'observe_mode'}"
            )
        prediction_dt = self.state_prediction_horizon_s
        prediction_blend = self.state_prediction_blend
        if not safety_check:
            prediction_blend *= 0.5
        if self.disable_prediction_on_estop and protective_stop:
            prediction_blend = 0.0
        predicted_xy = self._predict_next_pose_xy(state_pose, state_speed, srvpose[0], prediction_dt)
        observed_xy = (1.0 - prediction_blend) * np.array(state_pose[:2], dtype=float) + prediction_blend * predicted_xy
        observed_xy = np.array(clip_limits(observed_xy[0], observed_xy[1], self.lims, self.edge_lims), dtype=float)
        state_pose_for_observation = np.array(state_pose, dtype=float)
        state_pose_for_observation[0] = observed_xy[0]
        state_pose_for_observation[1] = observed_xy[1]
        state_speed_for_observation = np.array(state_speed, dtype=float)

        if self._should_debug_control():
            cmd_xy = np.array(srvpose[0][:2], dtype=float)
            if self._last_observed_xy is None:
                one_step_obs_err = float("nan")
            else:
                one_step_obs_err = float(np.linalg.norm(np.array(state_pose[:2], dtype=float) - self._last_observed_xy))
            print(
                "[control_debug] "
                f"step={self.timestep} prediction_dt={prediction_dt:.4f} blend={prediction_blend:.3f} "
                f"predicted_xy=({predicted_xy[0]:.4f},{predicted_xy[1]:.4f}) "
                f"observed_xy=({observed_xy[0]:.4f},{observed_xy[1]:.4f}) "
                f"actual_xy=({state_pose[0]:.4f},{state_pose[1]:.4f}) "
                f"cmd_xy=({cmd_xy[0]:.4f},{cmd_xy[1]:.4f}) "
                f"actual_to_cmd={np.linalg.norm(np.array(state_pose[:2], dtype=float) - cmd_xy):.4f} "
                f"observed_to_cmd={np.linalg.norm(observed_xy - cmd_xy):.4f} "
                f"prev_observed_to_current_actual={one_step_obs_err:.4f}"
            )
        self._last_observed_xy = np.array(observed_xy, dtype=float)

        if protective_stop:
            next_state = self._compute_state(state_pose_for_observation, state_speed_for_observation, self.timestep, self.puck_history)
            paddle_position = next_state["paddles"]["paddle_ego"]["position"]
            self.paddle_history.append(list(paddle_position) + [0])
            if self._should_debug_control():
                print(
                    "[control_debug] "
                    f"step={self.timestep} returned_state_xy=({paddle_position[0]:.4f},{paddle_position[1]:.4f}) "
                    f"state_source=actual_plus_prediction protective_stop=True"
                )
            return next_state

        # print("servl", np.abs(polx - tcp_target_pose[0]), np.abs(poly - tcp_target_pose[1]), pixel_coord, srvpose[0], rcv.isProtectiveStopped())# , tcp_target_speed, actual_tcp_force, measured_acc, )
        # print("time", time.time() - start)
        self.timestep += 1
        self.runtime = time.time() - self.transition_start
        next_state = self._compute_state(state_pose_for_observation, state_speed_for_observation, self.timestep, self.puck_history)
        paddle_position = next_state["paddles"]["paddle_ego"]["position"]
        self.paddle_history.append(list(paddle_position) + [0])
        if self._should_debug_control():
            print(
                "[control_debug] "
                f"step={self.timestep} returned_state_xy=({paddle_position[0]:.4f},{paddle_position[1]:.4f}) "
                f"state_source=actual_plus_prediction protective_stop=False"
            )
        return next_state

    def spawn_puck(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass
    