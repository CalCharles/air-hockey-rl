import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from collections import deque
import numpy as np
from .real.multiprocessing import ProtectedArray, NonBlockingConsole
from .real.control_parameters import camera_callback, save_callback, mimic_control, save_collect
from .real.trajectory_merging import merge_trajectory, clear_images, write_trajectory, get_trajectory_idx
from .real.robot_control import MotionPrimitive, apply_negative_z_force, filter_update
from .real.coordinate_transform import compute_rect, compute_pol
from .real.proprioceptive_state import get_state_array
from .real.image_detection import find_red_hockey_puck
import multiprocessing
import cv2


class AirHockeyReal:
    def __init__(self,
                 absorb_target, 
                 length, 
                 width,
                 puck_radius, 
                 paddle_radius, 
                 block_width,
                 max_force_timestep, 
                 force_scaling, 
                 paddle_damping, 
                 puck_damping,
                 render_size,
                 seed,
                 render_masks=False, 
                 gravity=-5,
                 paddle_density=1000,
                 puck_density=250,
                 block_density=1000,
                 max_paddle_vel=2,
                 time_frequency=20):
        # physics / world params
        # TODO: special config for real
        self.length, self.width = length, width
        self.paddle_radius = paddle_radius
        self.puck_radius = puck_radius
        self.block_width = block_width
        self.max_force_timestep = max_force_timestep
        self.time_frequency = time_frequency
        self.time_per_step = 1 / self.time_frequency
        self.force_scaling = force_scaling
        self.absorb_target = absorb_target
        self.paddle_damping = paddle_damping
        self.puck_damping = puck_damping
        self.gravity = gravity
        self.puck_min_height = (-length / 2) + (length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = width
        self.min_speed_start = -width
        self.paddle_density = paddle_density
        self.puck_density = puck_density
        self.block_density = block_density
        # these assume 2d, in 3d since we have height it would be higher mass
        self.paddle_mass = self.paddle_density * np.pi * self.paddle_radius ** 2
        self.puck_mass = self.puck_density * np.pi * self.puck_radius ** 2

        # these 2 will depend on the other parameters
        self.max_paddle_vel = max_paddle_vel # m/s. This will be dependent on the robot arm
        # compute maximum force based on max paddle velocity
        max_a = self.max_paddle_vel / self.time_per_step
        max_f = self.paddle_mass * max_a
        # assume maximum force transfer
        puck_max_a = max_f / self.puck_mass
        self.max_puck_vel = puck_max_a * self.time_per_step

        # box2d visualization params (but the visualization is done in the Render file)
        self.ppm = render_size / self.width
        self.render_width = int(render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = render_masks

        self.table_x_min = -self.width / 2
        self.table_x_max = self.width / 2
        self.table_y_min = -self.length / 2
        self.table_y_max = self.length / 2

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

        self.metadata = {}

        self.transition_start = time.time()
        rtde_frequency = 500.0
        self.control_mode = 'mouse' # mouse, mimic, keyboard, RL, BC, IQL, rnet, reach
        self.control_type = 'rect' # rect, pol or prim
        # input modes: state force_acc puck_vals goal goal_vel
        # algo options: iql, ppo
        self.additional_args = {"image_input": False, "frame_stack": 1, "algo": "iql", "goal_type": "goal_vel", "input_mode": "puck_vals",
                        "normalize": True} # Goal conditoned args


        self.ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
        self.rcv = RTDEReceive("172.22.22.2")

        teleoperation_modes = ['mouse', 'mimic', 'keyboard']
        autonomous_modes = ['BC', 'RL', 'IQL', 'rnet', 'reach']
        autonomous_model = None
        # if control_mode in autonomous_modes:
        #     autonomous_model = initialize_agent(control_mode, load_path, additional_args=additional_args)
        # control_mode = 'mouse' # 'mimic'
        # control_mode = 'mimic'

        # TODO: we should have these come in as parameters
        self.puck_history_len = 5
        self.puck_detector = find_red_hockey_puck
        self.image_path = "./temp/images/"
        self.save_path = "./data/mouse/expert_avoid_fixed_start_goal"
        self.tidx = get_trajectory_idx(self.save_path)


        shared_mouse_pos = multiprocessing.Array("f", 3)
        shared_paddle_pos = multiprocessing.Array("f", 3)
        shared_image_check = multiprocessing.Array("f", 1)
        shared_mouse_pos[0] = 0
        shared_mouse_pos[1] = 0
        shared_mouse_pos[2] = 1
        shared_image_check[0] = 0
        self.protected_mouse_pos = ProtectedArray(shared_mouse_pos)
        self.protected_img_check = ProtectedArray(shared_image_check)
        self.protected_paddle_pos = ProtectedArray(shared_paddle_pos)
        self.cap, self.camera_process, self.mimic_process = None, None, None
        if self.control_type == "prim":
            self.motion_primitive = MotionPrimitive()

        self.images = list() # image data of the trajectory
        self.vals = list() # proprioceptive data of the trajectory
        # self.num_trajectories = num_trajectories
        self.vel = 0.8 # velocity limit
        self.acc = 0.8 # acceleration limit 

        # rmax_x = 0.23
        # rmax_y = 0.12
        # fast limits
        self.rmax_x = 0.26
        self.rmax_y = 0.12

        # safe limits 
        # rmax_x = 0.1
        # rmax_y = 0.05

        # servol control parameters and general frame rate (20Hz)
        self.block_time = 0.049 # time for the robot to reach a position (blocking)
        self.runtime = 0
        if self.control_mode == "mimic":
            self.compute_time = 0.004
        elif self.control_mode == "mouse":
            self.compute_time = 0.002
        elif self.control_mode == "keyboard":
            self.compute_time = 0.025
        # compute_time = 0.004 if control_mode == 'mimic' else 0.002 # TODO: figure out the numbers for learned policies
        self.lookahead = 0.2 # smooths more with larger values (0.03-0.2)
        self.gain = 700 # 100-2000
        
        # may need to calibrate angle of end effector
        # angle = [-0.05153677648744038, -2.9847520618606172, 0.]
        self.angle = [-0.00153677648744038, -3.0647520618606172, 0.]

        # if z is used to compute angles
        self.zslope = 0.02577
        self.computez = lambda x: self.zslope * (x + 0.310) - 0.310

        # homography offsets
        self.offset_constants = np.array((2100, 500))
        
        # max workspace limits
        self.x_offset = 1
        self.x_min_lim = -0.8
        self.x_max_lim = -0.33
        # y_min = -0.3382
        # y_max = 0.388
        # y_min = -0.3782
        # y_max = 0.360
        self.y_min = -0.3582
        self.y_max = 0.350
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
        self.xv_min = -0.5
        self.xv_max = 0.5
        # y_min = -0.3382
        # y_max = 0.388
        self.yv_min = -0.3
        self.yv_max = 0.3


        # robot reset pose
        # reset_pose = ([-0.68, 0., 0.34] + angle, vel,acc)
        # TODO: make the reset pose not hardcoded but from the high level environment
        # self.reset_pose = ([-0.68, 0., 0.33] + self.angle, self.vel,self.acc)
        self.reset_pose = ([-0.38, -0.345, 0.33] + self.angle, self.vel,self.acc)
        self.lims = (self.x_min_lim, self.x_max_lim, self.y_min, self.y_max)
        self.move_lims = (self.rmax_x, self.rmax_y)

        # smooth_history
        self.hist_len = 2


        # creating the ground -- need to only call once! otherwise it can be laggy
        # self.reset(seed)
    
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyReal(**state_dict)


    def start_callbacks(self, **kwargs):
        if self.control_mode == 'mouse':
            region_info = kwargs["region_info"] if "region_info" in kwargs else None
            goal_info = kwargs["goal_info"] if "goal_info" in kwargs else None
            self.camera_process = multiprocessing.Process(target=camera_callback, args=(self.protected_mouse_pos,self.protected_img_check, self.protected_paddle_pos, region_info, goal_info))
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
        puck[0] += self.x_offset
        self.puck = puck
        self.pose = pose
        self.speed = speed

        state_info = self.get_current_state()

        return state_info

    def get_current_state(self):
        state_info = dict()
        state_info['paddles'] = dict()
        state_info['paddles']['paddle_ego'] = dict()
        state_info['paddles']['paddle_ego']['position'] = self.pose[:2]
        state_info['paddles']['paddle_ego']['position'][0] += self.x_offset
        state_info['paddles']['paddle_ego']['velocity'] = self.speed[:2]
        state_info["pucks"] = list()
        state_info["pucks"].append({"history": self.puck_history[- self.puck_history_len:], 
                                    "position": self.puck, 
                                    "velocity": np.array(self.puck_history[-1])[:2] - np.array(self.puck_history[-2])[:2], 
                                    "occluded": np.array(self.puck_history[-1])[-1:]})
        return state_info


    def take_action(self, action, pose, speed, force, acc, estop, image, images, puck_history, lims, move_lims):
        # converts an action from the agent to an action in the robot space
        if self.puck_detector is not None: puck = self.puck_detector(image, puck_history)
        else: puck = (puck_history[-1][0],puck_history[-1][1],0)
        puck_vals = np.concatenate( [np.array(puck_history[self.puck_history_len-i]) for i in range(1,self.puck_history_len)] + [np.array(puck)])
        puck_vel = (np.array(puck)[:2] - np.array(puck_history[-self.puck_history_len])[:2])
        paddle_puck_rel = np.array(pose[:2]) - np.array(puck[:2])
        delta_x, delta_y = action
        move_vector = np.array((delta_x,delta_y)) * np.array(move_lims)
        x, y = move_vector + pose[:2]
        # x, y = clip_limits(delta_vector[0], delta_vector[1],lims)
        print(action, move_vector, delta_x, delta_y, pose[:2],  x,y)
        return x, y, puck




    def reset(self, seed, **kwargs):
        self.ctrl.servoStop(6)
        self.ctrl.forceModeStop()
        print("write_traj" in kwargs)
        if "write_traj" in kwargs:
            print( kwargs["write_traj"])
        if "write_traj" in kwargs and kwargs["write_traj"]: imgs, vals = merge_trajectory(self.image_path, self.images, self.vals)
        clear_images(folder=self.image_path)
        if "write_traj" in kwargs and kwargs["write_traj"] and imgs is not None: write_trajectory(self.save_path, self.tidx, imgs, vals) # TODO: not necessarily the best place to do writing
        self.images = list()
        self.vals = list()
        self.timestep = 0
        self.pose_hist, self.dpose_hist = deque(maxlen=self.hist_len), deque(maxlen=self.hist_len)
        self.puck_history = [(-1,0,0) for i in range(5)] # pretend that the puck starts at the other end of the table, but is occluded, for 5 frames
        self.total = time.time()
        self.runtime = 0.0

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
        with NonBlockingConsole() as nbc:

            # Setting a reset pose for the robot
            reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
            apply_negative_z_force(self.ctrl, self.rcv)
            print("reset to initial pose:", reset_success)
            count = 0
            time.sleep(0.7)
            # wait to start moving
            print("Press space to start")
            for j in range(10000):
                time.sleep(0.01)  # To prevent high CPU usage
                if nbc.get_data() == ' ':  # x1b is ESC
                    break
        self.protected_img_check[0] = 1 and bool(self.save_path)
        time.sleep(0.1)

        reset_success = self.ctrl.moveL(self.reset_pose[0], self.reset_pose[1], self.reset_pose[2], False)
        print("reset to initial pose:", reset_success)
        count = 0
        time.sleep(0.7)
        true_pose = self.rcv.getTargetTCPPose()
        true_speed = self.rcv.getTargetTCPSpeed()
        state_info = self._compute_state(true_pose, true_speed, 0, self.puck_history) # TODO: not sure if i=0 is correct

        print("To exit press 'q'")

        return state_info

    def instantiate_objects(self):
        # TODO: put telling the human where to reset physical objects
        # Do this here. Also have option for running automatic recovery
        pass
    
    def get_transition(self, action):
        # TODO: change self.block_time if additional computation happens outside of get_transition
        runtime = time.time() - self.transition_start 
        time.sleep(max(0,self.block_time - runtime))
        print(time.time() - self.total, runtime)
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
        apply_negative_z_force(self.ctrl, self.rcv)

        # get image data
        if self.cap is not None:
            image, save_img = save_collect(self.cap)
            self.images.append(save_img)
        
        # acquire useful statistics
        true_pose = self.rcv.getTargetTCPPose()
        true_speed = self.rcv.getTargetTCPSpeed()
        true_force = self.rcv.getActualTCPForce()
        measured_acc = self.rcv.getActualToolAccelerometer()
        self.protected_paddle_pos[0] = true_pose[0]
        self.protected_paddle_pos[1] = true_pose[1]
        self.protected_paddle_pos[2] = self.paddle_radius

        
        if self.control_mode in ["mouse", "mimic"]:
            x, y = (pixel_coord - self.offset_constants) * 0.001
            y= -y
            self.puck_history.append((-2,0,0))
        else:
            x,y, puck = self.take_action(action, true_pose, true_speed, true_force, measured_acc, self.rcv.isProtectiveStopped(), image, self.images, self.puck_history, self.lims, self.move_lims) # TODO: add image handling
            print("puck", puck)
            self.puck_history.append(puck)
            srvpose = [[x, y, 0.30] + self.angle, self.vel,self.acc]
        ###### servoL #####
        if self.control_type == "pol":
            polx, poly = compute_pol(x, y, true_pose, self.lims, self.move_lims)
            srvpose = [[polx, poly, 0.30] + self.angle, self.vel,self.acc]
        elif self.control_type == "rect":
            # x,y = true_pose[:2] + (np.random.rand(2) * ((np.random.randint(2) - 0.5) * 2)) # uncomment to test random actions
            recx, recy = compute_rect(x, y, true_pose, self.lims, self.move_lims)
            # print(recx - true_pose[0], recy -true_pose[1], true_pose[:2],recx, recy,  x,y)
            srvpose = [[recx, recy, 0.30] + self.angle, self.vel,self.acc]
        elif self.control_type == "prim":
            x, y = self.motion_primitive.compute_primitive(action, true_pose, self.lims, self.move_lims)
            srvpose = [[x, y, 0.30] + self.angle, self.vel,self.acc]
        
        # TODO: change of direction is currently very sudden, we need to tune that
        # print("servl", srvpose[0][1], true_speed, true_force, measured_acc, ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain))
        self.pose_hist.append(true_pose)
        self.dpose_hist.append(srvpose[0])
        srvpose[0] = filter_update(true_speed, self.pose_hist, self.dpose_hist)
        safety_check = self.ctrl.isPoseWithinSafetyLimits(srvpose[0])
        values = get_state_array(time.time(), self.tidx, self.timestep, true_pose, true_speed, true_force, measured_acc, srvpose, self.rcv.isProtectiveStopped(), safety_check)
        self.vals.append(values), #frames.append(np.array(protected_img[:]).reshape(640,480,3))

        print("servl", true_speed[:2], srvpose[0][:2], x,y, safety_check)# srvpose[0][:2], x,y, true_pose[:2], rcv.isProtectiveStopped())# , true_speed, true_force, measured_acc, )
        if safety_check: self.ctrl.servoL(srvpose[0], self.vel, self.acc, self.block_time, self.lookahead, self.gain)
        if self.rcv.isProtectiveStopped():
            return None

        # print("servl", np.abs(polx - true_pose[0]), np.abs(poly - true_pose[1]), pixel_coord, srvpose[0], rcv.isProtectiveStopped())# , true_speed, true_force, measured_acc, )
        # print("time", time.time() - start)
        self.timestep += 1
        self.runtime = time.time() - self.transition_start
        return self._compute_state(srvpose[0], true_speed, self.timestep, self.puck_history) # TODO: populate this with the names of objects

    def spawn_puck(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass
    