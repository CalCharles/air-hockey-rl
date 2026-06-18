from types import SimpleNamespace
import numpy as np
import math
# Compatibility fix for newer Robosuite versions
try:
    from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
except ImportError:
    # Fallback to ManipulationEnv for newer Robosuite versions
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv as SingleArmEnv
from .airhockey_sim import AirHockeySim
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjmod import DynamicsModder
from robosuite.utils.mjcf_utils import xml_path_completion as robosuite_xml_path_completion
from robosuite.robots import ROBOT_CLASS_MAPPING
import yaml
import xmltodict
import time
import datetime
from collections import namedtuple
from robosuite.utils.control_utils import trans
import inspect
import os

import numpy as np
from robosuite.models.arenas import Arena
from airhockey.sims.utils import custom_xml_path_completion
from ..utils import dict_to_namespace

class AirHockeyRobosuite(AirHockeySim):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (puck) information in
            the observation.
        # Get robot prefix and define observables modality

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(self, **kwargs):
        # breakpoint()
        defaults = {
            'action_x_scaling': 1.0,
            'action_y_scaling': 1.0,
            'render_masks': False,
            'gravity': -5,
            'paddle_density': 1000,
            'puck_density': 250,
            'block_density': 1000,
            'max_paddle_vel': 2,
            'time_frequency': 20,
            'paddle_bounds': [],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2,
            'robots': ['UR5e'],  # Use standard UR5e instead of AirHockeyUR5e
            'env_configuration': "default",
            'controller_configs': {'arm': 'OSC_POSE'},  # Use OSC controller for position-based control
            'gripper_types': None,  # Disable gripper to avoid joint name conflicts
            'initialization_noise': "default",
            'table_friction': (1.0, 5e-3, 1e-4),
            'use_camera_obs': True,
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'render_camera': "frontview",
            'render_collision_mesh': False,
            'render_visual_mesh': True,
            'render_gpu_device_id': -1,
            'control_freq': 20,
            'step_frequency': 20,
            'horizon': 400,
            'ignore_done': False,
            'hard_reset': True,
            'camera_names': ["birdview", "sideview"],
            'camera_heights': 512,
            'camera_widths': 512,
            'camera_depths': False,
            'camera_segmentations': None,  # {None, instance, class, element}
            'renderer': "mujoco",
            'renderer_config': None,
            'task': "JUGGLE_PUCK",
            'table_xml': "arenas/air_hockey_table.xml",  # relative to assets dir
            # 'table_xml': "arenas/air_hockey_table_robola_compile.xml",  # relative to assets dir
            'puck_radius': 0.03165,
            'puck_damping': 0.01,
            'puck_density': 30,
            'seed': 0,
            'absorb_target': False,
            'force_scaling': 1000,
            'paddle_damping': 1,
            'paddle_density': 1,
            'block_density': 1,
            'gravity': 1,
            'max_force_timestep': 1,
            'paddle_bounds': [],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2,
            'depth': 0.0505,  # Table depth from XML comment          # TODO: was 0.0505
            'table_elevation': 0.0,  # Table elevation          # TODO: changed just for the love of the game - was 0.0
            'table_tilt': 0.0,  # Table tilt angle
            'rim_width': 0.05,  # Rim width for table offsets   # TODO:L just for the love of the game - was 0.05
            'max_puck_vel': 10.0,  # Maximum puck velocity
            'max_paddle_vel': 2.0,  # Maximum paddle velocity
            'time_frequency': 20,  # Time frequency
            'render_size': 360,  # Render size
            
            # OSC Controller Parameters
            'osc_kp': 150,  # Position gain
            'osc_damping_ratio': 1,  # Damping ratio
            'osc_input_max': 1,  # Maximum input value
            'osc_input_min': -1,  # Minimum input value
            'osc_output_max_pos': [0.05, 0.05, 0.05],  # Max position output [x, y, z]
            'osc_output_min_pos': [-0.05, -0.05, -0.05],  # Min position output [x, y, z]
            'osc_output_max_ori': [0.5, 0.5, 0.5],  # Max orientation output [roll, pitch, yaw]
            'osc_output_min_ori': [-0.5, -0.5, -0.5],  # Min orientation output [roll, pitch, yaw]
            'osc_uncouple_pos_ori': True,  # Uncouple position and orientation
            'osc_input_type': 'delta',  # Input type (delta or absolute)
            'osc_input_ref_frame': 'base',  # Reference frame
            'osc_ramp_ratio': 0.2,  # Ramp ratio for smooth transitions
            'osc_impedance_mode': 'fixed'  # Impedance mode
        }

        kwargs = {**defaults, **kwargs}
        config = dict_to_namespace(kwargs)
        # settings for table top
        table_full_size = (config.length / 2, config.width / 2, config.depth / 2)
        self.table_full_size = table_full_size
        self.table_friction = config.table_friction
        self.table_offset = np.array((0, 0, config.table_elevation))
        
        self.length = config.length
        self.width = config.width
        self.ppm = config.render_size / self.width
        self.render_width = int(config.render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = False

        self.gripper_types = config.gripper_types

        self.table_tilt = config.table_tilt
        self.table_elevation = config.table_elevation
        self.table_depth = config.depth
        self.x_to_x_prime_ratio = math.cos(self.table_tilt)
        self.x_prime_to_x_ratio = 1 / self.x_to_x_prime_ratio
        self.x_to_z_ratio = math.sin(self.table_tilt)
        self.transform_z = lambda x: self.x_to_z_ratio * x + self.table_elevation - config.depth
        self.transform_x = lambda x: self.x_to_x_prime_ratio * x
        self.inverse_transform_x = lambda x: self.x_prime_to_x_ratio * x
        
        self.high_level_table_x_top = -self.length / 2
        self.high_level_table_x_bot = self.length / 2
        self.high_level_table_y_right = self.width / 2
        self.high_level_table_y_left = -self.width / 2
        self.center_offset_constant = config.center_offset_constant
        
        self.table_x_offset = 2 * config.rim_width
        self.table_y_offset = 2 * config.rim_width
        
        # where the playable area starts
        self.table_x_top = self.length - self.table_x_offset
        self.table_x_bot = self.table_x_offset
        self.table_y_right = -self.width / 2 + self.table_y_offset
        self.table_y_left = self.width / 2 - self.table_y_offset

        self.table_q = T.axisangle2quat(np.array([0, self.table_tilt, 0]))
        self.table_transform = T.quat2mat(self.table_q)
        self.inv_table_transform = np.linalg.inv(self.table_transform)

        self.initial_puck_vels = dict()
        self.initial_block_positions = dict()
        self.table_xml = config.table_xml

        self.puck_radius = config.puck_radius
        self.puck_damping = config.puck_damping
        self.puck_density = config.puck_density
        self.puck_height = 0.009
        self.puck_z_offset = math.sin(self.table_tilt) * self.puck_radius
        self.action_x_scaling = config.action_x_scaling
        self.action_y_scaling = config.action_y_scaling

        # FIXME make these parameters do something, right now it's a placeholder to make calls to robosuite work
        self.seed = config.seed
        self.paddle_radius = config.paddle_radius
        self.block_width = config.block_width
        self.max_paddle_vel = config.max_paddle_vel
        self.max_puck_vel = config.max_puck_vel
        self.control_freq = config.control_freq
        self.step_frequency = config.step_frequency
        self.last_action = np.zeros(2)
        
        self.robosuite_env = None
        self.robosuite_env_cfg = {'robots': config.robots, 'env_configuration': config.env_configuration, 
                              'controller_configs': self._build_controller_config(config),  # Build custom OSC controller config
                              'base_types': "default", 'gripper_types': config.gripper_types, 'initialization_noise': config.initialization_noise,
                              'use_camera_obs': config.use_camera_obs, 'has_renderer': config.has_renderer, 'has_offscreen_renderer': config.has_offscreen_renderer,
                              'render_camera': config.render_camera, 'render_collision_mesh': config.render_collision_mesh, 'render_visual_mesh': config.render_visual_mesh,
                              'render_gpu_device_id': config.render_gpu_device_id, 'control_freq': config.control_freq, 'horizon': config.horizon, 'ignore_done': config.ignore_done,
                              'hard_reset': config.hard_reset, 'camera_names': config.camera_names, 'camera_heights': config.camera_heights, 'camera_widths': config.camera_widths,
                              'camera_depths': config.camera_depths, 'camera_segmentations': config.camera_segmentations, 'renderer': config.renderer, 'renderer_config': config.renderer_config}
        
        self.initialized_objects = False
        current_time = datetime.datetime.fromtimestamp(time.time())
        # formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
        formatted_time = np.random.randint(1000000000000000000)
        # self.tmp_xml_fp = robosuite_xml_path_completion(self.table_xml + f"_{formatted_time}.xml")
        # breakpoint()
        self.tmp_xml_fp = robosuite_xml_path_completion(self.table_xml)


        
    def _build_controller_config(self, config):
        """
        Build OSC controller configuration from config parameters.
        
        Args:
            config: Configuration namespace with OSC parameters
            
        Returns:
            dict: Controller configuration compatible with Robosuite
        """
        # Build OSC controller configuration for the right arm
        osc_config = {
            'type': 'OSC_POSE',
            'input_max': config.osc_input_max,
            'input_min': config.osc_input_min,
            'output_max': config.osc_output_max_pos + config.osc_output_max_ori,
            'output_min': config.osc_output_min_pos + config.osc_output_min_ori,
            'kp': config.osc_kp,
            'damping_ratio': config.osc_damping_ratio,
            'impedance_mode': config.osc_impedance_mode,
            'kp_limits': [0, 300],  # Standard limits
            'damping_ratio_limits': [0, 10],  # Standard limits
            'position_limits': None,
            'orientation_limits': None,
            'uncouple_pos_ori': config.osc_uncouple_pos_ori,
            'input_type': config.osc_input_type,
            'input_ref_frame': config.osc_input_ref_frame,
            'interpolation': None,
            'ramp_ratio': config.osc_ramp_ratio,
            'gripper': {'type': 'GRIP'} if config.gripper_types is not None else None
        }
        
        # Build the complete controller configuration
        controller_config = {
            'type': 'BASIC',
            'body_parts': {
                'right': osc_config
            }
        }
        
        return controller_config
        
    def __del__(self):
        if self.robosuite_env is not None:
            self.robosuite_env.close()
        if self.initialized_objects and os.path.exists(self.tmp_xml_fp):
            os.remove(self.tmp_xml_fp)

    @staticmethod
    def from_dict(state_dict):
        # create a dictionary of only the relevant parameters
        return AirHockeyRobosuite(**state_dict)

    def get_contacts(self):
        return self.robosuite_env.get_contacts() if self.robosuite_env is not None else None

    def start_callbacks(self, **kwargs):
        return

    def reset(self, seed=None, **kwargs):
        if self.robosuite_env is not None:
            self.robosuite_env.reset()
        
        self.timestep = 0
        self.puck_history = [(-2 + self.center_offset_constant,0,1) for i in range(5)]
        self.paddle_history = [(-2 + self.center_offset_constant,0,1) for i in range(5)]
        self.last_action = np.zeros(2)

        if not self.initialized_objects:
            self.puck_names = {}
            self.block_names = {}
            self.initial_obj_configurations = {'paddles': {}, 'pucks': {}, 'blocks': {}}
            
            # Load and configure XML
            xml_fp = custom_xml_path_completion(self.table_xml)
            with open(xml_fp, "r") as file:
                self.xml_config = xmltodict.parse(file.read())

            # update table config
            # assert self.xml_config['mujoco']['worldbody']['body']['@name'] == 'table'
            # self.xml_config['mujoco']['worldbody']['body']['@pos'] = f"{self.table_full_size[0]} 0 {self.table_elevation}"

            # # update table surface config
            # table_surface_idx = None
            # for i, body in enumerate(self.xml_config['mujoco']['worldbody']['body']['body']):
            #     if body['@name'] == 'table_surface':
            #         table_surface_idx = i
            #         break
            # self.xml_config['mujoco']['worldbody']['body']['body'][table_surface_idx]['geom']['@size'] = f"{self.table_full_size[0]} {self.table_full_size[1]} {self.table_full_size[2]}"


            # TODO: make it such that we use the table from our XML not override it like above ^^^
            # READ table geometry from XML instead of overwriting it with YAML values.
            # The XML is the ground truth for the 3D sim geometry.
            table_body = self.xml_config['mujoco']['worldbody']['body']
            assert table_body['@name'] == 'table', f"Expected 'table' body, got {table_body['@name']}"

            # Parse table position from XML to get table_elevation and x-offset
            table_pos_str = table_body.get('@pos', f"{self.table_full_size[0]} 0 {self.table_elevation}")
            table_pos = [float(v) for v in table_pos_str.split()]
            xml_table_x_offset = table_pos[0]   # half-length in robosuite coords
            xml_table_elevation = table_pos[2]

            # Parse table tilt from XML if present
            axisangle_str = table_body.get('@axisangle', None)
            if axisangle_str is not None:
                parts = [float(v) for v in axisangle_str.split()]
                # axisangle is "ax ay az angle" — table uses "0 1 0 -tilt"
                xml_table_tilt = -parts[3] if (parts[0]==0 and parts[1]==1 and parts[2]==0) else 0.0
            else:
                xml_table_tilt = 0.0

            # Find table_surface geom and parse its size
            table_surface_idx = None
            for i, body in enumerate(table_body['body']):
                if body['@name'] == 'table_surface':
                    table_surface_idx = i
                    break
            
            if table_surface_idx is not None:
                geom_size_str = table_body['body'][table_surface_idx]['geom']['@size']
                geom_size = [float(v) for v in geom_size_str.split()]
                xml_half_length = geom_size[0]
                xml_half_width  = geom_size[1]
                xml_half_depth  = geom_size[2]

                # Update sim geometry to match XML — makes coordinate transforms correct
                self.table_full_size = (xml_half_length, xml_half_width, xml_half_depth)
                self.table_offset = np.array((0, 0, xml_table_elevation))
                self.table_elevation = xml_table_elevation
                self.table_tilt = xml_table_tilt
                self.table_depth = xml_half_depth * 2

                # Recompute trig ratios from actual XML tilt
                self.x_to_x_prime_ratio = math.cos(self.table_tilt)
                self.x_prime_to_x_ratio = 1.0 / self.x_to_x_prime_ratio if self.x_to_x_prime_ratio != 0 else 1.0
                self.x_to_z_ratio = math.sin(self.table_tilt)
                self.transform_z = lambda x: self.x_to_z_ratio * x + self.table_elevation - self.table_depth
                self.transform_x = lambda x: self.x_to_x_prime_ratio * x
                self.inverse_transform_x = lambda x: self.x_prime_to_x_ratio * x
                self.puck_z_offset = math.sin(self.table_tilt) * self.puck_radius

                # Recompute playable area bounds from XML geometry
                self.length = xml_half_length * 2
                self.width  = xml_half_width  * 2
                self.high_level_table_x_top = -self.length / 2
                self.high_level_table_x_bot =  self.length / 2
                self.high_level_table_y_right =  self.width / 2
                self.high_level_table_y_left  = -self.width / 2
                self.table_x_top = self.length - self.table_x_offset
                self.table_x_bot = self.table_x_offset
                self.table_y_right = -self.width / 2 + self.table_y_offset
                self.table_y_left  =  self.width / 2 - self.table_y_offset

                # Update the robosuite_env_cfg table_offset so RobosuiteEnv gets the right value
                self.table_offset = np.array((0, 0, xml_table_elevation))




    def _disable_problematic_collisions(self):
        """Disable collisions that should never happen with the air-hockey
        scene. Runs on every reset because robosuite hard_reset rebuilds the
        model and resets contype/conaffinity to XML defaults.

        - Pedestal/mount: the robot's own shoulder body sits inside the
          pedestal collision volume by default, generating huge constraint
          forces that explode qvel on the first sim.step.
        - Default gripper fingers: a gripper is added even with
          gripper_types=None, and its inner fingers self-collide with their
          knuckles in the rest pose.
        """
        sim_model = self.robosuite_env.sim.model
        for i in range(sim_model.ngeom):
            name = sim_model.geom_id2name(i) or ''
            if 'pedestal' in name or 'mount' in name:
                sim_model.geom_contype[i] = 0
                sim_model.geom_conaffinity[i] = 0
            if 'gripper' in name and ('finger' in name or 'knuckle' in name) and 'collision' in name:
                sim_model.geom_contype[i] = 0
                sim_model.geom_conaffinity[i] = 0

    def set_obj_configs(self):
        # Reset robot joint pose to a tabletop-reach config so the EEF starts
        # ON the table (~5mm above the surface at the agent's intended paddle
        # start position), not below it. The robosuite UR5e default
        # init_qpos lands the EEF ~10cm under the table, which collapses
        # OSC dynamics (NaN/Inf in QACC) and is physically wrong.
        # Pose was solved via damped-least-squares IK on the position Jacobian,
        # target = (0.333, 0.0, 0.795) in robosuite world coords (= env paddle
        # start (0.79, 0) at table_top + 5mm).
        if self.robosuite_env is not None and len(self.robosuite_env.robots) > 0:
            # hard_reset rebuilds the mujoco model with the original XML on
            # every reset; reapply our contype/conaffinity overrides first.
            self._disable_problematic_collisions()
            robot = self.robosuite_env.robots[0]
            tabletop_qpos = np.array(getattr(self, 'tabletop_init_qpos',
                [-0.3388, -1.553, 2.1471, -2.4853, -1.3923, -1.991]))
            joint_idx = robot.joint_indexes
            self.robosuite_env.sim.data.qpos[joint_idx] = tabletop_qpos
            self.robosuite_env.sim.data.qvel[joint_idx] = 0
            self.robosuite_env.sim.forward()
            # Resync the OSC controller's references and goal to the new pose.
            # Otherwise OSC's ref_pos / ref_ori_mat / goal_pos / goal_ori still
            # point at the old sub-table home pose and the controller drags the
            # EEF back there on the first env.step (huge ~0.4 m jumps).
            #
            # Three things have to be in sync, in order:
            #   1. composite_controller.update_state() refreshes origin_pos /
            #      origin_ori from the (post-qpos) sim.
            #   2. each part controller's .update(force=True) refreshes
            #      ref_pos / ref_ori_mat from the (post-qpos) sim.
            #   3. goal_pos / goal_ori are then written in the controller's
            #      input_ref_frame ("base") via world_to_origin_frame(ref_pos).
            #      Robosuite's stock reset_goal writes ref_pos directly, but
            #      ref_pos is in world frame and goal_pos is in base frame —
            #      that mismatch is what caused the EEF to fly off.
            cc = robot.composite_controller
            if hasattr(cc, 'update_state'):
                cc.update_state()
            for pc in cc.part_controllers.values():
                if hasattr(pc, 'update_initial_joints'):
                    pc.update_initial_joints(tabletop_qpos)
                if hasattr(pc, 'update'):
                    pc.update(force=True)
                if hasattr(pc, 'world_to_origin_frame') and hasattr(pc, 'ref_pos'):
                    pc.goal_pos = pc.world_to_origin_frame(pc.ref_pos)
                    pc.goal_ori = np.array(pc.ref_ori_mat)

        for name in self.initial_obj_configurations['pucks'].keys():
            # Free joint: qpos addr returns (start, end) where the slice is
            # 7 elements long [x, y, z, qw, qx, qy, qz] (world position +
            # orientation quaternion). qvel is 6 elements [vx, vy, vz, wx, wy, wz].
            free_joint_name = name + "_free"
            qpos_addr = self.robosuite_env.sim.model.get_joint_qpos_addr(free_joint_name)
            qvel_addr = self.robosuite_env.sim.model.get_joint_qvel_addr(free_joint_name)
            qpos_start, qpos_end = qpos_addr if isinstance(qpos_addr, tuple) else (qpos_addr, qpos_addr + 7)
            qvel_start, qvel_end = qvel_addr if isinstance(qvel_addr, tuple) else (qvel_addr, qvel_addr + 6)

            pos = self.initial_obj_configurations['pucks'][name]['position']
            vel = self.initial_obj_configurations['pucks'][name]['velocity']
            # Set world pose (free joint pose IS the world pose, so write
            # directly without subtracting body_xpos).
            self.robosuite_env.sim.data.qpos[qpos_start:qpos_start + 3] = pos
            self.robosuite_env.sim.data.qpos[qpos_start + 3:qpos_end] = [1.0, 0.0, 0.0, 0.0]  # identity quat
            # Velocity: vel is a 2-element [vx, vy] from the high-level frame.
            # Write into linear vel; leave angular vel zero.
            self.robosuite_env.sim.data.qvel[qvel_start:qvel_start + 2] = vel[:2]
            self.robosuite_env.sim.data.qvel[qvel_start + 2:qvel_end] = 0

            # Honor the per-puck affected_by_gravity flag by toggling the
            # body's gravcomp (1.0 = full antigravity, 0.0 = normal gravity).
            grav_flag = getattr(self, 'puck_gravity_flags', {}).get(name, True)
            puck_body_id = self.robosuite_env.sim.model.body_name2id(name)
            if hasattr(self.robosuite_env.sim.model, 'body_gravcomp'):
                self.robosuite_env.sim.model.body_gravcomp[puck_body_id] = 0.0 if grav_flag else 1.0
        for name in self.initial_block_positions.keys():
            xpos = self.robosuite_env.sim.data.body_xpos[self.robosuite_env.sim.model.body_name2id(name)]
            
            pos = self.initial_block_positions[name]
            desired_qpos = pos - xpos
            
            joint_key  = self.robosuite_env.sim.model.get_joint_qpos_addr(name + "_x")
            self.robosuite_env.sim.data.qpos[joint_key] = desired_qpos[0]
            joint_key  = self.robosuite_env.sim.model.get_joint_qpos_addr(name + "_y")
            self.robosuite_env.sim.data.qpos[joint_key] = desired_qpos[1]
            joint_key  = self.robosuite_env.sim.model.get_joint_qpos_addr(name + "_yaw")
            self.robosuite_env.sim.data.qpos[joint_key] = desired_qpos[2]
        # Zero ctrl before settling so leftover torques from a prior episode
        # don't yank the freshly-positioned EEF off the table on the first
        # physics step. forward() refreshes derived quantities (xpos/xvel)
        # without integrating, then a single sim.step() integrates with ctrl=0.
        self.robosuite_env.sim.data.ctrl[:] = 0
        self.robosuite_env.sim.forward()
        self.robosuite_env.sim.step()

    def set_object_links(self):
        # set up object names TODO: might not be working
        self.paddle_name_list = ["gripper0_right_eef"] # Updated to match actual body name

        
        self.puck_name_list = list(self.puck_names.keys())
        self.block_name_list = list(self.block_names.keys())

        self.paddle_name_list.sort()
        self.puck_name_list.sort()
        self.block_name_list.sort()

    def instantiate_objects(self):
        if self.initialized_objects:
            self.set_obj_configs()
            return
        # this is only for the first time
        with open(self.tmp_xml_fp, 'w') as file:
            file.write(xmltodict.unparse(self.xml_config, pretty=True))
        self.robosuite_env = RobosuiteEnv(xml_fp=self.tmp_xml_fp,
                                          table_full_size=self.table_full_size,
                                          table_friction=self.table_friction,
                                          table_offset=self.table_offset,
                                          puck_names=self.puck_names,
                                          block_names=self.block_names,
                                          robosuite_env_params=self.robosuite_env_cfg)

        # set_obj_configs (called below and on every subsequent reset)
        # also re-applies the collision overrides — needed because hard_reset
        # rebuilds the MuJoCo model.
        self.set_obj_configs()
        self.initialized_objects = True
        
    def high_level_to_robosuite_coords(self, pos, object_type):
        # uses high_level_table_x_top, high_level_table_x_bot, high_level_table_y_right, high_level_table_y_left
        # and table_x_top, table_x_bot, table_y_right, table_y_left
        # first convert both to negative
        # pos = -pos
        
        x = (pos[0] - self.high_level_table_x_top) / (self.high_level_table_x_bot - self.high_level_table_x_top) * (self.table_x_bot - self.table_x_top) + self.table_x_top
        y = (pos[1] - self.high_level_table_y_left) / (self.high_level_table_y_right - self.high_level_table_y_left) * (self.table_y_right - self.table_y_left) + self.table_y_left
        if object_type == 'puck':
            x -= self.puck_radius
            y -= self.puck_radius
        elif object_type == 'block':
            x -= self.block_width / 2
            y -= self.block_width / 2
        elif object_type == 'paddle':
            x -= 0 # self.paddle_radius
            y -= 0 # self.paddle_radius
        else:
            raise ValueError("Invalid object type")
        x = self.inverse_transform_x(x)
        
        return np.array([x, y])
    
    def robosuite_to_high_level_coords(self, pos, object_type):
        # uses high_level_table_x_top, high_level_table_x_bot, high_level_table_y_right, high_level_table_y_left
        # and table_x_top, table_x_bot, table_y_right, table_y_left
        x = (pos[0] - self.table_x_top) / (self.table_x_bot - self.table_x_top) * (self.high_level_table_x_bot - self.high_level_table_x_top) + self.high_level_table_x_top
        y = (pos[1] - self.table_y_left) / (self.table_y_right - self.table_y_left) * (self.high_level_table_y_right - self.high_level_table_y_left) + self.high_level_table_y_left
        if object_type == 'puck':
            # x += self.puck_radius
            # y += self.puck_radius
            x += 0
            y += 0
        elif object_type == 'block':
            x += self.block_width / 2
            y += self.block_width / 2
        elif object_type == 'paddle':
            x += 0 # self.paddle_radius
            y += 0 # self.paddle_radius
        else:
            raise ValueError("Invalid object type")
        return np.array([x, y])
    
    def high_level_to_robosuite_vel(self, vel, object_type):
        return np.array([-vel[0], -vel[1]])

    def robosuite_to_high_level_vel(self, vel, object_type):
        return np.array([-vel[0], -vel[1]])

    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        self.initial_block_positions[name] = pos
        self.initial_obj_configurations['blocks'][name] = {'position': pos}
        if self.initialized_objects:
            return
        
        # create puck object to add
        puck_mass = self.puck_density * math.pi * (self.puck_radius ** 2) * 0.009
        z_pos = self.transform_z(pos[0])
        x_pos = self.transform_x(pos[0])
        y_pos = pos[1]
        self.block_names[name] = name
        puck_dict = {
            "@name": "base",
            "@pos": f"{x_pos} {y_pos} {z_pos}",
            "@axisangle": "0 1 0 -0.09",
            "joint": [
                {
                    "@name": f"{name}_x",
                    "@type": "slide",
                    "@axis": "1 0 0",
                    "@damping": f"{self.puck_damping}",
                    "@limited": "false",
                },
                {
                    "@name": f"{name}_y",
                    "@type": "slide",
                    "@axis": "0 1 0",
                    "@damping": f"{self.puck_damping}",
                    "@limited": "false",
                },
                {
                    "@name": f"{name}_yaw",
                    "@type": "hinge",
                    "@axis": "0 0 1",
                    "@damping": "2e-6",
                    "@limited": "false",
                },
            ],
            "body": {
                "@name": f"{name}",
                "geom": [
                    {
                        "@pos": "0 0 -0.2", # believe this is relative to the base
                        "@name": f"{name}",
                        "@type": "cylinder",
                        "@material": "red",
                        "@size": f"{self.puck_radius} 0.009",
                        "@condim": "4",
                        "@priority": "0",
                        # "@contype": "0",
                        # "@conaffinity": "0",
                        "@group": "1",
                    }
                ],
                "inertial": {
                    "@pos": "0 0 0", # believe this is relative to the base
                    "@mass": f"{puck_mass}",
                    "@diaginertia": "2.5e-6 2.5e-6 5e-6",
                },
            }
        }
        
        if isinstance(self.xml_config['mujoco']['worldbody']['body'], list):
            self.xml_config['mujoco']['worldbody']['body'].append(puck_dict)
        else:
            self.xml_config['mujoco']['worldbody']['body'] = [self.xml_config['mujoco']['worldbody']['body'], puck_dict]
            
        # add contact
        if 'contact' in self.xml_config['mujoco']:
            if 'exclude' in self.xml_config['mujoco']['contact']:
                self.xml_config['mujoco']['contact']['exclude'].append({
                    "@body1": f"{name}",
                    "@body2": f"table_surface"
                })
            else:
                self.xml_config['mujoco']['contact']['exclude'] = {
                    "@body1": f"{name}",
                    "@body2": f"table_surface"
                }
        else:
            self.xml_config['mujoco']['contact'] = {
                "exclude": [{
                    "@body1": f"{name}",
                    "@body2": f"table_surface"
                }]
            }

    def spawn_puck(self, pos, vel, name, affected_by_gravity=True, movable=True):


        # TODO: New, which clips puck to be at valid location
        pos = self.high_level_to_robosuite_coords(pos, object_type='puck')
        # Clamp to valid table area instead of hard asserting — coordinate
        # transforms may have slight floating point overshoot at boundaries.
        pos[0] = np.clip(pos[0], self.table_x_bot + self.puck_radius, self.table_x_top - self.puck_radius)
        pos[1] = np.clip(pos[1], self.table_y_right + self.puck_radius, self.table_y_left - self.puck_radius)


        # TODO: Original
        # pos = self.high_level_to_robosuite_coords(pos, object_type='puck')


        assert pos[0] >= self.table_x_bot and pos[0] <= self.table_x_top, f"pos[0]: {pos[0]}, table_x_bot: {self.table_x_bot}, table_x_top: {self.table_x_top}"
        assert pos[1] <= self.table_y_left and pos[1] >= self.table_y_right, f"pos[1]: {pos[1]}, table_y_left: {self.table_y_left}, table_y_right: {self.table_y_right}"
        vel = self.high_level_to_robosuite_vel(vel, object_type='puck')
        # Track gravity preference per-puck so set_obj_configs can re-apply it.
        if not hasattr(self, 'puck_gravity_flags'):
            self.puck_gravity_flags = {}
        self.puck_gravity_flags[name] = bool(affected_by_gravity)
        
        puck_mass = self.puck_density * math.pi * (self.puck_radius ** 2) * self.puck_height
        x_pos = self.transform_x(pos[0])
        y_pos = pos[1]
        # The table is tilted (axisangle "0 1 0 -table_tilt"), so its top
        # surface z varies with x: it's HIGHER on the +x (puck-spawn) end and
        # LOWER on the -x (agent) end. The previous z_pos = table_elevation +
        # puck_height/2 used the table-center z and ignored tilt, which
        # placed the puck UNDER the table at the +x end (where it spawns) —
        # making the puck invisible from above and unable to move on the
        # surface. Add the tilt-induced rise: dz = sin(table_tilt) * (x_pos -
        # table_center_x), where table body is at world x = table_full_size[0].
        table_top_z = (
            self.table_elevation
            + math.sin(self.table_tilt) * (x_pos - self.table_full_size[0])
        )
        z_pos = table_top_z + self.puck_height/2 + 0.001  # +1mm clearance
        pos = np.array([x_pos, y_pos, z_pos])
        self.initial_obj_configurations['pucks'][name] = {'position': pos, 'velocity': vel}
        self.initial_puck_vels[name] = vel
        self.puck_names[name] = name
        if self.initialized_objects:
            return
        
        # Use a SINGLE FREE joint instead of three constrained slides. The
        # original (x-slide, y-slide, yaw-hinge) trio locked the puck at its
        # spawn z, which made the puck juggle task impossible: the puck could
        # never fall under gravity or bounce up off the paddle. With a free
        # joint, MuJoCo gravity acts naturally on the puck and the paddle
        # contact correctly propels it upward. Damping is set on the joint so
        # the puck slides smoothly when it's resting on the table.
        puck_dict = {
            "@name": "base",
            "@pos": f"{x_pos} {y_pos} {z_pos}",
            "@axisangle": f"0 1 0 {-self.table_tilt}",
            "joint": [
                {
                    "@name": f"{name}_free",
                    "@type": "free",
                    "@damping": f"{self.puck_damping}",
                    "@limited": "false",
                },
            ],
            "body": {
                "@name": f"{name}",
                "geom": [
                    {
                        "@pos": f"0 0 -{self.puck_z_offset}",
                        "@name": f"{name}",
                        "@type": "cylinder",
                        # Set rgba directly instead of @material="green" — material
                        # name lookup against the table XML's <asset> block was
                        # silently falling back to MuJoCo's default gray (0.5 0.5
                        # 0.5 1), making the puck invisible against the white table.
                        "@rgba": "0.05 0.85 0.15 1",
                        "@size": f"{self.puck_radius} {self.puck_height}",
                        "@condim": "4",
                        "@priority": "0",
                        "@group": "1",
                    }
                ],
                "inertial": {
                    "@pos": "0 0 0",
                    "@mass": f"{puck_mass}",
                    "@diaginertia": "2.5e-6 2.5e-6 5e-6",
                },
            }
        }

        if isinstance(self.xml_config['mujoco']['worldbody']['body'], list):
            self.xml_config['mujoco']['worldbody']['body'].append(puck_dict)
        else:
            self.xml_config['mujoco']['worldbody']['body'] = [self.xml_config['mujoco']['worldbody']['body'], puck_dict]
            
        # add contact
        # if 'contact' in self.xml_config['mujoco']:
        #     if 'exclude' in self.xml_config['mujoco']['contact']:
        #         self.xml_config['mujoco']['contact']['exclude'].append({
        #             "@body1": f"{name}",
        #             "@body2": f"table_surface"
        #         })
        #     else:
        #         self.xml_config['mujoco']['contact']['exclude'] = {
        #             "@body1": f"{name}",
        #             "@body2": f"table_surface"
        #         }
        # else:
        #     self.xml_config['mujoco']['contact'] = {
        #         "exclude": [{
        #             "@body1": f"{name}",
        #             "@body2": f"table_surface"
        #         }]
        #     }

    def update_table(self, top_solref=None, bot_solref=None, left_solref=None, right_solref=None):

        if isinstance(self.xml_config['mujoco']['worldbody']['body'], list):
            geoms = self.xml_config['mujoco']['worldbody']['body'][0]['body'][1]['geom']
        else:
            geoms = self.xml_config['mujoco']['worldbody']['body']['body'][1]['geom']

        for geom in geoms:
            geom_name = geom.get('@name', '') 
            if 'home' in geom_name:
                geom['@solref'] = f"{bot_solref} -250" if bot_solref is not None else "-80000 -250"
            elif 'away' in geom_name:
                geom['@solref'] = f"{top_solref} -250" if top_solref is not None else "-80000 -250"
            elif 'left' in geom_name:
                    geom['@solref'] = f"{left_solref} -250" if left_solref is not None else "-100000 -250"
            elif 'right' in geom_name:
                geom['@solref'] = f"{right_solref} -250" if right_solref is not None else "-100000 -250"

        if isinstance(self.xml_config['mujoco']['worldbody']['body'], list):
            self.xml_config['mujoco']['worldbody']['body'][0]['body'][1]['geom'] = geoms
        else:
            self.xml_config['mujoco']['worldbody']['body']['body'][1]['geom'] = geoms

    def spawn_paddle(self, pos, vel, name):
        # put the eef in pos
        self.initial_obj_configurations['paddles'][name] = {'position': pos, 'velocity': vel}
    
    def translate_action(self, action):
        """
        Converts 2D action to 6D robot action for OSC controller
        OSC expects [x, y, z, roll, pitch, yaw] or [x, y, z, qx, qy, qz, qw]
        """
        delta_pos_x = -action[0] * self.x_to_x_prime_ratio * self.action_x_scaling
        delta_pos_y = - action[1] * self.action_y_scaling
        # BUGFIX: previously called self.transform_z(...) which returns an
        # ABSOLUTE z position (≈ table_elevation - depth, ~0.69 m), not a delta.
        # OSC then saw a constant z-input of ~0.69 every step regardless of
        # action and pulled the EEF upward, causing run-away drift.
        # Tilt-compensated z-delta should be sin(table_tilt) * x-delta.
        delta_pos_z = -self.x_to_z_ratio * action[0] * self.action_x_scaling
        
        # For OSC controller, we need 6D actions: [x, y, z, roll, pitch, yaw]
        # Set orientation changes to 0 for now (no rotation)
        delta_roll = 0.0
        delta_pitch = 0.0
        delta_yaw = 0.0
        
        return np.array([delta_pos_x, delta_pos_y, delta_pos_z, delta_roll, delta_pitch, delta_yaw])

    def get_transition(self, action):
        """
        Takes a step in simulation with control command @action and returns the resulting transition.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
        Raises:
            ValueError: [Steps past episode termination]
        """
        # TODO: use self.last_action, self.step_frequency, self.time_frequency to implement action_lag
        # number of steps to take: self.time_frequency / self.step_frequency 
        # also need to set self.last_action properly
        # This may require some recalibrating of the gains, though hopefully not


        action = self.translate_action(action)
        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True
        initial_vel = self.robosuite_env._get_observations()['gripper_eef_vel']

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.robosuite_env.control_timestep / self.robosuite_env.model_timestep)):
            self.robosuite_env.sim.forward()
            self.robosuite_env._pre_action(action, policy_step)
            self.robosuite_env.sim.step()

            self.robosuite_env._update_observables()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.robosuite_env.cur_time += self.robosuite_env.control_timestep
        self.timestep += 1

        current_state = self.get_current_state()
        contact_forces = self.robosuite_env.sim.data.cfrc_ext
        eef_index = self.robosuite_env.sim.model.body_name2id('gripper0_right_eef')
        current_state['paddles']['paddle_ego']['force'] = contact_forces[eef_index][:2] # exclude torques and z force
        
        if 'pucks' in current_state: self.puck_history.append(list(current_state['pucks'][0]["position"]) + [0])
        else: self.puck_history.append([-2 + self.center_offset_constant,0,1])

        paddle_pos = current_state['paddles']['paddle_ego']['position']
        self.paddle_history.append([float(paddle_pos[0]), float(paddle_pos[1]), 0])

        final_vel = current_state['paddles']['paddle_ego']['velocity']
        
        current_state['paddles']['paddle_ego']['acceleration'] = final_vel - initial_vel[:1]

        return current_state

    def get_current_state(self):
        """
        Returns the current state of the environment
        """
        obs = self.robosuite_env._get_observations()
        state_info = {}
        # eef position and vel become paddle position and vel
        ego_paddle_pos = obs['gripper_eef_pos']
        ego_paddle_pos = self.robosuite_to_high_level_coords(ego_paddle_pos, object_type='paddle')
        ego_paddle_vel = obs['gripper_eef_vel']
        ego_paddle_vel = self.robosuite_to_high_level_vel(ego_paddle_vel, object_type='paddle')
        ego_paddle_x_pos = ego_paddle_pos[0]
        ego_paddle_y_pos = ego_paddle_pos[1]
        ego_paddle_x_vel = ego_paddle_vel[0]
        ego_paddle_y_vel = ego_paddle_vel[1]
        
        state_info['paddles'] = {'paddle_ego': {'position': (ego_paddle_x_pos, ego_paddle_y_pos),
                                                'velocity': (ego_paddle_x_vel, ego_paddle_y_vel),
                                                'acceleration': (0, 0),
                                                'force': [0, 0]}}
        if len(self.puck_names) > 0:
            state_info['pucks'] = []
            for puck_name in self.puck_names:
                puck_pos = obs[puck_name + '_pos']
                puck_pos = self.robosuite_to_high_level_coords(puck_pos, object_type='puck')
                puck_vel = obs[puck_name + '_vel']
                puck_vel = self.robosuite_to_high_level_vel(puck_vel, object_type='puck')
                puck_x_pos = puck_pos[0]
                puck_y_pos = puck_pos[1]
                puck_x_vel = puck_vel[0]
                puck_y_vel = puck_vel[1]
                state_info['pucks'].append({'position': (puck_x_pos, puck_y_pos), 
                                'velocity': (puck_x_vel, puck_y_vel)})

        if len(self.block_names) > 0:
            state_info['blocks'] = []
            for block_name in self.block_names:
                block_pos = obs[block_name + '_pos']
                block_pos = self.robosuite_to_high_level_coords(block_pos, object_type='block')
                block_x_pos = block_pos[0]
                block_y_pos = block_pos[1]
                state_info['blocks'].append({'position': (block_x_pos, block_y_pos)})
                
        for key in obs.keys():
            if 'image' in key:
                state_info[key] = obs[key]
        return state_info

    def quat2axisangle(self, quat):
        """
        Converts quaternion to axis-angle format.
        Returns a unit vector direction scaled by its angle in radians.

        Args:
            quat (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: (ax,ay,az) axis-angle exponential coordinates
        """
        quat = np.array(quat)
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den
    

class AirHockeyTableArena(Arena):
    """
    Workspace that contains an empty table.


    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(self, table_offset, xml):
        # breakpoint()
        arena_fp = robosuite_xml_path_completion(xml)
        super().__init__(arena_fp)
        self.center_pos = self.bottom_pos + np.array([0, 0, 0.0]) + table_offset
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.configure_location()
        # pass

    def configure_location(self):
        """Configures correct locations for this arena"""
        pass
    
class RobosuiteEnv(SingleArmEnv):
    def __init__(self, xml_fp, table_full_size, table_friction, table_offset, puck_names, block_names, robosuite_env_params):
        # load model for table top workspace
        mujoco_arena = AirHockeyTableArena(
            table_offset=table_offset,
            xml=xml_fp,
        )
        
        self.puck_names = puck_names
        self.block_names = block_names
        
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        robots = robosuite_env_params['robots']
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        self.num_robots = len(robots)
        robot_names = self.input2list(robots, self.num_robots)
        controller_configs = self.input2list(robosuite_env_params['controller_configs'], self.num_robots)
        base_types = self.input2list(robosuite_env_params['base_types'], self.num_robots)
        initialization_noise = self.input2list(robosuite_env_params['initialization_noise'], self.num_robots)
        control_freq = self.input2list(robosuite_env_params['control_freq'], self.num_robots)
        # Forward gripper_types from robosuite_env_params so each robot's
        # load_model attaches the requested gripper (e.g. RoundGripper) instead
        # of falling back to the robot's default (Robotiq85 for UR5e).
        gripper_types = self.input2list(
            robosuite_env_params.get('gripper_types', 'default'), self.num_robots
        )
        robot_configs = self.load_robots_configs(
            robot_names, controller_configs, base_types, initialization_noise, control_freq,
            gripper_types=gripper_types,
        )
        self.robots = self.get_robots(robot_names, robot_configs)

        # task includes arena, robot, and objects of interest
        self.task_model = ManipulationTask(mujoco_arena=mujoco_arena, mujoco_robots=[robot.robot_model for robot in self.robots])
        super().__init__(**robosuite_env_params)
    
    def get_robots(self, robot_names, robot_configs):
        """
        Instantiates robots and stores them within the self.robots attribute
        """
        # Loop through robots and instantiate Robot object for each
        robots_out = [None for _ in range(len(robot_names))]
        for idx, (name, config) in enumerate(zip(robot_names, robot_configs)):
            # Create the robot instance
            robots_out[idx] = ROBOT_CLASS_MAPPING[name](robot_type=name, idn=idx, **config)
            # Now, load the robot models
            robots_out[idx].load_model()
        return robots_out
    
    def _load_model(self):
        super()._load_model()
        self.model = self.task_model # Prevents the super call from making this None
            
    def load_robots_configs(self, robot_names, controller_configs, base_types, initialization_noise, control_freq,
                            gripper_types=None, robot_configs=None):
        num_robots = len(robot_names)
        if robot_configs is None:
            robot_configs = [{} for _ in range(num_robots)]
        if gripper_types is None:
            gripper_types = ['default'] * num_robots
        self.robot_configs = [
            dict(
                **{
                    # Robosuite's Robot.__init__ takes `composite_controller_config`,
                    # `gripper_type`, `base_type` (singular) — NOT `controller_config`,
                    # `gripper_types`, or `mount_type`. Map to the right keys.
                    "composite_controller_config": controller_configs[idx],
                    "base_type": base_types[idx],
                    "gripper_type": gripper_types[idx] if gripper_types[idx] is not None else "default",
                    "initialization_noise": initialization_noise[idx],
                    "control_freq": control_freq[idx] if isinstance(control_freq, list) else control_freq,
                },
                **robot_config,
            )
            for idx, robot_config in enumerate(robot_configs)
        ]
        # Return the populated configs — previous code returned the empty input list.
        return self.robot_configs
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        from functools import partial
        def obj_pos(obs_cache, obj_name):
            return self.sim.data.get_body_xpos(obj_name)
        
        def obj_vel(obs_cache, obj_name):
            return self.sim.data.get_body_xvelp(obj_name)

        def gripper_eef_vel(obs_cache):
            return self.sim.data.get_body_xvelp("gripper0_right_eef")
        
        def gripper_eef_pos(obs_cache):
            return self.sim.data.get_body_xpos("gripper0_right_eef")
        
        gripper_eef_vel.__modality__ = modality
        gripper_eef_pos.__modality__ = modality

        sensors = [gripper_eef_vel,
                   gripper_eef_pos]
        
        def add_sensor(name, sensors):
            pos_fn = partial(obj_pos, obj_name=name)
            pos_fn.__name__ = f"{name}_pos"
            pos_fn.__modality__ = modality
            vel_fn = partial(obj_vel, obj_name=name)
            vel_fn.__name__ = f"{name}_vel"
            vel_fn.__modality__ = modality
            sensors.append(pos_fn)
            sensors.append(vel_fn)
        
        for name in self.puck_names:
            add_sensor(name, sensors)
        for name in self.block_names:
            add_sensor(name, sensors)

        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables
    
    def input2list(self, inp, length):
        """
        Helper function that converts an input that is either a single value or a list into a list

        Args:
            inp (None or str or list): Input value to be converted to list
            length (int): Length of list to broadcast input to

        Returns:
            list: input @inp converted into a list of length @length
        """
        # convert to list if necessary
        return list(inp) if type(inp) is list or type(inp) is tuple else [inp for _ in range(length)]
    
    def visualize(self, vis_settings):
        """
        Super call to visualize.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
    
    # def _reset_internal(self):
    #     """
    #     Resets simulation internal configurations.
    #     """
    #     super()._reset_internal()

    #     # Reset all object positions using initializer sampler if we're not directly loading from an xml
    #     if not self.deterministic_reset:
    #         self.modder = DynamicsModder(sim=self.robosuite_env.sim)
    #         self.modder.mod_position("base", [0.8, np.random.uniform(-0.3, 0.3), 1.2])
    #         self.modder.update()
    