import numpy as np
import math
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
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

import os

import numpy as np
from robosuite.models.arenas import Arena
from airhockey.sims.utils import custom_xml_path_completion


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

    def __init__(
        self,
        paddle_radius,
        block_width,
        max_paddle_vel,
        max_puck_vel,
        length, 
        width,
        depth,
        table_tilt,
        table_elevation,
        rim_width,
        render_size,
        robots=['AirHockeyUR5e'],
        env_configuration="default",
        controller_configs=None,
        gripper_types="RoundGripper",
        initialization_noise="default",
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=400,
        ignore_done=False,
        hard_reset=True,
        camera_names=["birdview","sideview"],
        camera_heights=512,
        camera_widths=512,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        task="JUGGLE_PUCK",
        table_xml="arenas/air_hockey_table.xml", # relative to assets dir
        puck_radius=0.03165,
        puck_damping=0.8,
        puck_density=30,
        seed=0
    ):
        # settings for table top
        table_full_size = (length / 2, width / 2, depth / 2)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, table_elevation))
        
        self.length = length
        self.width = width
        self.ppm = render_size / self.width
        self.render_width = int(render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = False

        self.gripper_types = gripper_types

        self.table_tilt = table_tilt
        self.table_elevation = table_elevation
        self.table_depth = depth
        self.x_to_x_prime_ratio = math.cos(self.table_tilt)
        self.x_prime_to_x_ratio = 1 / self.x_to_x_prime_ratio
        self.x_to_z_ratio = math.sin(self.table_tilt)
        self.transform_z = lambda x: self.x_to_z_ratio * x + self.table_elevation - depth
        self.transform_x = lambda x: self.x_to_x_prime_ratio * x
        self.inverse_transform_x = lambda x: self.x_prime_to_x_ratio * x
        
        self.high_level_table_x_top = -self.length / 2
        self.high_level_table_x_bot = self.length / 2
        self.high_level_table_y_right = self.width / 2
        self.high_level_table_y_left = -self.width / 2
        
        self.table_x_offset = 2 * rim_width
        self.table_y_offset = 2 * rim_width
        
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
        self.table_xml = table_xml

        self.puck_radius = puck_radius
        self.puck_damping = puck_damping
        self.puck_density = puck_density
        self.puck_height = 0.009
        self.puck_z_offset = math.sin(self.table_tilt) * self.puck_radius

        # FIXME make these parameters do something, right now it's a placeholder to make calls to robosuite work
        self.seed = seed
        self.paddle_radius = paddle_radius
        self.block_width = block_width
        self.max_paddle_vel = max_paddle_vel
        self.max_puck_vel = max_puck_vel
        
        self.robosuite_env = None
        self.robosuite_env_cfg = {'robots': robots, 'env_configuration': env_configuration, 'controller_configs': controller_configs,
                              'mount_types': "default", 'gripper_types': gripper_types, 'initialization_noise': initialization_noise,
                              'use_camera_obs': use_camera_obs, 'has_renderer': has_renderer, 'has_offscreen_renderer': has_offscreen_renderer,
                              'render_camera': render_camera, 'render_collision_mesh': render_collision_mesh, 'render_visual_mesh': render_visual_mesh,
                              'render_gpu_device_id': render_gpu_device_id, 'control_freq': control_freq, 'horizon': horizon, 'ignore_done': ignore_done,
                              'hard_reset': hard_reset, 'camera_names': camera_names, 'camera_heights': camera_heights, 'camera_widths': camera_widths,
                              'camera_depths': camera_depths, 'camera_segmentations': camera_segmentations, 'renderer': renderer, 'renderer_config': renderer_config}
        
        self.initialized_objects = False
        current_time = datetime.datetime.fromtimestamp(time.time())
        formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
        self.tmp_xml_fp = robosuite_xml_path_completion(self.table_xml + f"_{formatted_time}.xml")
        
    def __del__(self):
        if self.robosuite_env is not None:
            self.robosuite_env.close()
        if self.initialized_objects and os.path.exists(self.tmp_xml_fp):
            os.remove(self.tmp_xml_fp)

    @staticmethod
    def from_dict(state_dict):
        state_dict_copy = state_dict.copy()
        return AirHockeyRobosuite(**state_dict_copy)

    def reset(self, seed=None):
        if self.robosuite_env is not None:
            self.robosuite_env.reset()
        
        self.timestep = 0
        
        if not self.initialized_objects:
            self.puck_names = {}
            self.block_names = {}
            self.initial_obj_configurations = {'paddles': {}, 'pucks': {}, 'blocks': {}}
            xml_fp = custom_xml_path_completion(self.table_xml)
            
            with open(xml_fp, "r") as file:
                self.xml_config = xmltodict.parse(file.read())

            # update table config
            assert self.xml_config['mujoco']['worldbody']['body']['@name'] == 'table'
            self.xml_config['mujoco']['worldbody']['body']['@pos'] = f"{self.table_full_size[0]} 0 {self.table_elevation}"

            # update table surface config
            table_surface_idx = None
            for i, body in enumerate(self.xml_config['mujoco']['worldbody']['body']['body']):
                if body['@name'] == 'table_surface':
                    table_surface_idx = i
                    break
            self.xml_config['mujoco']['worldbody']['body']['body'][table_surface_idx]['geom']['@size'] = f"{self.table_full_size[0]} {self.table_full_size[1]} {self.table_full_size[2]}"
        return {}
    
    def set_obj_configs(self):
        for name in self.initial_obj_configurations['pucks'].keys():
            body_id = self.robosuite_env.sim.model.body_name2id(name)
            
            xpos = self.robosuite_env.sim.data.body_xpos[body_id]
            pos = self.initial_obj_configurations['pucks'][name]['position']
            desired_qpos = pos - xpos
            
            xvel = self.robosuite_env.sim.data.get_body_xvelp(name)[:2]
            vel = self.initial_obj_configurations['pucks'][name]['velocity']
            desired_qvel = vel - xvel
            
            joint_key  = self.robosuite_env.sim.model.get_joint_qpos_addr(name + "_x")
            self.robosuite_env.sim.data.qpos[joint_key] = desired_qpos[0]
            self.robosuite_env.sim.data.qvel[joint_key] = desired_qvel[0]
            joint_key  = self.robosuite_env.sim.model.get_joint_qpos_addr(name + "_y")
            self.robosuite_env.sim.data.qpos[joint_key] = desired_qpos[1]
            self.robosuite_env.sim.data.qvel[joint_key] = desired_qvel[1]
            joint_key  = self.robosuite_env.sim.model.get_joint_qpos_addr(name + "_yaw")
            self.robosuite_env.sim.data.qpos[joint_key] = desired_qpos[2]
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
        self.robosuite_env.sim.step()
        
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

        # Adjust base pose accordingly
        # TODO: uncomment and use this code, it is currently done in the xml
        # xpos = self.robosuite_env.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        # xpos = (-0.48, 0, 0)
        # self.robosuite_env.robots[0].robot_model.set_base_xpos(xpos)

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
            x += self.puck_radius
            y += self.puck_radius
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

    def spawn_puck(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pos = self.high_level_to_robosuite_coords(pos, object_type='puck')
        assert pos[0] >= self.table_x_bot and pos[0] <= self.table_x_top, f"pos[0]: {pos[0]}, table_x_bot: {self.table_x_bot}, table_x_top: {self.table_x_top}"
        assert pos[1] <= self.table_y_left and pos[1] >= self.table_y_right, f"pos[1]: {pos[1]}, table_y_left: {self.table_y_left}, table_y_right: {self.table_y_right}"
        vel = self.high_level_to_robosuite_vel(vel, object_type='puck')
        
        puck_mass = self.puck_density * math.pi * (self.puck_radius ** 2) * self.puck_height
        z_pos = self.transform_z(pos[0]) + 0.025
        x_pos = self.transform_x(pos[0])
        y_pos = pos[1]
        pos = np.array([x_pos, y_pos, z_pos])
        self.initial_obj_configurations['pucks'][name] = {'position': pos, 'velocity': vel}
        self.initial_puck_vels[name] = vel
        self.puck_names[name] = name
        if self.initialized_objects:
            return
        
        puck_dict = {
            "@name": "base",
            "@pos": f"{x_pos} {y_pos} {z_pos}",
            "@axisangle": f"0 1 0 {-self.table_tilt}",
            "joint": [
                {
                    "@name": f"{name}_x",
                    "@type": "slide",
                    "@axis": "1 0 0",
                    "@damping": f"{self.puck_damping}",
                    "@damping": "0.01",
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
                        "@pos": f"0 0 -{self.puck_z_offset}",
                        "@name": f"{name}",
                        "@type": "cylinder",
                        "@material": "red",
                        "@size": f"{self.puck_radius} {self.puck_height}",
                        "@condim": "4",
                        "@priority": "0",
                        "@group": "1",
                    }
                ],
                "inertial": {
                    "@pos": "0 0 0",
                    # "@mass": f"{puck_mass}",
                    "@mass": 0.01,
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

    def spawn_paddle(self, pos, vel, name):
        # put the eef in pos
        self.initial_obj_configurations['paddles'][name] = {'position': pos, 'velocity': vel}
    
    def get_6d_action(self, action):
        """
        Converts 2D action to 6D robot action
        """
        delta_pos_x = action[0] * self.x_to_x_prime_ratio
        delta_pos_y = action[1]
        delta_pos_z = action[0] * self.x_to_z_ratio
        delta_pos = np.array([delta_pos_x, delta_pos_y, delta_pos_z])
        return np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, 0])

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
        action = self.get_6d_action(action)

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

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

        return self.get_current_state()

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
                                                'velocity': (ego_paddle_x_vel, ego_paddle_y_vel)}}
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
        mount_types = self.input2list(robosuite_env_params['mount_types'], self.num_robots)
        initialization_noise = self.input2list(robosuite_env_params['initialization_noise'], self.num_robots)
        control_freq = self.input2list(robosuite_env_params['control_freq'], self.num_robots)
        robot_configs = self.load_robots_configs(robot_names, controller_configs, mount_types, initialization_noise, control_freq)
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
        self.model = self.task_model # Prevents the super call from making this None lol
            
    def load_robots_configs(self, robot_names, controller_configs, mount_types, initialization_noise, control_freq, robot_configs=None):
        num_robots = len(robot_names)
        if robot_configs is None:
            robot_configs = [{} for _ in range(num_robots)]
        self.robot_configs = [
            dict(
                **{
                    "controller_config": controller_configs[idx],
                    "mount_type": mount_types[idx],
                    "initialization_noise": initialization_noise[idx],
                    "control_freq": control_freq,
                },
                **robot_config,
            )
            for idx, robot_config in enumerate(robot_configs)
        ]
        return robot_configs
    
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
            return self.sim.data.get_body_xvelp("gripper0_eef")
        
        def gripper_eef_pos(obs_cache):
            return self.sim.data.get_body_xpos("gripper0_eef")
        
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
    
