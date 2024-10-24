import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_tasks.utils import DynamicRewardRegion, DynamicGoalRegion
import copy
from airhockey.airhockey_rewards import AirHockeyPuckGoalPositionDynamicNegRegionsReward
import math
from types import SimpleNamespace

class AirHockeyPuckGoalPositionDynamicNegRegionsEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        # Set default values for parameters
        defaults = {
            'dense_goal': True,
            'goal_selector': 'stationary',
            'goal_type': 'static',
            'velocity_of_goal_min': [0, 0],
            'velocity_of_goal_max': [0, 0],
            'max_timesteps': 1000,
            'num_positive_reward_regions': 0,
            'positive_reward_range': [1, 1],
            'num_negative_reward_regions': 0,
            'negative_reward_range': [-1, -1],
            'reward_region_shapes': [],
            'reward_region_scale_range': [0, 0],
            'reward_normalized_radius_min': 0.1,
            'reward_normalized_radius_max': 0.1,
            'reward_velocity_limits_min': [0, 0],
            'reward_velocity_limits_max': [0, 0],
            'reward_movement_types': [],
            'initialization_description_pth': "",
            'obs_type': "negative_regions_puck_vel",
            'terminate_on_puck_pass_paddle': False,
            'terminate_on_puck_hit_bottom': False,
            'goal_radius_type': "fixed",
            'base_goal_radius': 0.15,
        }
        # Merge defaults with kwargs
        kwargs = {**defaults, **kwargs}
        config = SimpleNamespace(**kwargs)

        # Initialize parameters
        self.num_negative_reward_regions = config.num_negative_reward_regions
        self.negative_reward_range = config.negative_reward_range
        self.reward_region_shapes = config.reward_region_shapes
        self.reward_region_scale_range = config.reward_region_scale_range
        self.reward_normalized_radius_min = config.reward_normalized_radius_min
        self.reward_normalized_radius_max = config.reward_normalized_radius_max
        self.reward_velocity_limits_min = config.reward_velocity_limits_min
        self.reward_velocity_limits_max = config.reward_velocity_limits_max
        self.goal_type = config.goal_type
        self.velocity_of_goal_min = config.velocity_of_goal_min
        self.velocity_of_goal_max = config.velocity_of_goal_max
        self.reward_movement_types = config.reward_movement_types
        self.reward = AirHockeyPuckGoalPositionDynamicNegRegionsReward(self)
        self.goal_radius_type = config.goal_radius_type
        self.base_goal_radius = config.base_goal_radius
        
        # Initialize the superclass with the remaining kwargs
        super().__init__(**kwargs)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionDynamicNegRegionsEnv(**state_dict)

    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type.replace("negative_regions_", ""))

        goal_low = [self.table_x_top, self.table_y_left]        
        goal_high = [0, self.table_y_right]
        
        nrr_obs_low = [-math.inf] * 12 * self.num_negative_reward_regions
        nrr_obs_high = [-math.inf] * 12 * self.num_negative_reward_regions

        if self.return_goal_obs:
            low = low + nrr_obs_low
            high = high + nrr_obs_high
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + nrr_obs_low + goal_low
            high = high + nrr_obs_high + goal_high
            self.observation_space = self.get_obs_space(low, high)

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4
        self.goal_radius = self.base_goal_radius

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
        xrad, yrad = np.array([self.reward_normalized_radius_min[0], self.reward_normalized_radius_max[0]]), np.array([self.reward_normalized_radius_min[1], self.reward_normalized_radius_max[1]])
        radius_range = np.array((self.table_x_bot * xrad, (self.table_y_right - self.table_y_left) * yrad))
        self.reward_regions = [DynamicRewardRegion(self.negative_reward_range, 
                                                     self.reward_region_scale_range, 
                                                     [np.array([0, self.table_y_left]), np.array([self.table_x_bot,self.table_y_right])],
                                                     radius_range, shapes=self.reward_region_shapes, movement_patterns=self.reward_movement_types, 
                                                     velocity_limits=(np.array(self.reward_velocity_limits_min), np.array(self.reward_velocity_limits_max))) for _ in range(self.num_negative_reward_regions)]
        if self.goal_type == 'dynamic':
            self.dynamic_goal_region = DynamicGoalRegion([np.array([0, self.table_y_left]), np.array([self.table_x_bot,self.table_y_right])],
                                                         movement_patterns=self.reward_movement_types, 
                                                         velocity_limits=(np.array(self.velocity_of_goal_min), np.array(self.velocity_of_goal_max))) 
        
    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def get_achieved_goal(self, state_info):
        position = np.array(state_info['pucks'][0]['position'])
        return position.astype(float)
    
    def get_desired_goal(self):
        # import pdb; pdb.set_trace()
        if self.goal_type == 'dynamic':
            dgr = self.dynamic_goal_region.get_pose()
            return self.dynamic_goal_region.get_pose()
        
        position = self.goal_pos
        return position.astype(float)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info, obs_type ="negative_regions_puck_vel", **kwargs):
        state_info["negative_regions"] = [nrr.get_state() for nrr in self.reward_regions]
        self.ego_pos = np.array(copy.deepcopy(state_info['paddles']['paddle_ego']['position']))
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
    #     self.ego_pos = np.array([ego_paddle_x_pos, ego_paddle_y_pos])
    #     puck_x_pos = state_info['pucks'][0]['position'][0]
    #     puck_y_pos = state_info['pucks'][0]['position'][1]
    #     puck_x_vel = state_info['pucks'][0]['velocity'][0]
    #     puck_y_vel = state_info['pucks'][0]['velocity'][1]       
    #     reward_regions_states = [nrr.get_state() for nrr in self.reward_regions]


    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
    #     return np.concatenate([obs] + reward_regions_states)
    
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        if self.goal_type == 'dynamic':
            self.dynamic_goal_region.reset()
            self.goal_pos = self.dynamic_goal_region.get_pose()
            self.goal_radius = 0.05
            return
        
        self.goal_set = goal_set
        if goal_radius_type == 'fixed':
            # goal_radius = self.rng.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
            base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
            # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
            ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
            goal_radius = ratio * base_radius
            self.goal_radius = goal_radius
            
        if self.goal_selector == 'dynamic':
            self.goal_radius = 0.16

        if goal_pos is None and goal_set is None:
            min_y = self.table_y_left + self.goal_radius
            max_y = self.table_y_right - self.goal_radius
            max_x = 0 - self.goal_radius
            min_x = self.table_x_top + self.goal_radius
            self.goal_pos = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        else:
            self.goal_pos = goal_pos if self.goal_set is None else self.goal_set[0]
        
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set
    
    def step(self, action):
        obs, reward, is_finished, truncated, info = super().step(action)
        # import pdb; pdb.set_trace()
        for nrr in self.reward_regions:
            nrr.step()
        if self.goal_type == 'dynamic':
            self.dynamic_goal_region.step()
            self.goal_pos = self.dynamic_goal_region.get_pose()
            
        return obs, reward, is_finished, truncated, info
    
    def reset(self, seed=None):
        for nrr in self.reward_regions:
            nrr.reset()
        if self.goal_type == 'dynamic':
            self.dynamic_goal_region.reset()
            self.goal_pos = self.dynamic_goal_region.get_pose()
        return super().reset(seed)
