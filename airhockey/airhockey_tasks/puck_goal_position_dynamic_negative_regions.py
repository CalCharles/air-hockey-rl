import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_tasks.utils import DynamicRewardRegion, DynamicGoalRegion

class AirHockeyPuckGoalPositionDynamicNegRegionsEnv(AirHockeyGoalEnv):
    def __init__(self,
                 simulator, # box2d or robosuite
                 simulator_params,
                 task, 
                 num_pucks,
                 num_blocks,
                 num_obstacles,
                 num_targets,
                 num_paddles,
                 n_training_steps,
                 wall_bumping_rew,
                 direction_change_rew,
                 horizontal_vel_rew,
                 diagonal_motion_rew,
                 stand_still_rew,
                 terminate_on_out_of_bounds, 
                 terminate_on_enemy_goal, 
                 terminate_on_puck_stop,
                 truncate_rew,
                 goal_max_x_velocity, 
                 goal_min_y_velocity, 
                 goal_max_y_velocity,
                 return_goal_obs,
                 seed,
                 dense_goal=True,
                 goal_selector='stationary',
                 goal_type='static',
                 velocity_of_goal_min=[0,0],
                 velocity_of_goal_max=[0,0],
                 max_timesteps=1000,
                 num_positive_reward_regions=0,
                 positive_reward_range=[1,1],
                 num_negative_reward_regions=0,
                 negative_reward_range=[-1,-1],
                 reward_region_shapes=[],
                 reward_region_scale_range=[0,0],
                 reward_normalized_radius_min=0.1,
                 reward_normalized_radius_max=0.1,
                 reward_velocity_limits_min=[0,0],
                 reward_velocity_limits_max=[0,0],
                 reward_movement_types=[]):
        self.num_negative_reward_regions = num_negative_reward_regions
        self.negative_reward_range = negative_reward_range
        self.reward_region_shapes = reward_region_shapes
        self.reward_region_scale_range = reward_region_scale_range
        self.reward_normalized_radius_min = reward_normalized_radius_min
        self.reward_normalized_radius_max = reward_normalized_radius_max
        self.reward_velocity_limits_min = reward_velocity_limits_min
        self.reward_velocity_limits_max = reward_velocity_limits_max
        self.goal_type = goal_type
        self.velocity_of_goal_min = velocity_of_goal_min
        self.velocity_of_goal_max = velocity_of_goal_max
        self.reward_movement_types = reward_movement_types
        super().__init__(simulator, # box2d or robosuite
                 simulator_params,
                 task, 
                 num_pucks,
                 num_blocks,
                 num_obstacles,
                 num_targets,
                 num_paddles,
                 n_training_steps,
                 wall_bumping_rew,
                 direction_change_rew,
                 horizontal_vel_rew,
                 diagonal_motion_rew,
                 stand_still_rew,
                 terminate_on_out_of_bounds, 
                 terminate_on_enemy_goal, 
                 terminate_on_puck_stop,
                 truncate_rew,
                 goal_max_x_velocity, 
                 goal_min_y_velocity, 
                 goal_max_y_velocity,
                 return_goal_obs,
                 seed,
                 dense_goal=dense_goal,
                 goal_selector=goal_selector,
                 max_timesteps=max_timesteps)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionDynamicNegRegionsEnv(**state_dict)

    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        goal_low = [self.table_x_top, self.table_y_left]        
        goal_high = [0, self.table_y_right]
        
        if self.return_goal_obs:
            low = paddle_obs_low + puck_obs_low
            high = paddle_obs_high + puck_obs_high
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = paddle_obs_low + puck_obs_low + goal_low
            high = paddle_obs_high + puck_obs_high + goal_high
            self.observation_space = self.get_obs_space(low, high)

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

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
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        sigmoid_scale = 2
        radius = self.goal_radius
        reward_raw = 1 - (dist / radius) #self.max_goal_rew_radius * radius)
        reward_mask = dist >= radius
        reward_raw[reward_mask] = 0 # numerical stability, we will make these 0 later
        reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        reward[reward_mask] = 0

        if self.dense_goal:
            bonus = 10 if self.current_timestep > self.falling_time else 0 # this prevents the falling initiliazwed puck from triggering a success
            reward = -dist  + (bonus if dist < radius else 0) # add bonus if within radius

        if single:
            reward = reward[0]
            
        for nrr in self.reward_regions:
            reward += nrr.check_reward(achieved_goal)
            
        return reward

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        self.ego_pos = np.array([ego_paddle_x_pos, ego_paddle_y_pos])
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       
        reward_regions_states = [nrr.get_state() for nrr in self.reward_regions]


        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return np.concatenate([obs] + reward_regions_states)
    
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
    
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.0
        success = success.item()
        return reward, success
    
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
