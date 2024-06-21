import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPuckGoalPositionVelocityReward

class AirHockeyPuckGoalPositionVelocityEnv(AirHockeyGoalEnv):
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        puck_hist_low = [self.table_x_top, self.table_y_left, 0] * 5
        puck_hist_high = [self.table_x_bot, self.table_y_right, 0] * 5

        if obs_type == "paddle":
            low = paddle_obs_low
            high = paddle_obs_high
        elif obs_type == "vel":
            low = paddle_obs_low + puck_obs_low
            high = paddle_obs_high + puck_obs_high
        elif obs_type == "history":
            low = paddle_obs_low + puck_hist_low
            high = paddle_obs_high + puck_hist_high

        goal_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        goal_high = [0, self.table_y_right, self.max_puck_vel, self.max_puck_vel]
        
        if self.return_goal_obs:
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + goal_low
            high = high + goal_high
            self.observation_space = self.get_obs_space(low, high)

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPuckGoalPositionVelocityReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionVelocityEnv(**state_dict)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
        
    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def get_achieved_goal(self, state_info):
        position = state_info['pucks'][0]['position']
        velocity = state_info['pucks'][0]['velocity']
        return np.array([position[0], position[1], velocity[0], velocity[1]])
    
    def get_desired_goal(self):
        position = self.goal_pos
        velocity = self.goal_vel
        return np.array([position[0], position[1], velocity[0], velocity[1]])
    
    def get_observation(self, state_info, obs_type ="vel", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
    #     puck_x_pos = state_info['pucks'][0]['position'][0]
    #     puck_y_pos = state_info['pucks'][0]['position'][1]
    #     puck_x_vel = state_info['pucks'][0]['velocity'][0]
    #     puck_y_vel = state_info['pucks'][0]['velocity'][1]       

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
    #     return obs
    
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
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
            min_x_vel = self.goal_min_x_velocity
            max_x_vel = self.goal_max_x_velocity
            min_y_vel = self.goal_min_y_velocity
            max_y_vel = self.goal_max_y_velocity
            self.goal_vel = self.rng.uniform(low=(min_x_vel, min_y_vel), high=(max_x_vel, max_y_vel))
        else:
            self.goal_pos = goal_pos if self.goal_set is None else self.goal_set[0, :2]
            self.goal_vel = self.goal_vel if self.goal_set is None else self.goal_set[0, 2:]
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set