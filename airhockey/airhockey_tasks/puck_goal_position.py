import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPuckGoalPositionReward

class AirHockeyPuckGoalPositionEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        self.goal_radius_type = kwargs['goal_radius_type']
        self.success_bonus = kwargs['success_bonus']
        self.test_goal_condition = kwargs['test_goal_condition']
        super().__init__(**kwargs)
        
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        goal_low = [self.table_x_top, self.table_y_left]        
        goal_high = [0, self.table_y_right]

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
        self.reward = AirHockeyPuckGoalPositionReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionEnv(**state_dict)
        
    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)
        
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def get_achieved_goal(self, state_info):
        position = np.array(state_info['pucks'][0]['position'])
        # if self.test_goal_condition:
        #     position = np.array(state_info['paddles']['paddle_ego']['position'])
        return position.astype(float)
    
    def get_desired_goal(self):
        position = self.goal_pos
        
        return position.astype(float)
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    def get_observation(self, state_info, obs_type ="vel", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if self.goal_radius_type == 'linear_decay':
            base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
            # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
            ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
            goal_radius = ratio * base_radius
            self.goal_radius = goal_radius
        elif self.goal_radius_type == 'fixed':
            self.goal_radius = 0.15
            
        if self.test_goal_condition:
            self.goal_radius = 0.15

        # if goal_pos is None and goal_set is None:
        min_y = self.table_y_left + self.goal_radius
        max_y = self.table_y_right - self.goal_radius
        max_x = 0 - self.goal_radius
        min_x = self.table_x_top + self.goal_radius
        self.goal_pos = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y)) #np.array([max_x + min_x, min_y + max_y])/2   #self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        if self.test_goal_condition:
            self.goal_pos = np.array([-0.5, 0.0])

        