import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPaddleReachPositionReward

class AirHockeyPaddleReachPositionEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        self.goal_radius_type = kwargs['goal_radius_type']
        self.base_goal_radius = kwargs['base_goal_radius']
        super().__init__(**kwargs)
        
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]

        low = paddle_obs_low
        high = paddle_obs_high
        
        goal_low = [0, self.table_y_left]
        goal_high = [self.table_x_bot, self.table_y_right]
        
        if self.paddle_x_min is not None:
            goal_low = [self.paddle_x_min, self.paddle_y_min]
            goal_high = [self.paddle_x_max, self.paddle_y_max]
            # print(goal_low)
        
        if self.return_goal_obs:
            low = paddle_obs_low
            high = paddle_obs_high
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = paddle_obs_low + goal_low
            high = paddle_obs_high + goal_high
            self.observation_space = self.get_obs_space(low, high)
        
        self.goal_radius = self.base_goal_radius

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPaddleReachPositionReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPaddleReachPositionEnv(**state_dict)
        
    def create_world_objects(self):
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
        
    def create_world_objects_from_state(self, state_vector):

        name = 'paddle_ego'
        paddle_pos, paddle_vel = state_vector[:2], state_vector[2:4]
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, name)

    def validate_configuration(self):
        assert self.num_pucks == 0
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_achieved_goal(self, state_info):
        position = state_info['paddles']['paddle_ego']['position']
        return np.array([position[0], position[1]])
    
    def get_desired_goal(self):
        position = self.goal_pos
        return np.array([position[0], position[1]])

    def get_observation(self, state_info, obs_type ="paddle", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)
    
    def set_goals(self, goal_radius_type, goal_pos=None, goal_set=None):
        goal_low = [0, self.table_y_left]
        goal_high = [self.table_x_bot / 2, self.table_y_right] # set goal positions to be in the bottom half of the table.
        
        if self.paddle_x_min is not None:
            goal_low = [self.paddle_x_min, self.paddle_y_min]
            goal_high = [self.paddle_x_max, self.paddle_y_max]
        
        self.goal_set = goal_set
        
        # sample goal position
        min_y = goal_low[1]
        max_y = goal_high[1]
        min_x = goal_low[0]
        max_x = goal_high[0]
        
        goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        self.goal_pos = goal_position if self.goal_set is None else self.goal_set[0, :2]
        self.goal_pos = goal_pos if goal_pos is not None else self.goal_pos