import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPuckGoalPositionReward, AirHockeyPaddleReachPositionReward

class AirHockeyPuckGoalPositionEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        # TODO: rearrange defaults so that they happen before this assignment, instead of after
        self.goal_radius_type = kwargs['goal_radius_type']
        self.puck_goal_success_bonus = kwargs['puck_goal_success_bonus']
        self.paddle_puck_success_bonus = kwargs['paddle_puck_success_bonus']
        self.base_goal_radius = kwargs['base_goal_radius']
        super().__init__(**kwargs)
        
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        goal_low = [self.table_x_top, self.table_y_left]        
        goal_high = [0, self.table_y_right]
        
        if self.return_goal_obs:
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + goal_low
            high = high + goal_high
            self.observation_space = self.get_obs_space(low, high)

        # self.min_goal_radius = self.width / 16
        # self.max_goal_radius = self.width / 4
        
        self.goal_radius = self.base_goal_radius

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
        # Need a puck touching reward to 
        self.puck_touch_reward = AirHockeyPaddleReachPositionReward(self, paddle_success_bonus=self.paddle_puck_success_bonus)
        self.reward = AirHockeyPuckGoalPositionReward(self, puck_touch_reward=self.puck_touch_reward)

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

    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_achieved_goal(self, state_info):
        position = state_info['pucks'][0]['position']
        return np.array([position[0], position[1]])
    
    def get_desired_goal(self):
        position = self.goal_pos
        return np.array([position[0], position[1]])

    def get_observation(self, state_info, obs_type ="paddle", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)
    
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        goal_low = [-self.table_x_bot , self.table_y_left]
        goal_high = [0, self.table_y_right] # set goal positions to be in the top half of the table.
        
        # sample goal position
        min_y = goal_low[1]
        max_y = goal_high[1]
        min_x = goal_low[0]
        max_x = goal_high[0]
        
        goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y)) if goal_pos is None else goal_pos
        self.goal_pos = goal_position 