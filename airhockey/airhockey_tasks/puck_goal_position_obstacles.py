import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPuckGoalPositionReward, AirHockeyPaddleReachPositionReward

class AirHockeyPuckGoalPositionObstaclesEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        # TODO: rearrange defaults so that they happen before this assignment, instead of after
        self.goal_radius_type = kwargs['goal_radius_type']
        self.puck_goal_success_bonus = kwargs['puck_goal_success_bonus']
        self.paddle_puck_success_bonus = kwargs['paddle_puck_success_bonus']
        self.base_goal_radius = kwargs['base_goal_radius']
        self.num_obstacles = kwargs['num_obstacles']
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
        
        block_obs_low = [self.table_x_top, self.table_y_left, self.table_x_top, self.table_y_left]
        block_obs_high = [self.table_x_bot, self.table_y_right, self.table_x_bot, self.table_y_right]

        if obs_type == "paddle":
            low = paddle_obs_low
            high = paddle_obs_high
        elif obs_type == "vel":
            low = paddle_obs_low + puck_obs_low
            high = paddle_obs_high + puck_obs_high
        elif obs_type == "history":
            low = paddle_obs_low + puck_hist_low
            high = paddle_obs_high + puck_hist_high
        elif obs_type == "many_blocks_vel":
            low = paddle_obs_low + puck_obs_low + [block_obs_low[0], block_obs_low[1]] * self.num_blocks
            high = paddle_obs_high + puck_obs_high + [block_obs_high[0], block_obs_high[1]] * self.num_blocks
        elif obs_type == "many_blocks_history":
            low = paddle_obs_low + [block_obs_low[0], block_obs_low[1]] * self.num_blocks + puck_hist_low
            high = paddle_obs_high + [block_obs_high[0], block_obs_high[1]] * self.num_blocks + puck_hist_high
        
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
        return AirHockeyPuckGoalPositionObstaclesEnv(**state_dict)
        
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_obstacles == 0
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

    def create_world_objects(self):
        self.block_initial_positions = {}

        for i in range(self.num_blocks):
            # make sure obstacles are sufficiently large and dense
            center_y = self.rng.uniform(self.table_y_left, self.table_y_right) 
            center_x = self.rng.uniform(self.table_x_top, 0.0) 
            block_name = "block_" + str(i)
            pos = (center_x, center_y)
            self.block_initial_positions[block_name] = pos
            vel = (0, 0)
            self.simulator.spawn_block(pos, vel, block_name, affected_by_gravity=False)

        # pucks moving downwards that we want to hit directly
        name = 'puck_{}'.format(0)
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, name)

        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)

    def get_observation(self, state_info, obs_type='many_blocks_vel', **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)