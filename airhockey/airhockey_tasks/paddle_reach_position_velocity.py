import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPaddleReachPositionVelocityReward

class AirHockeyPaddleReachPositionVelocityEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        self.goal_radius_type = kwargs['goal_radius_type']
        self.base_goal_radius = kwargs['base_goal_radius']
        super().__init__(**kwargs)
        
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        goal_low = [0, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        goal_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]

        if self.return_goal_obs:
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + goal_low
            high = high + goal_high
            self.observation_space = self.get_obs_space(low, high)

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4
        self.goal_radius = self.base_goal_radius

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPaddleReachPositionVelocityReward(self)
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPaddleReachPositionVelocityEnv(**state_dict)
        
    def create_world_objects(self):
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
        
    def validate_configuration(self):
        assert self.num_pucks == 0
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    def get_achieved_goal(self, state_info):
        position = state_info['paddles']['paddle_ego']['position']
        velocity = state_info['paddles']['paddle_ego']['velocity']
        return np.array([position[0], position[1], velocity[0], velocity[1]])
    
    def get_desired_goal(self):
        position = self.goal_pos
        velocity = self.goal_vel
        return np.array([position[0], position[1], velocity[0], velocity[1]])
    
    def get_observation(self, state_info, obs_type ="paddle", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
    #     return obs
    
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        # sample goal position
        min_y = self.table_y_left + 2 * self.paddle_radius # Not too close to the wall
        max_y = self.table_y_right - 2 * self.paddle_radius # Not too close to the wall
        min_x = 0 - self.paddle_radius # some buffer space from halfway point
        max_x = self.table_x_bot + 2 * self.paddle_radius # Not too close to the wall
        goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        goal_velocity = self.rng.uniform(low=(-self.max_paddle_vel, -self.max_paddle_vel), high=(self.max_paddle_vel, self.max_paddle_vel))
        # x vel shouldn't vary much
        # "minimum" is upward at max speed, "maximum" is slightly upwards, otherwise can't reach goal
        x_vel = self.rng.uniform(low=-self.max_paddle_vel, high=-self.max_paddle_vel / 8) # only upwards
        y_vel = self.rng.uniform(low=-self.max_paddle_vel / 2, high=self.max_paddle_vel / 2)
        goal_velocity = np.array([x_vel, y_vel])
        # y vel should be positive
        self.goal_radius = self.min_goal_radius # not too important
        self.goal_pos = goal_position if self.goal_set is None else self.goal_set[0, :2]
        self.goal_vel = goal_velocity if self.goal_set is None else self.goal_set[0, 2:]