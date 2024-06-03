import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv

class AirHockeyPaddleReachPositionEnv(AirHockeyGoalEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]

        low = paddle_obs_low
        high = paddle_obs_high

        goal_low = [0, self.table_y_left]
        goal_high = [self.table_x_bot, self.table_y_right]

        if self.return_goal_obs:
            low = paddle_obs_low
            high = paddle_obs_high
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = paddle_obs_low + goal_low
            high = paddle_obs_high + goal_high
            self.observation_space = self.get_obs_space(low, high)
        
        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
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
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
        # reward for closer to goal
        reward = 1 - (dist / max_euclidean_distance)

        if single:
            reward = reward[0]
        return reward
    
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
        min_y = self.table_y_left
        max_y = self.table_y_right
        min_x = 0
        max_x = self.table_x_bot
        goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        self.goal_radius = self.min_goal_radius # not too important
        self.goal_pos = goal_position if self.goal_set is None else self.goal_set[0, :2]
        self.goal_pos = goal_pos if goal_pos is not None else self.goal_pos
        
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.9
        success = success.item()
        return reward, success