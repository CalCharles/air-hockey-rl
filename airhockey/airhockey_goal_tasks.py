import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .airhockey_base import AirHockeyBaseEnv
from abc import ABC, abstractmethod
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class AirHockeyGoalEnv(AirHockeyBaseEnv, ABC):        
    @abstractmethod
    def initialize_spaces(self):
        pass

    @abstractmethod
    def create_world_objects(self):
        pass
    
    @abstractmethod
    def get_achieved_goal(self, state_info):
        pass
    
    @abstractmethod
    def get_desired_goal(self):
        pass
    
    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    @abstractmethod
    def get_observation(self, state_info):
        pass
    
    @abstractmethod
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        pass
    
    @abstractmethod
    def validate_configuration(self):
        pass
    
    @abstractmethod
    def from_dict(state_dict):
        pass
    
    @abstractmethod
    def get_base_reward(self, state_info):
        pass
        
    def get_goal_obs_space(self, low: list, high: list, goal_low: list, goal_high: list):
        return spaces.Dict(dict(
            observation=Box(low=np.array(low), high=np.array(high), dtype=float),
            desired_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float),
            achieved_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float)
        ))
        
    def reset(self, seed=None):
        self.set_goals(self.goal_radius_type)
        obs, success = super().reset(seed)
        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, success
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, success
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set

    def step(self, action):
        obs, reward, is_finished, truncated, info = super().step(action)
        info['ego_goal'] = self.goal_pos
        achieved_goal = self.get_achieved_goal(self.current_state)
        desired_goal = self.get_desired_goal()
        if self.return_goal_obs:
            return {"observation": obs, "desired_goal": desired_goal, "achieved_goal": achieved_goal}, reward, is_finished, truncated, info
        else:
            obs = np.concatenate([obs, desired_goal])
            return obs, reward, is_finished, truncated, info

class AirHockeyPuckGoalPositionEnv(AirHockeyGoalEnv):
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
            reward = -dist  + (bonus if dist < radius else 0)

        if single:
            reward = reward[0]
        return reward

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs
    
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
        else:
            self.goal_pos = goal_pos if self.goal_set is None else self.goal_set[0]
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set
    
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.0
        success = success.item()
        return reward, success
    
class AirHockeyPuckGoalPositionVelocityEnv(AirHockeyGoalEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high
        goal_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        goal_high = [0, self.table_y_right, self.max_puck_vel, self.max_puck_vel]
        
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
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        # compute angle between velocities
        denom = np.linalg.norm(achieved_goal[:, 2:], axis=1) * np.linalg.norm(desired_goal[:, 2:], axis=1) + 1e-8
        vel_cos = np.sum(achieved_goal[:, 2:] * desired_goal[:, 2:], axis=1) / denom
        
        # numerical stability
        vel_cos = np.clip(vel_cos, -1, 1)
        vel_angle = np.arccos(vel_cos)
        # mag difference
        mag_achieved = np.linalg.norm(achieved_goal[:, 2:], axis=1)
        mag_desired = np.linalg.norm(desired_goal[:, 2:], axis=1)
        mag_diff = np.abs(mag_achieved - mag_desired)
        
        # # also return float from [0, 1] 0 being far 1 being the point
        # # use sigmoid function because being closer is much more important than being far
        sigmoid_scale = 2
        radius = self.goal_radius
        reward_raw = 1 - (dist / radius)#self.max_goal_rew_radius * radius)
        
        mask = dist >= radius
        reward_raw[mask] = 0 # numerical stability, we will make these 0 later
        reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        reward_mask = dist >= radius
        reward[reward_mask] = 0
        position_reward = reward

        vel_mag_reward = 1 - mag_diff / self.max_paddle_vel
        
        reward_mask = position_reward == 0
        norm_cos_sim = (vel_cos + 1) / 2
        vel_angle_reward = norm_cos_sim
        vel_angle_reward[reward_mask] = 0
        vel_mag_reward[reward_mask] = 0
        vel_reward = (vel_angle_reward + vel_mag_reward) / 2
        
        # reward = (position_reward + vel_reward + vel_mag_reward) / 3
        reward = 0.5 * position_reward + 0.5 * vel_reward

        if single:
            reward = reward[0]
        return reward

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs
    
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
    
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.0
        success = success.item()
        return reward, success

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

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
        return obs
    
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
        
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.9
        success = success.item()
        return reward, success

class AirHockeyPaddleReachPositionVelocityEnv(AirHockeyGoalEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]

        low = paddle_obs_low
        high = paddle_obs_high

        goal_low = [0, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        goal_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]

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
        pos_reward = 1 - (dist / max_euclidean_distance)
        # vel reward
        current_vel = achieved_goal[:, 2:]
        goal_vel = achieved_goal[:, 2:]
        # dist = np.linalg.norm(np.array(current_vel) - np.array(goal_vel))
        mag_current = np.linalg.norm(current_vel, axis=1)
        mag_goal = np.linalg.norm(goal_vel, axis=1)
        mag_diff = np.abs(mag_current - mag_goal)
        maximum_mag_diff = np.abs(np.linalg.norm(np.array([self.max_paddle_vel, self.max_paddle_vel]) - np.array([0, 0])))
        vel_mag_reward = 1 - mag_diff / maximum_mag_diff

        dist = np.linalg.norm(current_vel - goal_vel)
        # compute angle between velocities
        denom = mag_current * mag_goal + 1e-8
        vel_cos = np.sum(current_vel * goal_vel) / denom
            
        # numerical stability
        vel_cos = np.clip(vel_cos, -1, 1)
        # vel_angle = np.arccos(vel_cos)

        norm_cos_sim = (vel_cos + 1) / 2
        vel_angle_reward = norm_cos_sim
        vel_reward = (vel_angle_reward + vel_mag_reward) / 2
        
        # # let's try old vel rew lol
        # vel_distance = np.linalg.norm(np.array(current_vel) - np.array(goal_vel))
        # max_vel_distance = np.linalg.norm(np.array([self.max_paddle_vel, self.max_paddle_vel]))
        # vel_reward = 1 - (vel_distance / max_vel_distance)
        
        self.latest_pos_reward = pos_reward
        self.latest_vel_reward = vel_reward
        reward = 0.5 * pos_reward + 0.5 * vel_reward

        if single:
            reward = reward[0]
        return reward

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
        return obs
    
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

    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = self.latest_pos_reward >= 0.9 and self.latest_vel_reward >= 0.8
        success = success.item()
        return reward, success

class RewardRegion():
    def __init__(self, reward_value_range, scale_range, limits, rad_limits, shapes, reset=True):
        self.reward_value_range = reward_value_range
        self.scale_range = scale_range
        self.shapes = shapes
        self.shape_onehot_helper = np.eye(len(shapes))
        self.limits = limits
        self.limit_range = self.limits[1] - self.limits[0]
        self.rad_limits = rad_limits
        if reset: self.reset()

    def reset(self):
        self.state = np.random.rand(*self.limits[0].shape) * self.limit_range + self.limits[0]
        self.reward_value = np.random.rand() * (self.reward_value_range[1] - self.reward_value_range[0]) + self.reward_value_range[0]
        self.scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        self.shape_idx = np.random.randint(len(self.shapes))
        self.shape = self.shapes[self.shape_idx]
        self.radius = np.random.rand(*self.rad_limits[1].shape) * (self.rad_limits[1] - self.rad_limits[0]) + self.rad_limits[0]
        if (self.shape == "circle" or self.shape == "square") and len(self.rad_limits[0]) > 1: self.radius = self.radius[0]

    def get_state(self):
        if isinstance(self.radius, Iterable):
            radius = np.pad(self.radius, (0,3-len(self.radius)), constant_values=self.radius[0])
        else:
            radius = np.pad(np.array([self.radius]), (0,2), constant_values=self.radius)
        return np.concatenate([self.state, [self.scale], [self.reward_value], radius, self.shape_onehot_helper[self.shape_idx]])

    def check_reward(self, obj_state):
        if self.shape == "circle" or self.shape == "ellipse":
            norm_dist = np.sum(np.square(obj_state - self.state) / np.square(self.radius))
            # print(np.linalg.norm(obj_state - self.state), self.radius, norm_dist)
        elif self.shape == "diamond":
            norm_dist = np.sum(np.abs(obj_state - self.state) / self.radius)
        elif self.shape == "rect" or self.shape == "square":
            norm_dist = np.max(np.abs(obj_state - self.state) / self.radius)
        # print("region reward", float(norm_dist <= 1) * np.exp(-self.scale * norm_dist) * self.reward_value)
        return float(norm_dist <= 1) * np.exp(-self.scale * norm_dist) * self.reward_value

class DynamicRewardRegion(RewardRegion):
    def __init__(self, reward_value_range, scale_range, limits, rad_limits, shapes, movement_patterns, velocity_limits, use_reset=True):
        super().__init__(reward_value_range, scale_range, limits, rad_limits, shapes, reset=False)
        self.movement_patterns = movement_patterns
        self.movement_onehot_helper = np.eye(len(movement_patterns))
        self.velocity_limits = velocity_limits
        self.velocity_limit_range = self.velocity_limits[1] - self.velocity_limits[0]
        if use_reset: self.reset()

    def reset(self):
        super().reset()
        self.velocity = np.random.rand(self.velocity_limits[0].shape) * self.velocity_limit_range + self.velocity_limits[0]
        self.movement_idx = np.random.randint(len(self.movement_patterns))
        self.movement = self.movement_patterns[self.movement_idx]

    def get_state(self):
        return np.concatenate([self.state, self.velocity, [self.scale], [self.reward_value], [self.radius], self.shape_onehot_helper[self.shape_idx], self.movement_onehot_helper[self.movement_idx]])

    def step(self, env_state, action):
        next_state = self.state + self.velocity
        hit_top_lim = next_state[1] > self.limits[1][1]
        hit_bot_lim = next_state[1] < self.limits[0][1]
        hit_right_lim = next_state[0] > self.limits[1][0]
        hit_left_lim = next_state[0] < self.limits[0][0]
        hit = hit_top_lim and hit_bot_lim and hit_right_lim and hit_left_lim
        if hit:
            if self.movement == "bounce":
                if hit_top_lim or hit_bot_lim:
                    self.velocity[1] = -self.velocity[1]
                    next_state = self.state + self.velocity
                if hit_right_lim or hit_left_lim:
                    self.velocity[0] = -self.velocity[0]
                    next_state = self.state + self.velocity
            elif self.movement == "through":
                if hit_top_lim:
                    next_state[1] = self.limits[0][1]
                if hit_bot_lim:
                    next_state[1] = self.limits[1][1]
                if hit_right_lim:
                    next_state[0] = self.limits[0][0]
                if hit_left_lim:
                    next_state[0] = self.limits[1][0]
            elif self.movement == "top_reset":
                next_state[1] = self.limits[0][1]
                next_state[0] = np.random.rand() * (self.limits[0][1] - self.limits[0][0]) + self.limits[0][0]
        self.state = next_state

class AirHockeyPaddleReachPositionNegRegionsEnv(AirHockeyGoalEnv):
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
        self.reward_velocity_limits_min = reward_normalized_radius_min
        self.reward_velocity_limits_max = reward_normalized_radius_max
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
        return AirHockeyPaddleReachPositionNegRegionsEnv(**state_dict)


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

        radius_range = np.array((self.table_x_bot/ 2 * np.array(self.reward_normalized_radius_min), (self.table_y_right - self.table_y_left) * np.array(self.reward_normalized_radius_max) / 2))
        self.reward_regions = [RewardRegion(self.negative_reward_range, 
                                                     self.reward_region_scale_range, 
                                                     [np.array([0, self.table_y_left]), np.array([self.table_x_bot,self.table_y_right])],
                                                     radius_range, shapes=self.reward_region_shapes) for _ in range(self.num_negative_reward_regions)]

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
        assert self.num_negative_reward_regions > 0
    
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

        for nrr in self.reward_regions:
            reward += nrr.check_reward(achieved_goal)

        print(dist / max_euclidean_distance, 1 - (dist / max_euclidean_distance), reward)
        if single:
            reward = reward[0]
        return reward

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        reward_regions_states = [nrr.get_state() for nrr in self.reward_regions]

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
        return np.concatenate([obs] + reward_regions_states)
    
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
            
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.9
        success = success.item()
        return reward, success

    def reset(self, seed=None):
        for nrr in self.reward_regions:
            nrr.reset()
        return super().reset(seed)

class AirHockeyPuckReachPositionDynamicNegRegionsEnv(AirHockeyGoalEnv):
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
        return AirHockeyPuckReachPositionDynamicNegRegionsEnv(**state_dict)

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

        self.dynamic_virtual_objects = self.reward_regions
        
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
            reward = -dist  + (bonus if dist < radius else 0)

        if single:
            reward = reward[0]
        return reward

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       
        reward_regions_states = [nrr.get_state() for nrr in self.reward_regions]


        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return np.concatenate([obs] + reward_regions_states)
    
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
        else:
            self.goal_pos = goal_pos if self.goal_set is None else self.goal_set[0]
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set
    
    def get_base_reward(self, state_info):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = reward > 0.0
        success = success.item()
        return reward, success
    
    def reset(self, seed=None):
        for nrr in self.reward_regions:
            nrr.reset()
        return super().reset(seed)
