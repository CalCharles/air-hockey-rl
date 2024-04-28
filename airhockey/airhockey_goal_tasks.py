import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .airhockey_base import AirHockeyBaseEnv
from abc import ABC, abstractmethod


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
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None, goal_set=None):
        pass
    
    @abstractmethod
    def validate_configuration(self):
        pass
    
    @abstractmethod
    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                       puck_within_alt_home, puck_within_goal,
                       goal_pos, goal_radius):
        pass
        
    def get_goal_obs_space(self, low: list, high: list, goal_low: list, goal_high: list):
        return spaces.Dict(dict(
            observation=Box(low=np.array(low), high=np.array(high), dtype=float),
            desired_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float),
            achieved_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float)
        ))
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set


class AirHockeyPuckGoalPositionEnv(AirHockeyGoalEnv):
    def initialize_spaces(self):
        # setup observation / action / reward spaces
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel]

        low = paddle_obs_low + puck_obs_low
        high = paddle_obs_high + puck_obs_high
        self.observation_space = self.get_obs_space(low, high)

        goal_low = np.array([self.table_x_top, self.table_y_left])#, -self.max_paddle_vel, self.max_paddle_vel])
        goal_high = np.array([0, self.table_y_right])#, self.max_paddle_vel, self.max_paddle_vel])
        self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
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
        position = self.ego_goal_pos
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
        radius = self.ego_goal_radius
        reward_raw = 1 - (dist / radius) #self.max_goal_rew_radius * radius)
        reward_mask = dist >= radius
        reward_raw[reward_mask] = 0 # numerical stability, we will make these 0 later
        reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        reward[reward_mask] = 0

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
    
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_radius_type == 'fixed':
            # ego_goal_radius = self.rng.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
            base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
            # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
            ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
            ego_goal_radius = ratio * base_radius
            self.ego_goal_radius = ego_goal_radius

        if ego_goal_pos is None and goal_set is None:
            min_y = self.table_y_left + self.ego_goal_radius
            max_y = self.table_y_right - self.ego_goal_radius
            max_x = 0 - self.ego_goal_radius
            min_x = self.table_x_top + self.ego_goal_radius
            self.ego_goal_pos = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        else:
            self.ego_goal_pos = ego_goal_pos if self.goal_set is None else self.goal_set[0]
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set
    
    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                       puck_within_alt_home, puck_within_goal,
                       goal_pos, goal_radius):
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

        goal_low = np.array([self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel])
        goal_high = np.array([0, self.table_y_right, self.max_puck_vel, self.max_puck_vel])
        self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
    
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
        position = self.ego_goal_pos
        velocity = self.ego_goal_vel
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
        radius = self.ego_goal_radius
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
    
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_radius_type == 'fixed':
            # ego_goal_radius = self.rng.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
            base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
            # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
            ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
            ego_goal_radius = ratio * base_radius
            self.ego_goal_radius = ego_goal_radius

        if ego_goal_pos is None and goal_set is None:
            min_y = self.table_y_left + self.ego_goal_radius
            max_y = self.table_y_right - self.ego_goal_radius
            max_x = 0 - self.ego_goal_radius
            min_x = self.table_x_top + self.ego_goal_radius
            self.ego_goal_pos = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
            min_x_vel = self.goal_min_x_velocity
            max_x_vel = self.goal_max_x_velocity
            min_y_vel = self.goal_min_y_velocity
            max_y_vel = self.goal_max_y_velocity
            self.ego_goal_vel = self.rng.uniform(low=(min_x_vel, min_y_vel), high=(max_x_vel, max_y_vel))
        else:
            self.ego_goal_pos = ego_goal_pos if self.goal_set is None else self.goal_set[0]
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set
    
    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                    puck_within_alt_home, puck_within_goal,
                    goal_pos, goal_radius):
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

        goal_low = np.array([0, self.table_y_left])
        goal_high = np.array([self.table_x_bot, self.table_y_right])

        self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
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
        position = self.ego_goal_pos
        velocity = self.ego_goal_vel
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
    
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_radius_type == 'fixed':
            # ego_goal_radius = self.rng.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
            base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
            # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
            ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
            ego_goal_radius = ratio * base_radius
            self.ego_goal_radius = ego_goal_radius

        if ego_goal_pos is None and goal_set is None:
            min_y = self.table_y_left + self.ego_goal_radius
            max_y = self.table_y_right - self.ego_goal_radius
            max_x = 0 - self.ego_goal_radius
            min_x = self.table_x_top + self.ego_goal_radius
            self.ego_goal_pos = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
            min_x_vel = self.goal_min_x_velocity
            max_x_vel = self.goal_max_x_velocity
            min_y_vel = self.goal_min_y_velocity
            max_y_vel = self.goal_max_y_velocity
            self.ego_goal_vel = self.rng.uniform(low=(min_x_vel, min_y_vel), high=(max_x_vel, max_y_vel))
        else:
            self.ego_goal_pos = ego_goal_pos if self.goal_set is None else self.goal_set[0]
        
    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                    puck_within_alt_home, puck_within_goal,
                    goal_pos, goal_radius):
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

        goal_low = np.array([0, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel])
        goal_high = np.array([self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel])

        self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        
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
        position = self.ego_goal_pos
        velocity = self.ego_goal_vel
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
    
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_radius_type == 'fixed':
            # ego_goal_radius = self.rng.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
            base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
            # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
            ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
            ego_goal_radius = ratio * base_radius
            self.ego_goal_radius = ego_goal_radius

        if ego_goal_pos is None and goal_set is None:
            min_y = self.table_y_left + self.ego_goal_radius
            max_y = self.table_y_right - self.ego_goal_radius
            max_x = 0 - self.ego_goal_radius
            min_x = self.table_x_top + self.ego_goal_radius
            self.ego_goal_pos = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
            min_x_vel = self.goal_min_x_velocity
            max_x_vel = self.goal_max_x_velocity
            min_y_vel = self.goal_min_y_velocity
            max_y_vel = self.goal_max_y_velocity
            self.ego_goal_vel = self.rng.uniform(low=(min_x_vel, min_y_vel), high=(max_x_vel, max_y_vel))
        else:
            self.ego_goal_pos = ego_goal_pos if self.goal_set is None else self.goal_set[0]
        
    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                    puck_within_alt_home, puck_within_goal,
                    goal_pos, goal_radius):
        reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
        success = self.latest_pos_reward >= 0.9 and self.latest_vel_reward >= 0.8
        success = success.item()
        return reward, success
