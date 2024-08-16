import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from abc import ABC, abstractmethod

from airhockey.airhockey_rewards.goal_task_rewards.airhockey_three_angle_task_reward import AirHockeyPuckThreeAngleReward
from airhockey.airhockey_tasks.abstract_airhockey_goal_task import AirHockeyGoalEnv

class AirHockeyPuckThreeAngleEnv(AirHockeyGoalEnv):        

    def initialize_spaces(self, obs_type):
        paddle_obs_low = [self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]
        paddle_obs_high = [self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel]
        puck_obs_low = [self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel, -np.pi]
        puck_obs_high = [self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel, np.pi]
        puck_hist_low = [self.table_x_top, self.table_y_left, 0] * 5
        puck_hist_high = [self.table_x_bot, self.table_y_right, 0] * 5

        if obs_type == "paddle":
            low = paddle_obs_low
            high = paddle_obs_high
        elif obs_type == "vel":
            low = paddle_obs_low + puck_obs_low[:4]
            high = paddle_obs_high + puck_obs_high[:4]
        elif obs_type == "history":
            low = paddle_obs_low + puck_hist_low
            high = paddle_obs_high + puck_hist_high
        elif obs_type == "angle_vel":
            low = paddle_obs_low + puck_obs_low + [0] * 3
            high = paddle_obs_high + puck_obs_high + [1] * 3

        self.observation_space = self.single_observation_space = self.get_obs_space(low, high)
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.angles = [np.pi/2, np.pi/4, 3*np.pi/4]
        self.success_threshold_deg = 10
        self.reward = AirHockeyPuckThreeAngleReward(self)

    def create_world_objects(self):
        name = 'puck_{}'.format(0)
        y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)
        pos = (self.table_x_top + 1.1, y_pos)
        vel = (1, 0)
        self.simulator.spawn_puck(pos, vel, name)

        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
    
    def create_world_objects_from_state(self, state_vector):
        name = 'puck_{}'.format(0)
        puck_pos, puck_vel = state_vector[:2], state_vector[2:4]
        self.simulator.spawn_puck(puck_pos, puck_vel, name)

        name = 'paddle_ego'
        paddle_pos, paddle_vel = state_vector[4:6], state_vector[6:]
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, name)

    def get_achieved_goal(self, state_info):
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]

        puck_vel_angle = np.arctan2(puck_y_vel, puck_x_vel)

        return np.array([puck_vel_angle])
    
    def get_desired_goal(self):
        return self.goal_angle

    def get_observation(self, state_info, obs_type='angle_vel', puck_history=None):
        obs = self.get_observation_by_type(state_info, obs_type)
        return obs
    
    def set_goals(self, goal_radius_type=None, goal_pos=None, alt_goal_pos=None, goal_set=None):
        angle = self.rng.choice(self.angles)
        goal_one_hot = np.zeros(3)
        goal_one_hot[self.angles.index(angle)] = 1
        
        self.goal_angle = goal_one_hot
    
    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1
    
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckThreeAngleEnv(**state_dict)