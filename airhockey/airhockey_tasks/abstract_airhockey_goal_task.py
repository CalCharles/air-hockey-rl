import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from airhockey.airhockey_base import AirHockeyBaseEnv
from abc import ABC, abstractmethod
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
        
    def reset(self, seed=None, **kwargs):
        self.set_goals(self.goal_radius_type)
        obs, success = super().reset(seed, **kwargs)
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