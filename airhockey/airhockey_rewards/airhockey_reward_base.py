import math
import numpy as np
from abc import ABC, abstractmethod

class AirHockeyRewardBase(ABC):
    
    def __init__(self, task_env):
        self.task_env = task_env

    @abstractmethod
    def get_base_reward(self, state_info):
        pass