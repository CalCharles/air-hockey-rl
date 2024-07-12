import numpy as np
from .airhockey_reward_base import AirHockeyRewardBase

class AirHockeyMoveBlockReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # also reward hitting puck! some shaping here :)
        vel_reward = -state_info['pucks'][0]['velocity'][0]
        max_rew = 2 # estimated max vel
        min_rew = 0  # min acceptable good velocity
        if vel_reward <= min_rew:
            vel_reward = 0
        else:
            vel_reward = min(vel_reward, max_rew)
            vel_reward = (vel_reward - min_rew) / (vel_reward - min_rew)
        
        # more reward if we move the block away from initial position
        block_initial_pos = state_info['blocks'][0]['initial_position']
        block_pos = state_info['blocks'][0]['current_position']
        dist = np.linalg.norm(np.array(block_pos) - np.array(block_initial_pos))
        max_euclidean_distance = np.linalg.norm(np.array([self.task_env.table_x_bot, self.task_env.table_y_right]) - np.array([self.task_env.table_x_top, self.task_env.table_y_left]))
        reward = 5000 * dist / max_euclidean_distance # big reward since its sparse!
        success = reward > 1 and self.task_env.current_timestep > 5
        return vel_reward + reward, success

class AirHockeyStrikeCrowdReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # also reward hitting puck! some shaping here :)
        vel_reward = -state_info['pucks'][0]['velocity'][0]
        max_rew = 2 # estimated max vel
        min_rew = 0  # min acceptable good velocity
        if vel_reward <= min_rew:
            vel_reward = 0
        else:
            vel_reward = min(vel_reward, max_rew)
            vel_reward = (vel_reward - min_rew) / (vel_reward - min_rew)
        # check how much blocks deviate from initial position
        reward = 0.0
        for block in state_info['blocks']:
            initial_pos = block['initial_position']
            current_pos = block['current_position']
            dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
            max_euclidean_distance = np.linalg.norm(np.array([self.task_env.table_x_bot, self.task_env.table_y_right]) - np.array([self.task_env.table_x_top, self.task_env.table_y_left]))
            reward += 10 * dist / max_euclidean_distance
        success = reward > 1 and self.task_env.current_timestep > 3
        return reward, success
