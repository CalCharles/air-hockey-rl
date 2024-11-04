from .airhockey_reward_base import AirHockeyRewardBase
import numpy as np

class AirHockeyPuckVelReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))

        # reward for positive velocity towards the top of the board
        puck_vel = -state_info['pucks'][0]['velocity'][0]
        puck_height = -puck_pos[0]

        reward = max(puck_vel * 5, 0) # + 0.5 / dist
        success = puck_height > 0.5 and self.task_env.current_timestep > 25
        return reward, success


class AirHockeyPuckHeightReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        
    def get_base_reward(self, state_info):
        puck_height = -state_info['pucks'][0]['position'][0]
        puck_vel = -state_info['pucks'][0]['velocity'][0]
        puck_pos = state_info['pucks'][0]['position']

        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))

        reward = max(puck_vel, 0) * 5 if puck_height < 0 else max(puck_vel, 0) * -10
        success = puck_height > 0 and self.task_env.current_timestep > 25

        if dist - min_dist < 0.05:
            if not self.task_env.touching:
                reward += 20
                self.task_env.num_touches += 1
            self.task_env.touching = True
        else:
            self.task_env.touching = False

        if success:
            reward = 60
        return reward, success


class AirHockeyPuckCatchReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.task_env.width
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        success = reward >= 0.9 and self.task_env.current_timestep > 75
        return reward, success


class AirHockeyPuckJuggleReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        self.hit_counter = 0
        self.hit_cooldown = False

    def get_base_reward(self, state_info):
        reward = self.original_region_reward(state_info) + self.top_bumping_reward(state_info)
        success = reward > 0 and self.task_env.current_timestep > 50
        return reward, success
    
    def top_bumping_reward(self, state_info):
        bump_top = state_info['paddles']['paddle_ego']['position'][0] < 0 + 4 * self.task_env.paddle_radius
        
        if bump_top:
            return -5
        
        return 0

    def original_region_reward(self, state_info):
        reward = 0
        
        for puck in state_info["pucks"]:
            x_pos = puck['position'][0]
            x_higher = self.task_env.table_x_top
            x_lower = self.task_env.table_x_bot
            if x_higher / 4 < x_pos < 0:
                reward += 15 / len(state_info["pucks"])
            elif x_pos < x_higher / 4:
                reward -= 1 / len(state_info["pucks"])
        
        return reward

    def vel_reward(self, state_info):
        reward = 0
        max_vel = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            vel = puck['velocity'][0]
            if 0.1 < vel < 0.5:
                reward += (vel / max_vel) * 5 / len(state_info["pucks"])
            elif vel > 0.5:
                reward -= (vel / max_vel) * 5 / len(state_info["pucks"])
        return reward

    def low_vel_x_correc_region_reward(self, state_info, min_vel=0, max_vel=0.3):
        reward = 0
        max_expected = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            vel = puck['velocity'][0]
            x_pos = puck['position'][0]
            if min_vel < vel < max_vel and self.task_env.table_x_top / 4 < x_pos < 0:
                reward += 1 / len(state_info["pucks"])
            else:
                reward -= 0.05
        return reward
    
    def x_potential_reward(self, state_info):
        reward = 0
        max_distance = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            x_pos = puck['position'][0]
            target_pos = self.task_env.table_x_top * 1 / 2
            distance_to_target = abs(x_pos - target_pos)
            reward += (distance_to_target / max_distance) * 10 / len(state_info["pucks"])
        return reward

    def hit_reward(self, state_info):
        reward = 0
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius

        for puck in state_info["pucks"]:
            puck_pos = puck['position']
            dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
            if not self.hit_cooldown and abs(dist - min_dist) < 0.02:
                self.hit_counter += 1
                self.hit_cooldown = True
                reward += 2  # 2 points for each hit
            elif self.hit_cooldown and dist > (min_dist + 0.1):
                self.hit_cooldown = False
        
        return reward

    def hit_low_vel_potential(self, state_info, min_vel=0, max_vel=0.3):
        reward = self.hit_reward(state_info)
        max_distance = self.task_env.table_x_bot * 2
        for puck in state_info["pucks"]:
            x_pos = puck['position'][0]
            target_pos = self.task_env.table_x_top * 3 / 4
            distance_to_target = abs(x_pos - target_pos)
            vel = puck['velocity'][0]
            if min_vel < vel < max_vel:
                reward += (distance_to_target / max_distance) * 10 / len(state_info["pucks"])
        return reward

    def y_position_reward(self, state_info):
        reward = 0
        for puck in state_info["pucks"]:
            y_pos = puck['position'][1]
            if -self.task_env.width / 4 < y_pos < self.task_env.width / 4:
                reward += 5 / len(state_info["pucks"])
        return reward

    def combo_hits_reward(self, state_info):
        reward = self.hit_reward(state_info)
        if self.hit_counter > 1:
            reward = 5 * (self.hit_counter - 1)
        return reward

class AirHockeyPuckStrikeReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)
        
    def get_base_reward(self, state_info):
        x_vel = state_info['pucks'][0]['velocity'][0]
        y_vel = state_info['pucks'][0]['velocity'][1]
        vel_mag = np.linalg.norm(np.array([x_vel, y_vel]))
        reward = vel_mag
        max_rew = 2  # estimated max vel
        min_rew = 0  # min acceptable good velocity

        initial_pos = self.task_env.puck_initial_position
        current_pos = state_info['pucks'][0]['position']
        dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
        has_moved = dist > 0.01

        if reward <= min_rew and not has_moved:
            return -5, False  # negative rew for standing still and hasn't moved
        reward = min(reward, max_rew)
        reward = (reward - min_rew) / (max_rew - min_rew)
        success = reward > (0.1)  # means the puck is moving
        if reward > 0:
            reward *= 10
        return reward, success


class AirHockeyPuckTouchReward(AirHockeyRewardBase):
    def __init__(self, task_env):
        super().__init__(task_env)

    def get_base_reward(self, state_info):
        # reward for getting close to the puck, but make sure not to displace it
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        min_dist = self.task_env.paddle_radius + self.task_env.puck_radius
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        max_dist = 0.16 * self.task_env.width
        reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
        reward = max(reward, 0)
        # let's also make sure puck does not deviate from initial position
        puck_initial_position = self.task_env.puck_initial_position
        puck_current_position = state_info['pucks'][0]['position']
        delta = np.linalg.norm(np.array(puck_initial_position) - np.array(puck_current_position))
        epsilon = 0.01 + min_dist

        success = dist < (self.task_env.paddle_radius + self.task_env.puck_radius + 0.02)

        if reward > 0:
            reward *= 20  # make it more significant

        return reward, success
