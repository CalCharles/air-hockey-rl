from gymnasium import Env
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
import math


def get_box2d_simulator_fn():
    from airhockey.sims import AirHockeyBox2D
    return AirHockeyBox2D
    
def get_robosuite_simulator_fn():
    from air_hockey_challenge_robosuite.robosuite.wrappers.gym_wrapper import GymWrapper
    return GymWrapper


class AirHockeyEnv(Env):
    def __init__(self,
                 simulator, # box2d or robosuite
                 simulator_params,
                 task, 
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
                 seed,
                 max_timesteps=1000):
        
        if simulator == 'box2d':
            simulator_fn = get_box2d_simulator_fn()
        elif simulator == 'robosuite':
            simulator_fn = get_robosuite_simulator_fn()
        else:
            raise ValueError("Invalid simulator type. Must be 'box2d' or 'robosuite'.")

        simulator_params['task'] = task
        self.simulator = simulator_fn.from_dict(simulator_params)
        self.simulator_params = simulator_params

        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.n_training_steps = n_training_steps
        self.n_timesteps_so_far = 0
        self.rng = np.random.RandomState(seed)
        
        # termination conditions
        self.terminate_on_out_of_bounds = terminate_on_out_of_bounds
        self.terminate_on_enemy_goal = terminate_on_enemy_goal
        self.terminate_on_puck_stop = terminate_on_puck_stop
        
        # reward function
        self.goal_conditioned = True if 'goal' in task else False
        self.goal_radius_type = 'home'
        self.goal_min_x_velocity = -goal_max_x_velocity
        self.goal_max_x_velocity = goal_max_x_velocity
        self.goal_min_y_velocity = goal_min_y_velocity
        self.goal_max_y_velocity = goal_max_y_velocity
        self.reward_type = task
        self.multiagent = self.simulator_params['num_paddles'] == 2
        self.truncate_rew = truncate_rew
        self.wall_bumping_rew = wall_bumping_rew
        self.direction_change_rew = direction_change_rew
        self.horizontal_vel_rew = horizontal_vel_rew
        self.diagonal_motion_rew = diagonal_motion_rew
        self.stand_still_rew = stand_still_rew
        
        self.width = simulator_params['width']
        self.length = simulator_params['length']
        self.paddle_radius = simulator_params['paddle_radius']
        self.puck_radius = simulator_params['puck_radius']
        
        self.table_x_top = -self.length / 2
        self.table_x_bot = self.length / 2
        self.table_y_right = self.width / 2
        self.table_y_left = -self.width / 2
        self.max_paddle_vel = self.simulator.max_paddle_vel
        self.max_puck_vel = self.simulator.max_puck_vel
        self.goal_set = None
        self.initialize_spaces()
        
        self.metadata = {}
        self.reset()

    def initialize_spaces(self):
        # setup observation / action / reward spaces
        low = np.array([self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel, 
                        self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel])

        high = np.array([self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel, 
                         self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel])

        if self.reward_type == 'move_block':
            low = np.array([self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel,
                            self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel,
                            self.table_x_top, self.table_y_left, self.table_x_top, self.table_y_left])
            high = np.array([self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel,
                             self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel,
                             self.table_x_bot, self.table_y_right, self.table_x_bot, self.table_y_right])
            self.observation_space = Box(low=low, high=high, shape=(12,), dtype=float)
        elif self.reward_type == 'reach':
            # also include goal position, which will on bottom half of board
            low = np.array([self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel,
                            0, self.table_y_left]) 
            high = np.array([self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel,
                             self.table_x_bot, self.table_y_right])
            self.observation_space = Box(low=low, high=high, shape=(6,), dtype=float)
        elif self.reward_type == 'reach_vel':
            # also include goal position, which will on bottom half of board
            low = np.array([self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel,
                            0, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel]) 
            high = np.array([self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel,
                             self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel])
            self.observation_space = Box(low=low, high=high, shape=(8,), dtype=float)
            
        else:
            if not self.goal_conditioned:
                self.observation_space = Box(low=low, high=high, shape=(8,), dtype=float)
            else:

                if self.reward_type == 'goal_position':
                    # y, x
                    goal_low = np.array([self.table_x_top, self.table_y_left])#, -self.max_paddle_vel, self.max_paddle_vel])
                    goal_high = np.array([0, self.table_y_right])#, self.max_paddle_vel, self.max_paddle_vel])

                    self.observation_space = spaces.Dict(dict(
                        observation=Box(low=low, high=high, shape=(8,), dtype=float),
                        desired_goal=Box(low=goal_low, high=goal_high, shape=(2,), dtype=float),
                        achieved_goal=Box(low=goal_low, high=goal_high, shape=(2,), dtype=float)
                    ))

                elif self.reward_type == 'goal_position_velocity':
                    goal_low = np.array([self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel])
                    goal_high = np.array([0, self.table_y_right, self.max_puck_vel, self.max_puck_vel])
                    self.observation_space = spaces.Dict(dict(
                        observation=Box(low=low, high=high, shape=(8,), dtype=float),
                        desired_goal=Box(low=goal_low, high=goal_high, shape=(4,), dtype=float),
                        achieved_goal=Box(low=goal_low, high=goal_high, shape=(4,), dtype=float)
                    ))

                self.min_goal_radius = self.width / 16
                self.max_goal_radius = self.width / 4
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyEnv(**state_dict)

    def reset(self, seed=None, **kwargs):
        if seed is None: # determine next seed, in a deterministic manner
            seed = self.rng.randint(0, int(1e8))
        self.rng = np.random.RandomState(seed)
        state_info = self.simulator.reset(seed)
        # get initial observation
        self.set_goals(self.goal_radius_type)
        obs = self.get_observation(state_info)
        
        self.n_timesteps_so_far += self.current_timestep
        self.current_timestep = 0
        self.success_in_ep = False
        self.max_reward_in_single_step = -np.inf
        self.min_reward_in_single_step = np.inf
        
        self.puck_initial_position = state_info['pucks'][0]['position']
        
        if not self.goal_conditioned:
            return obs, {'success': False}
        else:
            return {"observation": obs, "desired_goal": self.get_desired_goal(), "achieved_goal": self.get_achieved_goal(state_info)}, {'success': False}
        
    def get_achieved_goal(self, state_info):
        if self.reward_type == 'goal_position':
            # numpy array containing puck position and vel
            position = np.array(state_info['pucks'][0]['position'])
            return position.astype(float)
        elif self.reward_type == 'goal_position_velocity':
            position = state_info['pucks'][0]['position']
            velocity = state_info['pucks'][0]['velocity']
            return np.array([position[0], position[1], velocity[0], velocity[1]])
        else:
            raise ValueError("Invalid reward type for goal conditioned environment. " +
                             "Should be goal_position or goal_position_velocity.")
    
    def get_desired_goal(self):
        position = self.ego_goal_pos
        if self.reward_type == 'goal_position':
            return position.astype(float)
        elif self.reward_type == 'goal_position_velocity':
            velocity = self.ego_goal_vel
            return np.array([position[0], position[1], velocity[0], velocity[1]])
        else:
            raise ValueError("Invalid reward type for goal conditioned environment. " +
                             "Should be goal_position or goal_position_velocity.")
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        if self.goal_conditioned:
            if achieved_goal.shape[1] == 2:
                # return euclidean distance between the two points
                dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
                sigmoid_scale = 2
                radius = self.ego_goal_radius
                reward_raw = 1 - (dist / radius) #self.max_goal_rew_radius * radius)
                reward_mask = dist >= radius
                reward_raw[reward_mask] = 0 # numerical stability, we will make these 0 later
                reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
                reward[reward_mask] = 0
            else:
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
        else:
            return self.get_reward(False, False, False, False, self.ego_goal_pos, self.ego_goal_radius)

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        
        if self.reward_type == 'reach' or self.reward_type == 'reach_vel':
            if self.reward_type == 'reach':
                goal_pos = self.reach_goal_pos
                obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, goal_pos[0], goal_pos[1]])
                return obs
            elif self.reward_type == 'reach_vel':
                goal_pos = self.reach_goal_pos
                goal_vel = self.reach_goal_vel
                obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, goal_pos[0], goal_pos[1], goal_vel[0], goal_vel[1]])
                return obs
        else:
            puck_x_pos = state_info['pucks'][0]['position'][0]
            puck_y_pos = state_info['pucks'][0]['position'][1]
            puck_x_vel = state_info['pucks'][0]['velocity'][0]
            puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        if self.reward_type == 'move_block':
            block_x_pos = state_info['blocks'][0]['current_position'][0]
            block_y_pos = state_info['blocks'][0]['current_position'][1]
            block_initial_x_pos = state_info['blocks'][0]['initial_position'][0]
            block_initial_y_pos = state_info['blocks'][0]['initial_position'][1]
            obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel, block_x_pos, block_y_pos, block_initial_x_pos, block_initial_y_pos])
            return obs
        if not self.multiagent:
            obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        else:
            alt_paddle_x_pos = state_info['paddles']['paddle_alt']['position'][0]
            alt_paddle_y_pos = state_info['paddles']['paddle_alt']['position'][1]
            alt_paddle_x_vel = state_info['paddles']['paddle_alt']['velocity'][0]
            alt_paddle_y_vel = state_info['paddles']['paddle_alt']['velocity'][1]
            
            obs_ego = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel,  puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
            obs_alt = np.array([-alt_paddle_x_pos, -alt_paddle_y_pos, alt_paddle_x_vel, alt_paddle_y_vel, -puck_x_pos, -puck_y_pos, -puck_x_vel, -puck_y_vel])
            obs = (obs_ego, obs_alt)
        return obs
    
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if self.reward_type == 'reach':
            # sample goal position
            min_y = self.table_y_left
            max_y = self.table_y_right
            min_x = 0
            max_x = self.table_x_bot
            goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
            self.reach_goal_pos = goal_position
        elif self.reward_type == 'reach_vel':
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
            self.reach_goal_pos = goal_position
            self.reach_goal_vel = goal_velocity
        if self.goal_conditioned:
            if goal_radius_type == 'fixed':
                # ego_goal_radius = self.rng.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
                base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
                # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
                ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
                ego_goal_radius = ratio * base_radius
                if self.multiagent:
                    alt_goal_radius = ego_goal_radius      
                self.ego_goal_radius = ego_goal_radius
                if self.multiagent:
                    self.alt_goal_radius = alt_goal_radius
            elif goal_radius_type == 'home':
                self.ego_goal_radius = 0.16 * self.width
                if self.multiagent:
                    self.alt_goal_radius = 0.16 * self.width
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
                
                if self.multiagent:
                    self.alt_goal_pos = self.rng.uniform(low=(0 - self.alt_goal_radius, self.table_y_left), high=(self.table_x_bot + self.alt_goal_radius, self.table_y_right))
            else:
                self.ego_goal_pos = ego_goal_pos if self.goal_set is None else self.goal_set[0]
                if self.multiagent:
                    self.alt_goal_pos = alt_goal_pos
        else:
            self.ego_goal_pos = None
            self.ego_goal_radius = None
            self.alt_goal_pos = None
            self.alt_goal_radius = None

    def has_finished(self, state_info, multiagent=False):
        truncated = False
        terminated = False
        puck_within_alt_home = False
        puck_within_home = False

        if self.current_timestep > self.max_timesteps:
            terminated = True
        else:
            if self.terminate_on_out_of_bounds:
                # check if we hit any walls or are above the middle of the board
                if state_info['paddles']['paddle_ego']['position'][0] < 0 + self.paddle_radius or \
                    state_info['paddles']['paddle_ego']['position'][0] > self.table_x_bot - self.paddle_radius or \
                    state_info['paddles']['paddle_ego']['position'][1] > self.table_y_right - self.paddle_radius or \
                    state_info['paddles']['paddle_ego']['position'][1] < self.table_y_left + self.paddle_radius:
                    truncated = True

        bottom_center_point = np.array([self.table_x_bot, 0])
        top_center_point = np.array([self.table_x_top, 0])
        
        if 'reach' not in self.reward_type:
            puck_within_home = self.is_within_home_region(bottom_center_point, state_info['pucks'][0]['position'])
            puck_within_alt_home = self.is_within_home_region(top_center_point, state_info['pucks'][0]['position'])
        else:
            puck_within_home = False
            puck_within_alt_home = False
        
        if self.terminate_on_enemy_goal:
            if not terminated and puck_within_home:
                truncated = True

        if multiagent:
            terminated = terminated or truncated or puck_within_alt_home or puck_within_home
            truncated = False
            
        if self.terminate_on_puck_stop:
            if not truncated and np.linalg.norm(state_info['pucks'][0]['velocity']) < 0.01:
                truncated = True

        puck_within_ego_goal = False
        puck_within_alt_goal = False

        if self.goal_conditioned:
            if self.is_within_goal_region(self.ego_goal_pos, state_info['pucks'][0]['position'], self.ego_goal_radius):
                puck_within_ego_goal = True
            if multiagent:
                if self.is_within_goal_region(self.alt_goal_pos, state_info['pucks'][0]['position'], self.alt_goal_radius):
                    puck_within_alt_goal = True

        return terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal
    
    def get_goal_region_reward(self, point, position, radius, discrete=True) -> float:
        point = np.array([point[0], point[1]])
        dist = np.linalg.norm(position - point)
        
        if discrete:
            return 1.0 if dist < radius else 0.0
        # also return float from [0, 1] 0 being far 1 being the point
        # use sigmoid function because being closer is much more important than being far
        sigmoid_scale = 2
        reward_raw = 1 - (dist / radius)
        reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        reward = 0 if dist >= radius else reward
        return reward

    def get_home_region_reward(self, point, position, discrete=True) -> float:
        return self.get_goal_region_reward(point, position, 0.16 * self.width, discrete=discrete)
    
    def is_within_goal_region(self, point, position, radius) -> bool:
        point = np.array([point[0], point[1]])
        dist = np.linalg.norm(position - point)
        return dist < radius
    
    def is_within_home_region(self, point, position) -> bool:
        return self.is_within_goal_region(point, position, 0.16 * self.width)
    
    def puck_reached(self, state_info):
        puck_pos = state_info['pucks'][0]['position']
        paddle_pos = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
        return dist <= self.paddle_radius + self.puck_radius

    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                       puck_within_alt_home, puck_within_goal,
                       goal_pos, goal_radius):
        if self.reward_type == 'goal_discrete':
            reward = self.get_goal_region_reward(goal_pos, state_info['pucks'][0]['position'], 
                                                 goal_radius, discrete=True)
            success = reward == 1
            return reward, success
        elif self.reward_type == 'goal_position' or self.reward_type == 'goal_position_velocity':
            
            reward = self.compute_reward(self.get_achieved_goal(state_info), self.get_desired_goal(), {})
            success = reward > 0.0
            # numpy bool to bool
            success = success.item()
            return reward, success
        elif self.reward_type == 'puck_juggle':
            reward = 0
            x_pos = state_info['pucks'][0]['position'][0]
            x_higher = self.table_x_top
            x_lower = self.table_x_bot
            if x_higher / 4 < x_pos < 0:
                reward += 15
            elif x_pos < x_higher / 4:
                reward -= 1
            success = reward > 0 and self.current_timestep > 50
            return reward, success
        elif self.reward_type == 'puck_height':
            reward = -state_info['pucks'][0]['position'][0]
            # min acceptable reward is 0 height and above
            reward = max(reward, 0)
            # let's normalize reward w.r.t. the top half length of the table
            # aka within the range [0, self.length / 2]
            max_rew = self.length / 2
            min_rew = 0
            reward = (reward - min_rew) / (max_rew - min_rew)
            success = reward > 0.5 and self.current_timestep > 25
            return reward, success
        elif self.reward_type == 'multipuck_juggle':
            reward = 0
            pos_rew = 15
            for puck in state_info['pucks']:
                x_pos = puck[0]['position'][0]
                x_higher = self.table_x_top
                x_lower = self.table_x_bot
                if x_higher / 4 < x_pos < 0:
                    reward += pos_rew
                elif x_pos < x_higher / 4 or x_pos > x_lower / 2:
                    reward -= 1
            success = reward >= pos_rew and self.current_timestep > 50
            return reward, success
        elif self.reward_type == 'strike':
            # alternative
            # let's check difference from initial position
            initial_pos = self.puck_initial_position
            current_pos = state_info['pucks'][0]['position']
            dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
            max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
            reward = 10 * dist / max_euclidean_distance
            success = reward > 2 and self.current_timestep > 3
            return reward, success
            
            # # reward for velocity
            # x_vel = state_info['pucks'][0]['velocity'][0]
            # y_vel = state_info['pucks'][0]['velocity'][1]
            # vel_mag = np.linalg.norm(np.array([x_vel, y_vel]))
            # reward = vel_mag
            # max_rew = 2 # estimated max vel
            # min_rew = 0  # min acceptable good velocity
            # if reward <= min_rew:
            #     return -5, False # negative rew for standing still
            # reward = min(reward, max_rew)
            # reward = (reward - min_rew) / (max_rew - min_rew)
            # success = reward > (0.3) # means the puck is moving at acceptable vel
            # return reward, success
        elif self.reward_type == 'strike_crowd':
            # check how much blocks deviate from initial position
            reward = 0.0
            for block in state_info['blocks']:
                initial_pos = block['initial_position']
                current_pos = block['current_position']
                dist = np.linalg.norm(np.array(initial_pos) - np.array(current_pos))
                max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
                reward += 10 * dist / max_euclidean_distance
            success = reward > 2 and self.current_timestep > 3
            return reward, success
        elif self.reward_type == 'puck_vel':
            # reward for positive velocity towards the top of the board
            reward = -state_info['pucks'][0]['velocity'][0]
            max_rew = 2 # estimated max vel
            min_rew = 0  # min acceptable good velocity
            if reward < min_rew:
                return 0, False
            reward = min(reward, max_rew)
            reward = (reward - min_rew) / (max_rew - min_rew)
            success = reward > 0.5 and self.current_timestep > 25
            return reward, success
        elif self.reward_type == 'puck_catch':
            # reward for getting close to the puck, but make sure not to displace it
            puck_pos = state_info['pucks'][0]['position']
            paddle_pos = state_info['paddles']['paddle_ego']['position']
            dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
            max_dist = 0.16 * self.width
            min_dist = self.paddle_radius + self.puck_radius
            reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
            reward = max(reward, 0)
            success = reward >= 0.9 and self.current_timestep > 75
            return reward, success
        elif self.reward_type == 'puck_touch':
            # reward for getting close to the puck, but make sure not to displace it
            puck_pos = state_info['pucks'][0]['position']
            paddle_pos = state_info['paddles']['paddle_ego']['position']
            min_dist = self.paddle_radius + self.puck_radius
            dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
            max_dist = 0.16 * self.width
            reward = 1 - ((dist - min_dist) / (max_dist - min_dist))
            reward = max(reward, 0)
            
            # let's also make sure puck does not deviate from initial position
            puck_initial_position = self.puck_initial_position
            puck_current_position = state_info['pucks'][0]['position']
            dist = np.linalg.norm(np.array(puck_initial_position) - np.array(puck_current_position))
            epsilon = 0.01
            if dist >= epsilon:
                reward -= 1
            success = reward >= 0.9 and dist < epsilon
            return reward, success
        elif self.reward_type == 'reach':
            # reward for getting close to target location
            paddle_position = state_info['paddles']['paddle_ego']['position']
            goal_position = self.reach_goal_pos
            dist = np.linalg.norm(np.array(paddle_position) - np.array(goal_position))
            max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
            # reward for closer to goal
            reward = 1 - (dist / max_euclidean_distance)
            success = reward >= 0.9
            return reward, success.item()
        elif self.reward_type == 'reach_vel':
            # reward for getting close to target location
            paddle_position = state_info['paddles']['paddle_ego']['position']
            goal_position = self.reach_goal_pos
            dist = np.linalg.norm(np.array(paddle_position) - np.array(goal_position))
            max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
            # reward for closer to goal
            pos_reward = 1 - (dist / max_euclidean_distance)
            # vel reward
            current_vel = state_info['paddles']['paddle_ego']['velocity']
            goal_vel = self.reach_goal_vel
            # dist = np.linalg.norm(np.array(current_vel) - np.array(goal_vel))
            mag_current = np.linalg.norm(current_vel)
            mag_goal = np.linalg.norm(goal_vel)
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
            
            success = pos_reward >= 0.9 and vel_reward >= 0.8 # a little easier for both since it's harder to do both in general
            return 0.5 * pos_reward + 0.5 * vel_reward, success.item()
        elif self.reward_type == 'puck_reach':
            puck_pos = state_info['pucks'][0]['position']
            paddle_pos = state_info['paddles']['paddle_ego']['position']
            dist = np.linalg.norm(np.array(puck_pos) - np.array(paddle_pos))
            if dist <= self.paddle_radius + self.puck_radius:
                reward = 1
            else:
                reward = 0
            success = reward == 1
            return reward, success
        elif self.reward_type == 'puck_touch':
            reward = 1 if hit_a_puck else 0
            success = reward == 1
            return reward, success
        elif self.reward_type == 'alt_home':
            reward = 1 if puck_within_alt_home else 0
            success = reward == 1
            return reward, success
        elif self.reward_type == 'move_block':
            # more reward if we move the block away from initial position
            block_initial_pos = state_info['blocks'][0]['initial_position']
            block_pos = state_info['blocks'][0]['current_position']
            dist = np.linalg.norm(np.array(block_pos) - np.array(block_initial_pos))
            max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
            reward = 500 * dist / max_euclidean_distance
            success = reward > 1 and self.current_timestep > 10
            return reward, success
        else:
            raise ValueError("Invalid reward type defined in config.")
        
    def get_reward_shaping(self, state_info):
        additional_rew = 0.0
        
        # small negative reward for changing direction
        if self.current_timestep > 0:
            old_vel = self.old_state['paddles']['paddle_ego']['velocity']
            new_vel = state_info['paddles']['paddle_ego']['velocity']
            vel_unit = old_vel / (np.linalg.norm(old_vel) + 1e-8)
            new_vel_unit = new_vel / (np.linalg.norm(new_vel) + 1e-8)
            cosine_sim = np.dot(vel_unit, new_vel_unit) / (np.linalg.norm(vel_unit) * np.linalg.norm(new_vel_unit) + 1e-8)
            norm_cosine_sim = (cosine_sim + 1) / 2
            max_change_dir_rew = self.direction_change_rew
            direction_rew = max_change_dir_rew * (1 - norm_cosine_sim)
            additional_rew += direction_rew
            
        # small negative reward for moving too fast in horizontal direction
        max_vel = self.max_paddle_vel
        max_vel_rew = self.horizontal_vel_rew
        normalized_y_vel = np.abs(state_info['paddles']['paddle_ego']['velocity'][1]) / max_vel
        additional_rew += max_vel_rew * normalized_y_vel
        
        # negative penalty for diagonal motion
        # angle of vector will be close to % 45 degrees if moving diagonally
        angle = np.arctan2(state_info['paddles']['paddle_ego']['velocity'][1], state_info['paddles']['paddle_ego']['velocity'][0])
        angle = np.abs(angle)
        # check if sufficiently close to pi/4, 3pi/4, 5pi/4, 7pi/4
        threshold = np.pi / 12
        # check if between (pi/4 - pi/12, pi/4 + pi/12), ...
        if np.abs(angle - -np.pi / 4) < threshold or np.abs(angle - 3 * -np.pi / 4) < threshold or \
            np.abs(angle - np.pi / 4) < threshold or np.abs(angle - 3 * np.pi / 4) < threshold:
            additional_rew += self.diagonal_motion_rew
        
        # small positive reward for keeping still
        if np.linalg.norm(state_info['paddles']['paddle_ego']['velocity']) < 0.01:
            additional_rew += self.stand_still_rew
            
        # determine if close to walls
        if self.wall_bumping_rew != 0:
            bump_right = state_info['paddles']['paddle_ego']['position'][1] > self.table_y_right - 2 * self.paddle_radius
            bump_left = state_info['paddles']['paddle_ego']['position'][1] < self.table_y_left + 2 * self.paddle_radius
            bump_top = state_info['paddles']['paddle_ego']['position'][0] < 0 + 4 * self.paddle_radius
            bump_bottom = state_info['paddles']['paddle_ego']['position'][0] > self.table_x_bot - 4 * self.paddle_radius
            if bump_left or bump_right or bump_top or bump_bottom:
                additional_rew += self.wall_bumping_rew
        
        # TODO: require simulators to send contact info in state
        return additional_rew

    def step(self, action):
        if not self.multiagent:
            obs, reward, is_finished, truncated, info = self.single_agent_step(action)
            if not self.goal_conditioned:
                return obs, reward, is_finished, truncated, info
            else:
                return {"observation": obs, "desired_goal": self.get_desired_goal(), "achieved_goal": self.get_achieved_goal(self.current_state)}, reward, is_finished, truncated, info
        else:
            return self.multi_step(action)
        
    def set_goal_set(self, goal_set):
        self.goal_set = goal_set

    def single_agent_step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        next_state = self.simulator.get_transition(action)
        if self.current_timestep > 0:
            self.old_state = self.current_state
        self.current_state = next_state
        success = self.success_in_ep 
        info = {}
        info['success'] = success

        hit_a_puck = False
        is_finished, truncated, puck_within_home, puck_within_alt_home, puck_within_goal, _ = self.has_finished(next_state)
        if not truncated:
            reward, success = self.get_base_reward(next_state, hit_a_puck, puck_within_home, 
                                     puck_within_alt_home, puck_within_goal,
                                     self.ego_goal_pos, self.ego_goal_radius)
            if not info['success'] and success:
                info['success'] = success
                self.success_in_ep = success
        else:
            reward = self.truncate_rew
        reward += self.get_reward_shaping(next_state)
        if self.reward_type == 'puck_reach':
            puck_reached_successfully = self.puck_reached(next_state)
            if not is_finished and puck_reached_successfully:
                is_finished = True
        
        self.max_reward_in_single_step = max(self.max_reward_in_single_step, reward)
        self.min_reward_in_single_step = min(self.min_reward_in_single_step, reward)        
        
        info['max_reward'] = self.max_reward_in_single_step
        info['min_reward'] = self.min_reward_in_single_step

        self.current_timestep += 1
        
        obs = self.get_observation(next_state)
        return obs, reward, is_finished, truncated, info
    
    def multi_step(self, joint_action):
        raise NotImplementedError("Multi-agent step function not implemented yet. But shouldn't take much work, it is mostly copy-pasting. But need to do specific rewards per player")

    def get_joint_reward(self, ego_hit_a_puck, alt_hit_a_puck, 
                         puck_within_ego_home, puck_within_alt_home,
                         puck_within_ego_goal, puck_within_alt_goal):
        NotImplementedError("Joint reward function not implemented yet.")