from gymnasium import Env
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from abc import ABC, abstractmethod
import math

from typing import Tuple


def get_box2d_simulator_fn():
    from airhockey.sims import AirHockeyBox2D
    return AirHockeyBox2D
    
def get_robosuite_simulator_fn():
    from airhockey.sims import AirHockeyRobosuite
    return AirHockeyRobosuite


class AirHockeyBaseEnv(ABC, Env):
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
                 terminate_on_puck_hit_bottom=False,  # TODO Specify this parameter in the yaml config
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
                 reward_movement_types=[],
                 compute_online_rewards=True):
        
        if simulator == 'box2d':
            simulator_fn = get_box2d_simulator_fn()
        elif simulator == 'robosuite':
            simulator_fn = get_robosuite_simulator_fn()
        else:
            raise ValueError("Invalid simulator type. Must be 'box2d' or 'robosuite'.")

        simulator_params['seed'] = seed
        self.simulator_name = simulator
        self.simulator = simulator_fn.from_dict(simulator_params)
        self.render_length = self.simulator.render_length
        self.render_width = self.simulator.render_width
        self.render_masks = self.simulator.render_masks
        self.ppm = self.simulator.ppm
        
        self.simulator_params = simulator_params

        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.n_training_steps = n_training_steps
        self.n_timesteps_so_far = 0
        self.rng = np.random.RandomState(seed)
        self.dynamic_virtual_objects = list() # if the environment has these, put them in at subclass initialization
        self.reward_regions = list()
        
        # termination conditions
        self.terminate_on_out_of_bounds = terminate_on_out_of_bounds
        self.terminate_on_enemy_goal = terminate_on_enemy_goal
        self.terminate_on_puck_stop = terminate_on_puck_stop
        self.terminate_on_puck_hit_bottom = terminate_on_puck_hit_bottom
        
        # reward function
        self.compute_online_rewards = compute_online_rewards
        self.goal_conditioned = True if 'goal' in task else False
        self.goal_radius_type = 'fixed'
        self.goal_min_x_velocity = -goal_max_x_velocity
        self.goal_max_x_velocity = goal_max_x_velocity
        self.goal_min_y_velocity = goal_min_y_velocity
        self.goal_max_y_velocity = goal_max_y_velocity
        self.return_goal_obs = return_goal_obs
        self.dense_goal = dense_goal
        self.task = task
        self.multiagent = num_paddles == 2
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
        
        self.paddle_radius = simulator_params['paddle_radius']
        self.puck_radius = simulator_params['puck_radius']
        self.block_width = simulator_params['block_width']
        
        self.table_x_top = -self.length / 2
        self.table_x_bot = self.length / 2
        self.table_y_right = self.width / 2
        self.table_y_left = -self.width / 2

        self.max_paddle_vel = self.simulator.max_paddle_vel
        self.max_puck_vel = self.simulator.max_puck_vel
        self.goal_set = None
        
        self.num_pucks = num_pucks
        self.multiagent = num_paddles > 1
        self.num_blocks = num_blocks
        self.num_obstacles = num_obstacles
        self.num_targets = num_targets
        self.num_paddles = num_paddles
        
        self.validate_configuration()

        self.goal_selector = goal_selector
        self.initialize_spaces()
        self.falling_time = 25
        self.metadata = {}
        self.reset()

    @abstractmethod
    def from_dict(state_dict):
        pass

    @abstractmethod
    def initialize_spaces(self):
        pass
    
    @abstractmethod
    def create_world_objects(self):
        pass
    
    @abstractmethod
    def validate_configuration(self):
        pass
    
    @abstractmethod
    def get_base_reward(self, state_info):
        pass

    @abstractmethod
    def get_observation(self, state_info):
        pass

    def get_obs_space(self, low: list, high: list):
        return Box(low=np.array(low), high=np.array(high), dtype=float)        

    def reset(self, seed=None, **kwargs):
        if seed is None: # determine next seed, in a deterministic manner
            seed = self.rng.randint(0, int(1e8))

        self.rng = np.random.RandomState(seed)
        sim_seed = self.rng.randint(0, int(1e8))
        self.simulator.reset(sim_seed) # no point in getting state since no spawning
        self.create_world_objects()
        self.simulator.instantiate_objects()
        state_info = self.simulator.get_current_state()
        self.current_state = state_info
        obs = self.get_observation(state_info)
        
        self.n_timesteps_so_far += self.current_timestep
        self.current_timestep = 0
        self.success_in_ep = False
        self.max_reward_in_single_step = -np.inf
        self.min_reward_in_single_step = np.inf
        
        if 'pucks' in state_info and len(state_info['pucks']) > 0:
            self.puck_initial_position = state_info['pucks'][0]['position']
            
        return obs, {'success': False}

    def reset_from_state(self, state_vector, seed=None):
        if seed is None: # determine next seed, in a deterministic manner
            seed = self.rng.randint(0, int(1e8))

        self.rng = np.random.RandomState(seed)
        sim_seed = self.rng.randint(0, int(1e8))
        self.simulator.reset(sim_seed) # no point in getting state since no spawning
        self.create_world_objects_from_state(state_vector)
        self.simulator.instantiate_objects()
        state_info = self.simulator.get_current_state()
        self.current_state = state_info
        obs = self.get_observation(state_info)
        return obs, {'success': False}

    def get_puck_configuration(self, bad_regions=None):
        y_pos = None
        if bad_regions is not None:
            while y_pos is None:
                for region in bad_regions:
                    proposed_y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)  # doesnt spawn at edges
                    if not (proposed_y_pos > region[0] and proposed_y_pos < region[1]):
                        y_pos = proposed_y_pos
        else:
            y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)
        pos = (self.table_x_top + 0.01, y_pos)
        vel = (1, 0)
        return pos, vel
    
    def get_block_configuration(self, bad_regions=None):
        y_pos = None
        if bad_regions is not None:
            while y_pos is None:
                for region in bad_regions:
                    proposed_y_pos = self.rng.uniform(low=self.table_y_left + 2 * self.block_width, high=self.table_y_right - 2 * self.block_width)
                    region_with_margin = (region[0] - self.block_width, region[1] + self.block_width)
                    if not (proposed_y_pos > region_with_margin[0] and proposed_y_pos < region_with_margin[1]):
                        y_pos = proposed_y_pos
        else:
            y_pos = self.rng.uniform(low=-self.width / 3, high=self.width / 3)
        x_pos = self.rng.uniform(low=self.table_x_top + 2 * self.block_width, high=0 - self.block_width)
        pos = (x_pos, y_pos)
        vel = (0, 0)
        return pos, vel
    
    def get_paddle_configuration(self, name):
        if name == 'paddle_ego':
            x_pos = self.table_x_bot - self.paddle_radius
        elif name == 'paddle_alt':
            x_pos = self.table_x_top + self.paddle_radius
        else:
            raise ValueError("Invalid paddle name")
        vel = (0, 0)
        return (x_pos, 0), vel

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
                if state_info['paddles']['paddle_ego']['position'][0] < 0 or \
                    state_info['paddles']['paddle_ego']['position'][0] > self.table_x_bot or \
                    state_info['paddles']['paddle_ego']['position'][1] > self.table_y_right or \
                    state_info['paddles']['paddle_ego']['position'][1] < self.table_y_left:
                    truncated = True
                    print("paddle out of bounds with position: ", state_info['paddles']['paddle_ego']['position'])
                    print("X_min, X_max, Y_min, Y_max: ", 0 + self.paddle_radius, self.table_x_bot - self.paddle_radius, self.table_y_left + self.paddle_radius, self.table_y_right - self.paddle_radius)

        bottom_center_point = np.array([self.table_x_bot, 0])
        top_center_point = np.array([self.table_x_top, 0])
        
        puck_within_home = False
        puck_within_alt_home = False

        if self.terminate_on_puck_hit_bottom:
            puck_pos = state_info['pucks'][0]['position']
            if abs(puck_pos[0] - self.table_x_bot) < self.puck_radius + 0.03:
                terminated = True

        if self.terminate_on_enemy_goal:
            if not terminated and puck_within_home:
                truncated = True

        if multiagent:
            terminated = terminated or truncated or puck_within_alt_home or puck_within_home
            truncated = False
            
        if self.terminate_on_puck_stop:
            if not truncated and np.linalg.norm(state_info['pucks'][0]['velocity']) < 0.01:
                truncated = True

        # puck passed the our paddle
        if state_info['pucks'][0]['position'][0] > (state_info['paddles']['paddle_ego']['position'][0] + self.paddle_radius):
            truncated = True

        # puck touched our paddle
        # if np.linalg.norm(state_info['pucks'][0]['position'][0] - state_info['paddles']['paddle_ego']['position'][0]) <= (self.paddle_radius + self.puck_radius + 0.1):
            # puck_within_home = True
            # terminated = True
        
        puck_within_ego_goal = False
        puck_within_alt_goal = False
                    
        return terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal

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
            return obs, reward, is_finished, truncated, info
        else:
            return self.multi_step(action)
    
    def single_step_dynamic_virtual(self, action):
        # step any dynamic virtual objects to update their state
        for dvo in self.dynamic_virtual_objects:
            dvo.step(self.current_state, action)
        

    def single_agent_step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:

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
            reward, success = self.get_base_reward(next_state)
            if not info['success'] and success:
                info['success'] = success
                self.success_in_ep = success
        else:
            reward = self.truncate_rew
        reward += self.get_reward_shaping(next_state)
        
        self.max_reward_in_single_step = max(self.max_reward_in_single_step, reward)
        self.min_reward_in_single_step = min(self.min_reward_in_single_step, reward)        
        
        info['max_reward'] = self.max_reward_in_single_step
        info['min_reward'] = self.min_reward_in_single_step

        self.current_timestep += 1
        
        # # DEBUG STATEMENETS 4 LINES BELOW!
        # is_finished = False
        # truncated = False
        # # only end if timesteps
        # if self.current_timestep >= self.max_timesteps:
        #     is_finished = True
        
        obs = self.get_observation(next_state)
        return obs, reward, is_finished, truncated, info
    
    def multi_step(self, joint_action):
        raise NotImplementedError("Multi-agent step function not implemented yet. But shouldn't take much work, it is mostly copy-pasting. But need to do specific rewards per player")

    def get_joint_reward(self, ego_hit_a_puck, alt_hit_a_puck, 
                         puck_within_ego_home, puck_within_alt_home,
                         puck_within_ego_goal, puck_within_alt_goal):
        NotImplementedError("Joint reward function not implemented yet.")