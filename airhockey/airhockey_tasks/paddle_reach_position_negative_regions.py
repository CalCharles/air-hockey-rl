import copy
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_tasks.utils import RewardRegion

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
                 reward_movement_types=[],
                 initialization_description_pth="",
                 paddle_offsets = [0,0,0,0],
                 paddle_clipping = [1,0,-0.1,-0.15]):
        self.init_dict = self.load_initialization(initialization_description_pth)
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

    def load_initialization(self, pth):
        '''
        We can initialize preset arrangements as a n x m grid with the hashes:
        s: paddle
        x: negative reward region
        a: allowed to initialize a negative region here
        o: nothing
        g: goal
        TODO: we could also look at all the files in the path and use all of them as initializations
        '''
        lines = list()
        # TODO: we only specify a single paddle/goal pos for now, but we could do multiple
        self.init_paddle_pos = None 
        self.init_goal_pos = None
        self.init_negative_pos = list()
        self.init_allowed_negative_pos = list()
        if len(pth):
            with open(pth, 'r') as file:
                line = file.readline()
                line_counter = 0
                while line:
                    for col, v in enumerate(line):
                        if v == 's':
                            self.init_paddle_pos = [line_counter, col]
                        elif v == 'x':
                            self.init_negative_pos.append([line_counter, col])
                        elif v == 'a':
                            self.init_allowed_negative_pos.append([line_counter, col])
                        elif v == 'g':
                            self.init_goal_pos = [line_counter, col]
                    line_counter += 1
                    lines.append(line)
                    line = file.readline()
            self.grid_dims = [len(lines), len(lines[0])]
        else: # initialize these values to -1, but probably unnecessary
            self.grid_dims = [-1,-1]

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

        if self.grid_dims is not None:
            self.grid_lengths = np.array([(self.table_x_bot - 0) / self.grid_dims[0], 
                                 (self.table_y_right - self.table_y_left) / self.grid_dims[1]])
            self.grid_midpoints = list()
            for i in np.arange(self.grid_dims[0]):
                midpoint_row = list()
                for j in np.arange(self.grid_dims[1]):
                    midpoint_row.append(np.array([0, self.table_y_left]) + np.array([i + 0.5,j + 1.0]) * self.grid_lengths)
                self.grid_midpoints.append(midpoint_row)
            self.grid_midpoints = np.array(self.grid_midpoints)

        if len(self.init_negative_pos):
            # TODO: ignores the num_negative_reward_region, could have those initialized randomly
            for pos in self.init_negative_pos:
                radius_range = [np.array(self.grid_lengths) / 2, np.array(self.grid_lengths) / 2]
                pos = copy.deepcopy(self.grid_midpoints[pos[0], pos[1]])
                self.reward_regions.append(RewardRegion(self.negative_reward_range, 
                                                            self.reward_region_scale_range, 
                                                            [np.array(pos), np.array(pos)],
                                                            radius_range, shapes=self.reward_region_shapes, object_radius = self.paddle_radius))
        # TODO: implement the allowed regions code
        # elif len(self.init_allowed_negative_pos.keys()):
        else:
            radius_range = np.array((self.table_x_bot/ 2 * np.array(self.reward_normalized_radius_min), (self.table_y_right - self.table_y_left) * np.array(self.reward_normalized_radius_max) / 2))
            self.reward_regions = [RewardRegion(self.negative_reward_range, 
                                                     self.reward_region_scale_range, 
                                                     [np.array([0, self.table_y_left]), np.array([self.table_x_bot,self.table_y_right])],
                                                     radius_range, shapes=self.reward_region_shapes, object_radius = self.paddle_radius) for _ in range(self.num_negative_reward_regions)]

    def create_world_objects(self):
        name = 'paddle_ego'
        if self.init_paddle_pos is None: pos, vel = self.get_paddle_configuration(name)
        else: 
            pos = copy.deepcopy(self.grid_midpoints[self.init_paddle_pos[0], self.init_paddle_pos[1]])
            vel = (0,0)
            print("init paddle", self.init_paddle_pos, pos)
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
        # import pdb; pdb.set_trace()
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        max_euclidean_distance = np.linalg.norm(np.array([self.table_x_bot, self.table_y_right]) - np.array([self.table_x_top, self.table_y_left]))
        # reward for closer to goal
        reward = - (dist / max_euclidean_distance)

        for nrr in self.reward_regions:
            reward += nrr.check_reward(achieved_goal)

        # print(achieved_goal, desired_goal, reward)
        # print(dist / max_euclidean_distance, 1 - (dist / max_euclidean_distance), reward)
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
        if self.init_goal_pos is None: goal_position = self.rng.uniform(low=(min_x, min_y), high=(max_x, max_y))
        else: 
            goal_position = copy.deepcopy(self.grid_midpoints[self.init_goal_pos[0], self.init_goal_pos[1]])
            print("init_goal", self.grid_midpoints,self.table_x_bot, self.init_goal_pos, goal_position)
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