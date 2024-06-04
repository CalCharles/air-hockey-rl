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
                    
        
        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1

        radius_range = np.array((self.table_x_bot/ 2 * np.array(self.reward_normalized_radius_min), (self.table_y_right - self.table_y_left) * np.array(self.reward_normalized_radius_max) / 2))
        self.reward_regions = [RewardRegion(self.negative_reward_range, 
                                                     self.reward_region_scale_range, 
                                                     [np.array([0, self.table_y_left]), np.array([self.table_x_bot,self.table_y_right])],
                                                     radius_range, shapes=self.reward_region_shapes) for _ in range(self.num_negative_reward_regions)]
        
        reward_regions_states_shape = np.concatenate([nrr.get_state() for nrr in self.reward_regions]).shape
        # these are misspecified but you can check later
        reward_region_states_low = [-100] * reward_regions_states_shape[0]
        reward_region_states_high = [100] * reward_regions_states_shape[0]

        if self.return_goal_obs:
            low = paddle_obs_low + reward_region_states_low
            high = paddle_obs_high + reward_region_states_high
            self.observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = paddle_obs_low + goal_low + reward_region_states_low
            high = paddle_obs_high + goal_high + reward_region_states_high
            
            self.observation_space = self.get_obs_space(low, high)
            
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
        # import pdb; pdb.set_trace()
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