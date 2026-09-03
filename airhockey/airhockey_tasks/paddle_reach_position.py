import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPaddleReachPositionSparseReward

class AirHockeyPaddleReachPositionEnv(AirHockeyGoalEnv):
    def __init__(self, **kwargs):
        self.goal_radius_type = kwargs['goal_radius_type']
        self.base_goal_radius = kwargs['base_goal_radius']
        # Inset of the goal-sampling box from the reachable paddle workspace.
        # Defaults to the goal radius so a goal is always fully reachable.
        self.goal_position_margin = float(
            kwargs.get('goal_position_margin', kwargs['base_goal_radius'])
        )
        # The reward is +1 only on the step the goal is met, so the episode
        # has to end once the paddle gets there.
        self.terminate_on_goal_reached = bool(
            kwargs.get('terminate_on_goal_reached', True)
        )
        super().__init__(**kwargs)
        
    def initialize_spaces(self, obs_type):
        # setup observation / action / reward spaces
        low, high = self.init_observation(obs_type)
        
        # Declared over the full reachable workspace; sampling insets it further.
        x_lo, x_hi, y_lo, y_hi = self.get_paddle_workspace_bounds()
        goal_low = [x_lo, y_lo]
        goal_high = [x_hi, y_hi]
        
        if self.return_goal_obs:
            self.observation_space = self.single_observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + goal_low
            high = high + goal_high
            self.observation_space = self.single_observation_space = self.get_obs_space(low, high)
        
        self.goal_radius = self.base_goal_radius

        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPaddleReachPositionSparseReward(self)
        
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPaddleReachPositionEnv(**state_dict)
        
    def create_world_objects(self):
        name = 'paddle_ego'
        pos, vel = self.get_paddle_configuration(name)
        self.simulator.spawn_paddle(pos, vel, name)
        
    def create_world_objects_from_state(self, state_vector):

        name = 'paddle_ego'
        paddle_pos, paddle_vel = state_vector[:2], state_vector[2:4]
        self.simulator.spawn_paddle(paddle_pos, paddle_vel, name)

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
        position = self.goal_pos
        return np.array([position[0], position[1]])

    def goal_reached(self, state_info):
        position = state_info['paddles']['paddle_ego']['position']
        dist = np.linalg.norm(np.array(position[:2]) - np.array(self.goal_pos[:2]))
        return bool(dist <= self.goal_radius)

    def get_observation(self, state_info, obs_type ="paddle", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)
    
    def set_goals(self, goal_radius_type, goal_pos=None, goal_set=None):
        self.goal_set = goal_set

        # Sample inside the reachable paddle workspace, inset by the goal
        # radius: y first, because the corner wedge makes max_x depend on y.
        margin = self.goal_position_margin
        _, _, min_y, max_y = self.get_paddle_workspace_bounds(margin=margin)
        goal_y = self.rng.uniform(low=min_y, high=max_y)
        min_x, max_x, _, _ = self.get_paddle_workspace_bounds(margin=margin, y=goal_y)
        goal_x = self.rng.uniform(low=min_x, high=max_x)
        goal_position = np.array([goal_x, goal_y])
        self.goal_pos = goal_position if self.goal_set is None else self.goal_set[0, :2]
        self.goal_pos = goal_pos if goal_pos is not None else self.goal_pos