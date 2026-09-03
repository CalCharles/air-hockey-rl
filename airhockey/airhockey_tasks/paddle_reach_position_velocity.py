import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
from .abstract_airhockey_goal_task import AirHockeyGoalEnv
from airhockey.airhockey_rewards import AirHockeyPaddleReachPositionVelocityReward

class AirHockeyPaddleReachPositionVelocityEnv(AirHockeyGoalEnv):
    # Distance the paddle needs to accelerate to (or decelerate from) the top
    # speed of 2 m/s: measured at ~0.12 m along x and ~0.10 m along y in Box2D
    # with the sysid PID.  Goals are kept at least this far from the workspace
    # edge, otherwise the workspace clip forces the paddle to be slow there and
    # the velocity part of the goal is unreachable by construction.
    DEFAULT_GOAL_POSITION_MARGIN = 0.12

    # Velocity tolerance (m/s) of the sparse goal check.
    DEFAULT_GOAL_VELOCITY_RADIUS = 0.5

    def __init__(self, **kwargs):
        self.goal_radius_type = kwargs['goal_radius_type']
        self.base_goal_radius = kwargs['base_goal_radius']
        self.goal_position_margin = float(
            kwargs.get('goal_position_margin', self.DEFAULT_GOAL_POSITION_MARGIN)
        )
        # Velocity tolerance of the goal check, in m/s over both components.
        self.goal_velocity_radius = float(
            kwargs.get('base_goal_velocity_radius', self.DEFAULT_GOAL_VELOCITY_RADIUS)
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
        # Position bounds are the reachable workspace, not the whole table;
        # sampling insets them further by ``goal_position_margin``.
        x_lo, x_hi, y_lo, y_hi = self.get_paddle_workspace_bounds()
        goal_low = [x_lo, y_lo, -self.max_paddle_vel, -self.max_paddle_vel]
        goal_high = [x_hi, y_hi, self.max_paddle_vel, self.max_paddle_vel]

        if self.return_goal_obs:
            self.observation_space = self.single_observation_space = self.get_goal_obs_space(low, high, goal_low, goal_high)
        else:
            low = low + goal_low
            high = high + goal_high
            self.observation_space = self.single_observation_space = self.get_obs_space(low, high)

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4
        self.goal_radius = self.base_goal_radius

        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1
        self.reward = AirHockeyPaddleReachPositionVelocityReward(self)
    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPaddleReachPositionVelocityEnv(**state_dict)
        
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
    
    def get_achieved_goal(self, state_info):
        position = state_info['paddles']['paddle_ego']['position']
        velocity = state_info['paddles']['paddle_ego']['velocity']
        return np.array([position[0], position[1], velocity[0], velocity[1]])
    
    def get_desired_goal(self):
        position = self.goal_pos
        velocity = self.goal_vel
        return np.array([position[0], position[1], velocity[0], velocity[1]])

    def goal_reached(self, state_info):
        paddle = state_info['paddles']['paddle_ego']
        achieved = np.array([
            paddle['position'][0], paddle['position'][1],
            paddle['velocity'][0], paddle['velocity'][1],
        ])
        desired = self.get_desired_goal()
        pos_dist = np.linalg.norm(achieved[:2] - desired[:2])
        vel_dist = np.linalg.norm(achieved[2:] - desired[2:])
        return bool(pos_dist <= self.goal_radius and vel_dist <= self.goal_velocity_radius)
    
    def get_observation(self, state_info, obs_type ="paddle", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    # def get_observation(self, state_info):
    #     ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    #     ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    #     ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    #     ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]

    #     obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
    #     return obs
    
    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        # Sample inside the reachable paddle workspace, inset by the distance
        # needed to be at the goal velocity there: y first, because the corner
        # wedge makes max_x depend on y.
        margin = self.goal_position_margin
        _, _, min_y, max_y = self.get_paddle_workspace_bounds(margin=margin)
        goal_y = self.rng.uniform(low=min_y, high=max_y)
        min_x, max_x, _, _ = self.get_paddle_workspace_bounds(margin=margin, y=goal_y)
        goal_x = self.rng.uniform(low=min_x, high=max_x)
        goal_position = np.array([goal_x, goal_y])
        # x vel shouldn't vary much
        # "minimum" is upward at max speed, "maximum" is slightly upwards, otherwise can't reach goal
        x_vel = self.rng.uniform(low=-self.max_paddle_vel, high=-self.max_paddle_vel / 8) # only upwards
        y_vel = self.rng.uniform(low=-self.max_paddle_vel / 2, high=self.max_paddle_vel / 2)
        goal_velocity = np.array([x_vel, y_vel])
        # The simulator clamps the paddle speed to max_paddle_vel, so a goal
        # whose components combine past that norm can never be matched.
        goal_speed = float(np.linalg.norm(goal_velocity))
        if goal_speed > self.max_paddle_vel:
            goal_velocity = goal_velocity * (self.max_paddle_vel / goal_speed)
        self.goal_pos = goal_position if self.goal_set is None else self.goal_set[0, :2]
        self.goal_vel = goal_velocity if self.goal_set is None else self.goal_set[0, 2:]