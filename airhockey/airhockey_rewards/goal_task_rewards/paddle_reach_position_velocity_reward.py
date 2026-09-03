import numpy as np
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPaddleReachPositionVelocityReward(AirHockeyRewardBase):
    """+1 on the step the paddle is at the goal *with* the goal velocity, else 0.

    "At the goal" means both tolerances are met at the same step: the paddle
    centre within ``goal_radius`` of the goal position, and its velocity within
    ``goal_velocity_radius`` (m/s, Euclidean over both components, so direction
    and magnitude are covered by one number) of the goal velocity.  The episode
    ends there (``terminate_on_goal_reached``), so an episode returns 1 if the
    paddle hit the goal state inside the ``max_timesteps`` budget and 0 if not.
    """

    def __init__(self, task_env):
        super().__init__(task_env)

    def _goal_met(self, achieved_goal, desired_goal):
        pos_dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        vel_dist = np.linalg.norm(achieved_goal[:, 2:] - desired_goal[:, 2:], axis=1)
        return (pos_dist <= self.task_env.goal_radius) & (
            vel_dist <= self.task_env.goal_velocity_radius
        )

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)

        reward = np.where(self._goal_met(achieved_goal, desired_goal), 1.0, 0.0)

        if single:
            return float(reward.reshape(-1)[0])
        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info).reshape(1, -1)
        dg = self.task_env.get_desired_goal().reshape(1, -1)
        success = bool(self._goal_met(ag, dg)[0])
        return (1.0 if success else 0.0), success
