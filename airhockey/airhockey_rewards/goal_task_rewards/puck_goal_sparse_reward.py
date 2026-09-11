"""Sparse goal-conditioned puck rewards (HER targets).

Both rewards pay ``GOAL_REWARD`` (10) on the step the puck is at the goal and 0
otherwise, exactly like the sparse paddle-reach rewards
(``AirHockeyPaddleReachPositionSparseReward`` /
``AirHockeyPaddleReachPositionVelocityReward``), so the reward scale is the
one every canonical task uses.  ``compute_reward`` is vectorised over
``(N, goal_dim)`` batches because hindsight relabelling calls it on whole
episodes at once; ``goal_met`` exposes the boolean success test the same way
so relabelled transitions can be marked terminal consistently with
``terminate_on_goal_reached``.
"""

import numpy as np

from airhockey.airhockey_rewards import AirHockeyRewardBase


def _as_batch(achieved_goal, desired_goal):
    achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
    desired_goal = np.asarray(desired_goal, dtype=np.float64)
    single = achieved_goal.ndim == 1
    if single:
        achieved_goal = achieved_goal.reshape(1, -1)
        desired_goal = desired_goal.reshape(1, -1)
    return achieved_goal, desired_goal, single


class AirHockeyPuckGoalPositionSparseReward(AirHockeyRewardBase):
    """+10 on the step the puck centre is within ``goal_radius`` of the goal
    after the paddle has touched the puck in this episode, else 0.

    ``achieved_goal`` is ``(x, y, vx, vy, contacted)``; ``desired_goal`` is
    ``(x, y)`` (extra desired components, if any, are ignored).  The contact
    flag stops a puck that spawns on the goal, or drifts through it, from
    scoring (see ``AirHockeyPuckGoalPositionSparseEnv``).
    """

    CONTACT_COL = 4

    # Same x10 as the sparse paddle-reach rewards, for the same reason (a +1
    # goal reward leaves the critic flat; see
    # notes/scratch/experiments/2026-09-04_01-05_sparse-task-collapse-diagnosis.md).
    GOAL_REWARD = 10.0

    def goal_met(self, achieved_goal, desired_goal):
        achieved_goal, desired_goal, single = _as_batch(achieved_goal, desired_goal)
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        contacted = achieved_goal[:, self.CONTACT_COL] > 0.5
        met = (dist <= self.task_env.goal_radius) & contacted
        return bool(met[0]) if single else met

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        achieved_goal, desired_goal, single = _as_batch(achieved_goal, desired_goal)
        reward = np.where(self.goal_met(achieved_goal, desired_goal), self.GOAL_REWARD, 0.0)
        if single:
            return float(reward.reshape(-1)[0])
        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        success = bool(self.goal_met(ag, dg))
        return (self.GOAL_REWARD if success else 0.0), success


class AirHockeyPuckGoalPositionVelocitySparseReward(AirHockeyRewardBase):
    """+10 when the puck is at the goal position *with* the goal velocity, else 0.

    Both tolerances have to hold on the same step: centre within
    ``goal_radius`` of the goal position and velocity within
    ``goal_velocity_radius`` (m/s, Euclidean over both components) of the goal
    velocity — the puck analogue of ``AirHockeyPaddleReachPositionVelocityReward``
    — and, as for the position task, the paddle must have touched the puck in
    this episode (``achieved_goal[4]``).
    """

    GOAL_REWARD = 10.0
    CONTACT_COL = 4

    def goal_met(self, achieved_goal, desired_goal):
        achieved_goal, desired_goal, single = _as_batch(achieved_goal, desired_goal)
        pos_dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        vel_dist = np.linalg.norm(achieved_goal[:, 2:4] - desired_goal[:, 2:4], axis=1)
        contacted = achieved_goal[:, self.CONTACT_COL] > 0.5
        met = (pos_dist <= self.task_env.goal_radius) & (
            vel_dist <= self.task_env.goal_velocity_radius
        ) & contacted
        return bool(met[0]) if single else met

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        achieved_goal, desired_goal, single = _as_batch(achieved_goal, desired_goal)
        reward = np.where(self.goal_met(achieved_goal, desired_goal), self.GOAL_REWARD, 0.0)
        if single:
            return float(reward.reshape(-1)[0])
        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        success = bool(self.goal_met(ag, dg))
        return (self.GOAL_REWARD if success else 0.0), success


class AirHockeyPuckGoalPositionSpeedSparseReward(AirHockeyRewardBase):
    """+10 when the puck is at the goal position at the goal *speed*, else 0.

    ``desired_goal`` = ``(x, y, speed)``; ``achieved_goal`` =
    ``(x, y, vx, vy, contacted)``.  Position within ``goal_radius``, speed
    (``|v|``) within ``goal_speed_radius`` of the goal speed, and contact made.
    The direction of the puck at the goal is left to the shot geometry.
    """

    GOAL_REWARD = 10.0
    CONTACT_COL = 4

    def goal_met(self, achieved_goal, desired_goal):
        achieved_goal, desired_goal, single = _as_batch(achieved_goal, desired_goal)
        pos_dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        speed = np.linalg.norm(achieved_goal[:, 2:4], axis=1)
        speed_err = np.abs(speed - desired_goal[:, 2])
        contacted = achieved_goal[:, self.CONTACT_COL] > 0.5
        met = (pos_dist <= self.task_env.goal_radius) & (speed_err <= self.task_env.goal_speed_radius) & contacted
        return bool(met[0]) if single else met

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        achieved_goal, desired_goal, single = _as_batch(achieved_goal, desired_goal)
        reward = np.where(self.goal_met(achieved_goal, desired_goal), self.GOAL_REWARD, 0.0)
        if single:
            return float(reward.reshape(-1)[0])
        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        success = bool(self.goal_met(ag, dg))
        return (self.GOAL_REWARD if success else 0.0), success
