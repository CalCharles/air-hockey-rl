import numpy as np
from airhockey.airhockey_rewards import AirHockeyRewardBase

class AirHockeyPaddleReachPositionReward(AirHockeyRewardBase):
    """Dense paddle-to-goal shaping.

    No longer used by ``paddle_reach_position`` itself (that task is sparse now,
    see ``AirHockeyPaddleReachPositionSparseReward``) — it survives as the
    paddle-to-puck proximity term of ``puck_goal_position`` /
    ``puck_goal_position_obstacles``.
    """

    def __init__(self, task_env, paddle_success_bonus=None):
        super().__init__(task_env)
        self.paddle_success_bonus = paddle_success_bonus

    def compute_reward(self, achieved_goal, desired_goal):
        # if not vectorized, convert to vector
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            
        # return euclidean distance between the two points
        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)

        radius = self.task_env.goal_radius
        bonus = self.paddle_success_bonus if self.paddle_success_bonus is not None else 0
        # Vectorised: per-sample -dist outside the radius, bonus inside.
        reward = np.where(dist > radius, -dist, float(bonus))

        if single:
            # Return a plain float like every other task (the previous
            # `-dist if dist > radius else bonus` returned a (1,) array far
            # from the goal and a scalar inside it, which broke trajectory
            # stacking in the trainer).
            reward = float(reward.reshape(-1)[0])

        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        reward = self.compute_reward(self.task_env.get_achieved_goal(state_info), self.task_env.get_desired_goal())
        dist = np.linalg.norm(ag - dg, axis=0)
        
        success = dist < self.task_env.goal_radius
        success = success.item()
        return reward, success


class AirHockeyPaddleReachPositionSparseReward(AirHockeyRewardBase):
    """+1 on the step the paddle reaches the goal, 0 on every other step.

    The episode ends on arrival (``terminate_on_goal_reached``), so an episode
    returns ``GOAL_REWARD`` (10) if the paddle got within ``goal_radius`` of the
    goal inside the ``max_timesteps`` budget and 0 if it ran out of time.
    """

    # Reward paid on the single step the goal is reached. 10 rather than 1: with
    # a reward of 1 the critic's Q values sit at ~0.01-0.1, the same size as
    # the clipped-double-Q bias and the optimizer's regularisation pull, and
    # the actor saturates on a flat critic (2026-09-04 diagnosis,
    # notes/scratch/experiments/2026-09-04_01-05_sparse-task-collapse-diagnosis.md).
    # x10 alone took the task from 9 % to 100 % success.
    GOAL_REWARD = 10.0

    def __init__(self, task_env):
        super().__init__(task_env)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        single = len(achieved_goal.shape) == 1
        if single:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)

        dist = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        reward = np.where(dist > self.task_env.goal_radius, 0.0, self.GOAL_REWARD)

        if single:
            return float(reward.reshape(-1)[0])
        return reward

    def get_base_reward(self, state_info):
        ag = self.task_env.get_achieved_goal(state_info)
        dg = self.task_env.get_desired_goal()
        dist = float(np.linalg.norm(ag[:2] - dg[:2]))
        success = dist <= self.task_env.goal_radius
        return (self.GOAL_REWARD if success else 0.0), success
