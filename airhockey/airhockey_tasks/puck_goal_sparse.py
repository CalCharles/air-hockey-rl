"""Sparse, goal-conditioned puck tasks: send the puck to a goal position
(optionally with a goal velocity) in the upper half of the table.

``puck_goal_position_sparse``
    Goal = puck position ``(x, y)`` sampled uniformly over the upper half of the
    table.  Reward +10 on the step the puck centre is within ``base_goal_radius``
    of the goal *after the paddle has touched the puck in this episode* (the
    episode ends there), 0 otherwise.  The contact condition makes the task
    "hit the puck to the goal": without it a puck that spawns on the goal, or
    drifts through it, scores for free (16 % of random-action episodes), and
    hindsight relabelling then rewards the policy for states its actions did
    not cause.  (An earlier variant used the repo's older puck-goal convention
    "puck moving up the table" instead; that still leaves the spawn drift —
    spawns launch the puck at up to 0.5 m/s in a random heading — as free
    reward.)  The achieved goal is therefore the puck state plus the contact
    flag, ``(x, y, vx, vy, contacted)``, while the desired goal is ``(x, y)``;
    the reward compares the first two components for distance and reads the
    flag from the achieved goal.

``puck_goal_position_velocity_sparse``
    Goal = ``(x, y, vx, vy)``: a puck position in the upper half plus the puck
    velocity there.  Reward +10 when both the position tolerance
    (``base_goal_radius``) and the velocity tolerance
    (``base_goal_velocity_radius``, m/s Euclidean) hold on the same step
    (and the paddle has touched the puck).  Two goal samplers
    (``goal_velocity_sampling``):

    * ``shot`` (default): the goal is a state of a simulated free flight
      launched from a random point of the paddle workspace at a random speed
      (``goal_shot_speed_range``) and angle from straight up
      (``goal_shot_max_angle_deg``), integrated with the table's gravity /
      puck damping and elastic side-wall bounces; a random state of that
      flight inside the upper-half box, still moving up, is the goal.  Every
      goal is therefore consistent with *some* straight shot from the paddle
      region.
    * ``intercept_shot``: like ``shot`` but the launch point is where *this
      episode's* puck will cross the paddle workspace (its spawn state is
      integrated forward to a random x inside the workspace), so the goal
      direction is reachable from the puck the agent actually gets.  The
      goal is resampled right after the world is spawned, so the goal
      distribution is conditioned on the initial state.
    * ``box``: position as above, velocity independent and uniform over
      ``goal_puck_vx_range`` × ``goal_puck_vy_range`` (clipped to
      ``goal_puck_max_speed``).  Mostly infeasible — the direction of the
      puck at the goal is dictated by where it was hit from — and kept only
      for the round-1/2/3 runs of 2026-09-10.

``puck_goal_position_speed_sparse``
    Goal = ``(x, y, speed)``: a shot-sampled position plus the puck *speed*
    there (direction left to the shot geometry).  Reward +10 when the
    position tolerance and the speed tolerance (``base_goal_speed_radius``,
    m/s) hold on the same step after contact.  The easier sibling of the
    velocity task: the paddle mostly controls how hard it hits.

Everything else follows the canonical five tasks: ``obs_type: history``
(30-dim) with the goal appended, canonical puck spawn (uniform over the top
``puck_spawn_top_fraction`` of the table at a slow random velocity), random
paddle spawn in the workspace, and the juggle termination set (puck hits the
bottom / passes the paddle).  With ``return_goal_obs: true`` the env returns
the gym ``GoalEnv`` dict (``observation`` / ``achieved_goal`` /
``desired_goal``) that hindsight experience replay
(``scripts/td3/td3_training_her.py``) relabels; the achieved goal is the
*true* puck state, not the noisy / delayed observation.

The dense ``puck_goal_position`` / ``puck_goal_position_velocity`` tasks are
untouched (the SGCRL real-robot baseline depends on them).
"""

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

from airhockey.airhockey_rewards.goal_task_rewards.puck_goal_sparse_reward import (
    AirHockeyPuckGoalPositionSparseReward,
    AirHockeyPuckGoalPositionSpeedSparseReward,
    AirHockeyPuckGoalPositionVelocitySparseReward,
)

from .abstract_airhockey_goal_task import AirHockeyGoalEnv


class AirHockeyPuckGoalPositionSparseEnv(AirHockeyGoalEnv):
    """Puck to a goal position in the upper half; +10 on arrival (moving up), episode ends."""

    random_paddle_spawn_default = True

    # Achieved goal = (x, y, vx, vy, contacted); desired goal = its first GOAL_DIM entries.
    GOAL_DIM = 2
    ACHIEVED_DIM = 5

    # Position tolerance (m).  A puck in the upper half moves 3-15 cm per
    # 20 Hz step (sysid physics, scripted hits), so a 10 cm radius keeps the
    # end-of-step point check from skipping over the goal.
    DEFAULT_GOAL_RADIUS = 0.10

    def __init__(self, **kwargs):
        self.goal_radius_type = kwargs.get("goal_radius_type", "fixed")
        self.base_goal_radius = float(kwargs.get("base_goal_radius", self.DEFAULT_GOAL_RADIUS))
        # Inset of the goal-sampling box from the upper-half boundary (on top
        # of the puck radius, which the puck centre can never cross).
        self.goal_position_margin = float(kwargs.get("goal_position_margin", self.base_goal_radius))
        # The reward is paid once, on the step the goal is met, so the episode
        # ends there (same as the sparse paddle-reach tasks).
        self.terminate_on_goal_reached = bool(kwargs.get("terminate_on_goal_reached", True))
        # Has the paddle touched the puck in the current episode?  Folded into
        # the achieved goal; cleared on reset.
        self._puck_contacted = False
        super().__init__(**kwargs)

    # ----------------------------------------------------------- spaces
    def _goal_position_bounds(self):
        low = [self.table_x_top, self.table_y_left]
        high = [0.0, self.table_y_right]
        return low, high

    def _achieved_goal_bounds(self):
        low, high = self._goal_position_bounds()
        return (
            low + [-self.max_puck_vel, -self.max_puck_vel, 0.0],
            high + [self.max_puck_vel, self.max_puck_vel, 1.0],
        )

    def _goal_bounds(self):
        """Bounds of the *desired* goal (the first GOAL_DIM achieved components)."""
        low, high = self._achieved_goal_bounds()
        return low[: self.GOAL_DIM], high[: self.GOAL_DIM]

    def initialize_spaces(self, obs_type):
        low, high = self.init_observation(obs_type)
        goal_low, goal_high = self._goal_bounds()
        achieved_low, achieved_high = self._achieved_goal_bounds()
        if self.return_goal_obs:
            self.observation_space = self.single_observation_space = spaces.Dict(
                dict(
                    observation=Box(low=np.array(low), high=np.array(high), dtype=float),
                    desired_goal=Box(low=np.array(goal_low), high=np.array(goal_high), dtype=float),
                    achieved_goal=Box(low=np.array(achieved_low), high=np.array(achieved_high), dtype=float),
                )
            )
        else:
            self.observation_space = self.single_observation_space = self.get_obs_space(
                list(low) + list(goal_low), list(high) + list(goal_high)
            )
        self.goal_radius = self.base_goal_radius
        self.action_space = self.single_action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = Box(low=-1, high=1)
        self.reward = self._build_reward()

    def _build_reward(self):
        return AirHockeyPuckGoalPositionSparseReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionSparseEnv(**state_dict)

    # ------------------------------------------------------------ world
    def get_puck_configuration(self, bad_regions=None):
        # Canonical spawn shared with the five benchmark tasks.
        return self.sample_puck_spawn_top_fraction(bad_regions=bad_regions)

    def create_world_objects(self):
        pos, vel = self.get_puck_configuration()
        self.simulator.spawn_puck(pos, vel, "puck_0")
        pos, vel = self.get_paddle_configuration("paddle_ego")
        self.simulator.spawn_paddle(pos, vel, "paddle_ego")

    def validate_configuration(self):
        assert self.num_pucks == 1
        assert self.num_blocks == 0
        assert self.num_obstacles == 0
        assert self.num_targets == 0
        assert self.num_paddles == 1

    # ------------------------------------------------------------ goals
    def achieved_to_desired(self, achieved):
        """Map achieved goals ``(N, ACHIEVED_DIM)`` to desired-goal space ``(N, GOAL_DIM)``.

        Hindsight relabelling proposes ``achieved_to_desired(ag)`` as goals.
        Default: the first ``GOAL_DIM`` achieved components.
        """
        achieved = np.asarray(achieved, dtype=np.float64)
        achieved = achieved.reshape(-1, achieved.shape[-1]) if achieved.ndim > 1 else achieved.reshape(1, -1)
        return achieved[:, : self.GOAL_DIM]

    def reset(self, seed=None, **kwargs):
        self._puck_contacted = False
        return super().reset(seed, **kwargs)

    def reset_from_state_and_goal(self, state_vector, goal_vector, seed=None):
        self._puck_contacted = False
        return super().reset_from_state_and_goal(state_vector, goal_vector, seed)

    def get_achieved_goal(self, state_info):
        """Puck state plus the episode's contact flag: ``(x, y, vx, vy, contacted)``.

        The flag latches on the first step whose state reports a paddle-puck
        collision (``paddle_puck_collision_count`` from the simulator).
        """
        if int(state_info.get("paddle_puck_collision_count", 0) or 0) > 0:
            self._puck_contacted = True
        puck = state_info["pucks"][0]
        return np.array(
            [puck["position"][0], puck["position"][1], puck["velocity"][0], puck["velocity"][1],
             1.0 if self._puck_contacted else 0.0],
            dtype=np.float64,
        )

    def get_desired_goal(self):
        return np.array(self.goal_pos[:2], dtype=np.float64)

    def goal_reached(self, state_info):
        return bool(self.reward.goal_met(self.get_achieved_goal(state_info), self.get_desired_goal()))

    def get_observation(self, state_info, obs_type="history", **kwargs):
        return self.get_observation_by_type(state_info, obs_type=obs_type, **kwargs)

    def _goal_position_sampling_box(self):
        inset = self.puck_radius + self.goal_position_margin
        min_x = self.table_x_top + inset
        max_x = 0.0 - self.goal_position_margin
        min_y = self.table_y_left + inset
        max_y = self.table_y_right - inset
        if min_x > max_x:
            min_x = max_x = 0.5 * (min_x + max_x)
        if min_y > max_y:
            min_y = max_y = 0.5 * (min_y + max_y)
        return min_x, max_x, min_y, max_y

    def _sample_goal_position(self):
        min_x, max_x, min_y, max_y = self._goal_position_sampling_box()
        return np.array(
            [self.rng.uniform(low=min_x, high=max_x), self.rng.uniform(low=min_y, high=max_y)],
            dtype=np.float64,
        )

    def goal_in_distribution(self, goals):
        """Whether desired goals ``(N, GOAL_DIM)`` lie in the goal-sampling region.

        Used by hindsight relabelling to propose only goals the task would
        ever ask for (an achieved puck state in the lower half is a valid
        *state* but never a sampled goal).  Position goals: inside the
        sampling box, widened by the goal radius.
        """
        goals = np.asarray(goals, dtype=np.float64)
        goals = goals.reshape(-1, goals.shape[-1]) if goals.ndim > 1 else goals.reshape(1, -1)
        min_x, max_x, min_y, max_y = self._goal_position_sampling_box()
        r = self.goal_radius
        return (
            (goals[:, 0] >= min_x - r) & (goals[:, 0] <= max_x + r)
            & (goals[:, 1] >= min_y - r) & (goals[:, 1] <= max_y + r)
        )

    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_set is not None:
            self.goal_pos = np.asarray(goal_set[0, :2], dtype=np.float64)
        elif goal_pos is not None:
            self.goal_pos = np.asarray(goal_pos, dtype=np.float64)[:2]
        else:
            self.goal_pos = self._sample_goal_position()


class AirHockeyPuckGoalPositionVelocitySparseEnv(AirHockeyPuckGoalPositionSparseEnv):
    """Puck to a goal position *with* a goal velocity; +10 when both hold, episode ends."""

    # Velocity tolerance (m/s), the same as the paddle reach_vel task.
    DEFAULT_GOAL_VELOCITY_RADIUS = 0.5
    # Goal-velocity sampling box (env frame; negative x = towards the top of
    # the table).  Measured with scripted hits under the sysid physics: the
    # puck's upward speed in the upper half is 0.6-2.9 m/s (p5-p95, median
    # 1.7) with |vy| below 1.9 m/s at p95.
    DEFAULT_GOAL_VX_RANGE = (-2.5, -0.5)
    DEFAULT_GOAL_VY_RANGE = (-1.5, 1.5)
    DEFAULT_GOAL_MAX_SPEED = 3.0
    # ``shot`` sampler: launch speed (m/s) and half-angle from straight up.
    DEFAULT_SHOT_SPEED_RANGE = (1.0, 3.0)
    DEFAULT_SHOT_MAX_ANGLE_DEG = 45.0
    # A shot state counts as a goal while the puck still moves up at least this fast.
    DEFAULT_SHOT_MIN_UPWARD_SPEED = 0.3
    SHOT_DT = 0.05
    SHOT_MAX_TIME = 3.0

    def __init__(self, **kwargs):
        self.goal_velocity_radius = float(
            kwargs.get("base_goal_velocity_radius", self.DEFAULT_GOAL_VELOCITY_RADIUS)
        )
        self.goal_vx_range = tuple(float(v) for v in kwargs.get("goal_puck_vx_range", self.DEFAULT_GOAL_VX_RANGE))
        self.goal_vy_range = tuple(float(v) for v in kwargs.get("goal_puck_vy_range", self.DEFAULT_GOAL_VY_RANGE))
        self.goal_max_speed = float(kwargs.get("goal_puck_max_speed", self.DEFAULT_GOAL_MAX_SPEED))
        self.goal_velocity_sampling = str(kwargs.get("goal_velocity_sampling", "shot"))
        if self.goal_velocity_sampling not in ("shot", "intercept_shot", "box"):
            raise ValueError("goal_velocity_sampling must be 'shot', 'intercept_shot' or 'box'")
        self.goal_shot_speed_range = tuple(float(v) for v in kwargs.get("goal_shot_speed_range", self.DEFAULT_SHOT_SPEED_RANGE))
        self.goal_shot_max_angle_deg = float(kwargs.get("goal_shot_max_angle_deg", self.DEFAULT_SHOT_MAX_ANGLE_DEG))
        self.goal_shot_min_upward_speed = float(kwargs.get("goal_shot_min_upward_speed", self.DEFAULT_SHOT_MIN_UPWARD_SPEED))
        super().__init__(**kwargs)

    GOAL_DIM = 4

    def _build_reward(self):
        return AirHockeyPuckGoalPositionVelocitySparseReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionVelocitySparseEnv(**state_dict)

    def get_desired_goal(self):
        return np.concatenate([np.asarray(self.goal_pos[:2]), np.asarray(self.goal_vel[:2])]).astype(np.float64)

    def _velocity_in_goal_set(self, vel, tol=0.0):
        """Velocity part of the goal set (widened by ``tol``)."""
        vx, vy = vel[:, 0], vel[:, 1]
        if self.goal_velocity_sampling in ("shot", "intercept_shot"):
            lo, hi = self.goal_shot_speed_range
            ok = vx <= -self.goal_shot_min_upward_speed + tol
            ok &= np.linalg.norm(vel, axis=1) <= hi + tol
            return ok
        ok = (vx >= self.goal_vx_range[0] - tol) & (vx <= self.goal_vx_range[1] + tol)
        ok &= (vy >= self.goal_vy_range[0] - tol) & (vy <= self.goal_vy_range[1] + tol)
        if self.goal_max_speed > 0:
            ok &= np.linalg.norm(vel, axis=1) <= self.goal_max_speed + tol
        return ok

    def goal_in_distribution(self, goals):
        """Position inside the sampling box and velocity inside the goal
        velocity set, each widened by the matching tolerance."""
        goals = np.asarray(goals, dtype=np.float64).reshape(-1, self.GOAL_DIM)
        ok = super().goal_in_distribution(goals[:, :2])
        ok &= self._velocity_in_goal_set(goals[:, 2:4], tol=self.goal_velocity_radius)
        return ok

    def _sample_goal_shot(self):
        """Goal = a random upper-half state of a simulated shot from the paddle workspace.

        Launch point uniform over the reachable paddle workspace, speed
        uniform in ``goal_shot_speed_range``, direction within
        ``goal_shot_max_angle_deg`` of straight up (negative x).  Free flight
        under the table's gravity (along +x) and Box2D-style linear damping,
        with elastic reflections off the side walls, sampled at the control
        rate.  Among the states inside the position box that still move up
        at ``goal_shot_min_upward_speed`` one is drawn uniformly; the shot
        is redrawn if none qualifies.
        """
        for _ in range(64):
            (lx, ly), _ = self.sample_paddle_spawn_in_workspace()
            goal = self._sample_shot_from(np.array([lx, ly], dtype=np.float64))
            if goal is not None:
                return goal
        # Fallback (should not happen): box sampling.
        return self._sample_goal_position(), self._sample_goal_velocity()

    def _flight_params(self):
        g = abs(float(getattr(self.simulator_params, "gravity", 0.661)))
        damping = float(getattr(self.simulator_params, "puck_damping", 0.0) or 0.0)
        wall_lo = self.table_y_left + self.puck_radius
        wall_hi = self.table_y_right - self.puck_radius
        return g, damping, wall_lo, wall_hi

    def _flight_step(self, pos, vel, g, damping, wall_lo, wall_hi, dt):
        vel[0] += g * dt
        vel *= 1.0 / (1.0 + damping * dt)
        pos = pos + vel * dt
        if pos[1] < wall_lo:
            pos[1] = 2 * wall_lo - pos[1]
            vel[1] = -vel[1]
        elif pos[1] > wall_hi:
            pos[1] = 2 * wall_hi - pos[1]
            vel[1] = -vel[1]
        return pos, vel

    def _sample_shot_from(self, launch_pos):
        """One simulated shot from ``launch_pos``; a random qualifying state or None."""
        g, damping, wall_lo, wall_hi = self._flight_params()
        min_x, max_x, min_y, max_y = self._goal_position_sampling_box()
        dt = self.SHOT_DT
        speed = self.rng.uniform(*self.goal_shot_speed_range)
        angle = np.deg2rad(self.rng.uniform(-self.goal_shot_max_angle_deg, self.goal_shot_max_angle_deg))
        pos = np.array(launch_pos, dtype=np.float64)
        vel = np.array([-speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float64)
        candidates = []
        for _ in range(int(self.SHOT_MAX_TIME / dt)):
            pos, vel = self._flight_step(pos, vel, g, damping, wall_lo, wall_hi, dt)
            if vel[0] > -self.goal_shot_min_upward_speed or pos[0] < self.table_x_top + self.puck_radius:
                break
            if min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y:
                candidates.append((pos.copy(), vel.copy()))
        if candidates:
            return candidates[self.rng.randint(len(candidates))]
        return None

    def _predict_intercept_point(self, state_info):
        """Where the spawned puck will be when it reaches a random x inside the
        paddle workspace (free flight from its current state); None if it
        never gets there within ``SHOT_MAX_TIME``."""
        g, damping, wall_lo, wall_hi = self._flight_params()
        x_lo, x_hi, _, _ = self.get_paddle_workspace_bounds()
        x_target = self.rng.uniform(x_lo, x_hi)
        puck = state_info["pucks"][0]
        pos = np.array(puck["position"][:2], dtype=np.float64)
        vel = np.array(puck["velocity"][:2], dtype=np.float64)
        if pos[0] >= x_target:
            return pos.copy()
        dt = self.SHOT_DT
        for _ in range(int(2 * self.SHOT_MAX_TIME / dt)):
            pos, vel = self._flight_step(pos, vel, g, damping, wall_lo, wall_hi, dt)
            if pos[0] >= x_target:
                return pos.copy()
        return None

    def _sample_goal_intercept_shot(self, state_info):
        launch = self._predict_intercept_point(state_info)
        if launch is not None:
            for _ in range(64):
                goal = self._sample_shot_from(launch)
                if goal is not None:
                    return goal
        return self._sample_goal_shot()

    def reset(self, seed=None, **kwargs):
        obs, success = super().reset(seed, **kwargs)
        if self.goal_velocity_sampling == "intercept_shot" and self.goal_set is None:
            # Now that the puck is spawned, condition the goal on its state.
            self.goal_pos, self.goal_vel = self._sample_goal_intercept_shot(self.current_state)
            self._sync_goal_marker_to_simulator()
            desired = self.get_desired_goal()
            if self.return_goal_obs:
                obs["desired_goal"] = desired
            else:
                obs = np.concatenate([obs[: -len(desired)], desired])
        return obs, success

    def _sample_goal_velocity(self):
        vel = np.array(
            [self.rng.uniform(*self.goal_vx_range), self.rng.uniform(*self.goal_vy_range)],
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(vel))
        if speed > self.goal_max_speed > 0:
            vel = vel * (self.goal_max_speed / speed)
        return vel

    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_set is not None:
            goal = np.asarray(goal_set[0], dtype=np.float64)
            self.goal_pos, self.goal_vel = goal[:2], goal[2:4]
        elif goal_pos is not None:
            goal = np.asarray(goal_pos, dtype=np.float64).reshape(-1)
            self.goal_pos = goal[:2]
            self.goal_vel = goal[2:4] if goal.shape[0] >= 4 else self._sample_goal_velocity()
        elif self.goal_velocity_sampling in ("shot", "intercept_shot"):
            # intercept_shot is re-sampled in reset() once the puck exists.
            self.goal_pos, self.goal_vel = self._sample_goal_shot()
        else:
            self.goal_pos = self._sample_goal_position()
            self.goal_vel = self._sample_goal_velocity()


class AirHockeyPuckGoalPositionSpeedSparseEnv(AirHockeyPuckGoalPositionVelocitySparseEnv):
    """Puck to a goal position at a goal *speed* (scalar); +10 when both hold, episode ends."""

    GOAL_DIM = 3
    DEFAULT_GOAL_SPEED_RADIUS = 0.5

    def __init__(self, **kwargs):
        self.goal_speed_radius = float(kwargs.get("base_goal_speed_radius", self.DEFAULT_GOAL_SPEED_RADIUS))
        super().__init__(**kwargs)

    def _goal_bounds(self):
        low, high = self._goal_position_bounds()
        return low + [0.0], high + [self.max_puck_vel]

    def _build_reward(self):
        return AirHockeyPuckGoalPositionSpeedSparseReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionSpeedSparseEnv(**state_dict)

    def get_desired_goal(self):
        return np.array([self.goal_pos[0], self.goal_pos[1], float(np.linalg.norm(self.goal_vel[:2]))], dtype=np.float64)

    def achieved_to_desired(self, achieved):
        achieved = np.asarray(achieved, dtype=np.float64)
        achieved = achieved.reshape(-1, achieved.shape[-1]) if achieved.ndim > 1 else achieved.reshape(1, -1)
        return np.concatenate([achieved[:, :2], np.linalg.norm(achieved[:, 2:4], axis=1, keepdims=True)], axis=1)

    def goal_in_distribution(self, goals):
        goals = np.asarray(goals, dtype=np.float64).reshape(-1, self.GOAL_DIM)
        ok = AirHockeyPuckGoalPositionSparseEnv.goal_in_distribution(self, goals[:, :2])
        tol = self.goal_speed_radius
        if self.goal_velocity_sampling in ("shot", "intercept_shot"):
            lo, hi = self.goal_shot_speed_range
            ok &= (goals[:, 2] >= self.goal_shot_min_upward_speed - tol) & (goals[:, 2] <= hi + tol)
        else:
            ok &= goals[:, 2] <= self.goal_max_speed + tol
        return ok
