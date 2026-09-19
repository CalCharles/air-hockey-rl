"""Sparse, goal-conditioned puck tasks: hit the puck to a goal in the upper
half of the table (optionally at a goal speed).

``puck_goal_position_sparse``
    Goal = puck position ``(x, y)`` sampled uniformly over the upper half of
    the table.  Reward +10 on the step the puck centre is within
    ``base_goal_radius`` of the goal *after the paddle has touched the puck in
    this episode* (the episode ends there), 0 otherwise.

``puck_goal_position_speed_sparse``
    Goal = ``(x, y, speed)``: a puck position plus the puck's speed there,
    both taken from a simulated shot launched from the paddle workspace
    (``goal_sampling: shot``) or from where this episode's puck will cross
    the workspace (``intercept_shot``), so every goal is consistent with some
    straight shot.  Reward +10 when the position tolerance
    (``base_goal_radius``) and the speed tolerance (``base_goal_speed_radius``,
    m/s) hold on the same step after contact.  The direction of the puck at
    the goal is left to the shot geometry: a full velocity-vector goal was
    tried and not learnt (notes/scratch/experiments/2026-09-10_06-30_her-puck-goal-tasks.md;
    code at commit 467da75).

The contact condition makes both tasks "hit the puck to the goal": without
it a puck that spawns on the goal, or drifts through it, scores for free,
and hindsight relabelling then rewards the policy for states its actions did
not cause.  The achieved goal is therefore the puck state plus the contact
flag, ``(x, y, vx, vy, contacted)``; ``achieved_to_desired`` maps it to the
desired-goal space.

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
)

from .abstract_airhockey_goal_task import AirHockeyGoalEnv


class AirHockeyPuckGoalPositionSparseEnv(AirHockeyGoalEnv):
    """Puck to a goal position in the upper half; +10 on arrival after contact, episode ends."""

    random_paddle_spawn_default = True

    # Achieved goal = (x, y, vx, vy, contacted); desired goal = achieved_to_desired(achieved).
    GOAL_DIM = 2
    ACHIEVED_DIM = 5
    CONTACT_COL = 4

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
        return [self.table_x_top, self.table_y_left], [0.0, self.table_y_right]

    def _achieved_goal_bounds(self):
        low, high = self._goal_position_bounds()
        return (
            low + [-self.max_puck_vel, -self.max_puck_vel, 0.0],
            high + [self.max_puck_vel, self.max_puck_vel, 1.0],
        )

    def _goal_bounds(self):
        """Bounds of the *desired* goal."""
        return self._goal_position_bounds()

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
    @staticmethod
    def _as_rows(goals):
        goals = np.asarray(goals, dtype=np.float64)
        return goals.reshape(-1, goals.shape[-1]) if goals.ndim > 1 else goals.reshape(1, -1)

    def achieved_to_desired(self, achieved):
        """Map achieved goals ``(N, ACHIEVED_DIM)`` to desired-goal space ``(N, GOAL_DIM)``.

        Hindsight relabelling proposes ``achieved_to_desired(ag)`` as goals.
        """
        return self._as_rows(achieved)[:, : self.GOAL_DIM]

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
        """Whether desired goals lie in the goal-sampling region (position box
        widened by the goal radius).

        Used by hindsight relabelling to propose only goals the task would
        ever ask for (an achieved puck state in the lower half is a valid
        *state* but never a sampled goal).
        """
        goals = self._as_rows(goals)
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


class AirHockeyPuckGoalPositionSpeedSparseEnv(AirHockeyPuckGoalPositionSparseEnv):
    """Puck to a goal position at a goal *speed*; +10 when both hold after contact, episode ends."""

    GOAL_DIM = 3

    # Speed tolerance (m/s), the same as the paddle reach_vel velocity tolerance.
    DEFAULT_GOAL_SPEED_RADIUS = 0.5
    # Shot sampler: launch speed (m/s) and half-angle from straight up
    # (negative x).  Measured with scripted hits under the sysid physics the
    # puck's upward speed in the upper half is 0.6-2.9 m/s (p5-p95).
    DEFAULT_SHOT_SPEED_RANGE = (1.0, 3.0)
    DEFAULT_SHOT_MAX_ANGLE_DEG = 45.0
    # A shot state counts as a goal while the puck still moves up at least this fast.
    DEFAULT_SHOT_MIN_UPWARD_SPEED = 0.3
    SHOT_DT = 0.05
    SHOT_MAX_TIME = 3.0
    GOAL_SAMPLERS = ("shot", "intercept_shot")

    def __init__(self, **kwargs):
        self.goal_speed_radius = float(kwargs.get("base_goal_speed_radius", self.DEFAULT_GOAL_SPEED_RADIUS))
        self.goal_sampling = str(kwargs.get("goal_sampling", kwargs.get("goal_velocity_sampling", "shot")))
        if self.goal_sampling not in self.GOAL_SAMPLERS:
            raise ValueError(f"goal_sampling must be one of {self.GOAL_SAMPLERS}, got {self.goal_sampling!r}")
        self.goal_shot_speed_range = tuple(float(v) for v in kwargs.get("goal_shot_speed_range", self.DEFAULT_SHOT_SPEED_RANGE))
        self.goal_shot_max_angle_deg = float(kwargs.get("goal_shot_max_angle_deg", self.DEFAULT_SHOT_MAX_ANGLE_DEG))
        self.goal_shot_min_upward_speed = float(kwargs.get("goal_shot_min_upward_speed", self.DEFAULT_SHOT_MIN_UPWARD_SPEED))
        # Velocity of the sampled shot at the goal (the goal itself only keeps its norm).
        self.goal_vel = np.zeros(2, dtype=np.float64)
        super().__init__(**kwargs)

    def _goal_bounds(self):
        low, high = self._goal_position_bounds()
        return low + [0.0], high + [self.max_puck_vel]

    def _build_reward(self):
        return AirHockeyPuckGoalPositionSpeedSparseReward(self)

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyPuckGoalPositionSpeedSparseEnv(**state_dict)

    # ------------------------------------------------------------ goals
    @property
    def goal_speed(self):
        return float(np.linalg.norm(self.goal_vel[:2]))

    def get_desired_goal(self):
        return np.array([self.goal_pos[0], self.goal_pos[1], self.goal_speed], dtype=np.float64)

    def achieved_to_desired(self, achieved):
        achieved = self._as_rows(achieved)
        return np.concatenate([achieved[:, :2], np.linalg.norm(achieved[:, 2:4], axis=1, keepdims=True)], axis=1)

    def goal_in_distribution(self, goals):
        goals = self._as_rows(goals)
        ok = super().goal_in_distribution(goals[:, :2])
        tol = self.goal_speed_radius
        ok &= (goals[:, 2] >= self.goal_shot_min_upward_speed - tol)
        ok &= (goals[:, 2] <= self.goal_shot_speed_range[1] + tol)
        return ok

    # ------------------------------------------------------ shot sampler
    def _flight_params(self):
        g = abs(float(getattr(self.simulator_params, "gravity", 0.661)))
        damping = float(getattr(self.simulator_params, "puck_damping", 0.0) or 0.0)
        wall_lo = self.table_y_left + self.puck_radius
        wall_hi = self.table_y_right - self.puck_radius
        return g, damping, wall_lo, wall_hi

    @staticmethod
    def _flight_step(pos, vel, g, damping, wall_lo, wall_hi, dt):
        """One step of free flight: gravity along +x, Box2D-style linear
        damping, elastic reflection off the side walls."""
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
        """One simulated shot from ``launch_pos`` (speed uniform in
        ``goal_shot_speed_range``, direction within ``goal_shot_max_angle_deg``
        of straight up); returns a random state of it inside the position box
        that still moves up at ``goal_shot_min_upward_speed``, or None."""
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

    def _sample_goal_shot(self):
        """Goal = a random upper-half state of a shot from a random workspace point."""
        for _ in range(64):
            (lx, ly), _ = self.sample_paddle_spawn_in_workspace()
            goal = self._sample_shot_from(np.array([lx, ly], dtype=np.float64))
            if goal is not None:
                return goal
        # Fallback (should not happen): slowest shot speed at a random position.
        return self._sample_goal_position(), np.array([-self.goal_shot_speed_range[0], 0.0])

    def _predict_intercept_point(self, state_info):
        """Where the spawned puck will be when it reaches a random x inside the
        paddle workspace (free flight from its current state); None if it
        never gets there within ``2 * SHOT_MAX_TIME``."""
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

    def set_goals(self, goal_radius_type, goal_pos=None, alt_goal_pos=None, goal_set=None):
        self.goal_set = goal_set
        if goal_set is not None:
            goal = np.asarray(goal_set[0], dtype=np.float64)
            self.goal_pos, self.goal_vel = goal[:2], self._vel_from_goal(goal)
        elif goal_pos is not None:
            goal = np.asarray(goal_pos, dtype=np.float64).reshape(-1)
            self.goal_pos = goal[:2]
            self.goal_vel = self._vel_from_goal(goal) if goal.shape[0] >= 3 else self._sample_goal_shot()[1]
        else:
            # intercept_shot is re-sampled in reset() once the puck exists.
            self.goal_pos, self.goal_vel = self._sample_goal_shot()

    @staticmethod
    def _vel_from_goal(goal):
        """A velocity with the goal's speed (``(x, y, speed)``) or the goal's own velocity (``(x, y, vx, vy)``)."""
        if goal.shape[0] >= 4:
            return np.asarray(goal[2:4], dtype=np.float64)
        return np.array([-float(goal[2]), 0.0], dtype=np.float64)

    def reset(self, seed=None, **kwargs):
        obs, success = super().reset(seed, **kwargs)
        if self.goal_sampling == "intercept_shot" and self.goal_set is None:
            # Now that the puck is spawned, condition the goal on its state.
            self.goal_pos, self.goal_vel = self._sample_goal_intercept_shot(self.current_state)
            self._sync_goal_marker_to_simulator()
            desired = self.get_desired_goal()
            if self.return_goal_obs:
                obs["desired_goal"] = desired
            else:
                obs = np.concatenate([obs[: -len(desired)], desired])
        return obs, success
