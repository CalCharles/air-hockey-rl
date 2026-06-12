"""
RobosuiteAirHockeyEnv — MuJoCo/robosuite multi-task air hockey env.

Adapted from the meta-rl-air-hockey reference implementation:
  - Robot changed from Panda → UR5e (closest robosuite model to physical UR5)
  - Table dimensions match AirHockeyReal (1.9304 × 0.8636 m)
  - Goal-line y derived from table width rather than hardcoded
  - Stripped camera configs not present on the real rig
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

# Local arena / object imports — you will need to copy or re-implement these
# from the reference repo if they don't already exist in your airhockey package.

from airhockey.sims.robosuite_3D.air_hockey_arena import AirHockeyArena
from airhockey.sims.robosuite_3D.puck import PuckObject

AIR_HOCKEY_TASKS = ("reach", "strike", "block")

class RobosuiteAirHockeyEnv(ManipulationEnv):
    """MuJoCo air hockey environment with UR5e and real-world table dimensions."""

    def __init__(
        self,
        robots="UR5e",
        env_configuration="default",
        controller_configs=None,
        gripper_types=None,               # paddle is a fixed tool, no gripper
        table_full_size=(0.9652, 0.4318, 0.02),   # half-extents for robosuite
        table_friction=(0.01, 0.0001, 0.00001),
        table_offset=(0, 0, 0.8),
        puck_radius=0.03175,
        fixed_task: Optional[str] = "reach",
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=500,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="overview",
        control_freq=20,
        seed=None,
        physics_overrides: Optional[dict] = None,
        **kwargs,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.puck_radius = puck_radius
        self.fixed_task = fixed_task
        self.active_task = fixed_task or "reach"
        self.use_object_obs = use_object_obs
        self.reward_shaping = reward_shaping

        # Goal line at 90 % of half-width (leaves 10 % buffer)
        self.goal_line_y = table_full_size[1] * 0.9
        self.physics_overrides = physics_overrides or {}

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            horizon=horizon,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # robosuite hooks
    # ------------------------------------------------------------------

    def reward(self, action=None):
        return self._compute_reward(action)

    def _compute_reward(self, action=None) -> float:
        if self.active_task == "reach":
            dist = self._gripper_to_puck_distance()
            r = -dist
            if self.reward_shaping:
                r += max(0.0, 1.0 - dist / 0.2)
            return float(r)
        if self.active_task == "strike":
            puck_y = self._puck_pos()[1]
            return float(1.0 if puck_y >= self.goal_line_y else -0.01)
        if self.active_task == "block":
            puck_y = self._puck_pos()[1]
            puck_vy = self._puck_vel()[1]
            return float(1.0 if (puck_y < self.goal_line_y and puck_vy <= 0) else -0.01)
        return 0.0

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = AirHockeyArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.table_bounds_x = (-self.table_full_size[0], self.table_full_size[0])
        self.table_bounds_y = (-self.table_full_size[1], self.table_full_size[1])
        self.table_top_z = self.table_offset[2] + self.table_full_size[2]

        self.puck = PuckObject(name="puck", radius=self.puck_radius)

        self.placement_initializer = UniformRandomSampler(
            name="PuckSampler",
            mujoco_objects=self.puck,
            x_range=[-0.10, 0.10],
            y_range=[-0.08, 0.08],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=self.table_full_size[2] + self.puck_radius,
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.puck,
        )

    def _setup_references(self):
        super()._setup_references()
        self.puck_body_id = self.sim.model.body_name2id(self.puck.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def puck_pos(obs_cache):
                return self._puck_pos()

            @sensor(modality=modality)
            def puck_vel(obs_cache):
                return self._puck_vel()

            for s in (puck_pos, puck_vel):
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self._apply_physics_overrides()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    def _check_success(self) -> bool:
        if self.active_task == "reach":
            return self._gripper_to_puck_distance() < 0.04
        if self.active_task == "strike":
            return self._puck_pos()[1] >= self.goal_line_y
        if self.active_task == "block":
            return self._puck_pos()[1] < self.goal_line_y and self._puck_vel()[1] <= 0.0
        return False

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _puck_pos(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.puck_body_id])

    def _puck_vel(self) -> np.ndarray:
        qvel_addr = self.sim.model.get_joint_qvel_addr(self.puck.joints[0])
        if isinstance(qvel_addr, tuple):
            return np.array(self.sim.data.qvel[qvel_addr[0]:qvel_addr[1]][:3])
        return np.array([self.sim.data.qvel[qvel_addr], 0.0, 0.0])

    # def _gripper_to_puck_distance(self) -> float:
    #     eef_pos = self.robots[0].recent_ee_pose.current[:3]
    #     puck_pos = self._puck_pos()
    #     return float(np.linalg.norm(eef_pos - puck_pos))

    def _gripper_to_puck_distance(self) -> float:
        eef_pose = self.robots[0].recent_ee_pose
        if isinstance(eef_pose, dict):
            arm = self.robots[0].arms[0] if hasattr(self.robots[0], "arms") else "right"
            eef_pos = eef_pose[arm].current[:3]
        else:
            eef_pos = eef_pose.current[:3]
        puck_pos = self._puck_pos()
        return float(np.linalg.norm(eef_pos - puck_pos))
    
    def _apply_physics_overrides(self):
        # """Apply per-episode physics overrides for domain randomization."""
        for key, val in self.physics_overrides.items():
            if key == "puck_damping":
                # Find the puck joint and set its damping
                jnt_id = self.sim.model.joint_name2id(self.puck.joints[0])
                self.sim.model.dof_damping[jnt_id] = float(val)
            elif key == "gravity":
                # MuJoCo gravity is a 3-vec; only z component
                self.sim.model.opt.gravity[2] = float(val)
            # paddle_density requires rebuilding the geom — skip for now or
            # approximate via body mass: sim.model.body_mass[paddle_body_id]