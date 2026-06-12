# airhockey/envs/robosuite_env.py
# Corrected to give right output dimension to work with existing support like Box2D

from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym

# try:
#     import robosuite as suite
#     from robosuite.controllers import load_controller_config
#     HAS_ROBOSUITE = True
# except ImportError:
#     HAS_ROBOSUITE = False


# import robosuite as suite
# from robosuite.controllers import load_controller_config
# HAS_ROBOSUITE = True

# try:
#     import robosuite as suite
#     # robosuite >= 1.5 moved this to robosuite.controllers.composite
#     # robosuite <  1.5 had it at robosuite.controllers directly
#     try:
#         from robosuite.controllers import load_composite_controller_config as load_controller_config
#     except ImportError:
#         from robosuite.controllers.composite_controller import load_composite_controller_config as load_controller_config
#     HAS_ROBOSUITE = True
# except ImportError:
#     HAS_ROBOSUITE = False

try:
    import robosuite as suite
    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False


from airhockey.sims.robosuite_3D.robosuite_air_hockey import RobosuiteAirHockeyEnv

_TABLE_LENGTH   = 1.9304
_TABLE_WIDTH    = 0.8636
_TABLE_HEIGHT   = 0.02
_PUCK_RADIUS    = 0.03175
_PADDLE_RADIUS  = 0.0508      # from AirHockeyReal simulator_params
_HISTORY_LEN    = 5           # matches Box2D obs layout
_OBS_DIM        = 30          # 5*3 paddle + 5*3 puck — matches Box2D

# Half-extents for robosuite (it uses half-sizes internally)
_TABLE_FULL_SIZE = (_TABLE_LENGTH / 2, _TABLE_WIDTH / 2, _TABLE_HEIGHT)

# TODO: Better name might be a Robosuite air hockey wrapper.
class RobosuiteAirHockeyAdapter(gym.Env):
    """
    Wraps the MuJoCo air hockey env to be a drop-in for AirHockeyEnv (Box2D).

    Observation space: Box(30,) — identical layout to Box2D history obs:
        [0:15]  paddle history  5 × [x, y, valid]  oldest→newest
        [15:30] puck   history  5 × [x, y, valid]  oldest→newest

    Action space: Box([-1,1], shape=(2,)) — normalised (dx, dy), same as Box2D.

    This means HistoryBuffer.extract_entry() works unchanged on observations
    from either env, since PADDLE_POS_SLICE and PUCK_POS_SLICE both index
    into the same 30-dim layout.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "reach",
        reward_shaping: bool = True,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
        control_freq: int = 20,
        has_offscreen_renderer: bool = False,   # add this
        camera_name: str = "overview",          # add this
        # Domain randomization (mirrors Box2D random_variables interface)
        domain_random: bool = False,
        random_variables: Optional[list] = None,
        random_variable_ranges: Optional[dict] = None,
        **kwargs,
    ):
        if not HAS_ROBOSUITE:
            raise ImportError("pip install robosuite")

        super().__init__()

        self.task = task
        self.reward_shaping = reward_shaping
        self.max_episode_steps = max_episode_steps
        self._seed = seed
        self.control_freq = control_freq
        self.domain_random = domain_random
        self.random_variables = random_variables or []
        self.random_variable_ranges = random_variable_ranges or {}

        self._has_offscreen_renderer = has_offscreen_renderer
        self._camera_name = camera_name
        self.has_offscreen_renderer = has_offscreen_renderer

        # Gymnasium spaces — identical shapes to Box2D so AsyncVectorEnv works
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Table geometry (used by training-loop introspection, mirrors AirHockeyReal)
        self.length = _TABLE_LENGTH
        self.width  = _TABLE_WIDTH
        self.puck_radius   = _PUCK_RADIUS
        self.paddle_radius = _PADDLE_RADIUS
        self.time_frequency = control_freq
        self.lims = (
            -_TABLE_LENGTH / 2, _TABLE_LENGTH / 2,
            -_TABLE_WIDTH  / 2, _TABLE_WIDTH  / 2,
        )

        # History ring-buffers (list of [x, y, valid])
        self._paddle_history: list = []
        self._puck_history:   list = []
        self._elapsed = 0

        self._env = self._build_env()

    # ------------------------------------------------------------------
    # Internal env construction (rebuilt on domain-rand resets)
    # ------------------------------------------------------------------

    def _sample_dr_overrides(self) -> dict:
        """Sample per-reset physics overrides when domain_random=True."""
        overrides = {}
        for var in self.random_variables:
            if var in self.random_variable_ranges:
                lo, hi = self.random_variable_ranges[var]
                overrides[var] = float(np.random.uniform(lo, hi))
        return overrides

    # def _build_env(self, physics_overrides: Optional[dict] = None):

    #     controller_config = load_controller_config(default_controller="OSC_POSE")
    #     # Match real UR5 workspace limits from AirHockeyReal
    #     controller_config["output_max"] = [0.26, 0.12, 0.05, 0.5, 0.5, 0.5]
    #     controller_config["output_min"] = [-0.26, -0.12, -0.05, -0.5, -0.5, -0.5]

    #     env = RobosuiteAirHockeyEnv(
    #         robots="UR5e",
    #         controller_configs=controller_config,
    #         gripper_types=None,
    #         table_full_size=_TABLE_FULL_SIZE,
    #         table_friction=(0.01, 0.0001, 0.00001),
    #         puck_radius=_PUCK_RADIUS,
    #         fixed_task=self.task,
    #         reward_shaping=self.reward_shaping,
    #         has_renderer=False,
    #         use_camera_obs=False,
    #         use_object_obs=True,
    #         control_freq=self.control_freq,
    #         horizon=self.max_episode_steps,
    #         physics_overrides=physics_overrides or {},
    #         seed=self._seed,
    #         has_offscreen_renderer=self._has_offscreen_renderer,  # use stored value
    #     )
    #     return env
    # Expose sim for direct access if needed
    
    @property
    def sim(self):
        return self._env.sim

    def _build_env(self, physics_overrides=None):
        from airhockey.sims.robosuite_3D.robosuite_air_hockey import RobosuiteAirHockeyEnv

        def _make_osc_controller_config() -> dict:
            """
            Composite BASIC controller config for UR5e in robosuite >= 1.5.
            BASIC wraps a single arm controller (OSC_POSE) for fixed-base robots.
            """
            from robosuite.controllers.composite.composite_controller_factory import (
                load_composite_controller_config,
            )
            cfg = load_composite_controller_config(controller="BASIC", robot="UR5e")

            # Find the arm part and patch it to OSC_POSE with our workspace limits
            # The exact key depends on the output of the debug command above
            for part_name, part_cfg in cfg.items():
                if isinstance(part_cfg, dict) and part_cfg.get("type") in (
                    "OSC_POSE", "OSC_POSITION", "JOINT_VELOCITY", "JOINT_TORQUE"
                ):
                    part_cfg["type"] = "OSC_POSE"
                    part_cfg["output_max"] = [0.26, 0.12, 0.05, 0.5, 0.5, 0.5]
                    part_cfg["output_min"] = [-0.26, -0.12, -0.05, -0.5, -0.5, -0.5]
                    part_cfg["control_delta"] = True
                    break

            return cfg
        
        return RobosuiteAirHockeyEnv(
            robots="UR5e",
            controller_configs=_make_osc_controller_config(),
            gripper_types=None,
            table_full_size=_TABLE_FULL_SIZE,
            table_friction=(0.01, 0.0001, 0.00001),
            puck_radius=_PUCK_RADIUS,
            fixed_task=self.task,
            reward_shaping=self.reward_shaping,
            has_renderer=False,
            has_offscreen_renderer=self._has_offscreen_renderer,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq=self.control_freq,
            horizon=self.max_episode_steps,
            physics_overrides=physics_overrides or {},
            seed=self._seed,
        )

    @classmethod
    def for_video(cls, config: dict, camera_name: str = "overview") -> "RobosuiteAirHockeyAdapter":
        """
        Build a video-capable copy from the same YAML config dict.
        Use this for eval recording; never for training (costs ~2x memory).
        """
        return cls(
            task=config.get("task", "reach"),
            reward_shaping=config.get("reward_shaping", True),
            max_episode_steps=config.get("max_timesteps", 500),
            seed=config.get("seed", None),
            control_freq=config.get("control_freq", 20),
            has_offscreen_renderer=True,
            camera_name=camera_name,
            # no domain_random for eval — fixed physics
        )


    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self._elapsed = 0

        # Domain randomization: rebuild env with new physics each episode
        if self.domain_random and self.random_variables:
            overrides = self._sample_dr_overrides()
            self._env.close()
            self._env = self._build_env(physics_overrides=overrides)

        obs_dict = self._env.reset()

        puck_pos   = self._extract_puck_pos(obs_dict)
        paddle_pos = self._extract_paddle_pos(obs_dict)

        # Pad history with occluded sentinels (valid=1 means occluded, matching Box2D)
        self._puck_history   = [[puck_pos[0],   puck_pos[1],   1]] * _HISTORY_LEN
        self._paddle_history = [[paddle_pos[0], paddle_pos[1], 1]] * _HISTORY_LEN

        obs = self._build_obs()
        info = {"task": self.task, "success": False, "paddle_puck_collision_count": 0}
        return obs, info

    def step(self, action: np.ndarray):
        # Scale normalised [-1,1] action to metric deltas, then pad to 6-D OSC
        full_action = np.zeros(6, dtype=np.float32)
        full_action[0] = float(action[0]) * 0.26   # rmax_x from AirHockeyReal
        full_action[1] = float(action[1]) * 0.12   # rmax_y from AirHockeyReal

        obs_dict, reward, done, info = self._env.step(full_action)
        self._elapsed += 1

        puck_pos   = self._extract_puck_pos(obs_dict)
        paddle_pos = self._extract_paddle_pos(obs_dict)

        # valid=0 means visible — same convention as Box2D
        self._puck_history.append([puck_pos[0], puck_pos[1], 0])
        self._paddle_history.append([paddle_pos[0], paddle_pos[1], 0])

        truncated  = self._elapsed >= self.max_episode_steps
        terminated = bool(done) and not truncated

        obs = self._build_obs()
        info = dict(info or {})
        info["success"]                   = bool(self._env._check_success())
        info["paddle_puck_collision_count"] = 0     # not tracked in MuJoCo yet
        info["task"]                      = self.task
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()

    # ------------------------------------------------------------------
    # Obs building — produces the same 30-dim layout as Box2D
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """
        Build the 30-dim flat observation that HistoryBuffer expects.

        Layout (identical to Box2D history obs):
            [0:15]  paddle: 5 × [x, y, valid]   (oldest first)
            [15:30] puck:   5 × [x, y, valid]   (oldest first)
        """
        paddle_hist = self._paddle_history[-_HISTORY_LEN:]
        puck_hist   = self._puck_history[-_HISTORY_LEN:]

        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        for i, entry in enumerate(paddle_hist):
            obs[i * 3 : i * 3 + 3] = entry          # [0:15]
        for i, entry in enumerate(puck_hist):
            obs[15 + i * 3 : 15 + i * 3 + 3] = entry  # [15:30]
        return obs

    def _extract_puck_pos(self, obs_dict) -> np.ndarray:
        if "puck_pos" in obs_dict:
            return np.array(obs_dict["puck_pos"][:2], dtype=np.float32)
        body_id = self._env.sim.model.body_name2id("puck_main")
        return np.array(self._env.sim.data.body_xpos[body_id][:2], dtype=np.float32)

    def _extract_paddle_pos(self, obs_dict) -> np.ndarray:
        if "robot0_eef_pos" in obs_dict:
            return np.array(obs_dict["robot0_eef_pos"][:2], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)