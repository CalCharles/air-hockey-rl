import airhockey.renderers as renderers
import airhockey.sims as sims
import os

from airhockey.airhockey_simple_tasks import (
    AirHockeyPuckVelEnv,
    AirHockeyPuckHeightEnv,
    AirHockeyPuckCatchEnv,
)
from airhockey.airhockey_simple_tasks import (
    AirHockeyPuckJuggleEnv,
    AirHockeyPuckJuggleLinearTopEnv,
    AirHockeyPuckJuggleNoBaseRewardEnv,
    AirHockeyPuckJuggleUpperHalfRewardEnv,
    AirHockeyPuckJugglePinballTriangleSidesEnv,
    AirHockeyPuckTopEdgeGoalTrianglesEnv,
    AirHockeyPuckScoreEnv,
    AirHockeyPuckJuggleUpperHalfMidBandRewardEnv,
    AirHockeyPuckStrikeEnv,
    AirHockeyPuckTouchEnv,
    AirHockeyPaddleFreeMovementEnv,
)
from airhockey.airhockey_hierarchical_tasks import AirHockeyMoveBlockEnv, AirHockeyStrikeCrowdEnv
from airhockey.airhockey_tasks.paddle_reach_position import AirHockeyPaddleReachPositionEnv
from airhockey.airhockey_tasks.puck_goal_position import AirHockeyPuckGoalPositionEnv
from airhockey.airhockey_tasks.paddle_reach_position_velocity import (
    AirHockeyPaddleReachPositionVelocityEnv,
)
from airhockey.airhockey_tasks.puck_goal_position_velocity import (
    AirHockeyPuckGoalPositionVelocityEnv,
)
from airhockey.airhockey_tasks.paddle_reach_position_negative_regions import (
    AirHockeyPaddleReachPositionNegRegionsEnv,
)
from airhockey.airhockey_tasks.puck_goal_position_dynamic_negative_regions import (
    AirHockeyPuckGoalPositionDynamicNegRegionsEnv,
)
from airhockey.airhockey_tasks.puck_goal_position_obstacles import (
    AirHockeyPuckGoalPositionObstaclesEnv,
)
from airhockey.sims.robosuite_3D.robosuite_env import RobosuiteAirHockeyAdapter


ASSETS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets"))


def custom_xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path

    Returns:
        str: Full (absolute) xml path
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(ASSETS_ROOT, xml_path)
    return full_path


def AirHockeyEnv(cfg):
    task = cfg["task"]
    if task == "puck_velocity":
        task_env = AirHockeyPuckVelEnv
    elif task == "puck_height":
        task_env = AirHockeyPuckHeightEnv
    elif task == "puck_catch":
        task_env = AirHockeyPuckCatchEnv
    elif task == "puck_juggle" or task == "multipuck_juggle":
        task_env = AirHockeyPuckJuggleEnv
    elif task == "puck_juggle_linear_top" or task == "multipuck_juggle_linear_top":
        task_env = AirHockeyPuckJuggleLinearTopEnv
    elif task == "puck_juggle_no_base_reward" or task == "multipuck_juggle_no_base_reward":
        task_env = AirHockeyPuckJuggleNoBaseRewardEnv
    elif task == "puck_juggle_upper_half_reward" or task == "multipuck_juggle_upper_half_reward":
        task_env = AirHockeyPuckJuggleUpperHalfRewardEnv
    elif task == "puck_juggle_pinball_triangle_sides" or task == "multipuck_juggle_pinball_triangle_sides":
        task_env = AirHockeyPuckJugglePinballTriangleSidesEnv
    elif task == "puck_goal_top_edge_slot_triangles" or task == "multipuck_goal_top_edge_slot_triangles":
        task_env = AirHockeyPuckTopEdgeGoalTrianglesEnv
    elif task == "puck_score" or task == "multipuck_score":
        task_env = AirHockeyPuckScoreEnv
    elif task == "puck_juggle_upper_half_mid_band_reward" or task == "multipuck_juggle_upper_half_mid_band_reward":
        task_env = AirHockeyPuckJuggleUpperHalfMidBandRewardEnv
    elif task == "puck_strike":
        task_env = AirHockeyPuckStrikeEnv
    elif task == "puck_touch":
        task_env = AirHockeyPuckTouchEnv
    elif task == "move_block":
        task_env = AirHockeyMoveBlockEnv
    elif task == "strike_crowd":
        task_env = AirHockeyStrikeCrowdEnv
    elif task == "puck_goal_position":
        task_env = AirHockeyPuckGoalPositionEnv
    elif task == "puck_goal_position_velocity":
        task_env = AirHockeyPuckGoalPositionVelocityEnv
    elif task == "paddle_reach_position":
        task_env = AirHockeyPaddleReachPositionEnv
    elif task == "paddle_reach_position_velocity":
        task_env = AirHockeyPaddleReachPositionVelocityEnv
    elif task == "paddle_reach_position_neg":
        task_env = AirHockeyPaddleReachPositionNegRegionsEnv
    elif task == "puck_goal_position_dynamic_neg":
        task_env = AirHockeyPuckGoalPositionDynamicNegRegionsEnv
    elif task == "puck_goal_position_obstacles":
        task_env = AirHockeyPuckGoalPositionObstaclesEnv
    elif task == "paddle_free_movement":
        task_env = AirHockeyPaddleFreeMovementEnv
    else:
        raise ValueError("Task {} not recognized".format(task))
    return task_env.from_dict(cfg)

# TODO: Change name to 3D simulator
def make_env(config: dict):
    sim = config.get("simulator", "box2d")
    if sim == "robosuite":
        return RobosuiteAirHockeyAdapter(
            task=config.get("task", "reach"),
            reward_shaping=config.get("reward_shaping", True),
            max_episode_steps=config.get("max_timesteps", 500),
            seed=config.get("seed", None),
            control_freq=config.get("control_freq", 20),
            domain_random=config.get("domain_random", False),
            random_variables=config.get("random_variables", []),
            random_variable_ranges=config.get("random_variable_ranges", {}),
        )
    return AirHockeyEnv(config)   # Box2D path unchanged