import airhockey.renderers as renderers
import airhockey.sims as sims
import os
import shutil

ROBOSUITE_AVAILABLE = False
robosuite_xml_path_completion = None
assets_root = None
try:
    from robosuite.utils.mjcf_utils import xml_path_completion as robosuite_xml_path_completion
    from robosuite.models import assets_root
    ROBOSUITE_AVAILABLE = True
except Exception:
    print("Robosuite not loaded. Robosuite-only components are unavailable.")
# import airhockey.sims # this registers the air hockey robosuite env
# Register optional robosuite extras independently so a failure in one (e.g.
# `controllers`/`robots` use the old robosuite 1.4 API and break on 1.5+)
# doesn't prevent the others (notably `grippers`, which contains the round
# paddle) from registering.
if ROBOSUITE_AVAILABLE:
    for _module in (
        "airhockey.sims.controllers",
        "airhockey.sims.robots",
        "airhockey.sims.grippers",
        "airhockey.sims.utils.RobosuiteTransforms",
    ):
        try:
            __import__(_module)
        except Exception as _e:
            print(f"airhockey: optional component {_module} not loaded ({type(_e).__name__}: {_e})")
from airhockey.airhockey_simple_tasks import AirHockeyPuckVelEnv, AirHockeyPuckHeightEnv, AirHockeyPuckCatchEnv
from airhockey.airhockey_simple_tasks import AirHockeyPuckJuggleEnv, AirHockeyPuckJuggleLinearTopEnv, AirHockeyPuckJuggleNoBaseRewardEnv, AirHockeyPuckJuggleUpperHalfRewardEnv, AirHockeyPuckJuggleUpperHalfMidBandRewardEnv, AirHockeyPuckStrikeEnv, AirHockeyPuckTouchEnv, AirHockeyPaddleFreeMovementEnv
from airhockey.airhockey_hierarchical_tasks  import AirHockeyMoveBlockEnv, AirHockeyStrikeCrowdEnv
# from airhockey.airhockey_goal_tasks import AirHockeyPuckGoalPositionEnv, AirHockeyPuckGoalPositionVelocityEnv, AirHockeyPuckReachPositionDynamicNegRegionsEnv
# from airhockey.airhockey_goal_tasks import AirHockeyPaddleReachPositionEnv, AirHockeyPaddleReachPositionVelocityEnv, AirHockeyPaddleReachPositionNegRegionsEnv
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


if ROBOSUITE_AVAILABLE:
    arena_fp = custom_xml_path_completion("arenas/air_hockey_table.xml")
    arena_fp_dst = os.path.join(assets_root, "arenas/air_hockey_table.xml")
    os.makedirs(os.path.dirname(arena_fp_dst), exist_ok=True)
    shutil.copyfile(arena_fp, arena_fp_dst)


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


if ROBOSUITE_AVAILABLE:
    robosuite_robot_assets_fp = robosuite_xml_path_completion(os.path.join("robots", "ur5e"))
    robot_xml_fp = custom_xml_path_completion(os.path.join("robots", "ur5e", "robot.xml"))
    new_folder_fp = robosuite_xml_path_completion(os.path.join("robots", "custom_ur5e"))
    out_robot_xml_fp = robosuite_xml_path_completion(os.path.join(new_folder_fp, "custom_robot.xml"))
    if not os.path.exists(new_folder_fp):
        shutil.copytree(robosuite_robot_assets_fp, new_folder_fp)
    shutil.copy(robot_xml_fp, out_robot_xml_fp)
