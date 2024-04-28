import airhockey.renderers as renderers
import airhockey.sims as sims
import os
import airhockey.sims # this registers the air hockey robosuite env
import airhockey.sims.controllers # this registers the custom controllers!
import airhockey.sims.robots # this registers the custom robot!
import airhockey.sims.grippers # this registers the roundgripper!
from airhockey.airhockey_simple_tasks import AirHockeyPuckVelEnv, AirHockeyPuckHeightEnv, AirHockeyPuckCatchEnv 
from airhockey.airhockey_simple_tasks import AirHockeyPuckJuggleEnv, AirHockeyPuckStrikeEnv, AirHockeyPuckTouchEnv
from airhockey.airhockey_hierarchical_tasks  import AirHockeyMoveBlockEnv, AirHockeyStrikeCrowdEnv
from airhockey.airhockey_goal_tasks import AirHockeyPuckGoalPositionEnv, AirHockeyPuckGoalPositionVelocityEnv
from airhockey.airhockey_goal_tasks import AirHockeyPaddleReachPositionEnv, AirHockeyPaddleReachPositionVelocityEnv

ASSETS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets"))

# let's also copy over Arena asset to robosuite, this is ugly but works fine!
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

from robosuite.models import assets_root
import os
arena_fp = custom_xml_path_completion("arenas/air_hockey_table.xml")
arena_fp_dst = os.path.join(assets_root, "arenas/air_hockey_table.xml")
os.makedirs(os.path.dirname(arena_fp_dst), exist_ok=True)
import shutil
shutil.copyfile(arena_fp, arena_fp_dst)

def AirHockeyEnv(cfg):
    # check what task
    task = cfg['task']
    # get corresponding env
    if task == "puck_velocity":
        task_env = AirHockeyPuckVelEnv
    elif task == "puck_height":
        task_env = AirHockeyPuckHeightEnv
    elif task == "puck_catch":
        task_env = AirHockeyPuckCatchEnv
    elif task == "puck_juggle":
        task_env = AirHockeyPuckJuggleEnv
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
    else:
        raise ValueError("Task {} not recognized".format(task))
    return task_env.from_dict(cfg)

