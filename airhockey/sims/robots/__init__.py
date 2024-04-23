# First register robot
from robosuite.models.robots.robot_model import register_robot
from .custom_ur5e import AirHockeyUR5e
register_robot(AirHockeyUR5e)

from robosuite.robots import ROBOT_CLASS_MAPPING
from airhockey.sims.robots.custom_single_arm import AirHockeySingleArm
ROBOT_CLASS_MAPPING["AirHockeyUR5e"] = AirHockeySingleArm
