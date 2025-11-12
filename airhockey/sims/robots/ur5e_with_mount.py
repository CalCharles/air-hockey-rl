"""from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.models.base import register_robot

@register_robot
class UR5eWithMount(UR5e):
    
   # UR5e robot with a visible gripper mount site for attaching a custom gripper (e.g. RoundGripper).
   

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        # Explicitly set end effector site name for gripper attachment
        self.eef_site_name = "flange"  # common UR5e attachment point in MJCF"""
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.models.base import register_robot
from robosuite.robots import ROBOT_CLASS_MAPPING

@register_robot
class UR5eWithMount(UR5e):
    """
    UR5e robot with a visible gripper mount site for attaching a custom gripper (e.g. RoundGripper).
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self.eef_site_name = "flange"  # typical UR5e tool mount site

# ✅ Manually register it in robosuite’s robot class mapping
ROBOT_CLASS_MAPPING["UR5eWithMount"] = UR5eWithMount
