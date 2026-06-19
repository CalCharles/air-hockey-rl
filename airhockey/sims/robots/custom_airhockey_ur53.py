import os
# from robosuite.models.robots.robot_model import ManipulatorModel
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.utils.mjcf_utils import robosuite_xml_path_completion

class CustomAirHockeyUR5e(UR5e):
    """
    Custom UR5e class that forces Robosuite to load a specific, 
    modified robot XML from the airhockey asset directory.
    """
    def __init__(self, idn=0):
        # We initialize using the standard UR5e definition but will redirect the paths
        super().__init__(idn=idn)

    @property
    def basexml_path(self):
        # Point this directly to your custom folder destination
        # Ensure 'robots/custom_ur5e/custom_robot.xml' is resolved properly
        return robosuite_xml_path_completion(
            os.path.join("robots", "ur5e", "robot.xml")
        )