import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion as robosuite_xml_path_completion


class AirHockeyUR5e(ManipulatorModel):
    """
    UR5e is a sleek and elegant new robot created by Universal Robots.
    This file customizes the UR5e for use with the Air Hockey environment.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(robosuite_xml_path_completion("robots/custom_ur5e/custom_robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "RoundGripper"

    @property
    def default_controller_config(self):
        return "default_ur5e"

    @property
    def init_qpos(self):
        return np.array([-0.470, -1.2, 2.3, -2.6, -1.590, -1.991]) # TODO: update this, right now it is initializes too high up

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0)) # TODO: Why is this 1.0? Double check.

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
