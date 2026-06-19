# airhockey/sims/robots/custom_ur5e.py

import numpy as np
# from robosuite.robots.ur5e import UR5e as UR5eRobot
from robosuite.models.robots.robot_model import RobotModel as UR5eRobot

# robosuite/models/robots/robot_model.py

from robosuite.models.robots.manipulators.ur5e_robot import UR5e as UR5eModel

CUSTOM_XML_FP = "/work/10993/rohanpatel01/vista/air-hockey-rl/assets/robots/ur5e/robot.xml"


class _CustomUR5eModel(UR5eModel):
    """Model wrapper that loads from our custom XML path."""

    @property
    def basexml_path(self):
        return CUSTOM_XML_FP

    @property
    def default_mount(self):
        # Our XML already includes the mount body — tell robosuite not to
        # attach another one on top of it.
        return "NoMount"


class AirHockeyUR5e(UR5eRobot):
    """
    Drop-in replacement for robosuite's UR5e robot that loads our custom
    robot.xml (which has the RethinkMount base baked in).
    """

    def load_model(self):
        # 1. Let robosuite do all its normal load_model wiring
        super().load_model()
        # 2. Swap out the model it built with ours
        #    idn is already set on self by __init__ before load_model is called
        self.robot_model = _CustomUR5eModel(idn=self.idn)

    @property
    def init_qpos(self):
        return np.array([-0.23487048, -0.98489984, 2.01435974, -2.74821211, -1.55431237, -3.37570874])