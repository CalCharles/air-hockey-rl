from .air_hockey_osc import AirHockeyOperationalSpaceController

from robosuite.controllers import CONTROLLER_INFO

CONTROLLER_INFO["AIR_HOCKEY_OSC_POSE"] = "Customized Operational Space Control (keeps eef on table) (Position + Orientation)"
CONTROLLER_INFO["AIR_HOCKEY_OSC_POSITION"] = "Customized Operational Space Control (keeps eef on table) (Position)"

import robosuite
robosuite.controllers.ALL_CONTROLLERS = CONTROLLER_INFO.keys()