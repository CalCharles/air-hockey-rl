from .airhockey_box2d import AirHockeyBox2D
from .airhockey_robosuite import AirHockeyRobosuite
from .air_hockey_real import AirHockeyReal
from robosuite.environments.base import register_env
register_env(AirHockeyRobosuite)
