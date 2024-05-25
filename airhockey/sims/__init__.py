from .airhockey_box2d import AirHockeyBox2D
try:
    from .airhockey_robosuite import AirHockeyRobosuite
    from robosuite.environments.base import register_env
    register_env(AirHockeyRobosuite)
except:
    print('Robosuite not loaded. Cannot use Robosuite environment.')