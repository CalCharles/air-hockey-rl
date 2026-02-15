from .airhockey_box2d import AirHockeyBox2D
from .air_hockey_real import AirHockeyReal

try:
    from .airhockey_robosuite import AirHockeyRobosuite
    from robosuite.environments.base import register_env
    register_env(AirHockeyRobosuite)
except Exception as e:
    import traceback
    print("⚠️ Robosuite failed to load! Detailed error below:")
    traceback.print_exc()
