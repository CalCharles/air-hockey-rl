from robosuite.models.grippers import gripper_factory
try:
    from robosuite.models.grippers import GRIPPER_MAPPING
except ImportError:
    from robosuite.models.grippers.gripper_factory import GRIPPER_MAPPING
from airhockey.sims.grippers.round_gripper import RoundGripper

if "RoundGripper" not in GRIPPER_MAPPING:
    print("✅ Registering RoundGripper globally")
    GRIPPER_MAPPING["RoundGripper"] = RoundGripper
else:
    print("ℹ️ RoundGripper already registered")
