"""
Set of functions that streamline controller initialization process
"""
from copy import deepcopy

from robosuite.controllers.interpolators.linear_interpolator import LinearInterpolator
from robosuite.controllers.joint_pos import JointPositionController
from robosuite.controllers.joint_tor import JointTorqueController
from robosuite.controllers.joint_vel import JointVelocityController
from robosuite.controllers.osc import OperationalSpaceController
from .air_hockey_osc import AirHockeyOperationalSpaceController

# Global var for linking pybullet server to multiple ik controller instances if necessary
pybullet_server = None


def custom_controller_factory(name, params):
    """
    Generator for controllers

    Creates a Controller instance with the provided @name and relevant @params.

    Args:
        name (str): the name of the controller. Must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            OSC_POSITION, OSC_POSE, IK_POSE}
        params (dict): dict containing the relevant params to pass to the controller
        sim (MjSim): Mujoco sim reference to pass to the controller

    Returns:
        Controller: Controller instance

    Raises:
        ValueError: [unknown controller]
    """

    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )

    if name == "AIR_HOCKEY_OSC_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(ori="euler")
        params["control_ori"] = True
        return AirHockeyOperationalSpaceController(interpolator_pos=interpolator, interpolator_ori=ori_interpolator, **params)

    if name == "AIR_HOCKEY_OSC_POSITION":
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos
        params["control_ori"] = False
        return AirHockeyOperationalSpaceController(interpolator_pos=interpolator, **params)

    if name == "OSC_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(ori="euler")
        params["control_ori"] = True
        return OperationalSpaceController(interpolator_pos=interpolator, interpolator_ori=ori_interpolator, **params)

    if name == "OSC_POSITION":
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos
        params["control_ori"] = False
        return OperationalSpaceController(interpolator_pos=interpolator, **params)

    if name == "IK_POSE":
        raise NotImplementedError("Not supported currently, but robosuite controller_factory does.")

    if name == "JOINT_VELOCITY":
        return JointVelocityController(interpolator=interpolator, **params)

    if name == "JOINT_POSITION":
        return JointPositionController(interpolator=interpolator, **params)

    if name == "JOINT_TORQUE":
        return JointTorqueController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))