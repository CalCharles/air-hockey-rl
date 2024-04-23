"""
Null Gripper (if we don't want to attach gripper to robot eef).
"""
from robosuite.models.grippers.gripper_model import GripperModel
import os

def custom_xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path

    Returns:
        str: Full (absolute) xml path
    """
    from airhockey import ASSETS_ROOT
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(ASSETS_ROOT, xml_path)
    return full_path

class RoundGripper(GripperModel):
    """
    Dummy Gripper class to represent no gripper

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(custom_xml_path_completion("grippers/round_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action
    
    @property
    def init_qpos(self):
        return None