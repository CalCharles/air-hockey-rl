from robosuite.models.robots.manipulators import UR5e

# Pretty much a dummy class, working around robosuite not supporting custom controllers
class AirHockeyUR5e(UR5e):
    @property # Arbitrary function here. Just wanted to create a subclass
    def default_gripper(self):
        return "Robotiq85Gripper"