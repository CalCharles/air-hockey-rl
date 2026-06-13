"""Free-joint puck object — flat disk."""

from robosuite.models.objects import CylinderObject


class PuckObject(CylinderObject):
    """
    Air hockey puck: thin flat cylinder with low friction.
    """

    def __init__(self, name="puck", radius=0.03175, height=0.012, **kwargs):
        defaults = dict(
            size=[radius, height / 2],  # CylinderObject: [radius, half-height]
            rgba=[0.05, 0.05, 0.05, 1],
            density=300,
            friction=(0.01, 0.0001, 0.00001),
            solref=(0.002, 1),
            joints="default",
        )
        defaults.update(kwargs)
        super().__init__(name=name, **defaults)