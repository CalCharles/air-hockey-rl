"""Free-joint puck object."""

from robosuite.models.objects import BallObject


class PuckObject(BallObject):
    """
    Air hockey puck: flat cylinder-like sphere with low friction.

    Uses robosuite BallObject (free joint by default) — no custom MJCF file required.
    """

    def __init__(self, name="puck", radius=0.025, **kwargs):
        defaults = dict(
            size=[radius],
            rgba=[0.05, 0.05, 0.05, 1],
            density=300,
            friction=(0.01, 0.0001, 0.00001),
            solref=(0.002, 1),
            joints="default",
        )
        defaults.update(kwargs)
        super().__init__(name=name, **defaults)
