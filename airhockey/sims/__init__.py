from .airhockey_box2d import AirHockeyBox2D

__all__ = ["AirHockeyBox2D", "AirHockeyReal"]


def __getattr__(name):
    if name == "AirHockeyReal":
        from .air_hockey_real import AirHockeyReal

        return AirHockeyReal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
