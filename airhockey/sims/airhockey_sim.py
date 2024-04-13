from abc import ABC, abstractmethod


class AirHockeySim(ABC):
    @abstractmethod
    def get_transition(self, action):
        pass

    @abstractmethod
    def spawn_puck(self, x, y, x_vel, y_vel, affected_by_gravity=False):
        pass

    @abstractmethod
    def spawn_paddle(selfx, y, x_vel, y_vel, affected_by_gravity=False):
        pass

    @abstractmethod
    def spawn_block(selfx, y, x_vel, y_vel, affected_by_gravity=False):
        pass

    @abstractmethod
    def reset(self):
        pass