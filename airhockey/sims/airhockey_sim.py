from abc import ABC, abstractmethod


class AirHockeySim(ABC):
    @abstractmethod
    def get_transition(self, action):
        pass

    @abstractmethod
    def spawn_puck(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    @abstractmethod
    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass

    @abstractmethod
    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pass
    
    @abstractmethod
    def instantiate_objects(self):
        pass

    @abstractmethod
    def reset(self, seed):
        pass