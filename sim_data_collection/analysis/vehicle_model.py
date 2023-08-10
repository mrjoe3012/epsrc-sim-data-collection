from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class VehicleModel:
    def __init__(self):
        """
        Abstract vehicle model class. Subclasses should override update_state() method
        to consume new data and the step() method to provide new estimated.
        """
        pass

    def __call__(self, delta_time: float = 1.0):
        return self.step(delta_time)

    @abstractmethod
    def update_state(self, state: Dict[str : Any]) -> None:
        """
        Update the vehicle model with new state variables.
        :param state: The state dictionary may contain one or more unique state variables.
        Possible state variables are:
        'steering_angle' : The vehicle's current steering angle : float
        'wheel_speeds' : The vehicle's current wheel speeds (fl, fr, rl, rr) : List[float, float, float, float]
        'steering_angle_request' : The requested steering angle : float
        'torque_request' : The requested torque : float
        """
        pass

    @abstractmethod
    def step(self, delta_time: float) -> Tuple[float, float, float]:
        """
        Return the pose deltas based on the current internal state.
        :param delta_time: The amount of time to step in seconds.
        :returns: A tuple containing delta x, y and heading in metres and radians.
        """
        pass