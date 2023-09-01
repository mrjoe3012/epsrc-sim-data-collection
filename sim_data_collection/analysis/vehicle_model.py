from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from pathlib import Path
import epsrc_vehicle_model.lib as nn_lib
import numpy as np
import torch

class VehicleModel:
    def __init__(self, name: str):
        """
        Abstract vehicle model class. Subclasses should override update_state() method
        to consume new data and the step() method to provide new estimated.
        :param name: A name for the vehicle model.
        """
        self._name = name

    def __call__(self, delta_time: float = 1.0):
        return self.step(delta_time)

    def get_name(self) -> str:
        return self._name

    @abstractmethod
    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update the vehicle model with new state variables.
        :param state: The state dictionary may contain one or more unique state variables.
        Possible state variables are:
        'steering_angle' : The vehicle's current steering angle : float
        'wheel_speeds' : The vehicle's current wheel speeds (fl, fr, rl, rr) : List[float, float, float, float]
        'steering_angle_request' : The requested steering angle : float
        'acceleration_request' : The requested acceleration : float
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

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the vehicle model.
        """
        pass

class KinematicBicycle(VehicleModel):
    def __init__(self):
        super().__init__("Kinematic Bicycle")
        self.min_steer = np.deg2rad(-21.0)
        self.max_steer = np.deg2rad(21.0)
        self.friction = 0.2
        self.mass = 200.0
        self.wheel_radius = 0.26
        self.wheelbase = 1.53
        self.state_size = 4
        self.dtype = torch.float32
        # state format;
        # 0: steering_angle
        # 1: velocity
        # 2: steering_request
        # 3: acceleration_request
        self.state = torch.zeros(
            (self.state_size,),
            dtype=self.dtype
        )

    def _calculate_velocity(self, wheel_speeds: List[float]) -> float:
        wheel_circumference = 2 * self.wheel_radius * np.pi
        avg_wheel_rpm = sum(wheel_speeds) / len(wheel_speeds)
        velocity = wheel_circumference * avg_wheel_rpm / 60.0
        return velocity

    def update_state(self, state: Dict[str, Any]) -> None:
        self.state[0] = max(self.min_steer, min(state.get("steering_angle", self.state[0]), self.max_steer))
        self.state[2] = max(self.min_steer, min(state.get("steering_angle_request", self.state[2]), self.max_steer))
        self.state[3] = state.get("acceleration_request", self.state[3])
        if "wheel_speeds" in state:
            self.state[1] = max(0.0, self._calculate_velocity(state["wheel_speeds"]))

    def step(self, delta_time: float) -> Tuple[float, float, float]:
        # update internal state
        state_derivatives = torch.tensor([
            (self.state[2] - self.state[0]) / delta_time,
            self.state[3] - self.state[1] * self.friction,
            0.0,
            0.0
        ], dtype=self.dtype)
        self.state += state_derivatives * delta_time
        self.state[0] = max(self.min_steer, min(self.state[0], self.max_steer))
        self.state[1] = max(0.0, self.state[1])
        # calculate deltas
        dx = delta_time * self.state[1]
        dy = 0.0
        dtheta = delta_time * self.state[1] * np.tan(self.state[0]) / self.wheelbase
        return dx, dy, dtheta

    def reset(self) -> None:
        self.state = torch.zeros(
            (self.state_size,),
            dtype=self.dtype
        )

class NNVehicleModel(VehicleModel):
    def __init__(self, model : str | Path):
        super().__init__("Neural Network")
        model = Path(model)
        self.model = torch.load(model)
        self.model.eval()
        self.x = torch.zeros((self.model._input_constraints.SIZE,), dtype=torch.float32, device="cuda:0")
        self.min_steer = np.deg2rad(-21.0)
        self.max_steer = np.deg2rad(21.0)

    def update_state(self, state: Dict[str, Any]) -> None:
        self.x[0] = state.get("steering_angle", self.x[0])
        self.x[1:5] = torch.tensor(state.get("wheel_speeds", self.x[1:5]), dtype=torch.float32)
        self.x[5] = state.get("steering_angle_request", self.x[5])
        self.x[6] = state.get("acceleration_reqeust", self.x[6])

    def step(self, delta_time: float) -> Tuple[float, float, float]:
        y = self.model(self.x) * delta_time
        dx = y[0].item()
        dy = y[1].item()
        dtheta = y[2].item()
        self.x[:5] += y[3:]
        self.x[:5] = torch.max(torch.zeros((5,), dtype=torch.float32, device="cuda:0"), torch.min(torch.full((5,), 100.0, dtype=torch.float32, device="cuda:0"), self.x[:5]))
        self.x[0] = max(self.min_steer, min(self.x[0], self.max_steer))
        return dx, dy, dtheta

    def reset(self) -> None: 
        self.x = torch.zeros((self.model._input_constraints.SIZE,), dtype=torch.float32, device="cuda:0")