from rclpy.node import Node as ROSNode
from eufs_msgs.msg import ConeArrayWithCovariance, CarState, ConeWithCovariance
from ugrdv_msgs.msg import Cone3dArray, Cone3d
from scipy.spatial.transform import Rotation
from typing import List, Tuple
import math
import numpy as np
import sim_data_collection.perception_model.model as perception_model
## TODO: parameterise the node

class Node(ROSNode):
    def __init__(self):
        super().__init__("simulated_perception_node")
        # set up publishers and subscriptions
        self.subs = {
            "ground_truth_cones" : self.create_subscription(
                ConeArrayWithCovariance,
                "/ground_truth/track",
                self.on_gt_cones,
                1
            ),
            "ground_truth_state" : self.create_subscription(
                CarState,
                "/ground_truth/state",
                self.on_gt_car_state,
                1
            ),
        } 
        self.pubs = {
            "simulated_perception" : self.create_publisher(
                ConeArrayWithCovariance,
                "/ugrdv/perception/epsrc_cones",
                1
            )
        }
        update_hz = 10
        self.timer = self.create_timer(
            1 / update_hz,
            self.timer_callback
        )
        self.last_car_state = {
            "x" : 0.0,
            "y" : 0.0,
            "yaw" : 0.0
        }
        self.last_gt_cones = Cone3dArray()
        self.declare_parameter("model", "realistic")
        model_name = self.get_parameter("model").value
        self.perception_model = self.load_model(model_name)

    def load_model(self, name):
        allowed_names = ["realistic", "poor", "good"]
        assert name in allowed_names
        if name == "realistic":
            probs = [
                perception_model.Probabilities(
                    detection=0.2,
                    colour_correct=1.0,
                    colour_incorrect=0.0,
                    false_positive=0.13,
                    var_x=0.5,
                    var_y=0.8,
                    distance=1.0
                ),
                perception_model.Probabilities(
                    detection=0.008,
                    colour_correct=0.98,
                    colour_incorrect=0.0,
                    false_positive=0.13,
                    var_x=0.5,
                    var_y=0.8,
                    distance=1.5
                ),
                perception_model.Probabilities(
                    detection=0.333,
                    colour_correct=0.98,
                    colour_incorrect=0.0,
                    false_positive=0.07,
                    var_x=0.05,
                    var_y=0.025,
                    distance=3.0
                ),
                perception_model.Probabilities(
                    detection=0.52,
                    colour_correct=0.98,
                    colour_incorrect=0.005,
                    false_positive=0.01,
                    var_x=0.05,
                    var_y=0.025,
                    distance=4.5
                ),
                perception_model.Probabilities(
                    detection=0.94,
                    colour_correct=0.98,
                    colour_incorrect=0.005,
                    false_positive=0.02,
                    var_x=0.07,
                    var_y=0.025,
                    distance=6.0
                ),
                perception_model.Probabilities(
                    detection=0.955,
                    colour_correct=0.98,
                    colour_incorrect=0.005,
                    false_positive=0.04,
                    var_x=0.01,
                    var_y=0.025,
                    distance=7.5
                ),
                perception_model.Probabilities(
                    detection=0.93,
                    colour_correct=0.98,
                    colour_incorrect=0.01,
                    false_positive=0.024,
                    var_x=0.125,
                    var_y=0.025,
                    distance=9.0
                ),
                perception_model.Probabilities(
                    detection=0.88,
                    colour_correct=0.98,
                    colour_incorrect=0.01,
                    false_positive=0.03,
                    var_x=0.15,
                    var_y=0.05,
                    distance=10.5
                ),
                perception_model.Probabilities(
                    detection=0.82,
                    colour_correct=0.98,
                    colour_incorrect=0.01,
                    false_positive=0.025,
                    var_x=0.2,
                    var_y=0.05,
                    distance=12.0
                ),
                perception_model.Probabilities(
                    detection=0.80,
                    colour_correct=0.98,
                    colour_incorrect=0.1,
                    false_positive=0.025,
                    var_x=0.2,
                    var_y=0.05,
                    distance=13.5
                ),
                perception_model.Probabilities(
                    detection=0.74,
                    colour_correct=0.96,
                    colour_incorrect=0.02,
                    false_positive=0.04,
                    var_x=0.17,
                    var_y=0.05,
                    distance=15.0
                ),
                perception_model.Probabilities(
                    detection=0.45,
                    colour_correct=0.95,
                    colour_incorrect=0.02,
                    false_positive=0.214,
                    var_x=0.17,
                    var_y=0.05,
                    distance=16.5
                ),
            ]
        elif name == "poor":
            probs = [
                perception_model.Probabilities(
                    detection=0.1,
                    colour_correct=0.95,
                    colour_incorrect=0.01,
                    false_positive=0.26,
                    var_x=1.0,
                    var_y=1.6,
                    distance=1.0
                ),
                perception_model.Probabilities(
                    detection=0.004,
                    colour_correct=0.95,
                    colour_incorrect=0.01,
                    false_positive=0.26,
                    var_x=1.0,
                    var_y=1.6,
                    distance=1.5
                ),
                perception_model.Probabilities(
                    detection=0.15,
                    colour_correct=0.95,
                    colour_incorrect=0.01,
                    false_positive=0.14,
                    var_x=0.1,
                    var_y=0.05,
                    distance=3.0
                ),
                perception_model.Probabilities(
                    detection=0.25,
                    colour_correct=0.95,
                    colour_incorrect=0.01,
                    false_positive=0.02,
                    var_x=0.1,
                    var_y=0.05,
                    distance=4.5
                ),
                perception_model.Probabilities(
                    detection=0.45,
                    colour_correct=0.95,
                    colour_incorrect=0.01,
                    false_positive=0.04,
                    var_x=0.14,
                    var_y=0.05,
                    distance=6.0
                ),
                perception_model.Probabilities(
                    detection=0.45,
                    colour_correct=0.95,
                    colour_incorrect=0.01,
                    false_positive=0.08,
                    var_x=0.02,
                    var_y=0.05,
                    distance=7.5
                ),
                perception_model.Probabilities(
                    detection=0.45,
                    colour_correct=0.95,
                    colour_incorrect=0.02,
                    false_positive=0.048,
                    var_x=0.25,
                    var_y=0.05,
                    distance=9.0
                ),
                perception_model.Probabilities(
                    detection=0.4,
                    colour_correct=0.95,
                    colour_incorrect=0.02,
                    false_positive=0.06,
                    var_x=0.3,
                    var_y=0.1,
                    distance=10.5
                ),
                perception_model.Probabilities(
                    detection=0.4,
                    colour_correct=0.98,
                    colour_incorrect=0.02,
                    false_positive=0.05,
                    var_x=0.4,
                    var_y=0.1,
                    distance=12.0
                ),
                perception_model.Probabilities(
                    detection=0.4,
                    colour_correct=0.95,
                    colour_incorrect=0.2,
                    false_positive=0.05,
                    var_x=0.4,
                    var_y=0.1,
                    distance=13.5
                ),
                perception_model.Probabilities(
                    detection=0.74,
                    colour_correct=0.93,
                    colour_incorrect=0.03,
                    false_positive=0.08,
                    var_x=0.34,
                    var_y=0.1,
                    distance=15.0
                ),
                perception_model.Probabilities(
                    detection=0.22,
                    colour_correct=0.93,
                    colour_incorrect=0.03,
                    false_positive=0.42,
                    var_x=0.34,
                    var_y=0.1,
                    distance=16.5
                ),
            ]
        elif name == "good":
            probs = [
                perception_model.Probabilities(
                    detection=0.4,
                    colour_correct=1.0,
                    colour_incorrect=0.0,
                    false_positive=0.05,
                    var_x=0.25,
                    var_y=0.4,
                    distance=1.0
                ),
                perception_model.Probabilities(
                    detection=0.016,
                    colour_correct=0.99,
                    colour_incorrect=0.0,
                    false_positive=0.06,
                    var_x=0.25,
                    var_y=0.4,
                    distance=1.5
                ),
                perception_model.Probabilities(
                    detection=0.66,
                    colour_correct=0.99,
                    colour_incorrect=0.0,
                    false_positive=0.03,
                    var_x=0.025,
                    var_y=0.0125,
                    distance=3.0
                ),
                perception_model.Probabilities(
                    detection=0.90,
                    colour_correct=0.99,
                    colour_incorrect=0.0025,
                    false_positive=0.005,
                    var_x=0.025,
                    var_y=0.0125,
                    distance=4.5
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.0025,
                    false_positive=0.01,
                    var_x=0.03,
                    var_y=0.0125,
                    distance=6.0
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.0025,
                    false_positive=0.02,
                    var_x=0.005,
                    var_y=0.0125,
                    distance=7.5
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.005,
                    false_positive=0.012,
                    var_x=0.06,
                    var_y=0.0125,
                    distance=9.0
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.005,
                    false_positive=0.015,
                    var_x=0.07,
                    var_y=0.025,
                    distance=10.5
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.005,
                    false_positive=0.0125,
                    var_x=0.1,
                    var_y=0.025,
                    distance=12.0
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.01,
                    false_positive=0.0125,
                    var_x=0.1,
                    var_y=0.025,
                    distance=13.5
                ),
                perception_model.Probabilities(
                    detection=0.99,
                    colour_correct=0.99,
                    colour_incorrect=0.01,
                    false_positive=0.02,
                    var_x=0.09,
                    var_y=0.025,
                    distance=15.0
                ),
                perception_model.Probabilities(
                    detection=0.88,
                    colour_correct=0.99,
                    colour_incorrect=0.01,
                    false_positive=0.01,
                    var_x=0.09,
                    var_y=0.025,
                    distance=16.5
                ),
            ]
        return perception_model.PerceptionModel(probs)
    
    def on_gt_cones(self, msg):
        self.last_gt_cones = self.convert_eufs_cones(msg)

    def on_gt_car_state(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        yaw = self.get_car_heading(msg)
        self.last_car_state["x"] = x
        self.last_car_state["y"] = y
        self.last_car_state["yaw"] = yaw

    def get_car_heading(self, car_state):
        quat = Rotation.from_quat((
            car_state.pose.pose.orientation.x,
            car_state.pose.pose.orientation.y,
            car_state.pose.pose.orientation.z,
            car_state.pose.pose.orientation.w,
        ))
        euler = quat.as_euler("XYZ")
        yaw = euler[2]
        return yaw

    def convert_eufs_cones(self, msg):
        def do_array(arr, colour):
            new_arr = []
            for eufs in arr:
                ugr = Cone3d()
                ugr.position.x = eufs.point.x
                ugr.position.y = eufs.point.y
                ugr.colour = colour
                new_arr.append(ugr)
            return new_arr

        result = Cone3dArray()
        result.header = msg.header
        result.cones = \
            do_array(msg.blue_cones, Cone3d.BLUE) + \
            do_array(msg.yellow_cones, Cone3d.YELLOW) + \
            do_array(msg.orange_cones, Cone3d.ORANGE) + \
            do_array(msg.big_orange_cones, Cone3d.LARGEORANGE) + \
            do_array(msg.unknown_color_cones ,Cone3d.UNKNOWN)
        return result

    def convert_ugr_cones(self, cones):
        new = ConeArrayWithCovariance()
        new.header = cones.header
        for cone in cones.cones:
            eufs = ConeWithCovariance()
            eufs.point.x = cone.position.x
            eufs.point.y = cone.position.y
            if cone.colour == Cone3d.BLUE:
                new.blue_cones.append(eufs)
            elif cone.colour == Cone3d.YELLOW:
                new.yellow_cones.append(eufs)
            elif cone.colour == Cone3d.ORANGE:
                new.orange_cones.append(eufs)
            elif cone.colour == Cone3d.LARGEORANGE:
                new.big_orange_cones.append(eufs)
            else:
                new.unknown_color_cones.append(eufs)
        return new

    def timer_callback(self):
        self.publish()

    def publish(self):
        fov = math.radians(110.0)
        distance = 12.0
        cropped_cones = self.crop_to_fov(
            self.last_gt_cones,
            self.last_car_state,
            fov,
            distance
        )
        fov = np.radians(110.0)
        simulated_perception = self.perception_model.process(cropped_cones.cones, fov).to_msg()
        simulated_perception = self.convert_ugr_cones(simulated_perception)
        self.pubs["simulated_perception"].publish(simulated_perception)

    def crop_to_fov(self, cones: List[Cone3d],
                    car_state: dict, fov: float,
                    max_distance: float) -> List[Cone3d]:
        conearray = Cone3dArray()
        result = conearray.cones
        x, y, yaw = car_state["x"], car_state["y"], car_state["yaw"]
        fmin, fmax = -fov/2, fov/2
        rot = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        carpos = np.array([
            [x],
            [y]
        ])
        for cone in cones.cones:
            # put into local frame
            conepos = np.array([
                [cone.position.x],
                [cone.position.y]
            ])
            conepos = rot @ (conepos - carpos)
            # check angle
            theta = math.atan2(conepos[1], conepos[0])
            # check distance
            dist = np.linalg.norm(conepos, ord=1) 
            # add to result if good
            if theta >= fmin and theta <= fmax and dist <= max_distance:
                cone_local = Cone3d()
                cone_local.colour = cone.colour
                cone_local.position.x = conepos[0,0]
                cone_local.position.y = conepos[1,0]
                result.append(cone_local)

        return conearray