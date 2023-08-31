from rclpy.node import Node as ROSNode
from eufs_msgs.msg import ConeArrayWithCovariance, CarState, ConeWithCovariance
from ugrdv_msgs.msg import Cone3dArray, Cone3d
from scipy.spatial.transform import Rotation
from typing import List, Tuple
from ament_index_python import get_package_share_directory
from pathlib import Path
import math
import numpy as np
import sim_data_collection.perception_model.model as perception_model

class Node(ROSNode):
    def __init__(self):
        super().__init__("simulated_perception_node")
        # parameters
        self.declare_parameter("gt-cones-topic", "/ground_truth/track")
        self.declare_parameter("gt-car-state-topic", "/ground_truth/state")
        self.declare_parameter("simulated-cones-topic", "/ugrdv/perception/epsrc_cones")
        self.declare_parameter("sensor-fov", 110.0)
        self.declare_parameter("sensor-range", 12.0)
        self.gt_cones_topic = self.get_parameter("gt-cones-topic").value
        self.gt_car_state_topic = self.get_parameter("gt-car-state-topic").value
        self.perception_cones_topic = self.get_parameter("simulated-cones-topic").value
        self.sensor_fov = math.radians(self.get_parameter("sensor-fov").value)
        self.sensor_range = self.get_parameter("sensor-range").value
        # set up publishers and subscriptions
        self.subs = {
            "ground_truth_cones" : self.create_subscription(
                ConeArrayWithCovariance,
                self.gt_cones_topic,
                self.on_gt_cones,
                1
            ),
            "ground_truth_state" : self.create_subscription(
                CarState,
                self.gt_car_state_topic,
                self.on_gt_car_state,
                1
            ),
        } 
        self.pubs = {
            "simulated_perception" : self.create_publisher(
                ConeArrayWithCovariance,
                self.perception_cones_topic,
                1
            )
        }
        self.last_car_state = None
        self.last_gt_cones = Cone3dArray()
        self.declare_parameter("perception-model", "realistic")
        model_name = self.get_parameter("perception-model").value
        self.perception_model = self.load_model(model_name)

    def load_model(self, name):
        allowed_names = ["realistic", "poor", "good"]
        assert name in allowed_names
        filename = Path(get_package_share_directory("sim_data_collection")) / 'models' / (name + '.json')
        probs = perception_model.Probabilities.from_json(filename)
        return perception_model.PerceptionModel(probs)
    
    def on_gt_cones(self, msg):
        self.last_gt_cones = self.convert_eufs_cones(msg)
        if self.last_car_state is None: return
        self.publish()

    def on_gt_car_state(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        yaw = self.get_car_heading(msg)
        if self.last_car_state is None: self.last_car_state = {}
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

    def publish(self):
        fov = self.sensor_fov
        distance = self.sensor_range
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