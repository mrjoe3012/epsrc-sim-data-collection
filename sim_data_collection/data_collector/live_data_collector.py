from rclpy.node import Node as RosNode
import ugrdv_msgs.msg as ugrdv_msgs
import eufs_msgs.msg as eufs_msgs
import std_srvs.srv as std_srvs
from rclpy.qos import QoSProfile

class LiveDataCollector(RosNode):
    def __init__(self):
        super().__init__("data_collector")
        self.get_logger().info("data_collector starting...") 
        self._init_qos()
        self._init_message_info()
        self._init_params()
        self._init_subscriptions()
        self._init_services()
        self._has_stopped = False

    def _init_qos(self):
        self._qos_profile = 1#QoSProfile()

    def _init_message_info(self):
        def _message_description(topic, type):
            d = {
                "topic" : topic,
                "type" : type,
                "callbacks" : [],
                "subscription" : None
            }
            return d
        self._messages = {
            "drive_request" : _message_description("/ugrdv/cmd", ugrdv_msgs.DriveRequest),
            "car_request" : _message_description("/ugrdv/car_request", ugrdv_msgs.CarRequest),
            "path_planning_path_velocity_request" : _message_description("/ugrdv/path_velocity_request", ugrdv_msgs.PathVelocityRequest),
            "mission_path_velocity_request" : _message_description("/ugrdv/mission_path_velocity_request", ugrdv_msgs.PathVelocityRequest),
            "perception_cones" : _message_description("/ugrdv/perception/map", ugrdv_msgs.Cone3dArray),
            "ground_truth_cones" : _message_description("/ground_truth/cones", eufs_msgs.ConeArrayWithCovariance),
            "ground_truth_state" : _message_description("/ground_truth/state", eufs_msgs.CarState),
            "vcu_status" : _message_description("/ugrdv/vcu_status", ugrdv_msgs.VCUStatus)
        }

    def _init_params(self):
        # use the topic names in self._messages as a default
        self._param_names_and_values = {
            f"{id}_topic" : data["topic"] for (id,data) in self._messages.items()
        } | {
            "database" : "database.db3"
        }

        # declare through ros
        for (name, default) in self._param_names_and_values.items():
            self.declare_parameter(name, default)
        # get values (or default if none provided)
        for name in self._param_names_and_values.keys():
            self._param_names_and_values[name] = self.get_parameter(name).value 

    def _init_subscriptions(self):
        def _make_callback(id):
            return lambda msg : self._fire_callbacks(id, msg)
        for (id, data) in self._messages.items():
            data["subscription"] = self.create_subscription(
                data["type"],
                data["topic"],
                _make_callback(id) ,
                self._qos_profile
            )

    def _init_services(self):
        self._stop_collection_srv = self.create_service(
            std_srvs.Trigger,
            "/sim_data_collection/stop_collection",
            self._stop_collection_srv_handler
        )

    def _fire_callbacks(self, id, msg):
        assert id in self._messages
        for fn in self._messages[id]["callbacks"]: fn(id, msg)

    def _stop_collection_srv_handler(self, request, response):
        if self._has_stopped == True:
            response.success = False
            response.message = "Data collection has already been stopped."
        else:
            self._has_stopped = True
            response.success = True
        self.get_logger().info(f"Received request to stop collection. Success={response.success}")
        return response

    def register_callback(self, id, fn):
        assert id in self._messages or id == "all"
        if id == "all":
            for id, data in self._messages.items():
                data["callbacks"].append(fn)
        else:
            self._messages[id]["callbacks"].append(fn)

    def get_params(self):
        return self._param_names_and_values

    def has_stopped(self):
        return self._has_stopped
