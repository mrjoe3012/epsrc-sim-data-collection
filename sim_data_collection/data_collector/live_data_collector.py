from rclpy.node import Node as RosNode
import ugrdv_msgs.msg as ugrdv_msgs
import eufs_msgs.msg as eufs_msgs
import std_srvs.srv as std_srvs
from rclpy.qos import QoSProfile

class LiveDataCollector(RosNode):
    def __init__(self):
        """
        A ROS2 node which receives data live from the ROS 2 network
        and exposes it to consumers via callbacks.
        """
        super().__init__("data_collector")
        self.get_logger().info("data_collector starting...") 
        self._init_qos()
        self._init_message_info()
        self._init_params()
        self._init_subscriptions()
        self._init_services()
        self._has_stopped = False

    def _init_qos(self):
        """
        Sets up the qos profile which will be used for all
        subscriptions and publishers. This should be tuned
        to miss as few messages as possible.
        """
        self._qos_profile = 1

    def _init_message_info(self):
        """
        This initialises all of the topics and message
        types that will be subscribed to by this node.
        These entries correspond directly to the parameters
        that are read by this node.
        """
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
        """
        Use the entries initialized by _init_message_info to set up
        the ROS 2 parameters to be declared and read by this node.
        """
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
        """
        Use the entries initialized by _init_message_info to
        create the ROS 2 subscriptions and their callback hooks.
        """

        # this callback simply redirects to user-defined callbacks
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
        """
        Sets up this node's ROS 2 services.

        _stop_collection_srv: Used to stop the data collection cleanly.
        """
        self._stop_collection_srv = self.create_service(
            std_srvs.Trigger,
            "/sim_data_collection/stop_collection",
            self._stop_collection_srv_handler
        )

    def _fire_callbacks(self, id, msg):
        """
        Fire user-defined callbacks for a given message id.
        
        :param id: The string key for the specific message.
        :param msg: The ROS 2 message object which was received.
        """
        assert id in self._messages
        for fn in self._messages[id]["callbacks"]: fn(id, msg)

    def _stop_collection_srv_handler(self, request, response):
        """
        Callback method for the _stop_collection_srv service. Simply
        signals to the driver logic that data collection / ROS loop should
        cease.
        """
        if self._has_stopped == True:
            response.success = False
            response.message = "Data collection has already been stopped."
        else:
            self._has_stopped = True
            response.success = True
        self.get_logger().info(f"Received request to stop collection. Success={response.success}")
        return response

    def register_callback(self, id, fn):
        """
        Register a callback for a specific message.
        
        :param id: The message id to trigger the callback.
        :param fn: The callback callable.
        """
        assert id in self._messages or id == "all"
        if id == "all":
            for id, data in self._messages.items():
                data["callbacks"].append(fn)
        else:
            self._messages[id]["callbacks"].append(fn)

    def get_params(self):
        """
        Returns the ROS 2 parameters read in by the node.
        
        :returns: Dictionary of parameters and their applied values.
        """
        return self._param_names_and_values

    def has_stopped(self):
        """
        Returns true if something has told the data collection tool to stop
        running.
        
        :returns: Whether or not to stop collecting data.
        """
        return self._has_stopped
