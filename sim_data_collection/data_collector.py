import rclpy
from sim_data_collection.live_data_collector import LiveDataCollector as Node

logger = None

def callback(msg): 
    logger.info(str(type(msg)))

def main():
    global logger
    rclpy.init()
    instance = Node()
    instance.register_callback("drive_request", callback)
    instance.register_callback("car_request", callback)
    logger = instance.get_logger()
    rclpy.spin(instance)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
