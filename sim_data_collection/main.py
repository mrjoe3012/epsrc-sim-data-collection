import rclpy
from sim_data_collection.live_data_collector import LiveDataCollector as Node

def main():
    rclpy.init()
    instance = Node()
    rclpy.spin(instance)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
