import rclpy
from sim_data_collection.perception_model.simulated_perception_node import Node

def main():
    rclpy.init()
    node = Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
