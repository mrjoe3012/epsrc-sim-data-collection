import rclpy
from sim_data_collection.live_data_collector import LiveDataCollector as Node
from sim_data_collection.sqlite_serializer import SQLiteSerializer as Serializer

def main():
    try:
        rclpy.init()
        node = Node()
        serializer = Serializer(verbose=True)
        serializer.open("database.db3")
        serializer.create_new_database()
        node.register_callback("all", serializer.serialize_message)
        rclpy.spin(node)
        rclpy.shutdown()
    finally:
        serializer.drop_unmet_dependencies()
        serializer.close()

if __name__ == '__main__':
    main()
