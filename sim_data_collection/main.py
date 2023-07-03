import rclpy, os
from sim_data_collection.live_data_collector import LiveDataCollector as Node
from sim_data_collection.sqlite_serializer import SQLiteSerializer as Serializer
from ament_index_python import get_package_share_directory

package_name = "sim_data_collection"

def main():
    try:
        rclpy.init()
        node = Node()
        serializer = Serializer(verbose=True)
        db_path = os.path.join(
            get_package_share_directory(package_name),
            node.get_params()["database"]
        )
        serializer.open(db_path)
        serializer.create_new_database()
        node.register_callback("all", serializer.serialize_message)
        while not node.has_stopped(): rclpy.spin_once(node)
    finally:
        serializer.drop_unmet_dependencies()
        serializer.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
