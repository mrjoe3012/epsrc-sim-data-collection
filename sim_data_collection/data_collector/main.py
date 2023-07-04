import rclpy, os
from sim_data_collection.data_collector.live_data_collector import LiveDataCollector as Node
from sim_data_collection.data_collector.sqlite_serializer import SQLiteSerializer as Serializer
from ament_index_python import get_package_share_directory

package_name = "sim_data_collection"

def main():
    try:
        rclpy.init()
        node = Node()
        # initialize the serializer object, open up a database
        serializer = Serializer(verbose=True)
        db_path = os.path.join(
            get_package_share_directory(package_name),
            node.get_params()["database"]
        )
        serializer.open(db_path)
        serializer.create_new_database()
        # respond to all messages with this callback
        node.register_callback("all", serializer.serialize_message)
        # spin until the stop service is triggered
        while not node.has_stopped(): rclpy.spin_once(node)
    finally:
        # trim the database down by deleting messages whose dependencies
        # aren't present
        serializer.drop_unmet_dependencies()
        serializer.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
