import sqlite3 as db
import ugrdv_msgs.msg as ugrdv_msgs
import eufs_msgs.msg as eufs_msgs
from rclpy.serialization import serialize_message
from rclpy.logging import get_logger
import sim_data_collection.utils as utils

class SQLiteSerializer:
    def __init__(self):
        self._open = False
        self._logger = get_logger("SQLliteSerializer")

    def open(self, path):
        assert self._open == False
        self._connection = db.connect(path, isolation_level=None)
        self._open = True

    def close(self):
        assert self._open == True
        self._connection.commit()
        self._connection.close()
        self._open = False

    def create_new_database(self):
        assert self._open == True
        cursor = self._connection.cursor()
        queries = ("""
        CREATE TABLE vcu_status(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        """
        CREATE TABLE perception_cones(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        """
        CREATE TABLE ground_truth_cones(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        """
        CREATE TABLE ground_truth_state(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        )
        for sql in queries: cursor.execute(sql)

    def serialize_message(self, id, msg):
        serialization_fns = {
            "drive_request" : self._serialize_drive_request,
            "car_request" : self._serialize_car_request,
            "vcu_status" : self._serialize_vcu_status,
            "path_planning_path_velocity_request" : self._serialize_path_planning_path_velocity_request,
            "mission_path_velocity_request" : self._serialize_mission_path_velocity_request,
            "perception_cones" : self._serialize_perception_cones,
            "ground_truth_cones" : self._serialize_ground_truth_cones,
            "ground_truth_state" : self._serialize_ground_truth_state,
        } 
        assert id in serialization_fns
        try:
            serialization_fns[id](msg)
        except db.IntegrityError as e:
            self._logger.error(f"Attempt to add a duplicate message. Exception: {str(e)}")

    def _serialize_drive_request(self, msg):
        pass

    def _serialize_car_request(self, msg):
        pass

    def _serialize_path_planning_path_velocity_request(self, msg):
        pass

    def _serialize_mission_path_velocity_request(self, msg):
        pass

    def _serialize_perception_cones(self, msg):
        self._serialize_basic_message(msg, "perception_cones", msg.meta.hash)

    def _serialize_basic_message(self, msg, table_name, hash):
        query = f"""
        INSERT INTO {table_name} (hash, timestamp, data)
        VALUES (?, ?, ?);
        """
        params = (hash,
                  utils.rosTimestampToMillis(msg.header.stamp),
                  serialize_message(msg))
        self._connection.execute(query, params)
        self._connection.commit()

    def _serialize_ground_truth_cones(self, msg):
       self._serialize_basic_message(msg, "ground_truth_cones", utils.getMessageHash(msg)) 

    def _serialize_ground_truth_state(self, msg):
        self._serialize_basic_message(msg, "ground_truth_state", utils.getMessageHash(msg))

    def _serialize_vcu_status(self, msg):
        self._serialize_basic_message(msg, "vcu_status", msg.meta.hash)

    def drop_unmet_dependencies(self):
        pass