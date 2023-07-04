import sqlite3 as db
import ugrdv_msgs.msg as ugrdv_msgs
import eufs_msgs.msg as eufs_msgs
from rclpy.serialization import serialize_message
from rclpy.logging import get_logger
import sim_data_collection.utils as utils

class SQLiteSerializer:
    def __init__(self, verbose=False):
        """
        Responsible for serializing messages into a sqlite3 database.
        
        :param verbose: Whether or not to log extra information.
        """
        self._open = False
        self._verbose = verbose
        self._logger = get_logger("SQLliteSerializer")

    def open(self, path):
        """
        Attempt to open an sqlite3 database.
        
        :param path: The path to an sqlite3 database.
        """
        assert self._open == False
        if self._verbose == True:
            self._logger.info(f"Attempting to open database '{path}'")
        self._connection = db.connect(path)
        self._open = True

    def close(self):
        """
        Close any open sqlite3 database.
        """
        assert self._open == True
        self._connection.commit()
        self._connection.close()
        self._open = False

    def create_new_database(self):
        """
        Initialise new database tables after having opened an sqlite3
        database. This only works if the database which has been opened
        hasn't already got these tables initialised.
        """
        assert self._open == True
        cursor = self._connection.cursor()
        # hardcoded schema, a bit grim
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
        """
        CREATE TABLE path_planning_path_velocity_request(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            perception_cones VARCHAR(32),
            data BLOB,
            FOREIGN KEY (perception_cones) REFERENCES perception_cones(hash)
        );
        """,
        """
        CREATE TABLE mission_path_velocity_request(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            path_planning_path_velocity_request VARCHAR(32),
            data BLOB,
            FOREIGN KEY (path_planning_path_velocity_request) REFERENCES path_planning_path_velocity_request(hash)
        );
        """,
        """
        CREATE TABLE car_request(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            mission_path_velocity_request VARCHAR(32),
            data BLOB,
            FOREIGN KEY (mission_path_velocity_request) REFERENCES mission_path_velocity_request(hash)
        );
        """,
        """
        CREATE TABLE drive_request(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            vcu_status VARCHAR(32),
            car_request VARCHAR(32),
            data BLOB,
            FOREIGN KEY(car_request) REFERENCES car_request(hash),
            FOREIGN KEY(vcu_status) REFERENCES vcu_status(hash)
        );
        """,
        )
        for sql in queries: cursor.execute(sql)

    def serialize_message(self, id, msg):
        """
        Take a message described by a string key and serialize
        it into the database.
        
        :param id: The string key describing the database. Coupled with live_data_collector
        """
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
            if self._verbose == True:
                self._logger.error(f"Attempt to add a duplicate message. Exception: {str(e)}")

    def _serialize_drive_request(self, msg):
        query = """
        INSERT INTO drive_request(hash, timestamp, vcu_status, car_request, data)
        VALUES (?, ?, ?, ?, ?);
        """
        params = (
            msg.meta.hash,
            utils.rosTimestampToMillis(msg.header.stamp),
            msg.meta.consumed_messages[0],
            msg.meta.consumed_messages[1],
            serialize_message(msg)
        )
        self._connection.execute(query, params) 

    def _serialize_car_request(self, msg):
        query = """
        INSERT INTO car_request(hash, timestamp, mission_path_velocity_request, data)
        VALUES (?, ?, ?, ?);
        """
        params = (
            msg.meta.hash,
            utils.rosTimestampToMillis(msg.header.stamp),
            msg.meta.consumed_messages[0],
            serialize_message(msg)
        )
        self._connection.execute(query, params)

    def _serialize_path_planning_path_velocity_request(self, msg):
        query = """
        INSERT INTO path_planning_path_velocity_request(hash, timestamp, perception_cones, data)
        VALUES(?, ?, ?, ?);
        """
        params = (
            msg.meta.hash,
            utils.rosTimestampToMillis(msg.header.stamp),
            msg.meta.consumed_messages[0],
            serialize_message(msg)
        )
        self._connection.execute(query, params)

    def _serialize_mission_path_velocity_request(self, msg):
        query = """
        INSERT INTO mission_path_velocity_request(hash, timestamp, path_planning_path_velocity_request, data)
        VALUES(?, ?, ?, ?);
        """
        params = (
            msg.meta.hash,
            utils.rosTimestampToMillis(msg.header.stamp),
            msg.meta.consumed_messages[0],
            serialize_message(msg)
        )
        self._connection.execute(query, params)

    def _serialize_perception_cones(self, msg):
        self._serialize_basic_message(msg, "perception_cones", msg.meta.hash)

    def _serialize_basic_message(self, msg, table_name, hash):
        """
        Utility function for serializing a message with no dependencies.
        
        :param msg: The ROS 2 message object.
        :param table_name: The table to serialize to.
        :param hash: The message's hash code.
        """
        query = f"""
        INSERT INTO {table_name} (hash, timestamp, data)
        VALUES (?, ?, ?);
        """
        params = (hash,
                  utils.rosTimestampToMillis(msg.header.stamp),
                  serialize_message(msg))
        self._connection.execute(query, params)

    def _serialize_ground_truth_cones(self, msg):
       self._serialize_basic_message(msg, "ground_truth_cones", utils.getMessageHash(msg)) 

    def _serialize_ground_truth_state(self, msg):
        self._serialize_basic_message(msg, "ground_truth_state", utils.getMessageHash(msg))

    def _serialize_vcu_status(self, msg):
        self._serialize_basic_message(msg, "vcu_status", msg.meta.hash)

    def drop_unmet_dependencies(self):
        """
        Prune database entries by deleting messages whose
        dependencies aren't present in the database.
        """
        assert self._open == True
        queries = (
            (
            "path_planning_path_velocity_request",
            """
            DELETE FROM path_planning_path_velocity_request
            WHERE perception_cones NOT IN (SELECT hash FROM perception_cones);
            """
            ),
            (
            "mission_path_velocity_request",
            """
            DELETE FROM mission_path_velocity_request
            WHERE path_planning_path_velocity_request NOT IN (SELECT hash FROM path_planning_path_velocity_request);
            """
            ),
            (
            "car_request",
            """
            DELETE FROM car_request
            WHERE mission_path_velocity_request NOT IN (SELECT hash FROM mission_path_velocity_request);
            """
            ),
            (
            "drive_request",
            """
            DELETE FROM drive_request
            WHERE car_request NOT IN (SELECT hash FROM car_request)
            OR
            vcu_status NOT IN (SELECT hash from vcu_status);
            """
            ),
        )
        for (id,sql) in queries:
            c = self._connection.cursor()
            c.execute(sql)
            dropped = c.rowcount
            self._logger.warn(f"Dropped {dropped} of '{id}'")
            self._connection.commit()
