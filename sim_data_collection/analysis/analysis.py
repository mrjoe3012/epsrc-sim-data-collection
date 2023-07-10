from sim_data_collection.analysis.dataset import Dataset
import sim_data_collection.utils as utils
import matplotlib.pyplot as plt
import math

msg_ids = [
    "ground_truth_state",
    "ground_truth_cones",
    "perception_cones",
    "vcu_status",
    "path_planning_path_velocity_request",
    "mission_path_velocity_request",
    "car_request",
    "drive_request",
]

class DatabaseIntegrityError(Exception):
    def __init__(self, name, tables):
        self._tables = tables
        self._name = name
    
    def __str__(self):
        return f"{self._name}:{self._tables} EMPTY TABLES!"

def integrity_check_db(database_path):
    """
    Performs an integrity check on a database.
    
    :param database_path: Path to the sqlite3 database.
    :raises DatabaseIntegrityError: If the database integrity
    check fails.
    """
    dataset = Dataset()
    dataset.open(database_path)
    bad_tables = []
    try:
        for msg_id in msg_ids:
            msgs = dataset.get_msgs(msg_id).fetchall()
            msgs.sort(key=lambda x: x[1])
            if len(msgs) == 0:
                bad_tables.append(msg_id)
    except Exception as e:
        raise e
    finally:
        dataset.close()
    if len(bad_tables) > 0:
        raise DatabaseIntegrityError(
            database_path,
            bad_tables
        )
