from sim_data_collection.analysis.dataset import Dataset
import sim_data_collection.utils as utils
import matplotlib.pyplot as plt
import math
import numpy as np
from pathlib import Path
from ament_index_python import get_package_share_directory

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

def integrity_check_db(database_path: str):
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

class Track:
    def __init__(self, blue, yellow):
        self.blue_cones = blue
        self.yellow_cones = yellow

    @staticmethod
    def read_csv(track_file: str):
        blue, yellow = [], []
        with open(track_file, "r") as f:
            for line in f:
                cols = line.strip().split(",")
                tag = cols[0]
                arr = None
                if tag == "blue":
                    arr = blue
                elif tag == "yellow":
                    arr = yellow
                if arr:
                    cone = (
                        float(tag[1]),
                        float(tag[2])
                    )
                    arr.append(cone)
        return Track(blue, yellow)

from matplotlib.pyplot import plt

class Line:
    def __init__(self, sx=0.0, sy=0.0, ex=0.0, ey=0.0):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey

def make_line_from_cones(cone1, cone2):
    return Line(
        cone1[0], cone1[1],
        cone2[0], cone2[1]
    )

def track_from_db_path(db_path):
    """
    Loads a track with the same UUID as
    that in the database path.
    
    :param db_path: Path to an sqlite3 database
    with the .db3 extension.
    :returns: The loaded track.
    """
    p = Path(db_path)
    assert p.is_file()
    filename = p.name
    filename.replace(".db3", ".csv")
    eufs_tracks_share = get_package_share_directory(
        "eufs_tracks"
    )
    track_path = Path(eufs_tracks_share) / "csv" / filename
    return Track.read_csv(track_path)

def intersection_check(dataset: Dataset, track: Track):
    cones = track.blue_cones + track.yellow_cones
    # calculate lines using each pair of consecutive cones
    cone_lines = [
        make_line_from_cones(cones[i - 1], cones[i]) \
        for i in range(1, len(cones))
    ] 
    # now for each car pose, check if it intersects with any of
    # these lines
    car_poses = [
        dataset.get_msgs("ground_truth_state").fetchall()
    ]
    print(car_poses)
    

