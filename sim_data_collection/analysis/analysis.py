from sim_data_collection.analysis.dataset import Dataset
import sim_data_collection.utils as utils
import matplotlib.pyplot as plt
import math
import numpy as np
from pathlib import Path
from ament_index_python import get_package_share_directory
from eufs_msgs.msg import CarState
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
import copy

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
    def __init__(self, blue, yellow,
                 car_start = None, direction = 1):
        """
        Represents a racetrack.
        
        :param blue: list of blue cone positions
        :param yellow: list of yellow cone positions
        :param car_start: Car starting pose (x, y, theta)
        """
        self.nyellow, self.nblue = Line(), Line()
        self.blue_cones = blue
        self.yellow_cones = yellow
        if car_start is None: car_start = (0.0, 0.0, 0.0)
        self.car_start = car_start
        self.direction = direction
        self.blue_cone_lines, self.yellow_cone_lines = self._get_cone_lines()
        (self.first_blue_line, _, _), _ = Line.project_to_nearest(
            self.car_start[0], self.car_start[1],
            self.blue_cone_lines
        )
        (self.first_yellow_line, _, _), _ = Line.project_to_nearest(
            self.car_start[0], self.car_start[1],
            self.yellow_cone_lines
        )
        assert self.first_blue_line is not None
        assert self.first_yellow_line is not None
        self.path_direction = self._get_path_direction()
        print(f"Track is {'clockwise' if self.direction == 1 else 'anticlockwise'}")
        print(f"Path is {'clockwise' if self.path_direction == 1 else 'anticlockwise'}")

    @staticmethod
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
        filename = filename.replace(".db3", ".csv")
        eufs_tracks_share = get_package_share_directory(
            "eufs_tracks"
        )
        track_path = Path(eufs_tracks_share) / "csv" / filename
        return Track.read_csv(track_path)

    @staticmethod
    def read_csv(track_file: str):
        """
        Initialises a track from a csv file.
        """
        blue, yellow = [], []
        start_x, start_y, start_heading = 0.0, 0.0, 0.0
        with open(track_file, "r") as f:
            for line in f:
                cols = line.strip().split(",")
                tag = cols[0]
                arr = None
                if tag == "blue":
                    arr = blue
                elif tag == "yellow":
                    arr = yellow
                elif tag == "car_start":
                    start_x = float(cols[1])
                    start_y = float(cols[2])
                    start_heading = float(cols[3])
                if arr is not None:
                    cone = (
                        float(cols[1]),
                        float(cols[2])
                    )
                    arr.append(cone)
        return Track(
            blue, yellow,
            car_start = (start_x, start_y, start_heading),
            direction = 1 if str.replace(Path(track_file).name, ".csv", "")[-1] == "r" else -1
        )

    def transform_car_pose(self, car_pose):
        """
        Takes a car pose message, relative to the
        starting pose and transforms it into the global
        frame.
        
        :param car_pose: The car pose to transform.
        :returns: Car pose in global frame.
        """
        start_heading = self.car_start[2]
        rotation_matrix = np.matrix([
            [math.cos(-start_heading), math.sin(-start_heading)],
            [-math.sin(-start_heading), math.cos(-start_heading)]
        ])
        pos = np.matrix([
            [car_pose.pose.pose.position.x],
            [car_pose.pose.pose.position.y]
        ])
        car_start = np.matrix([
            [self.car_start[0]],
            [self.car_start[1]]
        ])
        # rotate the position by start heading offset and
        # add the start position offset
        new_car_pos = rotation_matrix @ pos + car_start
        # add the heading offset and reconstruct the orientation
        # quaternion
        car_heading = Rotation.from_quat([
            car_pose.pose.pose.orientation.x,
            car_pose.pose.pose.orientation.y,
            car_pose.pose.pose.orientation.z,
            car_pose.pose.pose.orientation.w,
        ]).as_euler("XYZ")[2]
        new_heading = car_heading + start_heading
        new_orientation = Rotation.from_euler(
            "XYZ",
            [0.0, 0.0, new_heading]
        ).as_quat()
        new_car_pose = copy.deepcopy(car_pose) 
        new_car_pose.pose.pose.position.x = new_car_pos[0,0]
        new_car_pose.pose.pose.position.y = new_car_pos[1,0]
        new_car_pose.pose.pose.orientation.x = new_orientation[0]
        new_car_pose.pose.pose.orientation.y = new_orientation[1]
        new_car_pose.pose.pose.orientation.z = new_orientation[2]
        new_car_pose.pose.pose.orientation.w = new_orientation[3]
        return new_car_pose

    def _get_path_direction(self):
        return Line.get_direction(
            self.blue_cone_lines[1],
            self.blue_cone_lines[len(self.blue_cone_lines)//2]
        )

    def _get_cone_lines(self):
        """
        Calculate all lines between consecutive cones.
        
        :returns: Arrays of cone lines for each cone colour.
        """
        blue_cone_lines = [
            Line.make_line_from_cones(self.blue_cones[i], self.blue_cones[i + 1]) \
            for i in range(-1, len(self.blue_cones) - 1)
        ]
        yellow_cone_lines = [
            Line.make_line_from_cones(self.yellow_cones[i - 1], self.yellow_cones[i]) \
            for i in range(-1, len(self.yellow_cones) - 1)
        ]
        return blue_cone_lines, yellow_cone_lines

    def _get_completion(self, x: float, y: float,
                        lines: list, first_line):
        MAX_DIST = 10.0
        nearest_proj, dist = Line.project_to_nearest(
            x, y,
            lines
        )
        if not nearest_proj or dist > MAX_DIST: return math.nan, Line()
        (nearest, _, _) = nearest_proj
        first_idx = lines.index(first_line)
        distance = 0.0
        total_distance = 0.0
        reached_car = False
        for i in range(len(lines)):
            idx = (first_idx + i) % len(lines)
            line = lines[idx]
            l = line.get_length()
            if line == nearest:
                reached_car = True
            if reached_car == False:
                distance += l
            total_distance += l
        completion = distance / total_distance    
        # if self.path_direction == self.direction: completion = 1 - completion
        return completion, nearest

    def get_completion(self, car_pose):
       """
       Get the completion percentage represented by
       car_pose.
       
       :param car_pose: The pose to calculate the
       completion percentage from.
       :returns: float [0-1]
       """ 
       x = car_pose.pose.pose.position.x
       y = car_pose.pose.pose.position.y
       blue_completion, n = self._get_completion(
           x, y,
           self.blue_cone_lines,
           self.first_blue_line
       )
       self.nblue = n
       yellow_completion, n = self._get_completion(
           x, y,
           self.yellow_cone_lines,
           self.first_yellow_line
       )
       self.nyellow = n
       return 0.5 * blue_completion + 0.5 * yellow_completion

class Line:
    def __init__(self, sx=0.0, sy=0.0, ex=0.0, ey=0.0):
        """
        Data structure to represent a line segment.
        
        :param sx: Starting point x.
        :param sy: Starting point y.
        :param ex: End point x.
        :param ey: End point y.
        """
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        self.m, self.c = self._get_line_eqn()

    def _get_line_eqn(self):
        """
        Calculates the gradient and y-axis intercept
        of the line.
        
        :returns: m, c
        """
        try:
            m = (self.ey - self.sy) / (self.ex - self.sx)
        except:
            m = 0.0
        c = self.sy - m * self.sx
        return m, c

    def project(self, x, y):
        """
        Attempt to project a point onto the line segment.
        
        :param x: The x component of the point to project.
        :returns: Whether or not the intersection exists
        and its position. Bool, int_x, int_y
        """
        line_vec = np.matrix([
            [self.ex - self.sx],
            [self.ey - self.sy]
        ])
        mag = np.sqrt(line_vec[0,0]**2 + line_vec[1,0]**2)
        line_vec = (1/mag) * line_vec
        start_pt = np.matrix([
            [self.sx],
            [self.sy]
        ])
        point_vec = np.matrix([
            [x],
            [y]
        ]) - start_pt
        xproj = line_vec[0,0]*point_vec[0,0] + line_vec[1,0]*point_vec[1,0]
        intersect = xproj * line_vec + start_pt
        int_x, int_y = intersect[0,0], intersect[1,0]
        can_project = 0 <= xproj <= self.get_length()
        return (can_project, int_x, int_y)

    def get_length(self):
        return math.sqrt(
            (self.ex - self.sx) ** 2 + \
            (self.ey - self.sy) ** 2
        )

    @staticmethod
    def project_to_nearest(x, y, lines):
        nearest_successful_projection = None
        nearest_successful_projection_distance = math.inf
        for line in lines:
            proj, int_x, int_y = line.project(x, y)
            if proj == False: continue
            dist = math.sqrt(
                (int_y - y)**2 + (int_x - x)**2
            )
            if dist < nearest_successful_projection_distance:
                nearest_successful_projection_distance = dist
                nearest_successful_projection = (line, int_x, int_y)
        return nearest_successful_projection, nearest_successful_projection_distance

    @staticmethod
    def make_line_from_cones(cone1, cone2):
        """
        Initialises a line from a pair of cones.
        
        :param cone1: First cone.
        :param cone2: Second cone.
        :returns: The line between the cones.
        """
        return Line(
            cone1[0], cone1[1],
            cone2[0], cone2[1]
        )

    @staticmethod
    def make_line_from_car_state(car_state):
        """
        Initialises a line using a car's position
        and rotation.
        
        :param car_state: The car's position and rotation.
        :returns: The line representing the car.
        """
        length = 1.5 
        rot = Rotation.from_quat([
            car_state.pose.pose.orientation.x,
            car_state.pose.pose.orientation.y,
            car_state.pose.pose.orientation.z,
            car_state.pose.pose.orientation.w,
        ])
        yaw = -rot.as_euler("XYZ")[-1]
        rotmat = np.matrix([
            [math.cos(yaw), math.sin(yaw)],
            [-math.sin(yaw), math.cos(yaw)]
        ])
        pos = np.matrix([
            [car_state.pose.pose.position.x],
            [car_state.pose.pose.position.y]
        ])
        start = rotmat @ np.matrix([[length/2], [0.0]]) + pos
        end = rotmat @ np.matrix([[-length/2], [0.0]]) + pos
        return Line(start[0,0], start[1,0], end[0,0], end[1,0])

    @staticmethod
    def get_direction(line1, line2):
        mat = np.matrix([
            [1.0, line1.sx, line1.sy],
            [1.0, line1.ex, line1.ey],
            [1.0, line2.sx, line2.sy]
        ])
        return 1 if np.linalg.det(mat) > 0 else -1

    @staticmethod
    def _ccw(
        ax, ay, 
        bx, by,
        cx, cy):
        """
        Utility function used for intersection check.
        """
        return \
            (cy - ay) * (bx - ax) > \
            (by - ay) * (cx - ax)

    @staticmethod
    def intersection(l1, l2):
        """
        Returns true if two line segments intersect.
        """
        ax, ay = l1.sx, l1.sy
        bx, by = l1.ex, l1.ey
        cx, cy = l2.sx, l2.sy
        dx, dy = l2.ex, l2.ey
        return \
            Line._ccw(ax, ay, cx, cy, dx, dy) \
            != Line._ccw(bx, by, cx, cy, dx, dy) \
            and Line._ccw(ax, ay, bx, by, cx, cy) \
            != Line._ccw(ax, ay, bx, by, dx, dy)

def intersection_check(dataset: Dataset, track: Track, visualize = False):
    """
    Check a database for any track intersections.
    
    :param dataset: The dataset to check.
    :param track: The track which the dataset was created from.
    :param visualize: Plot the results.
    :returns: Whether or not there exists an intersection and the time at which it occurs.
    """
    # construct lines from track cones
    blue_cone_lines, yellow_cone_lines = track.blue_cone_lines, track.yellow_cone_lines
    cone_lines = blue_cone_lines + yellow_cone_lines

    # deserialize all car poses in the database and sort them by ascending
    # timestamp
    car_poses = sorted([
        (x[1], track.transform_car_pose(deserialize_message(x[2], CarState))) \
        for x in dataset.get_msgs("ground_truth_state").fetchall()
    ],
    key = lambda data: data[0]
    )

    # draw track and starting point
    if visualize:
        for line in blue_cone_lines:
            plt.plot([line.sx, line.ex], [line.sy, line.ey], "-", color="blue")
        for line in yellow_cone_lines:
            plt.plot([line.sx, line.ex], [line.sy, line.ey], "-", color="yellow")
        plt.plot([track.car_start[0]], [track.car_start[1]], "o", color="red", markersize=15)


    # iterate through each car pose and check for intersection with
    # all line segments.
    intersection = False
    intersection_idx = len(car_poses)
    for car_pose_idx, (timestamp, car_pose) in enumerate(car_poses):
        if intersection == True: break
        car_pose_line = Line.make_line_from_car_state(car_pose)
        for cone_line in cone_lines:
            if Line.intersection(car_pose_line, cone_line):
                intersection_time = (timestamp - car_poses[0][0]) / 1000 
                print(f"INTERSECTION! {intersection_time} seconds.")
                intersection = True
                intersection_idx = car_pose_idx
                break
    
    # plot the path up to first intersection,
    # or the entire path if there was none
    if visualize:
        plt.plot(
            [c.pose.pose.position.x for _,c in car_poses[:intersection_idx + 1]],
            [c.pose.pose.position.y for _,c in car_poses[:intersection_idx + 1]],
            "-", color="black")
            
        plt.show()

    return intersection, intersection_time

def get_lap_times(dataset: Dataset, track: Track):
    """
    Finds intersections with starting cones to
    determine lap times.
    
    :param dataset: The dataset to use.
    :param track: The track associated with the dataset.
    :returns: A list of lap times.
    """
    pass  # TODO
