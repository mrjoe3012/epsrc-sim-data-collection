import sim_data_collection.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sim_data_collection.analysis.dataset import Dataset, DatasetIterator
from sim_data_collection.analysis.vehicle_model import VehicleModel
from pathlib import Path
from ament_index_python import get_package_share_directory
from eufs_msgs.msg import CarState
from ugrdv_msgs.msg import VCUStatus, DriveRequest
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from typing import List, Dict, Optional, Tuple
import copy, math

msg_ids = [
    "ground_truth_state",
    "ground_truth_cones",
    "perception_cones",
    "vcu_status",
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
    def __init__(self, blue, yellow, large_orange, centreline, centreline_lines,
                 car_start = None, direction = 1):
        """
        Represents a racetrack.
        
        :param blue: list of blue cone positions
        :param yellow: list of yellow cone positions
        :param car_start: Car starting pose (x, y, theta)
        """
        self.ncent = Line()
        self.blue_cones = blue
        self.centreline = centreline
        self.centreline_lines = centreline_lines
        self.yellow_cones = yellow
        self.large_orange_cones = large_orange
        if car_start is None: car_start = (0.0, 0.0, 0.0)
        self.car_start = car_start
        self.direction = direction
        self.blue_cone_lines, self.yellow_cone_lines = self._get_cone_lines()
        self.finish_line =  self._get_finish_line()
        (self.first_centreline, _, _), _ = Line.project_to_nearest(
            self.car_start[0], self.car_start[1],
            self.centreline_lines
        )
        assert self.first_centreline is not None
        self.path_direction = self._get_path_direction()

    @staticmethod
    def extract_centreline(all_cones: List[Tuple[float, float, str]]) -> List[Tuple[float, float]]:
        centreline = []
        lines = []
        for i in range(len(all_cones)):
            j = (i - 1) % len(all_cones)
            prev_x, prev_y, prev_col = all_cones[j]
            cur_x, cur_y, cur_col = all_cones[i]
            if cur_col == prev_col: continue 
            cen_x = (cur_x + prev_x) / 2
            cen_y = (cur_y + prev_y) / 2
            centreline.append((
                cen_x, cen_y
            ))
        for i in range(len(centreline)):
            j = (i - 1) % len(centreline)
            prev_x, prev_y = centreline[j]
            cur_x, cur_y = centreline[i]
            lines.append(Line(
                prev_x, prev_y,
                cur_x, cur_y
            ))
        return centreline, lines

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
        blue, yellow, large_orange = [], [], []
        all = []
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
                elif tag == "big_orange":
                    arr = large_orange
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
                    all.append(
                        (cone[0], cone[1], tag)
                    )
        centreline, centreline_lines = Track.extract_centreline(all)
        return Track(
            blue, yellow, large_orange, centreline, centreline_lines,
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

    def get_length(self):
        dist = np.sum([l.get_length() for l in self.centreline_lines])
        return dist

    def _get_finish_line(self):
        c1 = self.large_orange_cones[1]  # these two cones are definitely not
        c2 = self.large_orange_cones[2]  # on the same side of the track
        return Line.make_line_from_cones(
            c1, c2
        )

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
        MAX_DIST = 15.0
        nearest_proj, dist = Line.project_to_nearest(
            x, y,
            lines
        )
        if not nearest_proj or dist > MAX_DIST: return 0.0, self.get_length(), Line()
        (nearest, int_x, int_y) = nearest_proj
        remainder = np.sqrt((int_x - nearest.sx)**2 + (int_y - nearest.sy)**2)
        first_idx = lines.index(first_line)
        distance = remainder
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
        return distance, total_distance, nearest

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
       completion, total_distance, nearest = self._get_completion(
           x, y, self.centreline_lines, self.first_centreline
       )
       self.ncent = nearest
       return completion, total_distance

class BackwardsDetector:
    def __init__(self, track: Track,
                 small_negative_t = 5.0,
                 large_negative_t = 0.8,
                 num_seconds_to_keep = 3.0,
                 verbose = False):
        """
        This is a class which encapsulates the algorithm for detecting
        backwards travelling during a simulation run. This works by
        sequentially analysing the car's track completion per timestep.
        If the track completion in the last N seconds exceeds some small
        negative threshold we consider this a violation. However, if the
        track completion exceeds a large negative threshold, we ignore this
        as the car has likely just completed a lap (gone from 100% completion to 0%).
        :param track: The track that is currently being driven.
        :param small_negative_t: A threshold in metres which must be exceeded
        negatively to consider the current state as a violation.
        :param large_negative_t: The percentage of the total track length
        to be exceeded in order for a potential violation to be ignored
        due to it likely being the completion of a lap.
        :param num_seconds_to_keep: Maximum number of seconds of completion
        data to be analysed at once.
        :param verbose: Set to true for debug prints.
        """
        self._small_negative_threshold = small_negative_t  # metres
        self._large_negative_threshold = large_negative_t  # percentage of track length
        self._number_of_seconds_to_analyse = num_seconds_to_keep  # how many seconds of completion data we store at once
        self._completions_buffer = []
        self._verbose = verbose
        self._track = track
        self._track_len = track.get_length()

    def _trim_buffer(self) -> None:
        """
        Ensures the completions buffer contains only data
        within the last n seconds. 
        """
        j = 0
        while j < len(self._completions_buffer) - 1 \
            and self._completions_buffer[-1][0] - self._completions_buffer[0][0] > self._number_of_seconds_to_analyse:
            j += 1
        self._completions_buffer = self._completions_buffer[j:]

    def _get_completion_sum(self) -> float:
        """
        Calculate the sum of completion deltas. 
        """
        c = self._completions_buffer
        return sum([
            self._track_len * (c[i + 1][1] - c[i][1]) for i in range(len(c) - 1)
        ])

    def add_completion(self, time: float, distance: float) -> None:
        """
        Register the car's completion at a timestep. These values must be
        added in chronological order.
        :param time: The time at which this completion occurs.
        :param distance: The track completion in metres.
        """
        self._completions_buffer.append((time, distance))

    def is_violating(self) -> Tuple[bool, float]:
        """
        Get whether or not the car is currently committing a violation
        by driving backwards. If a violation is being committed, the second
        element is the time at which the violation begins.
        :returns: Whether or not a violation is being committed, and the time
        of the violation.
        """
        self._trim_buffer()
        completions_delta = self._get_completion_sum()
        first_time = self._completions_buffer[0][0]
        last_time = self._completions_buffer[-1][0]
        if completions_delta < -self._small_negative_threshold:
            if self._verbose: print(f"{last_time}: More backwards than forwards")
            if completions_delta < -self._large_negative_threshold * self._track_len:
                if self._verbose: print(f"Likely that a lap was completed. Ignoring potential violation.")
            else:
                if self._verbose: print(f"No lap completion in sight. Violation has occured.")
                return (True, first_time)
        return (False, -1.0)

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
        can_project = -0.5 * self.get_length() <= xproj <= self.get_length() * 1.5
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

class ViolationInfo:
    def __init__(self, type: str, time: float, completion: float):
        """
        A struct describing a track violation.
        'type' can be one of the following:
          - 'none' : no violations, in this case 'time' is irrelevant
          - 'intersection' : car intersected track boundaries
          - 'backwards' : car went against the direction of the track
        :param type: The type of violation as detailed above.
        :param time: The time of the intersection in seconds.
        :param completion: The car's track completion in metres at the time of violation.
        """
        self.type = type
        self.time = time
        self.completion = completion
    
    def to_dict(self):
        return {
            'type' : self.type,
            'time' : self.time,
            'completion' : self.completion
        }
    @staticmethod
    def from_dict(dict: Dict):
        return ViolationInfo(
            dict['type'],
            dict['time'],
            dict['completion']
        )

class SimulationRun:
    def __init__(self, violation: ViolationInfo, lap_times: List[Tuple[float, float]]):
        """
        Contains all data about a simulation run.
        :param violation: The violation info produced by the check.
        :param lap_times:
        """
        self.violation = violation
        self.lap_times = lap_times
    def to_dict(self):
        return {
            'violation' : self.violation.to_dict(),
            'lap_times' : list(self.lap_times)
        }
    @staticmethod
    def from_dict(dict: Dict):
        return SimulationRun(
            ViolationInfo.from_dict(dict['violation']),
            list(dict['lap_times']) 
        )

def violation_check(dataset: Dataset, track: Track, visualise = False) -> ViolationInfo:
    """
    Check a database for any track violations.
    
    :param dataset: The dataset to check.
    :param track: The track which the dataset was created from.
    :param visualize: Plot the results.
    :returns: A ViolationInfo instance describing the violation.
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

    # iterate through each car pose and check for intersection with
    # all line segments.
    backwards_detector = BackwardsDetector(track=track)
    result = ViolationInfo("none", -1.0, -1.0)
    final_car_pose_line = None
    final_completion, final_time = None, None
    for car_pose_idx, (timestamp, car_pose) in enumerate(car_poses):
        time = (timestamp - car_poses[0][0]) / 1000 
        completion, _ = track.get_completion(car_pose)
        # check for intersection
        car_pose_line = Line.make_line_from_car_state(car_pose)
        final_car_pose_line = car_pose_line
        final_completion = completion
        final_time = time
        for cone_line in cone_lines:
            if Line.intersection(car_pose_line, cone_line):
                result.type = "intersection"
                result.completion = completion
                result.time = time
                break
        if result.type != "none": break  # early break
        # check for backwards driving
        backwards_detector.add_completion(completion, time)
        # backwards_driving, _ = backwards_detector.is_violating()
        backwards_driving = False
        if backwards_driving:
            result.type = "backwards"
            result.completion = completion
            result.time = time
        if result.type != "none": break  # early break

    # if no violation occured, use the final car pose
    # to set a completion and time
    if result.type == "none":
        result.completion = final_completion
        result.time = final_time

    # visualise
    if visualise == True:
        if result.type == "none":
            print("No violations.")
        elif result.type == "intersection":
            print(f"Intersection at {result.time}.2f seconds")
        elif result.type == "backwards":
            print(f"Backwards driving at {result.time}.2f seconds")
        for line in yellow_cone_lines:
            plt.plot(
                [line.sx, line.ex],
                [line.sy, line.ey],
                "-",
                color="yellow"
            )
        for line in blue_cone_lines:
            plt.plot(
                [line.sx, line.ex],
                [line.sy, line.ey],
                "-",
                color="blue"
            )
        if final_car_pose_line is not None:
            plt.plot(
                [final_car_pose_line.sx, final_car_pose_line.ex],
                [final_car_pose_line.sy, final_car_pose_line.ey],
                "-",
                color="black"
            )
        plt.show()

    return result
    
def get_lap_times(dataset: Dataset, track: Track, min_lap_time = 60.0) -> List[Tuple[float, float]]:
    """
    Finds intersections with starting cones to
    determine lap times.
    
    :param dataset: The dataset to use.
    :param track: The track associated with the dataset.
    :param min_lap_time: A minimum time between laps or start of simulation.
    This is not to discount fast laps but instead to prevent the system for
    registering the same finish line intersection as multiple fast laps /
    registering a lap in the first few seconds of the simulation
    :returns: A list of lap times.
    """
    car_poses = dataset.get_msgs("ground_truth_state").fetchall()
    car_poses = sorted([
        (x[1], deserialize_message(x[2], CarState)) for x in car_poses
    ],
    key = lambda x: x[0]
    )
    car_poses = [
        (ts, track.transform_car_pose(cp)) for ts, cp in car_poses
    ]

    first_timestamp = car_poses[0][0]
    finish_line = track.finish_line
    intersections = []

    for timestamp, car_pose in car_poses:
        t = (timestamp - first_timestamp) / 1e3
        car_line = Line.make_line_from_car_state(
            car_pose
        )
        if Line.intersection(car_line, finish_line):
            intersections.append(t)

    laps = []
    cur_intersection = 0.0
    for t in intersections:
        if t - cur_intersection >= min_lap_time:
            laps.append((cur_intersection, t))  # (start, end)
            laptime = t - cur_intersection
            # print(f"LAP!\nStart: {cur_intersection} seconds.\nEnd: {t} seconds.\nTime: {laptime} seconds.")
        cur_intersection = t

    return laps

def evaluate_vehicle_models(db_paths: List[str], vehicle_models: List[VehicleModel], vehicle_model_names: List[str]):
    def get_next(dataset, time):
        vcu_status = dataset._connection.execute(
            "SELECT timestamp, data FROM vcu_status WHERE timestamp > ? ORDER BY timestamp ASC",
            (time,)
        ).fetchone()
        drive_request = dataset._connection.execute(
            "SELECT timestamp, data FROM drive_request WHERE timestamp > ? ORDER BY timestamp ASC",
            (time,)
        ).fetchone()
        car_state = dataset._connection.execute(
            "SELECT timestamp, data FROM ground_truth_state WHERE timestamp > ? ORDER BY timestamp ASC",
            (time,)
        ).fetchone()
        if vcu_status is not None:
            vcu_status = deserialize_message(vcu_status[1], VCUStatus)
        if drive_request is not None:
            drive_request = deserialize_message(drive_request[1], DriveRequest)
        if car_state is not None:
            car_state = deserialize_message(car_state[1], CarState)
        return vcu_status, drive_request, car_state
    steps = 10
    step_size = 1 * 1e3
    # dim 0: vehicle model index [0,len(vehicle_models))
    # dim 1: step index [0,steps)
    # dim 2: error index [0,2] 0=xpos 1=ypos 2=heading
    # dim 3: row index (arbtrary)
    data = [[[[], [], []] for j in range(steps)] for i in range(len(vehicle_models))]
    for db_path in db_paths:
        dataset = Dataset()
        dataset.open(db_path)
        # get starting time
        time, end_time = dataset.get_start_and_end()
        try:
            while time + steps * step_size < end_time:
                # get next messages for each step
                for i in range(steps):
                    t = time
                    t_n = time + step_size * (i + 1)
                    t_c = time ## TODO: keep track of current time and stream through to update vehicle model properly
                    vcu_status, drive_request, car_state = get_next(dataset, t)
                    _, _, car_state_n = get_next(dataset, t_n)
                    heading_quat = Rotation.from_quat([
                        car_state.pose.pose.orientation.x,
                        car_state.pose.pose.orientation.y,
                        car_state.pose.pose.orientation.z,
                        car_state.pose.pose.orientation.w,
                    ])
                    heading = heading_quat.as_euler("XYZ")[2]
                    heading_n_quat = Rotation.from_quat([
                        car_state_n.pose.pose.orientation.x,
                        car_state_n.pose.pose.orientation.y,
                        car_state_n.pose.pose.orientation.z,
                        car_state_n.pose.pose.orientation.w,
                    ])
                    dx_global = car_state.pose.pose.position.x - car_state_n.pose.pose.position.x
                    dy_global = car_state.pose.pose.position.y - car_state_n.pose.pose.position.y
                    dheading = np.rad2deg((heading_n_quat * heading_quat.inv()).as_euler("XYZ")[2])
                    dx = dx_global * np.cos(-heading) - dy_global * np.sin(-heading)
                    dy = dx_global * np.sin(-heading) + dy_global * np.cos(-heading)
                    for j, vehicle_model in enumerate(vehicle_models):
                        vehicle_model.reset()
                        vehicle_model.update_state({
                            'steering_angle' : vcu_status.steering_angle,
                            'wheel_speeds' : [
                                vcu_status.wheel_speeds.fl_speed,
                                vcu_status.wheel_speeds.fr_speed,
                                vcu_status.wheel_speeds.rl_speed,
                                vcu_status.wheel_speeds.rr_speed,
                            ],
                            'steering_angle_request' : drive_request.ackermann.drive.steering_angle,
                            'acceleration_request' : drive_request.ackermann.drive.acceleration
                        })
                        dx_pred, dy_pred, dheading_pred = vehicle_model((t_n - t) / 1e3)
                        dx_err = dx - dx_pred
                        dy_err = dy - dy_pred
                        dheading_err = dheading - dheading_pred
                        data[j][i][0].append(dx_err)
                        data[j][i][1].append(dy_err)
                        data[j][i][2].append(dheading_err)
                time += step_size
        finally:
            dataset.close()
    
    data = np.array(data)

    fig, axes = plt.subplots(
        2, 2
    )

    fig.suptitle("Vehicle Model Evaluation")

    for ax in axes.flat:
        # ax.axis("equal")
        ax.set_xlabel("Time (Seconds)")

    axes[1,1].axis("off")
    axes[0,0].set_title("X Position")
    axes[0,1].set_title("Y Position")
    axes[1,0].set_title("Heading")
    axes[0,0].set_ylabel("RMSE (Metres)")
    axes[0,1].set_ylabel("RMSE (Metres)")
    axes[1,0].set_ylabel("RMSE (Degrees)")

    for i, name in enumerate(vehicle_model_names):
        rmse = np.sqrt(np.sum(np.square(data[i,:,:,:]), axis=2) / data.shape[-1])
        x = [step_size * (i+1) * 1e-3 for i in range(steps)]
        axes[0,0].plot(
            x,
            rmse[:,0],
            "-"
        )
        axes[0,1].plot(
            x,
            rmse[:,1],
            "-"
        )
        axes[1,0].plot(
            x,
            rmse[:,2],
            "-"
        )

    for ax in axes.flat:
        ax.legend(vehicle_model_names)

    plt.show()
    plt.close(fig)

def evaluate_vehicle_models(db_paths: List[str], vehicle_models: List[VehicleModel], vehicle_model_names: List[str]):
    def get_next(dataset: DatasetIterator) -> Tuple[CarState, VCUStatus, DriveRequest]:
        result = [None, None, None]
        indices = {
            CarState : 0,
            VCUStatus : 1,
            DriveRequest : 2
        }
        try:
            while not all(result):
                type, timestamp, msg = next(dataset)
                result[indices[type]] = (timestamp, msg)
        except StopIteration as e:
            pass
        return tuple(result)
    window_size = 5 * 1e3
    # dim 0: vehicle model index
    # dim 1: data 0=propragation time 1=x pos error 2=y pos error 3=heading error
    # dim 2: row index
    data = [[[] for i in range(4)] for i in range(len(vehicle_models))]
    for db_path in db_paths:
        dataset = Dataset()
        dataset.open(db_path)
        dataset_it = iter(dataset)
        # get starting time
        time, end_time = dataset.get_start_and_end()
        try:
            stop = False
            window_start_time = time
            window = []
            while stop == False:
                car_state, vcu_status, drive_request = get_next(dataset_it)
                if not all((car_state, vcu_status, drive_request)):
                    stop = True
                else:
                    timestamp = sum((car_state[0], vcu_status[0], drive_request[0])) / 3
                    # now, when we update the window we should also calculate errors, resetting
                    # the vehicle models upon creation of a new window
                    if len(window) == 0 or timestamp - window_start_time > window_size:
                        x = car_state[1].pose.pose.position.x
                        y = car_state[1].pose.pose.position.y
                        orientation = Rotation.from_quat([
                            car_state[1].pose.pose.orientation.x,
                            car_state[1].pose.pose.orientation.y,
                            car_state[1].pose.pose.orientation.z,
                            car_state[1].pose.pose.orientation.w,
                        ])
                        window = [{
                            'timestamp' : timestamp,
                            'car_state' : car_state[1],
                            'drive_request' : drive_request[1],
                            'vcu_status' : vcu_status[1],
                            'x' : x,
                            'y' : y,
                            'orientation' : orientation,
                            'model_states' : [
                                { 'x' : x, 'y' : y, 'orientation' : orientation} for vm in vehicle_models
                            ]
                        }]
                        window_start_time = timestamp
                        for vehicle_model in vehicle_models:
                            vehicle_model.reset()
                            vehicle_model.update_state({
                                'steering_angle' : vcu_status[1].steering_angle,
                                'wheel_speeds' : [
                                    vcu_status[1].wheel_speeds.fl_speed,
                                    vcu_status[1].wheel_speeds.fr_speed,
                                    vcu_status[1].wheel_speeds.rl_speed,
                                    vcu_status[1].wheel_speeds.rr_speed,
                                ]
                            })
                    else:
                        # get current state and record it
                        x = car_state[1].pose.pose.position.x
                        y = car_state[1].pose.pose.position.y
                        orientation = Rotation.from_quat([
                            car_state[1].pose.pose.orientation.x,
                            car_state[1].pose.pose.orientation.y,
                            car_state[1].pose.pose.orientation.z,
                            car_state[1].pose.pose.orientation.w,
                        ])
                        previous_data = window[-1]
                        window.append({
                            'timestamp' : timestamp,
                            'car_state' : car_state[1],
                            'drive_request' : drive_request[1],
                            'vcu_status' : vcu_status[1],
                            'x' : x,
                            'y' : y,
                            'orientation' : orientation,
                            'model_states' : []
                        })
                        current_data = window[-1]
                        # models states should be kept track of in the global frame, this should be what we're evaluating against
                        # update vehicle model and compare against ground truth
                        time_since_window_start = timestamp - window_start_time
                        delta_time = (timestamp - previous_data['timestamp']) / 1e3
                        for i, vehicle_model in enumerate(vehicle_models):
                            vehicle_model.update_state({
                                'acceleration_request' : previous_data['drive_request'].ackermann.drive.acceleration,
                                'steering_angle_request' : previous_data['drive_request'].ackermann.drive.steering_angle
                            })
                            prev_model_state = previous_data['model_states'][i]
                            dx_pred, dy_pred, dheading_pred = vehicle_model(delta_time) 
                            # if i == 0:
                            #     print("in: ", vehicle_model.x)
                            #     print("out: ", dx_pred, dy_pred, dheading_pred)
                            #     if abs(dx_pred) > 2:
                            #         print("===================================")
                            dheading_pred_quat = Rotation.from_euler("XYZ", [0.0,0.0,dheading_pred])
                            heading = (prev_model_state['orientation'] * dheading_pred_quat).as_euler("XYZ")[2]
                            dx_pred_global = dx_pred * np.cos(heading) - dy_pred * np.sin(heading)
                            dy_pred_global = dx_pred * np.sin(heading) + dy_pred * np.cos(heading)
                            new_model_state = {
                                'x' : prev_model_state['x'] + dx_pred_global,
                                'y' : prev_model_state['y'] + dy_pred_global,
                                'orientation' : prev_model_state['orientation'] * dheading_pred_quat
                            }
                            x_err = x - new_model_state['x']
                            y_err = y - new_model_state['y']
                            heading_err = np.rad2deg((orientation * new_model_state['orientation'].inv()).as_euler("XYZ")[2])
                            data[i][0].append(time_since_window_start / 1e3)
                            data[i][1].append(x_err)
                            data[i][2].append(y_err)
                            data[i][3].append(heading_err)
                            current_data['model_states'].append(new_model_state)
        finally:
            dataset.close()
    
    data = np.array(data)
    num_data_points = data.shape[-1]

    fig, axes = plt.subplots(
        2, 2
    )

    fig.suptitle("Vehicle Model Evaluation")

    for ax in axes.flat:
        # ax.axis("equal")
        ax.set_xlabel("Time (Seconds)")

    axes[1,1].axis("off")
    axes[0,0].set_title("X Position")
    axes[0,1].set_title("Y Position")
    axes[1,0].set_title("Heading")
    axes[0,0].set_ylabel("RMSE (Metres)")
    axes[0,1].set_ylabel("RMSE (Metres)")
    axes[1,0].set_ylabel("RMSE (Degrees)")

    for i, name in enumerate(vehicle_model_names):
        x = data[i,0,:]
        marker = "-"
        bin_edges = np.histogram_bin_edges(x, 'auto')
        n_bins = len(bin_edges)
        indices = np.digitize(x, bin_edges, right=True)
        rmse = np.zeros((n_bins,3), dtype=np.float32)
        for j in range(n_bins):
            bin_indices = indices == j
            if not any(bin_indices): continue
            bin_data = data[i,1:,bin_indices]
            row = np.sqrt(np.mean(np.square(bin_data), axis=0))
            rmse[j,:] = row 
        axes[0,0].plot(
            bin_edges,
            rmse[:,0],
            marker
        )
        axes[0,1].plot(
            bin_edges,
            rmse[:,1],
            marker
        )
        axes[1,0].plot(
            bin_edges,
            rmse[:,2],
            marker
        )

    for ax in axes.flat[:-1]:
        ax.legend(vehicle_model_names)

    plt.savefig("vehicle_model_evaluation")
    plt.close(fig)