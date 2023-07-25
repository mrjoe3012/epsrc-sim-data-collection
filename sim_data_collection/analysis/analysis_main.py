import rclpy, sys
import rclpy.logging as logging
import sim_data_collection.analysis.analysis as analysis
from sim_data_collection.analysis.dataset import Dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from rclpy.serialization import deserialize_message
from eufs_msgs.msg import CarState
from scipy.spatial.transform import Rotation
from eufs_msgs.msg import ConeArrayWithCovariance
from typing import List, Tuple

def visualise_data(db_paths: List[str],
                   time_factor=15.0):
    dataset = Dataset()
    figsize = (5,5)
    update_hz = 1000.0
    update_interval = 1.0 / update_hz
    plot_percentage = True
    for db_path in db_paths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        print(f"checking {db_path}")
        try:
            dataset.open(db_path)
            track = analysis.Track.track_from_db_path(db_path)
            t = time.time()
            start_time = dataset._connection.execute(
                "SELECT timestamp, data FROM ground_truth_state ORDER BY timestamp ASC"
            ).fetchone()[0] / 1e3
            completions = []

            def anim_callback(i):
                nonlocal t, completions
                ax1.cla()
                ax1.set_aspect("equal")
                # figure out elapsed time since we last
                # visualised
                dt = time.time() - t
                # use the time factor to figure out what
                # time we should be visualising
                sim_time = (start_time + time_factor*dt) * 1e3
                # plot the track's centreline
                ax1.plot(
                    [c[0] for c in track.centreline],
                    [c[1] for c in track.centreline],
                    "-", color="black", markersize=3
                )
                # plot the starting point of the car
                ax1.plot(
                    track.car_start[0], track.car_start[1],
                    "o", color="green"
                )
                # plot the cones on the track
                ax1.plot(
                    [x[0] for x in track.blue_cones],
                    [x[1] for x in track.blue_cones],
                    "o", color="blue", markersize=2
                )
                ax1.plot(
                    [x[0] for x in track.yellow_cones],
                    [x[1] for x in track.yellow_cones],
                    "o", color="yellow", markersize=2
                )
                ax1.plot(
                    [x[0] for x in track.large_orange_cones],
                    [x[1] for x in track.large_orange_cones],
                    "o", color="orange", markersize=4
                )
                # plot the finish line
                ax1.plot(
                    [track.finish_line.sx, track.finish_line.ex],
                    [track.finish_line.sy, track.finish_line.ey],
                    "-", color="green"
                )
                # find the next car pose to visualise
                # using the scaled time
                latest_car_pose = dataset._connection.execute(
                    "SELECT timestamp, data FROM ground_truth_state \
                        WHERE timestamp < ? ORDER BY timestamp DESC",
                        (sim_time,)
                ).fetchone()
                if latest_car_pose is None: return
                else: latest_car_pose = deserialize_message(latest_car_pose[1], CarState)
                # plot the latest car pose as a line
                # on the track (to vaguely resemble  car)
                latest_car_pose = track.transform_car_pose(latest_car_pose)
                latest_car_pose_line = analysis.Line.make_line_from_car_state(latest_car_pose)
                ax1.plot([latest_car_pose_line.sx, latest_car_pose_line.ex],
                            [latest_car_pose_line.sy, latest_car_pose_line.ey],
                            "-", linewidth=5)
                # visualise where the system thinks the car is
                # by colouring the nearest centreline red
                ax1.plot(
                    [track.ncent.sx, track.ncent.ex], [track.ncent.sy, track.ncent.ey],
                    "-", color="red"
                )
                # plot the percentage track completion
                completion, total_distance = track.get_completion(
                    latest_car_pose
                )
                if plot_percentage == True:
                    completion /= total_distance
                    ax2.set_ylim((0, 2))
                else:
                    ax2.set_ylim((0, total_distance*1.5))
                completions.append((sim_time / 1e3, completion))
                ax2.plot(
                    [c[0] for c in completions],
                    [c[1] for c in completions],
                    "-", color="black")

            anim = FuncAnimation(fig, anim_callback,
                                 interval=update_interval)
            plt.show()
        finally:
            dataset.close()

def analyse_data(db_paths: List[str]):
    dataset = Dataset()
    for db_path in db_paths:
        dataset.open(db_path)
        track = analysis.Track.track_from_db_path(
            db_path
        )
        try:
            analysis.intersection_check(
                dataset,
                track,
                visualize=True
            )
            analysis.get_lap_times(
                dataset,
                track
            )
        finally:
            dataset.close()

def usage():
    print("ros2 run sim_data_collection analysis <analyse|visualise> <db1> <db2> ...")

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        usage()
        sys.exit(1)
    logger = logging.get_logger("analaysis")
    verb = sys.argv[1]
    db_paths = sys.argv[2:]
    
    if verb == "visualise":
        logger.info(f"Analysis starting up. Visualising {len(db_paths)} databases.")
        visualise_data(db_paths)
    elif verb == "analyse":
        logger.info(f"Analysis starting up. Analysing {len(db_paths)} databases.")
        analyse_data(db_paths)
    else:
        print(f"Unrecognised verb '{verb}'")
        usage()
        sys.exit(1)
