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
import numpy as np
import fcntl, json

def visualise_data(db_paths: List[str],
                   time_factor=15.0):
    dataset = Dataset()
    figsize = (5,5)
    update_hz = 1000.0
    update_interval = 1.0 / update_hz
    plot_percentage = True
    for db_path in db_paths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
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

def analyse_data(output_file: str, db_paths: List[str]):
    dataset = Dataset()
    intersections = []
    completions = []
    finished_without_intersection = 0
    finished_with_intersection = 0
    show = False
    last_percentage = 0.0
    # TODO: lap time analysis
    for i,db_path in enumerate(db_paths):
        dataset.open(db_path)
        percentage = (i+1) / len(db_paths)
        if percentage - last_percentage >= 0.05:
            last_percentage = percentage
            print(f"{int(percentage*100)}%")
        track = analysis.Track.track_from_db_path(
            db_path
        )
        track_length = track.get_length()
        try:
            intersection, time, completion = analysis.intersection_check(
                dataset,
                track,
                visualize=False
            )
            if intersection == True:
                intersections.append(time)
                completions.append(completion)
                finished_with_intersection += 1
            else:
                finished_without_intersection += 1
            laps = analysis.get_lap_times(
                dataset,
                track
            )
        finally:
            dataset.close()

    with open(output_file, "r+", encoding="ascii") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
        except Exception as e:
            data = {
                "intersections" : [],
                "completions" : [],
                "finished_without_intersection" : 0,
                "finished_with_intersection" : 0
            }
            print(f"An exception occured whilst reading json file: {e}")
        data["intersections"].extend(intersections)
        data["completions"].extend(completions)
        data["finished_without_intersection"] += finished_without_intersection
        data["finished_with_intersection"] += finished_with_intersection
        f.truncate(0)
        json.dump(data, f)
        fcntl.lockf(f, fcntl.F_UNLCK) 

def plot(data_path, show=False):

    with open(data_path, "r", encoding="ascii") as f:
        data = json.load(f)

    finished_without_intersection = data["finished_without_intersection"]
    finished_with_intersection = data["finished_with_intersection"]
    completions = data["completions"]
    intersections = data["intersections"]

    fig, axes = plt.subplots(
        1, 3,
    )

    ax = axes[0] 
    ax.set_title("Failures")
    ax.set_ylabel("Number of runs")
    ax.hist(
        ["No violations" for i in range(finished_without_intersection)] + ["At least 1 violation" for i in range(finished_with_intersection)],
        bins="auto"
    ) 

    ax = axes[1]
    ax.set_title("Time to first intersection")
    ax.set_xlabel("Time (seconds)")
    ax.hist(
        intersections,
        bins="auto"
    )

    ax = axes[2]
    ax.set_title("Overall track completion")
    # TODO: if laps, we need to register this as part
    # of the completion
    # TODO: for this, add funciton aggregate_completion(time)
    # so we can just use the laptime for this. remove return value
    # of completion from intersection check after doing this
    ax.set_xlabel("Distance (metres)")
    ax.hist(
        completions,
        bins="auto"
    )
    if show == True:
        plt.show()
    else:
        plt.savefig("analysis")

def usage():
    print("ros2 run sim_data_collection analysis <output json> <db1> <db2> ...")
    print("ros2 run sim_data_collection visualise <db1> <db2> ...")
    print("ros2 run sim_data_collection plot <input json>")

def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        usage()
        sys.exit(1)
    logger = logging.get_logger("analaysis")
    verb = sys.argv[1]
    
    if verb == "visualise":
        db_paths = sys.argv[2:]
        logger.info(f"Analysis starting up. Visualising {len(db_paths)} databases.")
        visualise_data(db_paths)
    elif verb == "analyse":
        output_filename = sys.argv[2]
        db_paths = sys.argv[3:]
        logger.info(f"Analysis starting up. Analysing {len(db_paths)} databases.")
        analyse_data(output_filename, db_paths)
    elif verb == "plot":
        input_filename = sys.argv[2]
        logger.info(f"Analysis starting up. Visualising data from {input_filename}.")
        plot(input_filename, show=False)
    else:
        print(f"Unrecognised verb '{verb}'")
        usage()
        sys.exit(1)
