import rclpy, sys
import rclpy.logging as logging
import sim_data_collection.analysis.analysis as analysis
from sim_data_collection.analysis.dataset import Dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from rclpy.serialization import deserialize_message
from eufs_msgs.msg import CarState
from ugrdv_msgs.msg import VCUStatus, DriveRequest
from scipy.spatial.transform import Rotation
from eufs_msgs.msg import ConeArrayWithCovariance
from typing import List, Tuple
from sim_data_collection.analysis.vehicle_model import VehicleModel, KinematicBicycle, NNVehicleModel
from sim_data_collection.analysis.simulation_visualiser import SimulationVisualiser
import numpy as np
import fcntl, json, signal, copy

def visualise_data(db_paths: List[str],
                   time_factor=15.0,
                   vehicle_model: List[VehicleModel] | None = None):
    vis = SimulationVisualiser(db_paths, time_factor, vehicle_model)
    vis.visualise_all()

def analyse_data(output_file: str, db_paths: List[str]):
    dataset = Dataset()
    sim_runs = []
    show = False
    last_percentage = 0.0
    for i,db_path in enumerate(db_paths):
        dataset.open(db_path)
        percentage = (i+1) / len(db_paths)
        if percentage - last_percentage >= 0.05:
            last_percentage = percentage
            print(f"{int(percentage*100)}%")
        track = analysis.Track.track_from_db_path(
            db_path
        )
        try:
            violation_info = analysis.violation_check(
                dataset,
                track,
                visualise=show
            )
            laps = analysis.get_lap_times(
                dataset,
                track
            )
            # use the number of laps to correct the run's completion
            number_of_laps = len(
                [1 for (lapstart, lapend) in laps if violation_info.type == "none" or lapend <= violation_info.time]
            )
            violation_info.completion += number_of_laps * track.get_length()
            sim_run = analysis.SimulationRun(
                violation_info,
                laps
            )
            sim_runs.append(sim_run.to_dict())
        finally:
            dataset.close()

    with open(output_file, "r+", encoding="ascii") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        filecontents = f.read()
        try:
            data = json.loads(filecontents)
        except Exception as e:
            data = {
                "sim_runs" : [],
            }
            print(f"An exception occured whilst reading json file: {e}\nWriting to json anyway.")
        data["sim_runs"].extend(sim_runs)
        f.seek(0)
        f.truncate()
        filecontents = json.dumps(data)
        f.write(filecontents)
        fcntl.flock(f, fcntl.F_UNLCK) 

def plot(data_path, show=False):

    def show(title):
        if show == True:
            plt.title(title)
            plt.show()
        else:
            plt.savefig(title)

    with open(data_path, "r", encoding="ascii") as f:
        data = json.load(f)

    all_runs = [
        analysis.SimulationRun.from_dict(x) for x in data['sim_runs']
    ]

    intersections = [
        run.violation for run in all_runs if run.violation.type == "intersection"
    ]

    backwards = [
        run.violation for run in all_runs if run.violation.type == "backwards"
    ]

    success = [
        run.violation for run in all_runs if run.violation.type == "none"
    ]

    finished_with_intersection = len(intersections)
    finished_with_backwards = len(backwards)
    finished_without_violation = len(success)

    failure_time = [
        v.time for v in backwards + intersections
    ]

    completions = [
        run.violation.completion for run in all_runs
    ]

    fig, axes = plt.subplots(
        1,
    )

    ax = axes
    ax.set_ylabel("Number of runs")
    ax.hist(
        ["No violations" for i in range(finished_without_violation)] + ["Track intersection" for i in range(finished_with_intersection)] + ["Driving the wrong way" for i in range(finished_with_backwards)],
        bins="auto"
    ) 

    show("violations")

    fig, axes = plt.subplots(
        2, 1
    )

    ax = axes[0]
    ax.set_title("Time to first violation")
    ax.set_xlabel("Time (seconds)")
    ax.hist(
        failure_time,
        bins="auto"
    )

    ax = axes[1]
    ax.set_title("Overall track completion")
    ax.set_xlabel("Distance (metres)")
    ax.hist(
        completions,
        bins="auto"
    )

    show("completion")

    fig, axes = plt.subplots(
        1
    )

    ax = axes
    ax.set_ylabel("Distance (metres)")
    ax.set_xlabel("Time (seconds)")
    ax.plot(
        [v.time for v in backwards + intersections],
        [v.completion for v in backwards + intersections],
        "o"
    )

    show("intersection_completion")


def usage():
    print("ros2 run sim_data_collection analysis <output json> <db1> <db2> ...")
    print("ros2 run sim_data_collection visualise <db1> <db2> ...")
    print("ros2 run sim_data_collection plot <input json>")

def main():
    signal.signal(
        signal.SIGINT,
        lambda _, __: sys.exit(1)
    )

    if len(sys.argv) < 3:
        print("Not enough arguments")
        usage()
        sys.exit(1)
    logger = logging.get_logger("analaysis")
    verb = sys.argv[1]
    
    if verb == "visualise":
        db_paths = sys.argv[2:]
        logger.info(f"Analysis starting up. Visualising {len(db_paths)} databases.")
        visualise_data(db_paths, vehicle_model=[NNVehicleModel("/home/joe/Downloads/testmodel.pt"), KinematicBicycle()])
        # visualise_data(db_paths, vehicle_model=[KinematicBicycle(), KinematicBicycle()])
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
