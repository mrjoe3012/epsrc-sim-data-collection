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

def evaluate(model_path: str, db_paths: List[str]) -> None:
    nn = NNVehicleModel(model_path)
    kinematic = KinematicBicycle()
    analysis.evaluate_vehicle_models(
        db_paths, [nn, kinematic], ["Neural Network", "Kinematic"]
    )

def visualise_data(db_paths: List[str],
                   time_factor=3.0,
                   vehicle_models: List[VehicleModel] | None = None):
    vis = SimulationVisualiser(db_paths, time_factor, vehicle_models)
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

def plot(data_paths: list[str], labels: list[str], show=False):
    assert len(data_paths) == len(labels)
    figsize = (8, 4.5)
    num_files = len(labels)

    def show(title):
        if show == True:
            plt.title(title)
            plt.show()
        else:
            plt.savefig(title)

    data = []

    for data_path in data_paths:
        with open(data_path, "r", encoding="ascii") as f:
            data.append(json.load(f))

    all_runs = [
        [analysis.SimulationRun.from_dict(x) for x in d['sim_runs']] for d in data
    ]

    intersections = [
        [run.violation for run in runs if run.violation.type == "intersection"] for runs in all_runs
    ]

    backwards = [
        [run.violation for run in runs if run.violation.type == "backwards"] for runs in all_runs
    ]

    success = [
        [run.violation for run in runs if run.violation.type == "none"] for runs in all_runs
    ]

    finished_with_intersection = [len(x) for x in intersections]
    finished_with_backwards = [len(x) for x in backwards]
    finished_without_violation = [len(x) for x in success]

    failure_time = [
        np.array([v.time for v in b + i]) for b,i in zip(backwards, intersections)
    ]

    completions = [
        np.array([run.violation.completion for run in runs]) for runs in all_runs
    ]

    fig, axes = plt.subplots(
        1,
        figsize=figsize
    )

    ax = axes
    ax.set_title("Simulation Outcomes")
    ax.set_ylabel("Number of simulations")
    ax.set_xlabel("Perception profile")

    arrs = (finished_without_violation, finished_with_intersection)#, finished_with_backwards)
    bar_labels = ("No violations", "Track intersection")#, "Driving the wrong way")
    num_arrs = len(arrs)
    for i, (arr, label) in enumerate(zip(arrs, bar_labels, strict=True)):
        offset = i / num_arrs
        bar = ax.bar(
            labels,
            arr,
            align='center',
            label=label
        )

    ax.legend(loc="best")
    fig.tight_layout()
    show("violations")

    fig, axes = plt.subplots(
        2, 1,
        figsize=figsize
    )

    ax = axes[0]
    ax.set_title("Time to first violation")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of simulations")
    ax.set_xlim((0, 120))
    ax.set_ylim((0, 10000))
    for i, f in enumerate(failure_time):
        ax.bar(
            np.arange(5, 125, 10),
            np.histogram(
                f,
                np.arange(0, 130, 10),
                (0, 120)
            )[0],
            width=10,
            label=labels[i]
        )
    ax.legend(loc="best")

    ax = axes[1]
    ax.set_title("Overall track completion")
    ax.set_xlabel("Distance (metres)")
    ax.set_ylabel("Number of simulations")
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 10000))
    for i, c in enumerate(completions):
        ax.bar(
            np.arange(50, 1000, 100),
            np.histogram(
                c,
                np.arange(0, 1100, 100),
            )[0],
            width=100,
            label=labels[i]
        )
    ax.legend(loc="best")
    fig.tight_layout()
    show("completion")

    fig, axes = plt.subplots(
        1,
        figsize=figsize
    )

    ax = axes
    ax.set_title("Track Completion Over Time")
    ax.set_ylabel("Distance (metres)")
    ax.set_xlabel("Time (seconds)")
    ax.set_xlim((0, 120))
    ax.set_ylim((0, 500))
    for l, b, i in zip(labels, backwards, intersections):
        ax.plot(
            [v.time for v in b + i],
            [v.completion for v in b + i],
            "o",
            label=l,
            alpha=0.4,
            markersize=3
        )
    ax.legend(loc='best')
    fig.tight_layout()
    show("intersection_completion")


def usage():
    print("ros2 run sim_data_collection analysis analyse <output json> <db1> <db2> ...")
    print("ros2 run sim_data_collection analysis visualise <vehicle model nn (optional)> <db1> <db2> ...")
    print("ros2 run sim_data_collection analysis plot <input jsons> <json labels>")
    print("ros2 run sim_data_collection analysis evaluate <vehicle model neural network> <db1> <db2> ...")

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
        vehicle_models = [KinematicBicycle()]
        if db_paths[0].endswith(".pt"): ## first arg is vehicle model nn
            vehicle_model_path = db_paths[0]
            db_paths = db_paths[1:]
            vehicle_models.insert(0, NNVehicleModel(vehicle_model_path))
        logger.info(f"Analysis starting up. Visualising {len(db_paths)} databases.")
        visualise_data(db_paths, vehicle_models=vehicle_models)
    elif verb == "analyse":
        output_filename = sys.argv[2]
        db_paths = sys.argv[3:]
        logger.info(f"Analysis starting up. Analysing {len(db_paths)} databases.")
        analyse_data(output_filename, db_paths)
    elif verb == "plot":
        args = sys.argv[2:]
        num_args = len(args)
        assert num_args % 2 == 0, "Please enter as many labels as there are json input files. These will be used for adding a legend to the produced charts."
        input_filenames = args[:num_args//2]
        labels = args[num_args//2:]
        logger.info(f"Analysis starting up. Visualising data from {input_filenames}.")
        plot(input_filenames, labels, show=False)
    elif verb == "evaluate":
        model_path = sys.argv[2]
        db_paths = sys.argv[3:]
        logger.info(f"Analysis starting up. Evaluating {model_path} against KinematicBicycle using {len(db_paths)} databases.")
        evaluate(model_path, db_paths)
    else:
        print(f"Unrecognised verb '{verb}'")
        usage()
        sys.exit(1)
