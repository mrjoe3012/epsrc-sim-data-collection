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
import numpy as np
import fcntl, json, signal, copy

def visualise_data(db_paths: List[str],
                   time_factor=15.0,
                   vehicle_model: VehicleModel | None = None):
    dataset = Dataset()
    figsize = (5,5)
    update_hz = 1000.0
    time_factor = 3.0
    update_interval = 1.0 / update_hz
    plot_percentage = True
    for db_path in db_paths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        try:
            dataset.open(db_path)
            track = analysis.Track.track_from_db_path(db_path)
            backwards_detector = analysis.BackwardsDetector(
                track = track,
                verbose = True
            )
            t = time.time()
            start_time = dataset._connection.execute(
                "SELECT timestamp, data FROM ground_truth_state ORDER BY timestamp ASC"
            ).fetchone()[0] / 1e3
            completions = []
            vm_car_state = None # vehicle model's car state
            vm_last_update = None
            vm_reset_every = 150
            vm_last_reset = None

            def anim_callback(i):
                nonlocal t, completions, vm_car_state, vm_last_reset, vm_last_update
                ax1.cla()
                ax1.set_aspect("equal")
                # figure out elapsed time since we last
                # visualised
                dt = time.time() - t
                # use the time factor to figure out what
                # time we should be visualising
                sim_time = (start_time + time_factor*dt) * 1e3
                if vm_last_update is None: vm_last_update = sim_time
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
                latest_vcu_status = dataset._connection.execute(
                    "SELECT timestamp, data FROM vcu_status \
                        WHERE timestamp < ? ORDER BY timestamp DESC",
                        (sim_time,)
                ).fetchone()
                latest_drive_request = dataset._connection.execute(
                    "SELECT timestamp, data FROM drive_request \
                        WHERE timestamp < ? ORDER BY timestamp DESC",
                        (sim_time,)
                ).fetchone()
                if latest_car_pose is None or (vehicle_model is not None and (latest_vcu_status is None or latest_drive_request is None)):
                    return
                else:
                    latest_car_pose = deserialize_message(latest_car_pose[1], CarState)
                    latest_vcu_status = deserialize_message(latest_vcu_status[1], VCUStatus)
                    latest_drive_request = deserialize_message(latest_drive_request[1], DriveRequest)

                # plot the vehicle model if we're simulating one
                if vehicle_model is not None:
                    if vm_car_state is None or (i - vm_last_reset) >= vm_reset_every:
                        vm_last_reset = i
                        vm_car_state = track.transform_car_pose(copy.deepcopy(latest_car_pose))
                        vehicle_model.update_state({
                            'steering_angle' : latest_vcu_status.steering_angle,
                            'wheel_speeds' : [
                                latest_vcu_status.wheel_speeds.fl_speed,
                                latest_vcu_status.wheel_speeds.fr_speed,
                                latest_vcu_status.wheel_speeds.rl_speed,
                                latest_vcu_status.wheel_speeds.rr_speed,
                            ],
                        })
                    vm_dt = sim_time - vm_last_update
                    vm_last_update = sim_time
                    vehicle_model.update_state({
                        'steering_angle_request' : latest_drive_request.steering_angle,
                        'torque_request' : latest_drive_request.axle_torque
                    })
                    dx_local, dy_local, dtheta = vehicle_model(vm_dt/1e3)
                    vm_rot = Rotation.from_quat([
                        vm_car_state.pose.pose.orientation.x,
                        vm_car_state.pose.pose.orientation.y,
                        vm_car_state.pose.pose.orientation.z,
                        vm_car_state.pose.pose.orientation.w,
                    ])
                    heading = vm_rot.as_euler("XYZ")[2]
                    dx = dx_local * np.cos(heading) - dy_local * np.sin(heading)
                    dy = dx_local * np.sin(heading) + dy_local * np.cos(heading)
                    dtheta_quat = Rotation.from_euler("XYZ", (0.0, 0.0, dtheta))
                    vm_rot = dtheta_quat * vm_rot
                    vm_car_state.pose.pose.orientation.x = vm_rot.as_quat()[0]
                    vm_car_state.pose.pose.orientation.y = vm_rot.as_quat()[1]
                    vm_car_state.pose.pose.orientation.z = vm_rot.as_quat()[2]
                    vm_car_state.pose.pose.orientation.w = vm_rot.as_quat()[3]
                    vm_car_state.pose.pose.position.x += float(dx)
                    vm_car_state.pose.pose.position.y += float(dy)
                    line = analysis.Line.make_line_from_car_state(vm_car_state)
                    # plot the vehicle model's position as a line on the track
                    ax1.plot(
                        [line.sx, line.ex],
                        [line.sy, line.ey],
                        "-", linewidth=5, color="red"
                    )
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
                backwards_detector.add_completion(
                    completions[-1][0],
                    completions[-1][1]    
                )
                backwards_detector.is_violating()
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
        visualise_data(db_paths, vehicle_model=NNVehicleModel("/home/joe/Downloads/testmodel.pt"))
        # visualise_data(db_paths, vehicle_model=KinematicBicycle())
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
