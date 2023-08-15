from typing import List
from sim_data_collection.analysis.vehicle_model import VehicleModel
from sim_data_collection.analysis.dataset import Dataset
from sim_data_collection.analysis import analysis
from rclpy.serialization import deserialize_message
from ugrdv_msgs.msg import VCUStatus, DriveRequest
from eufs_msgs.msg import CarState
from scipy.spatial.transform import Rotation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time, copy

class VehicleModelState:
    def __init__(self, vehicle_model: VehicleModel):
        self._vehicle_model = vehicle_model
        self._car_state = None
        self._last_update = None
        self._last_reset = None
        self._reset_every = 150

class SimulationVisualiser:
    def __init__(self, db_paths: List[str], time_factor: float = 1.0,
                 vehicle_models: (List[VehicleModel] | None) = None):
        self._time_factor = 3.0
        self._db_paths = db_paths
        self._vehicle_models = [VehicleModelState(vm) for vm in vehicle_models]
        self._vehicle_model_colours = [
            "red",
            "orange",
            "yellow",
            "green",
            "pink"
        ]

    def _get_start_time(self, dataset: Dataset):
        start_time = dataset._connection.execute(
            "SELECT timestamp, data FROM ground_truth_state ORDER BY timestamp ASC"
        ).fetchone()[0] / 1e3
        return start_time

    def _get_next_messages(self, dataset: Dataset, millis: float):
        car_state = dataset._connection.execute(
            "SELECT timestamp, data FROM ground_truth_state \
                WHERE timestamp < ? ORDER BY timestamp DESC",
                (millis,)
        ).fetchone()
        vcu_status = dataset._connection.execute(
            "SELECT timestamp, data FROM vcu_status \
                WHERE timestamp < ? ORDER BY timestamp DESC",
                (millis,)
        ).fetchone()
        drive_request = dataset._connection.execute(
            "SELECT timestamp, data FROM drive_request \
                WHERE timestamp < ? ORDER BY timestamp DESC",
                (millis,)
        ).fetchone()
        if car_state is not None:
            car_state = deserialize_message(car_state[1], CarState)
        if vcu_status is not None:
            vcu_status = deserialize_message(vcu_status[1], VCUStatus)
        if drive_request is not None:
            drive_request = deserialize_message(drive_request[1], DriveRequest)
        return car_state, vcu_status, drive_request

    ####################
    # public interface #
    ####################

    def visualise_all(self):
        dataset = Dataset()
        figsize = (5,5)
        update_hz = 1000.0
        time_factor = self._time_factor
        update_interval = 1.0 / update_hz
        plot_percentage = True
        for db_path in self._db_paths:
            self._vehicle_models = [VehicleModelState(vm._vehicle_model) for vm in self._vehicle_models]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            try:
                dataset.open(db_path)
                track = analysis.Track.track_from_db_path(db_path)
                backwards_detector = analysis.BackwardsDetector(
                    track = track,
                    verbose = True
                )
                t = time.time()
                start_time = self._get_start_time(dataset)
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
                    latest_car_pose, latest_vcu_status, latest_drive_request = \
                        self._get_next_messages(dataset, sim_time)
                    if latest_car_pose is None or \
                        latest_vcu_status is None or \
                        latest_drive_request is None:
                        return

                    # plot the vehicle model if we're simulating one
                    if self._vehicle_models is not None:
                        for vm_idx, vm in enumerate(self._vehicle_models):
                            if vm._last_update is None: vm._last_update = sim_time
                            if vm._last_reset is None: vm._last_reset = sim_time
                            if vm._car_state is None or (i - vm._last_reset) >= vm._reset_every:
                                vm._last_reset = i
                                vm._car_state = track.transform_car_pose(copy.deepcopy(latest_car_pose))
                                vm._vehicle_model.update_state({
                                    'steering_angle' : latest_vcu_status.steering_angle,
                                    'wheel_speeds' : [
                                        latest_vcu_status.wheel_speeds.fl_speed,
                                        latest_vcu_status.wheel_speeds.fr_speed,
                                        latest_vcu_status.wheel_speeds.rl_speed,
                                        latest_vcu_status.wheel_speeds.rr_speed,
                                    ],
                                })
                            vm_dt = sim_time - vm._last_update
                            vm._last_update = sim_time
                            vm._vehicle_model.update_state({
                                'steering_angle_request' : latest_drive_request.steering_angle,
                                'torque_request' : latest_drive_request.axle_torque
                            })
                            dx_local, dy_local, dtheta = vm._vehicle_model(vm_dt/1e3)
                            vm_rot = Rotation.from_quat([
                                vm._car_state.pose.pose.orientation.x,
                                vm._car_state.pose.pose.orientation.y,
                                vm._car_state.pose.pose.orientation.z,
                                vm._car_state.pose.pose.orientation.w,
                            ])
                            heading = vm_rot.as_euler("XYZ")[2]
                            dx = dx_local * np.cos(heading) - dy_local * np.sin(heading)
                            dy = dx_local * np.sin(heading) + dy_local * np.cos(heading)
                            dtheta_quat = Rotation.from_euler("XYZ", (0.0, 0.0, dtheta))
                            vm_rot = dtheta_quat * vm_rot
                            vm._car_state.pose.pose.orientation.x = vm_rot.as_quat()[0]
                            vm._car_state.pose.pose.orientation.y = vm_rot.as_quat()[1]
                            vm._car_state.pose.pose.orientation.z = vm_rot.as_quat()[2]
                            vm._car_state.pose.pose.orientation.w = vm_rot.as_quat()[3]
                            vm._car_state.pose.pose.position.x += float(dx)
                            vm._car_state.pose.pose.position.y += float(dy)
                            line = analysis.Line.make_line_from_car_state(vm._car_state)
                            # plot the vehicle model's position as a line on the track
                            ax1.plot(
                                [line.sx, line.ex],
                                [line.sy, line.ey],
                                "-", linewidth=5, color=self._vehicle_model_colours[vm_idx]
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