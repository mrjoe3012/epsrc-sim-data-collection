import rclpy, sys
import rclpy.logging as logging
import sim_data_collection.analysis.analysis as analysis
from sim_data_collection.analysis.dataset import Dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from rclpy.serialization import deserialize_message
from eufs_msgs.msg import CarState

def main():
    assert len(sys.argv) > 1, "Please provide at least one database path."
    logger = logging.get_logger("analaysis")
    db_paths = sys.argv[1:]
    logger.info(f"Analysis starting up. Analysing {len(db_paths)} databases.")
    dataset = Dataset()
    
    for db_path in db_paths:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5,5))
        print(f"checking {db_path}")
        dataset.open(db_path)
        track = analysis.Track.track_from_db_path(db_path)
        try:
            # analysis.intersection_check(
            #     dataset,
            #     track,
            #     visualize=True
            # )
            t = time.time()
            factor = 5.0
            start_time = dataset._connection.execute(
                "SELECT timestamp, data FROM ground_truth_state ORDER BY timestamp ASC"
            ).fetchone()[0] / 1e3
            completions = []
            def cb(i):
                nonlocal t, completions
                ax1.cla()
                # ax1.set_xlim((-150, 150))
                # ax1.set_ylim((-150, 150))
                ax1.set_aspect("equal")
                ax2.set_ylim((0.0, 2.0))
                dt = time.time() - t
                sim_time = (start_time + factor*dt) * 1e3
                ax1.plot(
                    [c[0] for c in track.centreline],
                    [c[1] for c in track.centreline],
                    "-", color="black", markersize=3
                )
                ax1.plot(
                    track.car_start[0], track.car_start[1],
                    "o", color="green"
                )
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

                latest_car_pose = dataset._connection.execute(
                    "SELECT timestamp, data FROM ground_truth_state \
                     WHERE timestamp < ? ORDER BY timestamp DESC",
                     (sim_time,)
                ).fetchone()
                if latest_car_pose is None: return
                else: latest_car_pose = deserialize_message(latest_car_pose[1], CarState)
                latest_car_pose = track.transform_car_pose(latest_car_pose)
                x, y = latest_car_pose.pose.pose.position.x, latest_car_pose.pose.pose.position.y
                ax1.plot(x, y, "o", color="black")
                ax1.plot(
                    [track.ncent.sx, track.ncent.ex], [track.ncent.sy, track.ncent.ey],
                    "-", color="red"
                )
                completion = track.get_completion(
                    latest_car_pose
                )
                completions.append((sim_time / 1e3, completion))
                ax2.plot(
                    [c[0] for c in completions],
                    [c[1] for c in completions],
                    "-", color="black")


            anim = FuncAnimation(fig, cb, interval=100)
            plt.show()
        finally:
            dataset.close()