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
            def cb(i):
                nonlocal t
                ax1.cla()
                # ax1.set_xlim((-150, 150))
                # ax1.set_ylim((-150, 150))
                ax1.set_aspect("equal")
                ax2.set_ylim((0.0, 2.0))
                dt = time.time() - t
                sim_time = (start_time + factor*dt) * 1e3

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
                    # [track.first_blue_line.sx, track.first_blue_line.ex], [track.first_blue_line.sy, track.first_blue_line.ey],
                    [track.nblue.sx, track.nblue.ex], [track.nblue.sy, track.nblue.ey],
                    "-", color="pink"
                )
                ax1.plot(
                    # [track.first_yellow_line.sx, track.first_yellow_line.ex], [track.first_yellow_line.sy, track.first_yellow_line.ey],
                    [track.nyellow.sx, track.nyellow.ex], [track.nyellow.sy, track.nyellow.ey],
                    "-", color="purple"
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
                ax1.plot(x, y, "o", color="red")
                completion = track.get_completion(
                    latest_car_pose
                )
                ax2.plot(sim_time / 1e3, completion, "o", color="black")


            anim = FuncAnimation(fig, cb, interval=100)
            plt.show()
        finally:
            dataset.close()