from sim_data_collection.analysis.dataset import Dataset
import sim_data_collection.utils as utils
import matplotlib.pyplot as plt

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

def get_stats(data: Dataset):
    stats = {
    }

    for msg_id in msg_ids:
        stats[msg_id] = {
            "duration": 0.0,
            "count" : 0
        }
        rows = data.get_msgs(msg_id).fetchall()
        rows.sort(key=lambda x: x[1])
        stats[msg_id]["count"] = len(rows)
        stats[msg_id]["duration"] = utils.millisToSeconds(rows[-1][1] - rows[0][1])

    return stats

def bar_plot(stats, key, ax):
    keys = list(stats.keys())
    durations = [x[key] for x in stats.values()]
    ax.bar(keys, durations)
