from sim_data_collection.analysis.dataset import Dataset
import sim_data_collection.utils as utils
import matplotlib.pyplot as plt
import math

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

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def get_track_cones(csv_path):
    blue = [],
    yellow = []
    return blue, yellow

def check_for_violation(car_states, track_blue_cones, track_yellow_cones):
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    def intersection(ax, ay, bx, by, cx, cy, dx, dy):
        return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) \
            and ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)

    def get_car_line_segment(car_state):
        car_length = 1.5
        x, y = car_state.pose.position.x, car_state.pose.position.y
        _, _, heading = euler_from_quaternion(
            car_state.pose.orientation.x,
            car_state.pose.orientation.y,
            car_state.pose.orientation.z,
            car_state.pose.orientation.w,
        )
        # y is sin
        # x is cos
        sx = x - 0.5 * car_length * math.cos(heading)
        sy = y - 0.5 * car_length * math.sin(heading)
        ex = x + 0.5 * car_length * math.cos(heading)
        ey = y + 0.5 * car_length * math.sin(heading)
        return sx, sy, ex, ey

    def get_cones_line_segment(cone1, cone2):
        sx, sy = cone1[0], cone1[1]
        ex, ey = cone2[0], cone2[1]
        return sx, sy, ex, ey 

    def check_state_against_cones(car_state, cones):
        car_line_segment = get_car_line_segment(car_state)
        for i in range(len(cones)):
            j = (i + 1) % len(cones)
            cone1, cone2 = cones[i], cones[j]
            cone_line_segment = get_cones_line_segment(cone1, cone2)
            if intersection(
                car_line_segment[0], car_line_segment[1],
                car_line_segment[2], car_line_segment[3],
                cone_line_segment[0], cone_line_segment[1],
                cone_line_segment[2], cone_line_segment[3]
            ):
                return True

        return False

    for car_state in car_states:
        if check_state_against_cones(car_state, track_blue_cones) \
            or check_state_against_cones(track_yellow_cones):
            import matplotlib.pyplot as plt
            plt.plot(track_blue_cones, "o", color="blue")
            plt.plot(track_yellow_cones, "o", color="yellow")
            plt.plot((car_state.pose.position.x, car_state.pose.position.y), "o", color="black")
            ls = get_car_line_segment(car_state)
            plt.plot(
                [(ls[0], ls[1]), (ls[1], ls(2))],
                "-",
                color="black"
            )
            plt.show()
            return True
    
    return False
