import rclpy
import rclpy.logging as logging
import matplotlib.pyplot as plt

import sim_data_collection.analysis.analysis as analysis
import sim_data_collection.utils as utils
from sim_data_collection.analysis.dataset import Dataset

import sys

def main():
    assert len(sys.argv) > 1, "Please provide a database path."
    db_paths = sys.argv[1:]
    rclpy.init()
    dataset = Dataset()
    try:
        logger = logging.get_logger("health_check")
        logger.info("Hello from health_check!")

        mean_durations = {
            k : 0.0 for k in analysis.msg_ids
        }
        mean_counts = {
            k : 0.0 for k in analysis.msg_ids
        }
        n = 0

        for db_path in db_paths:
            dataset.open(db_path)
            n += 1
            for msg_id in analysis.msg_ids:
                msgs = dataset.get_msgs(msg_id).fetchall()
                msgs.sort(key=lambda x: x[1])
                if len(msgs) == 0:
                    logger.error(f"{db_path}:{msg_id} EMPTY TABLE!")
                    continue
                duration = utils.millisToSeconds(msgs[-1][1] - msgs[0][1])
                count = len(msgs)
                mean_durations[msg_id] += (duration - mean_durations[msg_id]) / n
                mean_counts[msg_id] += (count - mean_counts[msg_id]) / n
            dataset.close()

        logger.info("Mean Counts:")
        for msg_id in analysis.msg_ids:
            logger.info(f"{msg_id} : {mean_counts[msg_id]}")
        logger.info("Mean Durations:")
        for msg_id in analysis.msg_ids:
            logger.info(f"{msg_id} : {mean_durations[msg_id]}")

        # calculate mean durations and counts for each message

        progress = 0.0

        # look for outliers
        for i,db_path in enumerate(db_paths):
            new_progress = 100.0 * ((i+1) / len(db_paths))
            if new_progress - progress >= 1.0:
                logger.info(f"PROGRESS: {new_progress}%")
            progress = new_progress
            dataset.open(db_path)
            for msg_id in analysis.msg_ids:
                msgs = dataset.get_msgs(msg_id).fetchall()
                duration = utils.millisToSeconds(msgs[-1][1] - msgs[0][1])
                count = len(msgs)
                duration_err = abs(duration - mean_durations[msg_id])
                count_err = abs(count - mean_counts[msg_id])
                if duration_err > 5.0:
                    logger.warn(f"Outlier: {db_path}:{msg_id}, duration: {duration}")
                if count_err > 5.0:
                    logger.warn(f"Outlier: {db_path}:{msg_id}, count: {count}")
            dataset.close()

    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
