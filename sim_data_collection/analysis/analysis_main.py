import rclpy, sys
import rclpy.logging as logging
import sim_data_collection.analysis.analysis as analysis
from sim_data_collection.analysis.dataset import Dataset

def main():
    assert len(sys.argv) > 1, "Please provide at least one database path."
    logger = logging.get_logger("analaysis")
    db_paths = sys.argv[1:]
    logger.info("Analysis starting up. Analysing {len(db_paths)} databases.")
    dataset = Dataset()
    
    for db_path in db_paths:
        dataset.open(db_path)
        try:
            analysis.intersection_check(
                dataset,
                track
            )
        finally:
            dataset.close()