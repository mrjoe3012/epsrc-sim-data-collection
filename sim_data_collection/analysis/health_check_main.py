import rclpy
import rclpy.logging as logging

def main():
    rclpy.init()
    try:
       logger = logging.get_logger("health_check")
       logger.info("Hello from health_check!")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
