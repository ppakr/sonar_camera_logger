from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("sonar_camera_logger"),
        "config",
        "aruco_distance_node.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="sonar_camera_logger",
                executable="sonar_driver.py",
                name="sonar_driver",
                output="screen",
            ),
            Node(
                package="sonar_camera_logger",
                executable="aruco_distance_node",
                name="aruco_distance_node",
                parameters=[params_file],
                output="screen",
            ),
            Node(
                package="sonar_camera_logger",
                executable="sonar_camera_fusion.py",
                name="sonar_camera_fusion",
                output="screen",
                parameters=[
                    {"slop": 0.1},       # max time difference (s) to still pair messages
                    {"queue_size": 10},  # per-topic buffer depth
                ],
            ),
        ]
    )
