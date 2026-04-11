from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory("sonar_camera_logger"),
        "config",
        "camera_calibration_node.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="sonar_camera_logger",
                executable="camera_calibration_node",
                name="camera_calibration_node",
                parameters=[params_file],
                output="screen",
            ),
        ]
    )
