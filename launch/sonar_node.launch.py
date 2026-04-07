from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="sonar_camera_logger",
                executable="sonar_driver.py",
                name="sonar_driver",
                output="screen",
            ),
        ]
    )
