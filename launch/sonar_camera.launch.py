from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("sonar_camera_logger")

    aruco_params = os.path.join(pkg, "config", "aruco_distance_node.yaml")
    tf_params = os.path.join(pkg, "config", "aruco_tf_publisher.yaml")

    return LaunchDescription(
        [
            # Sonar driver — publishes point clouds in sonar_link frame
            # and the static world → sonar_link identity TF.
            Node(
                package="sonar_camera_logger",
                executable="sonar_driver.py",
                name="sonar_driver",
                output="screen",
            ),
            # ArUco detection — detects markers, publishes poses + TF in
            # camera_optical_frame, using the Microsoft LifeCam HD-3000 calibration.
            Node(
                package="sonar_camera_logger",
                executable="aruco_distance_node",
                name="aruco_distance_node",
                parameters=[aruco_params],
                output="screen",
            ),
            # ArUco TF publisher — publishes the full experiment TF tree:
            #   sonar_link → sonar_aruco  (static, fill offsets in aruco_tf_publisher.yaml)
            #   sonar_aruco → floater_aruco  (dynamic, updated each frame)
            #   floater_aruco → object_center  (static, fill offsets in aruco_tf_publisher.yaml)
            Node(
                package="sonar_camera_logger",
                executable="aruco_tf_publisher.py",
                name="aruco_tf_publisher",
                parameters=[tf_params],
                output="screen",
            ),
            # Fusion node — time-synchronises sonar + camera and republishes
            # on /fusion/ topics.
            Node(
                package="sonar_camera_logger",
                executable="sonar_camera_fusion.py",
                name="sonar_camera_fusion",
                output="screen",
                parameters=[
                    {"slop": 0.1},
                    {"queue_size": 10},
                ],
            ),
        ]
    )
