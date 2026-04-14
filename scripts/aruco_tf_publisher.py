#!/usr/bin/env python3
"""
aruco_tf_publisher — ROS 2 node

Subscribes to /aruco/poses (geometry_msgs/msg/PoseArray).
When exactly 2 poses are present (pose[0] = sonar_tag, pose[1] = object_tag,
both expressed in the camera frame), computes the relative transform

    T_sonar_tag_object_tag = T_cam_sonar_tag^{-1} * T_cam_object_tag

and broadcasts it as a TF transform:
    parent frame : sonar_tag
    child  frame : object_tag

Rapid-rotation filter: if the angular change from the previous accepted
transform exceeds `max_rotation_jump_deg` (default 30°), the message is
dropped as an outlier and the previous transform is kept.

Usage
-----
    source /home/aki/auv_ws/install/setup.bash
    ros2 run sonar_camera_logger aruco_tf_publisher.py

    # Custom threshold (degrees):
    ros2 run sonar_camera_logger aruco_tf_publisher.py \
        --ros-args -p max_rotation_jump_deg:=20.0
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3×3 rotation matrix."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z),  s * (x * y - z * w),      s * (x * z + y * w)],
        [s * (x * y + z * w),      1 - s * (x * x + z * z),  s * (y * z - x * w)],
        [s * (x * z - y * w),      s * (y * z + x * w),      1 - s * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray):
    """3×3 rotation matrix → quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = float(np.copysign(x, R[2, 1] - R[1, 2]))
    y = float(np.copysign(y, R[0, 2] - R[2, 0]))
    z = float(np.copysign(z, R[1, 0] - R[0, 1]))
    return x, y, z, float(w)


def rotation_angle_between(R_prev: np.ndarray, R_curr: np.ndarray) -> float:
    """Return the geodesic angle (radians) between two rotation matrices."""
    R_diff = R_prev.T @ R_curr
    # trace of R_diff = 1 + 2*cos(angle)
    cos_angle = (np.trace(R_diff) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.acos(cos_angle)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__("aruco_tf_publisher")

        self.declare_parameter("max_rotation_jump_deg", 30.0)
        self._max_jump_rad = math.radians(
            self.get_parameter("max_rotation_jump_deg").value
        )

        self._tf_broadcaster = TransformBroadcaster(self)
        self._prev_R_rel: np.ndarray | None = None  # last accepted R_rel

        self.create_subscription(
            PoseArray,
            "/aruco/poses",
            self._poses_callback,
            10,
        )
        self.get_logger().info(
            f"aruco_tf_publisher ready — listening on /aruco/poses  "
            f"(max_rotation_jump={math.degrees(self._max_jump_rad):.1f}°)"
        )

    def _poses_callback(self, msg: PoseArray) -> None:
        if len(msg.poses) < 2:
            return  # need both sonar_tag and object_tag

        p0 = msg.poses[0]  # sonar_tag  in camera frame
        p1 = msg.poses[1]  # object_tag in camera frame

        # Rotation matrices from camera frame
        R0 = quat_to_matrix(p0.orientation.x, p0.orientation.y,
                             p0.orientation.z, p0.orientation.w)
        R1 = quat_to_matrix(p1.orientation.x, p1.orientation.y,
                             p1.orientation.z, p1.orientation.w)

        t0 = np.array([p0.position.x, p0.position.y, p0.position.z])
        t1 = np.array([p1.position.x, p1.position.y, p1.position.z])

        # T_sonar_tag_object_tag = T_cam_sonar_tag^{-1} * T_cam_object_tag
        R_rel = R0.T @ R1
        t_rel = R0.T @ (t1 - t0)

        # ── Rapid-rotation filter ──────────────────────────────────────────
        if self._prev_R_rel is not None:
            jump = rotation_angle_between(self._prev_R_rel, R_rel)
            if jump > self._max_jump_rad:
                self.get_logger().warn(
                    f"Rotation jump {math.degrees(jump):.1f}° > "
                    f"{math.degrees(self._max_jump_rad):.1f}° — dropping frame"
                )
                return

        self._prev_R_rel = R_rel

        qx, qy, qz, qw = matrix_to_quat(R_rel)

        ts = TransformStamped()
        ts.header.stamp = msg.header.stamp
        ts.header.frame_id = "sonar_tag"
        ts.child_frame_id = "object_tag"
        ts.transform.translation.x = float(t_rel[0])
        ts.transform.translation.y = float(t_rel[1])
        ts.transform.translation.z = float(t_rel[2])
        ts.transform.rotation.x = qx
        ts.transform.rotation.y = qy
        ts.transform.rotation.z = qz
        ts.transform.rotation.w = qw

        self._tf_broadcaster.sendTransform(ts)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
