#!/usr/bin/env python3
"""
aruco_tf_publisher — ROS 2 node

Publishes the TF tree for the wet-lab sonar experiment:

    world ──(static, identity)──> sonar_link          published by sonar_driver.py
    sonar_link ──(static)──> sonar_aruco               rigid offset, measured on experiment day
    sonar_aruco ──(dynamic)──> floater_aruco            computed each frame from camera detection
    floater_aruco ──(static)──> object_center           rigid offset, measured on experiment day

The dynamic transform is derived from /aruco/poses (PoseArray) and /aruco/ids
by matching markers to their configured IDs (sonar_aruco_id, floater_aruco_id).

Static offsets are loaded from ROS parameters — set them in aruco_tf_publisher.yaml.
Until measured they default to zero (identity), which still gives a valid TF tree.

Rapid-rotation filter: frames where the rotation changes by more than
max_rotation_jump_deg in one step are dropped as outliers.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, TransformStamped
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rpy_to_quat(roll_rad: float, pitch_rad: float, yaw_rad: float):
    """Roll-pitch-yaw (radians, ZYX extrinsic) → quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
    cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
    cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3×3 rotation matrix."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
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
    """Geodesic angle (radians) between two rotation matrices."""
    cos_angle = (np.trace(R_prev.T @ R_curr) - 1.0) / 2.0
    return math.acos(float(np.clip(cos_angle, -1.0, 1.0)))


def make_static_tf(parent: str, child: str,
                   x: float, y: float, z: float,
                   qx: float, qy: float, qz: float, qw: float) -> TransformStamped:
    ts = TransformStamped()
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(x)
    ts.transform.translation.y = float(y)
    ts.transform.translation.z = float(z)
    ts.transform.rotation.x = float(qx)
    ts.transform.rotation.y = float(qy)
    ts.transform.rotation.z = float(qz)
    ts.transform.rotation.w = float(qw)
    return ts


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ArucoTFPublisher(Node):
    def __init__(self):
        super().__init__("aruco_tf_publisher")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("max_rotation_jump_deg", 30.0)
        # Which ArUco IDs correspond to each physical marker
        self.declare_parameter("sonar_aruco_id", 0)
        self.declare_parameter("floater_aruco_id", 1)

        # Static offset: sonar_link → sonar_aruco  (measure on experiment day)
        self.declare_parameter("sonar_aruco_offset_x", 0.0)
        self.declare_parameter("sonar_aruco_offset_y", 0.0)
        self.declare_parameter("sonar_aruco_offset_z", 0.0)
        self.declare_parameter("sonar_aruco_offset_roll_deg", 0.0)
        self.declare_parameter("sonar_aruco_offset_pitch_deg", 0.0)
        self.declare_parameter("sonar_aruco_offset_yaw_deg", 0.0)

        # Static offset: floater_aruco → object_center  (measure on experiment day)
        self.declare_parameter("object_center_offset_x", 0.0)
        self.declare_parameter("object_center_offset_y", 0.0)
        self.declare_parameter("object_center_offset_z", 0.0)
        self.declare_parameter("object_center_offset_roll_deg", 0.0)
        self.declare_parameter("object_center_offset_pitch_deg", 0.0)
        self.declare_parameter("object_center_offset_yaw_deg", 0.0)

        self._max_jump_rad = math.radians(
            self.get_parameter("max_rotation_jump_deg").value
        )
        self._sonar_aruco_id: int = self.get_parameter("sonar_aruco_id").value
        self._floater_aruco_id: int = self.get_parameter("floater_aruco_id").value

        # ── Static TF broadcaster ──────────────────────────────────────────
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # ── Dynamic TF broadcaster ─────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)
        self._prev_R_rel: np.ndarray | None = None

        # ── Subscriptions ──────────────────────────────────────────────────
        # Cache latest IDs so the poses callback can look up by marker ID.
        self._latest_ids: list[int] | None = None
        self.create_subscription(Int32MultiArray, "/aruco/ids", self._ids_callback, 10)
        self.create_subscription(PoseArray, "/aruco/poses", self._poses_callback, 10)

        self.get_logger().info(
            f"aruco_tf_publisher ready "
            f"(sonar_aruco_id={self._sonar_aruco_id}, "
            f"floater_aruco_id={self._floater_aruco_id}, "
            f"max_rotation_jump={math.degrees(self._max_jump_rad):.1f}°)"
        )

    # ── Static TFs ────────────────────────────────────────────────────────────

    def _publish_static_transforms(self) -> None:
        def _rad(name: str) -> float:
            return math.radians(self.get_parameter(name).value)

        def _val(name: str) -> float:
            return float(self.get_parameter(name).value)

        # sonar_link → sonar_aruco
        qx, qy, qz, qw = rpy_to_quat(
            _rad("sonar_aruco_offset_roll_deg"),
            _rad("sonar_aruco_offset_pitch_deg"),
            _rad("sonar_aruco_offset_yaw_deg"),
        )
        sonar_aruco_tf = make_static_tf(
            "sonar_link", "sonar_aruco",
            _val("sonar_aruco_offset_x"),
            _val("sonar_aruco_offset_y"),
            _val("sonar_aruco_offset_z"),
            qx, qy, qz, qw,
        )

        # floater_aruco → object_center
        qx, qy, qz, qw = rpy_to_quat(
            _rad("object_center_offset_roll_deg"),
            _rad("object_center_offset_pitch_deg"),
            _rad("object_center_offset_yaw_deg"),
        )
        object_center_tf = make_static_tf(
            "floater_aruco", "object_center",
            _val("object_center_offset_x"),
            _val("object_center_offset_y"),
            _val("object_center_offset_z"),
            qx, qy, qz, qw,
        )

        self._static_broadcaster.sendTransform([sonar_aruco_tf, object_center_tf])
        self.get_logger().info(
            "Published static TFs: sonar_link → sonar_aruco, floater_aruco → object_center"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _ids_callback(self, msg: Int32MultiArray) -> None:
        self._latest_ids = list(msg.data)

    def _poses_callback(self, msg: PoseArray) -> None:
        if self._latest_ids is None or len(msg.poses) < 2:
            return

        ids = self._latest_ids
        if self._sonar_aruco_id not in ids or self._floater_aruco_id not in ids:
            return  # one or both markers not visible this frame

        sonar_idx = ids.index(self._sonar_aruco_id)
        floater_idx = ids.index(self._floater_aruco_id)

        p_sonar = msg.poses[sonar_idx]    # sonar ArUco in camera frame
        p_floater = msg.poses[floater_idx]  # floater ArUco in camera frame

        R_sonar = quat_to_matrix(
            p_sonar.orientation.x, p_sonar.orientation.y,
            p_sonar.orientation.z, p_sonar.orientation.w,
        )
        R_floater = quat_to_matrix(
            p_floater.orientation.x, p_floater.orientation.y,
            p_floater.orientation.z, p_floater.orientation.w,
        )
        t_sonar = np.array([p_sonar.position.x, p_sonar.position.y, p_sonar.position.z])
        t_floater = np.array([p_floater.position.x, p_floater.position.y, p_floater.position.z])

        # T_sonar_aruco_floater_aruco = T_cam_sonar_aruco^{-1} * T_cam_floater_aruco
        R_rel = R_sonar.T @ R_floater
        t_rel = R_sonar.T @ (t_floater - t_sonar)

        # ── Rapid-rotation filter ─────────────────────────────────────────
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
        ts.header.frame_id = "sonar_aruco"
        ts.child_frame_id = "floater_aruco"
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
