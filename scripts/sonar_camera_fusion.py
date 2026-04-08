#!/usr/bin/env python3
"""Fuse sonar point-cloud and camera ArUco data using approximate time sync.

Subscribes to:
  /sonar_3d/pointcloud_intensity  (sensor_msgs/PointCloud2)
  /aruco/image_debug              (sensor_msgs/Image)

Publishes synchronised pairs on:
  /fusion/pointcloud   (sensor_msgs/PointCloud2)  -- sonar cloud, stamp = common stamp
  /fusion/image        (sensor_msgs/Image)         -- camera frame, stamp = common stamp

The common stamp is the ROS wall-clock time at which the pair is matched,
which lets downstream nodes treat both messages as contemporaneous.

ApproximateTimeSynchronizer tolerates the residual jitter between the sonar
hardware clock (corrected by sonar_driver) and the camera wall-clock timer.
Tune `slop` (seconds) if pairs are not being formed -- raise it if the sonar
rate is low; lower it for tighter temporal accuracy.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import message_filters


class SonarCameraFusion(Node):

    def __init__(self):
        super().__init__("sonar_camera_fusion")

        self.declare_parameter("slop", 0.1)
        self.declare_parameter("queue_size", 10)

        slop = self.get_parameter("slop").get_parameter_value().double_value
        queue_size = self.get_parameter("queue_size").get_parameter_value().integer_value

        # ── Subscribers wired through message_filters ─────────────────────────
        self._sonar_sub = message_filters.Subscriber(
            self, PointCloud2, "/sonar_3d/pointcloud_intensity"
        )
        self._cam_sub = message_filters.Subscriber(
            self, Image, "/aruco/image_debug"
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sonar_sub, self._cam_sub],
            queue_size=queue_size,
            slop=slop,
        )
        self._sync.registerCallback(self._fused_callback)

        # ── Publishers ────────────────────────────────────────────────────────
        self._sonar_pub = self.create_publisher(PointCloud2, "/fusion/pointcloud", 10)
        self._cam_pub = self.create_publisher(Image, "/fusion/image", 10)

        self.get_logger().info(
            f"Sonar-camera fusion node ready  (slop={slop:.3f} s, queue={queue_size})"
        )

    def _fused_callback(self, sonar_msg: PointCloud2, cam_msg: Image) -> None:
        """Called when a temporally matched sonar+camera pair is found."""
        common_stamp = self.get_clock().now().to_msg()

        delta_s = abs(
            (sonar_msg.header.stamp.sec - cam_msg.header.stamp.sec)
            + (sonar_msg.header.stamp.nanosec - cam_msg.header.stamp.nanosec) * 1e-9
        )
        self.get_logger().debug(f"Fused pair | Δt = {delta_s*1000:.1f} ms")

        # Re-stamp both messages with the common wall-clock time so that any
        # downstream subscriber sees identical stamps on both topics.
        sonar_out = sonar_msg
        sonar_out.header.stamp = common_stamp

        cam_out = cam_msg
        cam_out.header.stamp = common_stamp

        self._sonar_pub.publish(sonar_out)
        self._cam_pub.publish(cam_out)


def main(args=None):
    rclpy.init(args=args)
    node = SonarCameraFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
