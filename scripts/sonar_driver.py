#!/usr/bin/env python3
"""ROS 2 Python driver for the WaterLinked 3D Sonar 3D-15."""

import math
import select as select_module
from typing import List, Tuple

import numpy as np
import rclpy
import rclpy.executors
import wlsonar
import wlsonar.range_image_protocol as rip
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster


class SonarDriver(Node):
    """ROS 2 node that receives WaterLinked sonar data and publishes ROS messages."""

    def __init__(self):
        """Initialize the sonar driver node."""
        super().__init__("WaterLinked_3D_Sonar_Driver")

        self.sonar_ip = "192.168.194.96"
        self.frame_id = "sonar_link"

        # The sonar head is the world origin — publish a static identity TF
        # so every other frame can reference "world" in RViz / TF tree.
        _static_br = StaticTransformBroadcaster(self)
        _world_tf = TransformStamped()
        _world_tf.header.stamp = self.get_clock().now().to_msg()
        _world_tf.header.frame_id = "world"
        _world_tf.child_frame_id = self.frame_id
        _world_tf.transform.rotation.w = 1.0  # identity rotation
        _static_br.sendTransform(_world_tf)

        self.publisher_pointcloud = self.create_publisher(
            PointCloud2, "/sonar_3d/pointcloud", 10
        )
        self.publisher_intensity_img = self.create_publisher(
            Image, "/sonar_3d/strength_image", 10
        )
        self.publisher_pointcloud_intensity = self.create_publisher(
            PointCloud2, "/sonar_3d/pointcloud_intensity", 10
        )

        # These are set to None until _try_connect() succeeds.
        self.sonar = None
        self.sock = None

        # Try to connect immediately. If the sonar is not reachable, a retry
        # timer will keep trying every 5 seconds until it succeeds.
        self._connect_timer = None
        self._try_connect()
        if self.sonar is None:
            self._connect_timer = self.create_timer(5.0, self._try_connect)

        self.get_logger().info("Sonar driver initialized and running.")

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
        ]

        self.dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("intensity", np.uint8),
            ]
        )

        self.range_buffer = {}
        self.strength_buffer = {}
        self.max_buffer_size = 50

        # One-time offset to align the sonar hardware clock to the ROS wall clock.
        # Computed from the first packet received; applied to every timestamp thereafter.
        self._clock_offset_ns: int = 0
        self._offset_initialized: bool = False

        self.get_logger().info("Sonar driver setup complete, waiting for data...")

    def _try_connect(self) -> None:
        """Attempt to connect to the sonar over HTTP and open the UDP socket.

        Called once at startup and then every 5 seconds by a timer until the
        sonar is reachable. On success the timer is cancelled and the driver
        starts receiving data normally.
        """
        try:
            self.sonar = wlsonar.Sonar3D(self.sonar_ip)
            self.sonar.set_acoustics_enabled(True)
            self.sonar.set_udp_multicast()
            self.sock = wlsonar.open_sonar_udp_multicast_socket(iface_ip="192.168.194.90")
            self.get_logger().info(f"Connected to sonar at {self.sonar_ip}.")
            # Stop retrying now that we are connected.
            if self._connect_timer is not None:
                self._connect_timer.cancel()
                self._connect_timer = None
        except Exception as e:
            self.sonar = None
            self.sock = None
            self.get_logger().warn(
                f"Could not connect to sonar at {self.sonar_ip}: {e}. "
                "Retrying in 5 s..."
            )

    def _adjusted_stamp(self, hw_sec: int, hw_ns: int) -> Time:
        """Return a ROS Time aligned to the ROS wall clock.

        On the first call, compute the offset between the sonar hardware clock
        and the ROS wall clock and store it.  Every subsequent call applies the
        same offset so that relative timing between sonar frames is preserved.
        """
        if not self._offset_initialized:
            hw_ns_total = hw_sec * 10**9 + hw_ns
            ros_ns_total = self.get_clock().now().nanoseconds
            self._clock_offset_ns = ros_ns_total - hw_ns_total
            self._offset_initialized = True
            self.get_logger().info(
                f"Sonar clock offset calibrated: {self._clock_offset_ns / 1e9:.3f} s"
            )
        corrected_ns = hw_sec * 10**9 + hw_ns + self._clock_offset_ns
        return Time(sec=int(corrected_ns // 10**9), nanosec=int(corrected_ns % 10**9))

    def handle_packet(self, packet: bytes) -> None:
        """Parse one UDP packet and publish whatever it contains."""
        try:
            msg = rip.unpackb(packet)
        except (
            rip.UnknownProtobufTypeError,
            rip.BadIDError,
            rip.CRCMismatchError,
            rip.ExtraDataError,
            ValueError,
        ):
            return

        if isinstance(msg, rip.RangeImage):
            self._handle_range_image(msg)
        elif isinstance(msg, rip.BitmapImageGreyscale8):
            self._handle_strength_image(msg)
        else:
            raise RuntimeError(f"Unhandled message type: {type(msg)}")

        self._buffer_handler()

    def _handle_range_image(self, msg: rip.RangeImage) -> None:
        self.range_buffer[msg.header.sequence_id] = msg

        xyz_points = wlsonar.range_image_to_xyz(msg)
        header = Header()
        header.stamp = self._adjusted_stamp(
            msg.header.timestamp.seconds,
            msg.header.timestamp.nanos,
        )
        header.frame_id = self.frame_id
        self.publisher_pointcloud.publish(
            point_cloud2.create_cloud_xyz32(header, xyz_points)
        )

    def _handle_strength_image(self, msg: rip.BitmapImageGreyscale8) -> None:
        if msg.type != rip.BitmapImageType.SIGNAL_STRENGTH_IMAGE:
            self.get_logger().warn(
                "Skipping BitmapImageGreyscale8 of type other than"
                " SIGNAL_STRENGTH_IMAGE"
            )
            return

        self.strength_buffer[msg.header.sequence_id] = msg

        ros_image = Image()
        ros_image.header.stamp = self._adjusted_stamp(
            msg.header.timestamp.seconds,
            msg.header.timestamp.nanos,
        )
        ros_image.header.frame_id = self.frame_id
        ros_image.width = msg.width
        ros_image.height = msg.height
        ros_image.encoding = "mono8"
        ros_image.step = msg.width

        img_array = np.frombuffer(msg.image_pixel_data, dtype=np.uint8).reshape(
            msg.height, msg.width
        )
        ros_image.data = np.flipud(img_array).tobytes()

        self.publisher_intensity_img.publish(ros_image)

    @staticmethod
    def _range_image_to_xyzi(
        range_image: rip.RangeImage,
        strength_image: rip.BitmapImageGreyscale8,
    ) -> List[Tuple[float, float, float, float]]:
        """Convert a RangeImage + strength image pair to a list of (x,y,z,i) tuples."""
        voxels = []
        max_pixel_x = range_image.width - 1
        max_pixel_y = range_image.height - 1
        fov_h = math.radians(range_image.fov_horizontal)
        fov_v = math.radians(range_image.fov_vertical)

        for pixel_x in range(range_image.width):
            for pixel_y in range(range_image.height):
                pixel_idx = pixel_y * range_image.width + pixel_x
                pixel_value = range_image.image_pixel_data[pixel_idx]
                if pixel_value == 0:
                    continue
                pixel_intensity = strength_image.image_pixel_data[pixel_idx]
                distance_meters = pixel_value * range_image.image_pixel_scale
                yaw_rad = (pixel_x / max_pixel_x) * fov_h - fov_h / 2
                pitch_rad = (pixel_y / max_pixel_y) * fov_v - fov_v / 2
                voxels.append(
                    (
                        distance_meters * math.cos(pitch_rad) * math.cos(yaw_rad),
                        distance_meters * math.cos(pitch_rad) * math.sin(yaw_rad),
                        -distance_meters * math.sin(pitch_rad),
                        pixel_intensity,
                    )
                )
        return voxels

    def _buffer_handler(self) -> None:
        common_ids = set(self.range_buffer) & set(self.strength_buffer)
        for seq_id in sorted(common_ids):
            r_msg = self.range_buffer.pop(seq_id)
            s_msg = self.strength_buffer.pop(seq_id)
            self._create_intensity_pointcloud(r_msg, s_msg)

        for buf, name in [
            (self.range_buffer, "Range"),
            (self.strength_buffer, "Strength"),
        ]:
            while len(buf) > self.max_buffer_size:
                oldest = min(buf)
                buf.pop(oldest)
                self.get_logger().warn(
                    f"Buffer overflow: dropped orphaned {name}"
                    f" message (ID={oldest})"
                )

    def _create_intensity_pointcloud(
        self,
        range_img: rip.RangeImage,
        strength_img: rip.BitmapImageGreyscale8,
    ) -> None:
        points_xyzi = self._range_image_to_xyzi(range_img, strength_img)
        header = Header()
        header.stamp = self._adjusted_stamp(
            range_img.header.timestamp.seconds,
            range_img.header.timestamp.nanos,
        )
        header.frame_id = self.frame_id
        points_np = np.array(points_xyzi, dtype=self.dtype)
        self.publisher_pointcloud_intensity.publish(
            point_cloud2.create_cloud(
                header=header, fields=self.fields, points=points_np
            )
        )

    def cleanup(self) -> None:
        """Disable acoustics and close the UDP socket."""
        if self.sonar is not None:
            try:
                self.sonar.set_acoustics_enabled(False)
            except Exception:
                pass
        if self.sock is not None:
            self.sock.close()


def main(args=None):
    """Run the sonar driver node using a select()-based event loop."""
    rclpy.init(args=args)
    node = SonarDriver()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok():
            # If the sonar socket is not yet open (still waiting for connection),
            # just service ROS2 callbacks so the retry timer keeps firing.
            if node.sock is None:
                executor.spin_once(timeout_sec=0.01)
                continue

            # Block up to 10ms waiting for a UDP packet OR a ROS2 event.
            # select() asks the OS to watch sock for incoming data. When a
            # packet arrives the OS wakes us immediately; if nothing arrives
            # within 10ms we fall through so spin_once() keeps ROS2 alive
            # (timers, Ctrl-C handling, etc.).
            readable, _, _ = select_module.select([node.sock], [], [], 0.01)

            if readable:
                # recvfrom() is guaranteed non-blocking here because select()
                # already confirmed data is available on the socket.
                packet, _ = node.sock.recvfrom(wlsonar.UDP_MAX_DATAGRAM_SIZE)
                node.handle_packet(packet)

            # Let ROS2 dispatch any pending callbacks (timeout=0 → never blocks).
            executor.spin_once(timeout_sec=0)

    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
