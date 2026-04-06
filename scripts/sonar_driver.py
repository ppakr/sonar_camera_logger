#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField
import numpy as np
import wlsonar
import wlsonar.range_image_protocol as rip
import threading
import socket
import struct
from typing import List, Tuple
import math

class SonarDriver(Node):
    def __init__(self):
        super().__init__('WaterLinked_3D_Sonar_Driver')

        # initialize values and publisher
        self.sonar_ip = "192.168.1.96"
        self.frame_id = "sonar_link"  # Important for RViz
        self.publisher_pointcloud = self.create_publisher(PointCloud2, '/blueye/sonar_3d/pointcloud', 10)
        self.publisher_intensity_img = self.create_publisher(Image, '/blueye/sonar_3d/strength_image', 10)
        self.publisher_pointcloud_intensity = self.create_publisher(PointCloud2, 'blueye/sonar_3d/pointcloud_intensity', 10)

        # connect, enable acoustics, configure to send images over UDP multicast
        self.sonar = wlsonar.Sonar3D(self.sonar_ip)
        self.sonar.set_acoustics_enabled(True)      # So remember that the start up of this node turns on the sonar
        self.sonar.set_udp_multicast()

        # receive UDP packets, parse them into protobuf, and extract voxels
        self.ethernet_connection = False
        if self.ethernet_connection:
            self.sock = self._open_socket()
        else:
            # self.sock = wlsonar.open_sonar_udp_multicast_socket(iface_ip="192.168.194.100") # has bug on line 30 of _udp_helper, should bind differently
            self.sock = wlsonar.open_sonar_udp_multicast_socket()

        # set up thread for socket listener
        self._thread = threading.Thread(target=self._socket_loop, daemon=True)
        self._thread.start()

        # keep track of sequence ID's of range and greybit image
        self.range_img = None
        self.strength_img = None

        self.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.UINT8,   count=1),
        ]

        self.dtype = np.dtype([
            ('x',         np.float32),
            ('y',         np.float32),
            ('z',         np.float32),
            ('intensity', np.uint8),
        ])

        self.printed_first = False
        # buffers to send out pointcloud with intensity in case of UDP dropped packets
        self.range_buffer = {}
        self.strength_buffer = {}
        self.max_buffer_size = 50

    def _socket_loop(self):
        while True:
            # Try to receive a packet of information over the socket
            try:
                packet, addr = self.sock.recvfrom(wlsonar.UDP_MAX_DATAGRAM_SIZE)
            except OSError:
                # Socket was closed by cleanup()
                break

            # Try to unpack the received packet
            try:
                msg = rip.unpackb(packet)
            except rip.UnknownProtobufTypeError:
                continue

            if isinstance(msg, rip.RangeImage):
                self.range_img = msg

                self.range_buffer[msg.header.sequence_id] = msg

                xyz_points = wlsonar.range_image_to_xyz(msg)

                header = Header()
                header.stamp = Time(
                    sec=msg.header.timestamp.seconds,
                    nanosec=msg.header.timestamp.nanos
                )
                header.frame_id = self.frame_id

                pc2_msg = point_cloud2.create_cloud_xyz32(header, xyz_points)
                self.publisher_pointcloud.publish(pc2_msg)

            elif isinstance(msg, rip.BitmapImageGreyscale8):
                self.strength_img = msg

                self.strength_buffer[msg.header.sequence_id] = msg

                # They have a shaded and strength image, we only want the strength image
                if msg.type != rip.BitmapImageType.SIGNAL_STRENGTH_IMAGE:
                    print(
                        "WARNING: Skipping BitmapImageGreyscale8 message of "
                        + "type other than SIGNAL_STRENGTH_IMAGE"
                    )
                    continue

                ros_image = Image()
                ros_image.header.stamp = Time(
                    sec=msg.header.timestamp.seconds,
                    nanosec=msg.header.timestamp.nanos
                )
                ros_image.header.frame_id = self.frame_id
                ros_image.width = msg.width
                ros_image.height = msg.height
                ros_image.encoding = "mono8"  # 8-bit greyscale
                ros_image.step = msg.width    # bytes per row

                # Flip the array because of difference axes image
                img_array = np.frombuffer(msg.image_pixel_data, dtype=np.uint8).reshape(msg.height, msg.width)
                flipped = np.flipud(img_array)  # flip top-bottom
                ros_image.data = flipped.tobytes()

                self.publisher_intensity_img.publish(ros_image)

            else:
                raise RuntimeError(f"Unhandled message type: {type(msg)}")

            self._buffer_handler()


    @staticmethod
    def _range_image_to_xyzi(range_image: rip.RangeImage, strength_image: rip.BitmapImageGreyscale8) -> List[Tuple[float, float, float, float]]:
        """range_image_to_xyz returns voxels of RangeImage as x,y,z in meters, with the intensity i.

        Args:
            range_image: the RangeImage and the BitmapImageGreyscale8

        Returns:
            List of (x, y, z, i) tuples in meters and intensity is some crazy log scale
        """
        voxels = []

        max_pixel_x = range_image.width - 1
        max_pixel_y = range_image.height - 1

        fov_h = math.radians(range_image.fov_horizontal)
        fov_v = math.radians(range_image.fov_vertical)

        for pixel_x in range(range_image.width):
            for pixel_y in range(range_image.height):
                pixel_idx = pixel_y * range_image.width + pixel_x
                pixel_value = range_image.image_pixel_data[pixel_idx]
                pixel_intesity = strength_image.image_pixel_data[pixel_idx]

                if pixel_value == 0:
                    # No data for this pixel
                    continue

                distance_meters = pixel_value * range_image.image_pixel_scale
                yaw_rad = (pixel_x / max_pixel_x) * fov_h - fov_h / 2
                pitch_rad = (pixel_y / max_pixel_y) * fov_v - fov_v / 2

                x = distance_meters * math.cos(pitch_rad) * math.cos(yaw_rad)
                y = distance_meters * math.cos(pitch_rad) * math.sin(yaw_rad)
                z = -distance_meters * math.sin(pitch_rad)
                i = pixel_intesity
                voxels.append((x, y, z, i))

        return voxels

    def _buffer_handler(self):
        # Find IDs that exist in both "buckets"
        common_ids = set(self.range_buffer.keys()) & set(self.strength_buffer.keys())

        # Process them in order (Ping 1, 2, 3...)
        for seq_id in sorted(common_ids):
            # Pull the matching pair out
            r_msg = self.range_buffer.pop(seq_id)
            s_msg = self.strength_buffer.pop(seq_id)

            # Pass the verified pair to your creation function
            self._create_intensity_pointcloud(range_img=r_msg, strength_img=s_msg)

        # Keep memory lean: Remove oldest orphans if buffer > max_buffer_size
        for buffer_obj, name in [(self.range_buffer, "Range"), (self.strength_buffer, "Strength")]:
            while len(buffer_obj) > self.max_buffer_size:
                oldest = min(buffer_obj.keys())
                buffer_obj.pop(oldest)
                self.get_logger().warn(
                    f"Buffer Overflow: Dropped orphaned {name} message (ID: {oldest}). "
                    "Processing might be too slow!"
                )

    def _create_intensity_pointcloud(self, range_img: rip.RangeImage, strength_img: rip.BitmapImageGreyscale8):
        points_xyzi = self._range_image_to_xyzi(range_image=range_img, strength_image=strength_img)

        header = Header()
        header.stamp = Time(
            sec=range_img.header.timestamp.seconds,
            nanosec=range_img.header.timestamp.nanos
        )
        header.frame_id = self.frame_id

        points_xyzi_np = np.array(points_xyzi, dtype=self.dtype)
        point_cloud_msg_intensity = point_cloud2.create_cloud(header=header, fields=self.fields, points=points_xyzi_np)
        self.publisher_pointcloud_intensity.publish(point_cloud_msg_intensity)

    def cleanup(self):
        self.sonar.set_acoustics_enabled(False) # if the node is shutdown turn off acoustics
        self.sock.close()
        self._thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = SonarDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
