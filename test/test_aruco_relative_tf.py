#!/usr/bin/env python3
"""
test_aruco_relative_tf.py — re-run ArUco detection on a recorded bag's
``/aruco/image_raw`` topic and publish the relative TF between two markers.

The detection pipeline mirrors ``src/aruco_detection.cpp``:
  * DICT_4X4_50
  * detector parameters loaded from ``config/aruco_detector_parameters.yaml``
  * camera intrinsics loaded from ``config/microsoft_livecam_hd3000.yaml``
  * solvePnP with the same 4-corner object-point layout

Marker ID 1 (sonar mount) is the reference (parent) frame; for every other
detected marker N a TF ``aruco_marker_1 -> aruco_marker_N`` is broadcast and
also printed to stdout. Final summary reports per-pair frame counts and the
mean / std-dev translation across the whole bag.

Usage
-----
    source ~/auv_ws/install/setup.bash
    python3 test/test_aruco_relative_tf.py \\
        --bag /media/aki/2C76C6780AEDB4DB/wl_wetlab_apr22_raw/1_apr22_static_brick

Optional flags
--------------
    --reference-id    parent marker ID (default 1)
    --marker-size     marker side length in metres (default 0.16,
                      matches config/aruco_distance_node.yaml)
    --image-topic     image topic in the bag (default /aruco/image_raw)
    --print-only      skip ROS init and TF broadcasting
    --max-frames N    stop after processing N image frames
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import rclpy
import rosbag2_py
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster


CONFIG_DIR = Path("/home/aki/auv_ws/src/sonar_camera_logger/config")
DEFAULT_CALIB = CONFIG_DIR / "microsoft_livecam_hd3000.yaml"
DEFAULT_DETECTOR_PARAMS = CONFIG_DIR / "aruco_detector_parameters.yaml"
DEFAULT_MARKER_SIZE = 0.16  # matches aruco_distance_node.yaml


def load_calibration(path: Path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file: {path}")
    K = fs.getNode("cameraMatrix").mat()
    D = fs.getNode("distCoeffs").mat()
    fs.release()
    if K is None or D is None:
        raise RuntimeError(
            f"Calibration file missing cameraMatrix or distCoeffs: {path}"
        )
    return K, D


def load_detector_params(path: Path):
    params = cv2.aruco.DetectorParameters()
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        params.readDetectorParameters(fs.root())
        fs.release()
    else:
        print(f"[warn] detector params file not found, using defaults: {path}")
    return params


def rotation_to_quat(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z). Same convention as the C++ source."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = float(np.copysign(x, R[2, 1] - R[1, 2]))
    y = float(np.copysign(y, R[0, 2] - R[2, 0]))
    z = float(np.copysign(z, R[1, 0] - R[0, 1]))
    return float(w), x, y, z


def rot_to_rpy_deg(R):
    pitch = np.arcsin(-R[2, 0])
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image into a BGR ndarray without cv_bridge.

    Avoids the NumPy 1.x/2.x ABI mismatch in the prebuilt ROS Jazzy
    cv_bridge module on this machine.
    """
    enc = msg.encoding
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if enc == "bgr8":
        return arr
    if enc == "rgb8":
        return arr[..., ::-1].copy()
    if enc in ("mono8", "8UC1"):
        return cv2.cvtColor(arr.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {enc!r}")


def open_bag_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bag", required=True, type=Path,
                        help="Path to the rosbag2 / mcap directory")
    parser.add_argument("--reference-id", type=int, default=1,
                        help="Marker ID used as the parent TF frame (default 1)")
    parser.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
                        help=f"Marker side length in metres (default {DEFAULT_MARKER_SIZE})")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--detector-params", type=Path,
                        default=DEFAULT_DETECTOR_PARAMS)
    parser.add_argument("--image-topic", default="/aruco/image_raw")
    parser.add_argument("--print-only", action="store_true",
                        help="Skip rclpy init and TF broadcast (useful in CI)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if not args.bag.exists():
        sys.exit(f"Bag not found: {args.bag}")

    K, D = load_calibration(args.calibration)
    print(f"[calib] cameraMatrix:\n{K}")
    print(f"[calib] distCoeffs:  {D.ravel()}")

    detector_params = load_detector_params(args.detector_params)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    half = float(args.marker_size) / 2.0
    obj_pts = np.array(
        [
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )

    # ROS 2 setup (skipped in --print-only)
    node = None
    tf_broadcaster = None
    if not args.print_only:
        rclpy.init()
        node = rclpy.create_node("aruco_relative_tf_test")
        tf_broadcaster = TransformBroadcaster(node)

    reader = open_bag_reader(args.bag)

    n_total = 0
    n_with_ref = 0
    rel_translations = {}  # other_id -> list of np.array(3,)

    while reader.has_next():
        topic, raw_data, _t_ns = reader.read_next()
        if topic != args.image_topic:
            continue
        n_total += 1
        if args.max_frames is not None and n_total > args.max_frames:
            n_total -= 1
            break

        msg = deserialize_message(raw_data, Image)
        try:
            frame = imgmsg_to_bgr(msg)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] image decode failed on frame {n_total}: {e}")
            continue

        corners, ids, _rejected = detector.detectMarkers(frame)
        if ids is None or len(ids) == 0:
            continue
        ids = ids.flatten().tolist()

        # solvePnP per detected marker, same object-point layout as the C++ node.
        poses = {}
        for c, mid in zip(corners, ids):
            ok, rvec, tvec = cv2.solvePnP(obj_pts, c[0], K, D)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            poses[int(mid)] = (R, tvec.flatten())

        ref_id = args.reference_id
        if ref_id not in poses:
            continue
        n_with_ref += 1

        R_ref, t_ref = poses[ref_id]
        R_ref_T = R_ref.T

        for mid, (R_o, t_o) in poses.items():
            if mid == ref_id:
                continue
            R_rel = R_ref_T @ R_o
            t_rel = R_ref_T @ (t_o - t_ref)

            rel_translations.setdefault(mid, []).append(t_rel.copy())

            qw, qx, qy, qz = rotation_to_quat(R_rel)
            rpy = rot_to_rpy_deg(R_rel)
            dist = float(np.linalg.norm(t_rel))
            print(
                f"frame {n_total:4d}  ID{ref_id}->ID{mid}  "
                f"t=[{t_rel[0]:+.3f}, {t_rel[1]:+.3f}, {t_rel[2]:+.3f}] m  "
                f"|t|={dist:.3f}  "
                f"RPY=[{rpy[0]:+6.1f}, {rpy[1]:+6.1f}, {rpy[2]:+6.1f}] deg"
            )

            if tf_broadcaster is not None:
                tf = TransformStamped()
                tf.header.stamp = node.get_clock().now().to_msg()
                tf.header.frame_id = f"aruco_marker_{ref_id}"
                tf.child_frame_id = f"aruco_marker_{mid}"
                tf.transform.translation.x = float(t_rel[0])
                tf.transform.translation.y = float(t_rel[1])
                tf.transform.translation.z = float(t_rel[2])
                tf.transform.rotation.w = qw
                tf.transform.rotation.x = qx
                tf.transform.rotation.y = qy
                tf.transform.rotation.z = qz
                tf_broadcaster.sendTransform(tf)
                rclpy.spin_once(node, timeout_sec=0.0)

    print("\n=== Summary ===")
    print(f"image frames processed             : {n_total}")
    print(f"frames with reference ID {args.reference_id}         : {n_with_ref}")
    if not rel_translations:
        print("no reference->other pairs detected — check the bag and marker IDs")
        rc = 1
    else:
        rc = 0
        for mid, samples in sorted(rel_translations.items()):
            arr = np.stack(samples)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            print(
                f"ID{args.reference_id}->ID{mid}: n={len(samples):4d}  "
                f"mean t=[{mean[0]:+.3f}, {mean[1]:+.3f}, {mean[2]:+.3f}] m  "
                f"|mean|={np.linalg.norm(mean):.3f}  "
                f"std=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}] m"
            )

    if node is not None:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
