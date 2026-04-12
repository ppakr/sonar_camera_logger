#!/usr/bin/env python3
"""
Build corrected MCAP bags from original bags + corrected camera calibration.

For each original bag this script:
  1. Copies /sonar_3d/pointcloud_intensity and /sonar_3d/strength_image verbatim.
  2. Re-reads /aruco/image_debug frames and re-runs ArUco detection with the
     corrected camera calibration.
  3. Writes the corrected topics:
       /aruco/image_debug   — raw camera frame (no overlay drawn on top)
       /aruco/poses         — full 6-DOF PoseArray with quaternions, matching
                              the format produced by aruco_detection.cpp
       /aruco/distances     — N×N symmetric Float64MultiArray, same layout as
                              aruco_detection.cpp

NOTE: The stored /aruco/image_debug frames were originally recorded with ArUco
overlays already baked in by aruco_detection.cpp.  Those pixels cannot be
removed, but this script does NOT add a second layer of new-detection overlay.

Usage (source workspace first):
    source /home/aki/auv_ws/install/setup.bash
    python3 build_corrected_bags.py [--help]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

SONAR_TOPICS = {"/sonar_3d/pointcloud_intensity", "/sonar_3d/strength_image"}
ARUCO_IMAGE_TOPIC = "/aruco/image_debug"
ARUCO_POSES_TOPIC = "/aruco/poses"
ARUCO_DISTANCES_TOPIC = "/aruco/distances"
ARUCO_IDS_TOPIC = "/aruco/ids"

# Topics written to every output bag (order determines create_topic calls)
OUTPUT_TOPICS = [
    ("/sonar_3d/pointcloud_intensity", "sensor_msgs/msg/PointCloud2"),
    ("/sonar_3d/strength_image",       "sensor_msgs/msg/Image"),
    ("/aruco/image_debug",             "sensor_msgs/msg/Image"),
    ("/aruco/poses",                   "geometry_msgs/msg/PoseArray"),
    ("/aruco/distances",               "std_msgs/msg/Float64MultiArray"),
    ("/aruco/ids",                     "std_msgs/msg/Int32MultiArray"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_calibration(yaml_path: str):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file: {yaml_path}")
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("distCoeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def rvec_to_quaternion(rvec: np.ndarray):
    """Rodrigues vector → quaternion [w, x, y, z].

    Mirrors the rotMatToQuat() implementation in aruco_detection.cpp so that
    the output is numerically identical to what the original node would have
    produced with the corrected calibration.
    """
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = float(np.copysign(x, R[2, 1] - R[1, 2]))
    y = float(np.copysign(y, R[0, 2] - R[2, 0]))
    z = float(np.copysign(z, R[1, 0] - R[0, 1]))
    return float(w), x, y, z


def image_msg_to_cv2(msg):
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    channels = len(msg.data) // (msg.height * msg.width)
    return arr.reshape(msg.height, msg.width, channels)


# ── Per-bag processing ────────────────────────────────────────────────────────

def build_corrected_bag(
    bag_path: str,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size: float,
    output_bag_dir: str,
    camera_frame: str,
):
    """Convert one bag.  Returns (total_aruco_frames, frames_with_detections)."""
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Pose, PoseArray
    from std_msgs.msg import (
        Float64MultiArray,
        Int32MultiArray,
        MultiArrayDimension,
        MultiArrayLayout,
    )

    # solvePnP object points — same layout as aruco_detection.cpp
    half = marker_size / 2.0
    obj_pts = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    ).reshape(4, 1, 3)

    # ── Reader ────────────────────────────────────────────────────────────────
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(
        rosbag2_py.StorageFilter(topics=list(SONAR_TOPICS) + [ARUCO_IMAGE_TOPIC])
    )

    # ── Writer ────────────────────────────────────────────────────────────────
    # Do NOT mkdir here — rosbag2_py creates the bag directory itself and
    # raises if it already exists.
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_bag_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for idx, (topic_name, topic_type) in enumerate(OUTPUT_TOPICS):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=idx,
                name=topic_name,
                type=topic_type,
                serialization_format="cdr",
            )
        )

    total_frames = 0
    detect_frames = 0

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        # ── Sonar: copy raw CDR bytes, no deserialization needed ──────────────
        if topic in SONAR_TOPICS:
            writer.write(topic, data, t_ns)
            continue

        # ── ArUco image: re-detect and write corrected messages ───────────────
        if topic != ARUCO_IMAGE_TOPIC:
            continue

        msg = deserialize_msg(data, Image)
        frame = image_msg_to_cv2(msg)
        total_frames += 1

        sec = int(t_ns // 1_000_000_000)
        nanosec = int(t_ns % 1_000_000_000)

        # Write raw frame unchanged — no new overlay added
        writer.write(ARUCO_IMAGE_TOPIC, data, t_ns)

        corners, ids, _ = detector.detectMarkers(frame)

        # Empty detections: write empty messages to keep topics time-aligned
        if ids is None or len(ids) == 0:
            pose_msg = PoseArray()
            pose_msg.header.stamp.sec     = sec
            pose_msg.header.stamp.nanosec = nanosec
            pose_msg.header.frame_id      = camera_frame
            writer.write(ARUCO_POSES_TOPIC,     serialize_message(pose_msg), t_ns)
            writer.write(ARUCO_DISTANCES_TOPIC, serialize_message(Float64MultiArray()), t_ns)
            writer.write(ARUCO_IDS_TOPIC,       serialize_message(Int32MultiArray()), t_ns)
            continue

        detect_frames += 1
        ids_flat = ids.flatten().tolist()
        N = len(ids_flat)

        tvecs, rvecs = [], []
        for i in range(N):
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, corners[i], camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            tvecs.append(tvec.flatten() if ok else np.zeros(3))
            rvecs.append(rvec.flatten() if ok else np.zeros(3))

        # ── PoseArray ─────────────────────────────────────────────────────────
        pose_msg = PoseArray()
        pose_msg.header.stamp.sec     = sec
        pose_msg.header.stamp.nanosec = nanosec
        pose_msg.header.frame_id      = camera_frame
        for i in range(N):
            w, x, y, z = rvec_to_quaternion(rvecs[i])
            pose = Pose()
            pose.position.x    = float(tvecs[i][0])
            pose.position.y    = float(tvecs[i][1])
            pose.position.z    = float(tvecs[i][2])
            pose.orientation.w = w
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            pose_msg.poses.append(pose)
        writer.write(ARUCO_POSES_TOPIC, serialize_message(pose_msg), t_ns)

        # ── Float64MultiArray: N×N symmetric distance matrix ──────────────────
        # Layout matches aruco_detection.cpp: data[i*N + j] = metres(i, j)
        dist_data = [0.0] * (N * N)
        for i in range(N):
            for j in range(i + 1, N):
                d = float(np.linalg.norm(tvecs[i] - tvecs[j]))
                dist_data[i * N + j] = d
                dist_data[j * N + i] = d
        dist_msg = Float64MultiArray()
        dist_msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="marker_i", size=N, stride=N * N),
                MultiArrayDimension(label="marker_j", size=N, stride=N),
            ],
            data_offset=0,
        )
        dist_msg.data = dist_data
        writer.write(ARUCO_DISTANCES_TOPIC, serialize_message(dist_msg), t_ns)

        # ── Int32MultiArray: marker IDs in the same order as PoseArray ────────
        ids_msg = Int32MultiArray()
        ids_msg.data = [int(i) for i in ids_flat]
        writer.write(ARUCO_IDS_TOPIC, serialize_message(ids_msg), t_ns)

    return total_frames, detect_frames


# rosbag2_py does not auto-import rclpy's deserialize — pull it lazily so
# the script can import without a full ROS init.
def deserialize_msg(data, msg_type):
    from rclpy.serialization import deserialize_message
    return deserialize_message(data, msg_type)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build corrected MCAP bags: sonar topics copied verbatim, "
            "ArUco re-detected with corrected calibration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9",
        help="Directory containing the original rosbag2 sub-folders.",
    )
    parser.add_argument(
        "--calibration",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/calibration_parameters.yaml"
        ),
        help="OpenCV FileStorage YAML with cameraMatrix and distCoeffs.",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_corrected_bags",
        help="Directory where corrected bag sub-folders are written.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=0.15,
        help="Physical side length of one ArUco marker in metres.",
    )
    parser.add_argument(
        "--detector_params",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/aruco_detector_parameters.yaml"
        ),
        help="ArUco DetectorParameters YAML (OpenCV defaults used if missing).",
    )
    parser.add_argument(
        "--camera_frame",
        default="camera",
        help="frame_id written into PoseArray and Image headers.",
    )
    parser.add_argument(
        "--bag",
        default=None,
        help="Process only this bag name (e.g. rosbag2_2026_04_09-14_34_35). "
             "Omit to process all bags.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    print("Calibration loaded:")
    print(f"  fx={camera_matrix[0, 0]:.4f}  fy={camera_matrix[1, 1]:.4f}")
    print(f"  cx={camera_matrix[0, 2]:.4f}  cy={camera_matrix[1, 2]:.4f}")
    print(f"  dist={dist_coeffs.flatten().tolist()}")

    # ── Detector ──────────────────────────────────────────────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det_params = cv2.aruco.DetectorParameters()
    fs_p = cv2.FileStorage(args.detector_params, cv2.FILE_STORAGE_READ)
    if fs_p.isOpened():
        det_params.readDetectorParameters(fs_p.root())
        fs_p.release()
        print(f"Detector params: {args.detector_params}")
    else:
        print("Detector params file not found — using OpenCV defaults.")
    detector = cv2.aruco.ArucoDetector(dictionary, det_params)

    # ── Bag list ──────────────────────────────────────────────────────────────
    all_dirs = sorted(
        p for p in Path(args.bags_dir).iterdir()
        if p.is_dir() and (p / "metadata.yaml").exists()
    )
    if not all_dirs:
        print(f"No bags found in {args.bags_dir}", file=sys.stderr)
        sys.exit(1)

    if args.bag:
        bag_dirs = [p for p in all_dirs if p.name == args.bag]
        if not bag_dirs:
            print(f"Bag '{args.bag}' not found in {args.bags_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        bag_dirs = all_dirs

    print(f"\nProcessing {len(bag_dirs)} of {len(all_dirs)} bag(s)")
    print(f"Output dir: {args.output_dir}\n")

    # ── Process ───────────────────────────────────────────────────────────────
    for bag_dir in bag_dirs:
        name = bag_dir.name
        out_dir = os.path.join(args.output_dir, name)
        print(f"[{name}]", flush=True)

        if os.path.exists(out_dir):
            print(f"  already exists — skipping (remove directory to reprocess)\n")
            continue

        try:
            total, detected = build_corrected_bag(
                str(bag_dir),
                detector,
                camera_matrix,
                dist_coeffs,
                args.marker_size,
                out_dir,
                args.camera_frame,
            )
            rate = 100.0 * detected / max(total, 1)
            print(f"  {total} aruco frames | {detected} with detections ({rate:.1f}%) → {out_dir}\n")
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print()

    print("Done.")


if __name__ == "__main__":
    main()
