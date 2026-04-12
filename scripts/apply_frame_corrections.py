#!/usr/bin/env python3
"""
Apply static geometric corrections to ArUco tag poses and write corrected
MCAP bags that include all original topics plus two new topics.

Background
----------
The ArUco tags are lying flat on the XY (horizontal) surface.  solvePnP gives
us the pose of each TAG ORIGIN in the camera frame.  The actual sonar head and
the actual object are not at the tag origin -- there is a fixed rigid offset
between each tag and its attached component.  These offsets are expressed in
the tag's local frame, which is aligned with ENU (East-North-Up, positive Z
upward) because the tags are horizontal.

Correction formula (per marker):

    p_corrected_in_cam = t_tag_in_cam + R_tag_cam @ offset_in_tag_frame

where
    t_tag_in_cam   = tvec from solvePnP  (tag origin in camera frame)
    R_tag_cam      = rotation matrix from rvec via Rodrigues (tag→camera)
    offset_in_tag_frame = fixed ENU offset from tag to component

Output bags contain:
  - All topics from the input corrected bag (passed through verbatim)
  - /aruco/corrected_poses  (geometry_msgs/msg/PoseArray)
        Poses in camera frame after applying tag→component offsets.
        Ordered [sonar_head, object_center] for detected markers only;
        same ordering as /aruco/ids but only for the two assigned IDs.
  - /aruco/sonar_object_distance  (std_msgs/msg/Float64)
        Euclidean sonar-to-object distance in metres.
        -1.0 when either marker is not detected in that frame.

Default marker assignments (from detection frequency in the Apr-9 dataset):
    ID 0 → floating object  (most frequently detected)
    ID 1 → sonar mount      (less frequently detected, on pool edge)

Default offsets (from 3d_sonar_data_collection.docx, ENU frame):
    Object offset (tag → object):  ( 0.00,  0.00, -0.15) m
    Sonar  offset (tag → sonar):   ( 0.00,  0.05, -0.70) m

Usage:
    source /home/aki/auv_ws/install/setup.bash
    python3 apply_frame_corrections.py [--help]
    python3 apply_frame_corrections.py --bag rosbag2_2026_04_09-14_34_35
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── New topics written into every output bag ──────────────────────────────────

ARUCO_IMAGE_TOPIC = "/aruco/image_debug"
CORRECTED_POSES_TOPIC = "/aruco/corrected_poses"
SONAR_OBJECT_DISTANCE_TOPIC = "/aruco/sonar_object_distance"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_calibration(yaml_path: str):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file: {yaml_path}")
    K = fs.getNode("cameraMatrix").mat()
    D = fs.getNode("distCoeffs").mat()
    fs.release()
    return K, D


def corrected_position(tvec: np.ndarray, rvec: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Return the component position in camera frame after applying the tag→component offset.

    Args:
        tvec:   (3,) tag origin in camera frame  (solvePnP translation)
        rvec:   (3,) Rodrigues rotation vector    (solvePnP rotation)
        offset: (3,) offset from tag to component in the tag's ENU-aligned frame

    Returns:
        (3,) corrected component position in camera frame
    """
    R, _ = cv2.Rodrigues(rvec)          # tag frame → camera frame
    return tvec + R @ offset


def rvec_to_quaternion(rvec: np.ndarray):
    """Rodrigues vector → quaternion (w, x, y, z).

    Mirrors the rotMatToQuat() in aruco_detection.cpp for consistency.
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

def process_bag(
    bag_path: str,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size: float,
    sonar_marker_id: int,
    object_marker_id: int,
    sonar_offset: np.ndarray,
    object_offset: np.ndarray,
    output_bag_dir: str,
    camera_frame: str,
):
    """Reprocess one bag: pass through all topics and append corrected pose topics.

    Returns (total_aruco_frames, frames_with_both_tags).
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Pose, PoseArray
    from std_msgs.msg import Float64

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
    # Discover all topics present in the input bag
    input_topic_meta = reader.get_all_topics_and_types()

    # ── Writer ────────────────────────────────────────────────────────────────
    # rosbag2_py creates the bag directory itself — do NOT mkdir beforehand.
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_bag_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    # Register all input topics
    for idx, tm in enumerate(input_topic_meta):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=idx,
                name=tm.name,
                type=tm.type,
                serialization_format="cdr",
            )
        )

    # Register new corrected topics
    base_id = len(input_topic_meta)
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=base_id,
            name=CORRECTED_POSES_TOPIC,
            type="geometry_msgs/msg/PoseArray",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=base_id + 1,
            name=SONAR_OBJECT_DISTANCE_TOPIC,
            type="std_msgs/msg/Float64",
            serialization_format="cdr",
        )
    )

    total_frames = 0
    both_detected = 0

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        # Always pass through the original message unchanged
        writer.write(topic, data, t_ns)

        # Only process ArUco image frames for correction
        if topic != ARUCO_IMAGE_TOPIC:
            continue

        total_frames += 1
        msg = deserialize_message(data, Image)
        frame = image_msg_to_cv2(msg)

        sec = int(t_ns // 1_000_000_000)
        nanosec = int(t_ns % 1_000_000_000)

        corners, ids, _ = detector.detectMarkers(frame)

        # Build {marker_id: (tvec, rvec)} for this frame
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten().tolist()):
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners[i], camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok:
                    poses[int(mid)] = (tvec.flatten(), rvec.flatten())

        has_sonar  = sonar_marker_id  in poses
        has_object = object_marker_id in poses

        # ── /aruco/corrected_poses ────────────────────────────────────────────
        # One Pose per detected assigned marker: [sonar_head, object_center]
        # Position = corrected component location in camera frame.
        # Orientation = tag orientation (same as /aruco/poses for that marker).
        corrected_pose_msg = PoseArray()
        corrected_pose_msg.header.stamp.sec     = sec
        corrected_pose_msg.header.stamp.nanosec = nanosec
        corrected_pose_msg.header.frame_id      = camera_frame

        for marker_id, offset in [
            (sonar_marker_id,  sonar_offset),
            (object_marker_id, object_offset),
        ]:
            if marker_id not in poses:
                continue
            tvec, rvec = poses[marker_id]
            p = corrected_position(tvec, rvec, offset)
            w, x, y, z = rvec_to_quaternion(rvec)
            pose = Pose()
            pose.position.x    = float(p[0])
            pose.position.y    = float(p[1])
            pose.position.z    = float(p[2])
            pose.orientation.w = w
            pose.orientation.x = x
            pose.orientation.y = y
            pose.orientation.z = z
            corrected_pose_msg.poses.append(pose)

        writer.write(
            CORRECTED_POSES_TOPIC,
            serialize_message(corrected_pose_msg),
            t_ns,
        )

        # ── /aruco/sonar_object_distance ──────────────────────────────────────
        dist_msg = Float64()
        if has_sonar and has_object:
            both_detected += 1
            t_s, r_s = poses[sonar_marker_id]
            t_o, r_o = poses[object_marker_id]
            p_s = corrected_position(t_s, r_s, sonar_offset)
            p_o = corrected_position(t_o, r_o, object_offset)
            dist_msg.data = float(np.linalg.norm(p_s - p_o))
        else:
            dist_msg.data = -1.0   # sentinel: one or both markers not visible

        writer.write(
            SONAR_OBJECT_DISTANCE_TOPIC,
            serialize_message(dist_msg),
            t_ns,
        )

    return total_frames, both_detected


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply static tag→component offsets to ArUco poses and write "
            "corrected MCAP bags with all original topics plus "
            "/aruco/corrected_poses and /aruco/sonar_object_distance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_corrected_bags",
        help="Directory containing corrected rosbag2 sub-folders (input).",
    )
    parser.add_argument(
        "--calibration",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/calibration_parameters.yaml"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_final_bags",
        help="Directory where output bag sub-folders are written.",
    )
    parser.add_argument(
        "--marker_size", type=float, default=0.15,
        help="Physical side length of one ArUco marker in metres.",
    )
    parser.add_argument(
        "--detector_params",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/aruco_detector_parameters.yaml"
        ),
    )
    parser.add_argument(
        "--camera_frame",
        default="camera",
        help="frame_id written into corrected PoseArray headers.",
    )
    parser.add_argument(
        "--sonar_marker_id", type=int, default=1,
        help="ArUco marker ID attached to the sonar mount.",
    )
    parser.add_argument(
        "--object_marker_id", type=int, default=0,
        help="ArUco marker ID attached to the floating object.",
    )
    # Offsets: tag → component, expressed in the tag's ENU-aligned local frame (metres)
    parser.add_argument(
        "--sonar_offset", nargs=3, type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.05, -0.70],
        help="Offset from sonar ArUco tag to sonar head (ENU, metres).",
    )
    parser.add_argument(
        "--object_offset", nargs=3, type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, -0.15],
        help="Offset from object ArUco tag to object centre (ENU, metres).",
    )
    parser.add_argument(
        "--bag", default=None,
        help="Process only this bag name; omit to process all.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────────
    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    sonar_offset  = np.array(args.sonar_offset,  dtype=np.float64)
    object_offset = np.array(args.object_offset, dtype=np.float64)

    print("Calibration:")
    print(f"  fx={camera_matrix[0,0]:.4f}  fy={camera_matrix[1,1]:.4f}")
    print(f"  cx={camera_matrix[0,2]:.4f}  cy={camera_matrix[1,2]:.4f}")
    print("Marker assignments:")
    print(f"  sonar  → ID {args.sonar_marker_id:2d}  offset {sonar_offset.tolist()} m (ENU)")
    print(f"  object → ID {args.object_marker_id:2d}  offset {object_offset.tolist()} m (ENU)")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det_params = cv2.aruco.DetectorParameters()
    fs_p = cv2.FileStorage(args.detector_params, cv2.FILE_STORAGE_READ)
    if fs_p.isOpened():
        det_params.readDetectorParameters(fs_p.root())
        fs_p.release()
    detector = cv2.aruco.ArucoDetector(dictionary, det_params)

    # ── Bag list ──────────────────────────────────────────────────────────────
    all_dirs = sorted(
        p for p in Path(args.bags_dir).iterdir()
        if p.is_dir() and (p / "metadata.yaml").exists()
    )
    if not all_dirs:
        print(f"No bags found in {args.bags_dir}", file=sys.stderr)
        sys.exit(1)

    bag_dirs = [p for p in all_dirs if p.name == args.bag] if args.bag else all_dirs
    if not bag_dirs:
        print(f"Bag '{args.bag}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing {len(bag_dirs)} bag(s)\n")

    # ── Process ───────────────────────────────────────────────────────────────
    summary = []
    for bag_dir in bag_dirs:
        name = bag_dir.name
        out_bag = os.path.join(args.output_dir, name)
        print(f"[{name}]", flush=True)

        if os.path.exists(out_bag):
            print(f"  already exists — skipping (remove to reprocess)\n")
            summary.append({"bag": name, "total_frames": "?",
                            "both_tags_frames": "?", "rate_pct": "?",
                            "status": "skipped"})
            continue

        try:
            total, both = process_bag(
                str(bag_dir), detector, camera_matrix, dist_coeffs,
                args.marker_size,
                args.sonar_marker_id, args.object_marker_id,
                sonar_offset, object_offset,
                out_bag,
                args.camera_frame,
            )
            rate = 100.0 * both / max(total, 1)
            print(f"  {total} aruco frames | {both} with both tags ({rate:.1f}%) → {out_bag}\n")
            summary.append({"bag": name, "total_frames": total,
                            "both_tags_frames": both, "rate_pct": f"{rate:.1f}",
                            "status": "ok"})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            summary.append({"bag": name, "total_frames": 0,
                            "both_tags_frames": 0, "rate_pct": "0.0",
                            "status": f"error: {exc}"})

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["bag", "total_frames", "both_tags_frames", "rate_pct", "status"]
        )
        w.writeheader()
        w.writerows(summary)

    print(f"Done.  Summary → {summary_path}")


if __name__ == "__main__":
    main()
