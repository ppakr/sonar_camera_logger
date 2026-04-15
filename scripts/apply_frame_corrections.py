#!/usr/bin/env python3
"""
Build TF-tree MCAP bags from corrected wet-lab bags.

TF Tree (output)
----------------
    world  (= sonar_head, the sonar transducer position)
        └── sonar_tag          [/tf_static]  fixed rigid offset
                └── object_tag [/tf]         dynamic — updated each frame both tags visible
                        └── object_center    [/tf_static]  fixed rigid offset

All transforms are expressed parent → child (ROS tf2 convention):
  transform.translation = position of child origin in parent frame
  transform.rotation    = rotation from child axes to parent axes

Dynamic transform derivation (sonar_tag → object_tag)
------------------------------------------------------
Camera gives us (via solvePnP) for each detected tag:
    T_cam_sonar_tag :  p_cam = R_s @ p_stag + tvec_s
    T_cam_obj_tag   :  p_cam = R_o @ p_otag + tvec_o

Inverse of T_cam_sonar_tag:
    T_sonar_tag_cam :  p_stag = R_s^T @ (p_cam − tvec_s)
                            R = R_s^T,   t = −R_s^T @ tvec_s

Compose to get T_sonar_tag_obj_tag = T_sonar_tag_cam ∘ T_cam_obj_tag:
    R = R_s^T @ R_o
    t = R_s^T @ (tvec_o − tvec_s)

Static transforms
-----------------
    world → sonar_tag:
        sonar_offset = position of sonar_head in sonar_tag ENU frame
        → sonar_tag is at −sonar_offset in world (ENU-aligned, identity rotation)
    object_tag → object_center:
        object_offset = position of object_center in object_tag ENU frame  (direct)

New topics written into every output bag
-----------------------------------------
    /tf_static   tf2_msgs/msg/TFMessage   two static transforms (once at t=0)
    /tf          tf2_msgs/msg/TFMessage   sonar_tag→object_tag (per-frame, both visible)
    /aruco/sonar_object_distance   std_msgs/msg/Float64
                 Euclidean distance sonar_head → object_center in world frame.
                 −1.0 when either marker is not visible.

Usage
-----
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


# ── Topic names ───────────────────────────────────────────────────────────────

ARUCO_IMAGE_TOPIC = "/aruco/image_debug"
TF_STATIC_TOPIC = "/tf_static"
TF_TOPIC = "/tf"
DIST_TOPIC = "/aruco/sonar_object_distance"


# ── TF helpers ────────────────────────────────────────────────────────────────


def rotation_matrix_to_quaternion(R: np.ndarray):
    """3×3 rotation matrix → (w, x, y, z) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = float(np.copysign(x, R[2, 1] - R[1, 2]))
    y = float(np.copysign(y, R[0, 2] - R[2, 0]))
    z = float(np.copysign(z, R[1, 0] - R[0, 1]))
    return float(w), x, y, z


def make_transform_stamped(parent_id, child_id, translation, R, sec, nanosec):
    """Build a geometry_msgs/TransformStamped from a rotation matrix and translation."""
    from geometry_msgs.msg import TransformStamped

    ts = TransformStamped()
    ts.header.stamp.sec = int(sec)
    ts.header.stamp.nanosec = int(nanosec)
    ts.header.frame_id = parent_id
    ts.child_frame_id = child_id
    ts.transform.translation.x = float(translation[0])
    ts.transform.translation.y = float(translation[1])
    ts.transform.translation.z = float(translation[2])
    w, x, y, z = rotation_matrix_to_quaternion(R)
    ts.transform.rotation.w = w
    ts.transform.rotation.x = x
    ts.transform.rotation.y = y
    ts.transform.rotation.z = z
    return ts


# ── Per-bag processing ────────────────────────────────────────────────────────


def process_bag(
    bag_path: str,
    detector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size: float,
    sonar_marker_id: int,
    object_marker_id: int,
    sonar_offset: np.ndarray,  # sonar_tag → sonar_head,  in tag ENU frame
    object_offset: np.ndarray,  # object_tag → object_center, in tag ENU frame
    output_bag_dir: str,
    world_frame: str = "world",
    sonar_tag_frame: str = "sonar_tag",
    object_tag_frame: str = "object_tag",
    object_center_frame: str = "object_center",
):
    """
    Reprocess one bag: pass through all original topics and add TF + distance.

    Returns (total_aruco_frames, frames_with_both_tags).
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64
    from tf2_msgs.msg import TFMessage

    half = marker_size / 2.0
    obj_pts = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    ).reshape(4, 1, 3)

    # ── Pre-compute static transforms ─────────────────────────────────────────
    # world (sonar_head) → sonar_tag
    #   sonar_offset = position of sonar_HEAD in sonar_TAG's ENU frame
    #   → sonar_tag origin is at (−sonar_offset) in the world (sonar_head) frame
    #   Orientation: identity — both frames share the same ENU orientation
    t_world_sonar_tag = -sonar_offset  # e.g. [0, −0.05,  0.70]
    R_world_sonar_tag = np.eye(3)

    # object_tag → object_center
    #   object_offset = position of object_CENTER in object_TAG's ENU frame → direct
    t_obj_tag_obj_ctr = object_offset  # e.g. [0,  0.00, −0.15]
    R_obj_tag_obj_ctr = np.eye(3)

    # ── Reader ────────────────────────────────────────────────────────────────
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    input_topic_meta = reader.get_all_topics_and_types()

    # ── Writer ────────────────────────────────────────────────────────────────
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_bag_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    for idx, tm in enumerate(input_topic_meta):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=idx,
                name=tm.name,
                type=tm.type,
                serialization_format="cdr",
            )
        )

    base_id = len(input_topic_meta)
    for extra_idx, (tname, ttype) in enumerate(
        [
            (TF_STATIC_TOPIC, "tf2_msgs/msg/TFMessage"),
            (TF_TOPIC, "tf2_msgs/msg/TFMessage"),
            (DIST_TOPIC, "std_msgs/msg/Float64"),
        ]
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=base_id + extra_idx,
                name=tname,
                type=ttype,
                serialization_format="cdr",
            )
        )

    static_written = False
    total_frames = 0
    both_detected = 0

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        # Pass through all original messages unchanged
        writer.write(topic, data, t_ns)

        sec = int(t_ns // 1_000_000_000)
        nanosec = int(t_ns % 1_000_000_000)

        # Write static TF once at the timestamp of the very first message
        if not static_written:
            # Write static link from world (sonar_head) → sonar_tag, and object_tag → object_center
            ts1 = make_transform_stamped(
                world_frame,
                sonar_tag_frame,
                t_world_sonar_tag,
                R_world_sonar_tag,
                sec,
                nanosec,
            )
            ts2 = make_transform_stamped(
                object_tag_frame,
                object_center_frame,
                t_obj_tag_obj_ctr,
                R_obj_tag_obj_ctr,
                sec,
                nanosec,
            )
            statics = TFMessage()
            statics.transforms = [ts1, ts2]
            writer.write(TF_STATIC_TOPIC, serialize_message(statics), t_ns)
            # static_written = True

        # Only ArUco image frames carry pose information
        if topic != ARUCO_IMAGE_TOPIC:
            continue

        total_frames += 1

        img_msg = deserialize_message(data, Image)
        arr = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)
        ch = max(1, len(img_msg.data) // (img_msg.height * img_msg.width))
        frame = arr.reshape(img_msg.height, img_msg.width, ch)

        corners, ids, _ = detector.detectMarkers(frame)

        # Build per-marker (tvec, R) from solvePnP
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten().tolist()):
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    corners[i],
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok:
                    R_mat, _ = cv2.Rodrigues(rvec.flatten())
                    poses[int(mid)] = (tvec.flatten(), R_mat)

        has_sonar = sonar_marker_id in poses
        has_object = object_marker_id in poses

        # ── Dynamic TF: sonar_tag → object_tag ───────────────────────────────
        dist_val = -1.0
        if has_sonar and has_object:
            both_detected += 1
            print(both_detected, "both tags detected at", sec + nanosec * 1e-9, "s")

            tvec_s, R_s = poses[sonar_marker_id]  # sonar_tag in camera frame
            tvec_o, R_o = poses[object_marker_id]  # object_tag in camera frame

            # T_sonar_tag_object_tag = T_cam_sonar_tag^{-1} @ T_cam_object_tag
            R_st_ot = R_s.T @ R_o
            t_st_ot = R_s.T @ (tvec_o - tvec_s)

            dyn_tf = TFMessage()
            dyn_tf.transforms = [
                make_transform_stamped(
                    sonar_tag_frame,
                    object_tag_frame,
                    t_st_ot,
                    R_st_ot,
                    sec,
                    nanosec,
                )
            ]
            writer.write(TF_TOPIC, serialize_message(dyn_tf), t_ns)

            # ── Distance: sonar_head (world origin) → object_center ───────────
            # T_world_object_tag:
            #   R = R_world_sonar_tag @ R_st_ot  = I @ R_st_ot
            #   t = t_world_sonar_tag + R_world_sonar_tag @ t_st_ot
            #     = t_world_sonar_tag + t_st_ot
            R_world_obj_tag = R_st_ot  # R_world_sonar_tag = I
            t_world_obj_tag = t_world_sonar_tag + t_st_ot

            # object_center in world:
            #   p = t_world_obj_tag + R_world_obj_tag @ object_offset
            p_obj_center_world = t_world_obj_tag + R_world_obj_tag @ object_offset
            dist_val = float(np.linalg.norm(p_obj_center_world))

        dist_msg = Float64()
        dist_msg.data = dist_val
        writer.write(DIST_TOPIC, serialize_message(dist_msg), t_ns)

    return total_frames, both_detected


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build TF-tree MCAP bags. "
            "Writes /tf_static (world→sonar_tag, object_tag→object_center), "
            "/tf (sonar_tag→object_tag, dynamic), "
            "and /aruco/sonar_object_distance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9/",
        help="Directory of corrected rosbag2 sub-folders (input from build_corrected_bags.py).",
    )
    parser.add_argument(
        "--calibration",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/microsoft_livecam_hd3000.yaml"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_final_bags",
        help="Directory where output bag sub-folders are written.",
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
    )
    parser.add_argument(
        "--sonar_marker_id", type=int, default=1, help="ArUco ID on the sonar mount."
    )
    parser.add_argument(
        "--object_marker_id",
        type=int,
        default=0,
        help="ArUco ID on the floating object.",
    )
    parser.add_argument(
        "--sonar_offset",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.05, -0.70],
        help="Offset sonar_tag → sonar_head in tag ENU frame (metres).",
    )
    parser.add_argument(
        "--object_offset",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, -0.15],
        help="Offset object_tag → object_center in tag ENU frame (metres).",
    )
    # TF frame name overrides
    parser.add_argument("--world_frame", default="world")
    parser.add_argument("--sonar_tag_frame", default="sonar_tag")
    parser.add_argument("--object_tag_frame", default="object_tag")
    parser.add_argument("--object_center_frame", default="object_center")
    parser.add_argument(
        "--bag", default=None, help="Process only this bag name; omit to process all."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load calibration ──────────────────────────────────────────────────────
    fs = cv2.FileStorage(args.calibration, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"ERROR: cannot open calibration: {args.calibration}", file=sys.stderr)
        sys.exit(1)
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("distCoeffs").mat()
    fs.release()

    sonar_offset = np.array(args.sonar_offset, dtype=np.float64)
    object_offset = np.array(args.object_offset, dtype=np.float64)

    print("Calibration:")
    print(f"  fx={camera_matrix[0,0]:.4f}  fy={camera_matrix[1,1]:.4f}")
    print(f"  cx={camera_matrix[0,2]:.4f}  cy={camera_matrix[1,2]:.4f}")
    print("TF tree:")
    print(
        f"  {args.world_frame} → {args.sonar_tag_frame}  "
        f"offset {(-sonar_offset).tolist()} m  [static]"
    )
    print(f"  {args.sonar_tag_frame} → {args.object_tag_frame}  [dynamic]")
    print(
        f"  {args.object_tag_frame} → {args.object_center_frame}  "
        f"offset {object_offset.tolist()} m  [static]"
    )

    # ── Detector ──────────────────────────────────────────────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det_params = cv2.aruco.DetectorParameters()
    fs_p = cv2.FileStorage(args.detector_params, cv2.FILE_STORAGE_READ)
    if fs_p.isOpened():
        det_params.readDetectorParameters(fs_p.root())
        fs_p.release()
    detector = cv2.aruco.ArucoDetector(dictionary, det_params)

    # ── Bag list ──────────────────────────────────────────────────────────────
    all_dirs = sorted(
        p
        for p in Path(args.bags_dir).iterdir()
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
            print("  already exists — skipping (remove to reprocess)\n")
            summary.append(
                {
                    "bag": name,
                    "total_frames": "?",
                    "both_tags_frames": "?",
                    "rate_pct": "?",
                    "status": "skipped",
                }
            )
            continue

        try:
            total, both = process_bag(
                str(bag_dir),
                detector,
                camera_matrix,
                dist_coeffs,
                args.marker_size,
                args.sonar_marker_id,
                args.object_marker_id,
                sonar_offset,
                object_offset,
                out_bag,
                world_frame=args.world_frame,
                sonar_tag_frame=args.sonar_tag_frame,
                object_tag_frame=args.object_tag_frame,
                object_center_frame=args.object_center_frame,
            )
            rate = 100.0 * both / max(total, 1)
            print(
                f"  {total} aruco frames | {both} with both tags ({rate:.1f}%) → {out_bag}\n"
            )
            summary.append(
                {
                    "bag": name,
                    "total_frames": total,
                    "both_tags_frames": both,
                    "rate_pct": f"{rate:.1f}",
                    "status": "ok",
                }
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            summary.append(
                {
                    "bag": name,
                    "total_frames": 0,
                    "both_tags_frames": 0,
                    "rate_pct": "0.0",
                    "status": f"error: {exc}",
                }
            )

    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bag",
                "total_frames",
                "both_tags_frames",
                "rate_pct",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(summary)

    print(f"Done.  Summary → {summary_path}")


if __name__ == "__main__":
    main()
