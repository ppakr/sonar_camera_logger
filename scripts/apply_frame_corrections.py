#!/usr/bin/env python3
"""
Re-process wet-lab bags: re-run ArUco solvePnP with a new calibration and rebuild TF.

TF Tree (output) — matches the live sonar_camera.launch.py tree
----------------
    world  (= sonar_link = sonar transducer position, identity)
        └── sonar_link         [/tf_static]  identity — world IS sonar_link
                └── sonar_aruco    [/tf_static]  fixed rigid offset (sonar_offset)
                        └── float_aruco [/tf]  dynamic — updated each frame both tags visible
                                └── object_top [/tf_static]  fixed rigid offset (object_offset)

Original /tf and /tf_static from the input bag are NOT passed through — they are
replaced entirely by the freshly computed transforms above.

Dynamic transform derivation (sonar_aruco → float_aruco)
----------------------------------------------------------
Camera gives us (via solvePnP) for each detected tag:
    T_cam_sonar_aruco :  p_cam = R_s @ p_s + tvec_s
    T_cam_float_aruco: p_cam = R_o @ p_o + tvec_o

Relative transform in sonar_aruco marker local frame:
    R = R_s^T @ R_o
    t = R_s^T @ (tvec_o − tvec_s)

New topics written into every output bag
-----------------------------------------
    /tf_static   tf2_msgs/msg/TFMessage   three static transforms (once, at first message)
    /tf          tf2_msgs/msg/TFMessage   sonar_aruco→float_aruco (per-frame, both visible)
    /aruco/sonar_object_distance   std_msgs/msg/Float64
                 Euclidean distance sonar_head → object_top (metres).
                 −1.0 when either marker is not visible.

Usage
-----
    source /home/aki/auv_ws/install/setup.bash
    python3 apply_frame_corrections.py [--help]
    python3 apply_frame_corrections.py --bags_dir ~/wetlab_raw --output_dir ~/wetlab_processed
    python3 apply_frame_corrections.py --bags_dir ~/wetlab_raw --bag cylinder_plastic --output_dir ~/wetlab_processed
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# ── Topic names ───────────────────────────────────────────────────────────────

ARUCO_IMAGE_TOPIC = "/aruco/image_raw"
TF_STATIC_TOPIC = "/tf_static"
TF_TOPIC = "/tf"
DIST_TOPIC = "/aruco/sonar_object_distance"

DEFAULT_TF_CONFIG = (
    "/home/aki/auv_ws/src/sonar_camera_logger/config/aruco_tf_publisher.yaml"
)


# ── TF config (aruco_tf_publisher.yaml) ───────────────────────────────────────


def rpy_to_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Roll-pitch-yaw (radians, ZYX extrinsic) → 3×3 rotation matrix.

    Same convention as aruco_tf_publisher.py:rpy_to_quat — ensures the
    static transforms produced here line up exactly with what the live
    TF publisher would emit for the same YAML values.
    """
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_tf_config(yaml_path: str):
    """Load static frame offsets from aruco_tf_publisher.yaml.

    Returns
    -------
    dict with keys:
        sonar_offset       (np.ndarray, 3,)  metres, sonar_link → sonar_aruco
        R_sonar_aruco      (np.ndarray, 3, 3) rotation, sonar_link → sonar_aruco
        object_offset      (np.ndarray, 3,)  metres, float_aruco → object_top
        R_object_top    (np.ndarray, 3, 3) rotation, float_aruco → object_top
        sonar_aruco_id     (int)
        floater_aruco_id   (int)
    """
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)
    try:
        params = doc["aruco_tf_publisher"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{yaml_path}: expected 'aruco_tf_publisher.ros__parameters' root"
        ) from exc

    def get(key, default=None, required=False):
        if required and key not in params:
            raise KeyError(f"{yaml_path}: missing required key '{key}'")
        return params.get(key, default)

    sonar_offset = np.array([
        float(get("sonar_aruco_offset_x", 0.0)),
        float(get("sonar_aruco_offset_y", 0.0)),
        float(get("sonar_aruco_offset_z", 0.0)),
    ])
    R_sonar_aruco = rpy_to_matrix(
        math.radians(float(get("sonar_aruco_offset_roll_deg", 0.0))),
        math.radians(float(get("sonar_aruco_offset_pitch_deg", 0.0))),
        math.radians(float(get("sonar_aruco_offset_yaw_deg", 0.0))),
    )
    object_offset = np.array([
        float(get("object_center_offset_x", 0.0)),
        float(get("object_center_offset_y", 0.0)),
        float(get("object_center_offset_z", 0.0)),
    ])
    R_object_top = rpy_to_matrix(
        math.radians(float(get("object_center_offset_roll_deg", 0.0))),
        math.radians(float(get("object_center_offset_pitch_deg", 0.0))),
        math.radians(float(get("object_center_offset_yaw_deg", 0.0))),
    )
    return {
        "sonar_offset": sonar_offset,
        "R_sonar_aruco": R_sonar_aruco,
        "object_offset": object_offset,
        "R_object_top": R_object_top,
        "sonar_aruco_id": int(get("sonar_aruco_id", 1)),
        "floater_aruco_id": int(get("floater_aruco_id", 0)),
    }


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
    sonar_offset: np.ndarray,  # sonar_link → sonar_aruco translation (metres)
    R_sonar_aruco: np.ndarray,  # sonar_link → sonar_aruco rotation
    object_offset: np.ndarray,  # float_aruco → object_top translation (metres)
    R_object_top: np.ndarray,  # float_aruco → object_top rotation
    output_bag_dir: str,
    world_frame: str = "world",
    sonar_link_frame: str = "sonar_link",
    sonar_tag_frame: str = "sonar_aruco",
    object_tag_frame: str = "float_aruco",
    object_top_frame: str = "object_top",
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

    # Topics we recompute from scratch — drop originals to avoid duplicates.
    REPLACED_TOPICS = {TF_STATIC_TOPIC, TF_TOPIC, DIST_TOPIC}

    passthrough_topics = [tm for tm in input_topic_meta if tm.name not in REPLACED_TOPICS]
    for idx, tm in enumerate(passthrough_topics):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=idx,
                name=tm.name,
                type=tm.type,
                serialization_format="cdr",
            )
        )

    base_id = len(passthrough_topics)
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

        # Pass through all original messages EXCEPT /tf and /tf_static which we recompute.
        if topic not in (TF_STATIC_TOPIC, TF_TOPIC):
            writer.write(topic, data, t_ns)

        sec = int(t_ns // 1_000_000_000)
        nanosec = int(t_ns % 1_000_000_000)

        # Write three static TFs once at the timestamp of the very first message.
        if not static_written:
            # world → sonar_link  (identity: the sonar transducer IS the world origin)
            ts1 = make_transform_stamped(
                world_frame, sonar_link_frame, np.zeros(3), np.eye(3), sec, nanosec
            )
            # sonar_link → sonar_aruco  (rigid offset + rotation from aruco_tf_publisher.yaml)
            ts2 = make_transform_stamped(
                sonar_link_frame, sonar_tag_frame, sonar_offset, R_sonar_aruco, sec, nanosec
            )
            # float_aruco → object_top  (rigid offset + rotation from aruco_tf_publisher.yaml)
            ts3 = make_transform_stamped(
                object_tag_frame, object_top_frame,
                object_offset, R_object_top, sec, nanosec,
            )
            statics = TFMessage()
            statics.transforms = [ts1, ts2, ts3]
            writer.write(TF_STATIC_TOPIC, serialize_message(statics), t_ns)
            static_written = True

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

            # ── Distance: sonar_head (world origin) → object_top ───────────
            # Full TF chain: world → sonar_link → sonar_aruco → float_aruco → object_top
            # t_st_ot is in sonar_aruco marker frame; rotate through R_sonar_aruco
            # (from aruco_tf_publisher.yaml) to express it in the world/sonar_link frame.
            t_world_floater = sonar_offset + R_sonar_aruco @ t_st_ot
            R_world_floater = R_sonar_aruco @ R_st_ot
            p_obj_center_world = t_world_floater + R_world_floater @ object_offset
            dist_val = float(np.linalg.norm(p_obj_center_world))

        dist_msg = Float64()
        dist_msg.data = dist_val
        writer.write(DIST_TOPIC, serialize_message(dist_msg), t_ns)

    return total_frames, both_detected


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Re-process bags with a new camera calibration: re-runs ArUco solvePnP and "
            "rebuilds /tf_static (world→sonar_link→sonar_aruco, float_aruco→object_top), "
            "/tf (sonar_aruco→float_aruco, dynamic), and /aruco/sonar_object_distance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        required=True,
        help="Directory whose sub-folders are rosbag2 bags (each contains metadata.yaml).",
    )
    parser.add_argument(
        "--calibration",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/microsoft_livecam_hd3000.yaml"
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where reprocessed bag sub-folders are written.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=0.16,
        help="Physical side length of one ArUco marker in metres.",
    )
    parser.add_argument(
        "--detector_params",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/aruco_detector_parameters.yaml"
        ),
    )
    parser.add_argument(
        "--tf_config",
        default=DEFAULT_TF_CONFIG,
        help=(
            "Path to aruco_tf_publisher.yaml. Provides static frame offsets "
            "(sonar_link → sonar_aruco and float_aruco → object_top, "
            "translation + RPY) and the marker IDs."
        ),
    )
    parser.add_argument(
        "--sonar_marker_id",
        type=int,
        default=None,
        help="Override sonar mount ArUco ID (defaults to value in --tf_config).",
    )
    parser.add_argument(
        "--object_marker_id",
        type=int,
        default=None,
        help="Override floater ArUco ID (defaults to value in --tf_config).",
    )
    # TF frame name overrides
    parser.add_argument("--world_frame", default="world")
    parser.add_argument("--sonar_link_frame", default="sonar_link")
    parser.add_argument("--sonar_tag_frame", default="sonar_aruco")
    parser.add_argument("--object_tag_frame", default="float_aruco")
    parser.add_argument("--object_top_frame", default="object_top")
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

    # ── Static frame offsets (translation + rotation) from aruco_tf_publisher.yaml
    try:
        tf_cfg = load_tf_config(args.tf_config)
    except (OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: cannot load tf_config: {exc}", file=sys.stderr)
        sys.exit(1)

    sonar_offset = tf_cfg["sonar_offset"]
    R_sonar_aruco = tf_cfg["R_sonar_aruco"]
    object_offset = tf_cfg["object_offset"]
    R_object_top = tf_cfg["R_object_top"]
    sonar_marker_id = (
        args.sonar_marker_id if args.sonar_marker_id is not None else tf_cfg["sonar_aruco_id"]
    )
    object_marker_id = (
        args.object_marker_id if args.object_marker_id is not None else tf_cfg["floater_aruco_id"]
    )

    print("Calibration:")
    print(f"  fx={camera_matrix[0,0]:.4f}  fy={camera_matrix[1,1]:.4f}")
    print(f"  cx={camera_matrix[0,2]:.4f}  cy={camera_matrix[1,2]:.4f}")
    print(f"TF config: {args.tf_config}")
    print(f"  sonar_aruco_id={sonar_marker_id}  floater_aruco_id={object_marker_id}")
    print("TF tree:")
    print(f"  {args.world_frame} → {args.sonar_link_frame}  identity  [static]")
    print(
        f"  {args.sonar_link_frame} → {args.sonar_tag_frame}  "
        f"t={sonar_offset.tolist()} m  R≠I={not np.allclose(R_sonar_aruco, np.eye(3))}  [static]"
    )
    print(f"  {args.sonar_tag_frame} → {args.object_tag_frame}  [dynamic]")
    print(
        f"  {args.object_tag_frame} → {args.object_top_frame}  "
        f"t={object_offset.tolist()} m  R≠I={not np.allclose(R_object_top, np.eye(3))}  [static]"
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
                sonar_marker_id,
                object_marker_id,
                sonar_offset,
                R_sonar_aruco,
                object_offset,
                R_object_top,
                out_bag,
                world_frame=args.world_frame,
                sonar_link_frame=args.sonar_link_frame,
                sonar_tag_frame=args.sonar_tag_frame,
                object_tag_frame=args.object_tag_frame,
                object_top_frame=args.object_top_frame,
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
