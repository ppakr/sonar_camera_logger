#!/usr/bin/env python3
"""
Apply static geometric corrections to ArUco tag poses and compute corrected
sonar-to-object distances.

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

The sonar-object distance is then computed as the Euclidean distance between
the two corrected positions in the camera frame (distance is frame-invariant).

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


# ── CSV fields ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp_ns",
    "timestamp_sec",
    # raw tag positions in camera frame
    "sonar_tag_tx", "sonar_tag_ty", "sonar_tag_tz",
    "object_tag_tx", "object_tag_ty", "object_tag_tz",
    # corrected component positions in camera frame
    "sonar_tx", "sonar_ty", "sonar_tz",
    "object_tx", "object_ty", "object_tz",
    # distances
    "sonar_object_distance_m",     # between corrected positions (primary result)
    "tag_tag_distance_m",          # raw tag-to-tag distance (for comparison)
    # detection flags
    "sonar_tag_detected",
    "object_tag_detected",
]


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
    output_csv: str,
):
    """Reprocess one bag and write corrected poses + distances to CSV.

    Returns (total_frames, frames_with_both_tags).
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    # solvePnP object points — same corner layout as aruco_detection.cpp
    half = marker_size / 2.0
    obj_pts = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    ).reshape(4, 1, 3)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/aruco/image_debug"]))

    total_frames = 0
    both_detected = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            msg = deserialize_message(data, Image)
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            channels = len(msg.data) // (msg.height * msg.width)
            frame = arr.reshape(msg.height, msg.width, channels)
            total_frames += 1
            t_sec = t_ns * 1e-9

            corners, ids, _ = detector.detectMarkers(frame)

            # Build a {marker_id: (tvec, rvec)} lookup for this frame
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

            row: dict = {
                "timestamp_ns":  t_ns,
                "timestamp_sec": f"{t_sec:.9f}",
                "sonar_tag_detected":  int(has_sonar),
                "object_tag_detected": int(has_object),
            }

            # Fill sonar fields
            if has_sonar:
                t_s, r_s = poses[sonar_marker_id]
                p_s = corrected_position(t_s, r_s, sonar_offset)
                row.update({
                    "sonar_tag_tx": f"{t_s[0]:.6f}",
                    "sonar_tag_ty": f"{t_s[1]:.6f}",
                    "sonar_tag_tz": f"{t_s[2]:.6f}",
                    "sonar_tx": f"{p_s[0]:.6f}",
                    "sonar_ty": f"{p_s[1]:.6f}",
                    "sonar_tz": f"{p_s[2]:.6f}",
                })

            # Fill object fields
            if has_object:
                t_o, r_o = poses[object_marker_id]
                p_o = corrected_position(t_o, r_o, object_offset)
                row.update({
                    "object_tag_tx": f"{t_o[0]:.6f}",
                    "object_tag_ty": f"{t_o[1]:.6f}",
                    "object_tag_tz": f"{t_o[2]:.6f}",
                    "object_tx": f"{p_o[0]:.6f}",
                    "object_ty": f"{p_o[1]:.6f}",
                    "object_tz": f"{p_o[2]:.6f}",
                })

            # Compute distances only when both are visible
            if has_sonar and has_object:
                both_detected += 1
                t_s, _ = poses[sonar_marker_id]
                t_o, _ = poses[object_marker_id]
                row["tag_tag_distance_m"] = f"{np.linalg.norm(t_s - t_o):.6f}"
                row["sonar_object_distance_m"] = f"{np.linalg.norm(p_s - p_o):.6f}"

            writer.writerow(row)

    return total_frames, both_detected


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply static tag→component offsets to ArUco poses and write "
            "corrected sonar-object distances to CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_corrected_bags",
        help="Directory containing corrected rosbag2 sub-folders.",
    )
    parser.add_argument(
        "--calibration",
        default=(
            "/home/aki/auv_ws/src/sonar_camera_logger/config/calibration_parameters.yaml"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_corrected_distances",
        help="Directory where per-bag CSVs are written.",
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
    print(f"Marker assignments:")
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
        out_csv = os.path.join(args.output_dir, f"{name}.csv")
        print(f"[{name}]", flush=True)
        try:
            total, both = process_bag(
                str(bag_dir), detector, camera_matrix, dist_coeffs,
                args.marker_size,
                args.sonar_marker_id, args.object_marker_id,
                sonar_offset, object_offset,
                out_csv,
            )
            rate = 100.0 * both / max(total, 1)
            print(f"  {total} frames | {both} with both tags ({rate:.1f}%) → {out_csv}")
            summary.append({"bag": name, "total_frames": total,
                            "both_tags_frames": both, "rate_pct": f"{rate:.1f}", "status": "ok"})
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

    print(f"\nDone.  Summary → {summary_path}")


if __name__ == "__main__":
    main()
