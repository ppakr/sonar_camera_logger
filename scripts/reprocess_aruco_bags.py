#!/usr/bin/env python3
"""
Reprocess ArUco distances from recorded bags using corrected camera calibration.

For every bag under --bags_dir this script:
  1. Reads /aruco/image_debug frames.
  2. Re-runs ArUco corner detection with default detector parameters.
  3. Calls solvePnP with the corrected camera matrix and distortion coefficients.
  4. Computes pairwise 3-D distances between all detected markers.
  5. Writes one CSV per bag plus a summary CSV.

Usage (source your workspace first):
    source /home/aki/auv_ws/install/setup.bash
    python3 reprocess_aruco_bags.py

All paths default to their experiment locations; override with flags if needed.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration(yaml_path: str):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file: {yaml_path}")
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("distCoeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


# ── Image conversion ──────────────────────────────────────────────────────────

def image_msg_to_cv2(msg):
    """Convert a sensor_msgs/msg/Image to an (H, W, C) uint8 numpy array."""
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    channels = len(msg.data) // (msg.height * msg.width)
    return arr.reshape(msg.height, msg.width, channels)


# ── Per-bag processing ────────────────────────────────────────────────────────

def reprocess_bag(
    bag_path: str,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size: float,
    output_csv: str,
):
    """Read image_debug from one bag, re-detect markers, write corrected CSV.

    Returns (total_frames, frames_with_detections).
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/aruco/image_debug"]))

    # Object points for solvePnP — same layout as aruco_detection.cpp
    half = marker_size / 2.0
    obj_pts = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    ).reshape(4, 1, 3)

    CSV_FIELDS = [
        "timestamp_ns",
        "timestamp_sec",
        "num_markers",
        "marker_ids",
        "pair_i",
        "pair_j",
        "distance_m",
        "tx_i", "ty_i", "tz_i",
        "tx_j", "ty_j", "tz_j",
    ]

    total_frames = 0
    detect_frames = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            msg = deserialize_message(data, Image)
            frame = image_msg_to_cv2(msg)
            total_frames += 1
            t_sec = t_ns * 1e-9

            corners, ids, _ = detector.detectMarkers(frame)

            base = {
                "timestamp_ns": t_ns,
                "timestamp_sec": f"{t_sec:.9f}",
                "num_markers": 0,
                "marker_ids": "",
                "pair_i": "", "pair_j": "",
                "distance_m": "",
                "tx_i": "", "ty_i": "", "tz_i": "",
                "tx_j": "", "ty_j": "", "tz_j": "",
            }

            if ids is None or len(ids) == 0:
                writer.writerow(base)
                continue

            detect_frames += 1
            ids_flat = ids.flatten().tolist()
            N = len(ids_flat)
            ids_str = ",".join(str(x) for x in ids_flat)

            # Estimate pose for every detected marker
            tvecs = []
            for i in range(N):
                ok, _, tvec = cv2.solvePnP(
                    obj_pts, corners[i], camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                tvecs.append(tvec.flatten() if ok else np.zeros(3))

            # Write one row per pair (one row if only one marker visible)
            if N == 1:
                row = {**base,
                    "num_markers": 1,
                    "marker_ids": ids_str,
                    "pair_i": ids_flat[0],
                    "tx_i": f"{tvecs[0][0]:.6f}",
                    "ty_i": f"{tvecs[0][1]:.6f}",
                    "tz_i": f"{tvecs[0][2]:.6f}",
                }
                writer.writerow(row)
            else:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = float(np.linalg.norm(tvecs[i] - tvecs[j]))
                        writer.writerow({
                            "timestamp_ns": t_ns,
                            "timestamp_sec": f"{t_sec:.9f}",
                            "num_markers": N,
                            "marker_ids": ids_str,
                            "pair_i": ids_flat[i],
                            "pair_j": ids_flat[j],
                            "distance_m": f"{dist:.6f}",
                            "tx_i": f"{tvecs[i][0]:.6f}",
                            "ty_i": f"{tvecs[i][1]:.6f}",
                            "tz_i": f"{tvecs[i][2]:.6f}",
                            "tx_j": f"{tvecs[j][0]:.6f}",
                            "ty_j": f"{tvecs[j][1]:.6f}",
                            "tz_j": f"{tvecs[j][2]:.6f}",
                        })

    return total_frames, detect_frames


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reprocess ArUco bags with corrected camera calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9",
        help="Directory containing the rosbag2 sub-folders.",
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
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_corrected",
        help="Directory where per-bag CSVs and summary.csv are written.",
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
        help="ArUco DetectorParameters YAML (optional — defaults used if missing).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load calibration ──────────────────────────────────────────────────────
    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    print("Calibration loaded:")
    print(f"  fx={camera_matrix[0, 0]:.4f}  fy={camera_matrix[1, 1]:.4f}")
    print(f"  cx={camera_matrix[0, 2]:.4f}  cy={camera_matrix[1, 2]:.4f}")
    print(f"  distCoeffs={dist_coeffs.flatten().tolist()}")

    # ── Build detector ────────────────────────────────────────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    det_params = cv2.aruco.DetectorParameters()
    fs_p = cv2.FileStorage(args.detector_params, cv2.FILE_STORAGE_READ)
    if fs_p.isOpened():
        det_params.readDetectorParameters(fs_p.root())
        fs_p.release()
        print(f"Detector params loaded from: {args.detector_params}")
    else:
        print("Detector params file not found — using OpenCV defaults.")
    detector = cv2.aruco.ArucoDetector(dictionary, det_params)

    # ── Find all bag sub-directories ──────────────────────────────────────────
    bag_dirs = sorted(
        p for p in Path(args.bags_dir).iterdir()
        if p.is_dir() and (p / "metadata.yaml").exists()
    )

    if not bag_dirs:
        print(f"No bags found in {args.bags_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(bag_dirs)} bag(s) in {args.bags_dir}")

    # ── Process each bag ──────────────────────────────────────────────────────
    summary_rows = []
    for bag_dir in bag_dirs:
        name = bag_dir.name
        out_csv = os.path.join(args.output_dir, f"{name}.csv")
        print(f"\n[{bag_dir.name}]", flush=True)

        try:
            total, detected = reprocess_bag(
                str(bag_dir), detector, camera_matrix, dist_coeffs,
                args.marker_size, out_csv,
            )
            rate = 100.0 * detected / max(total, 1)
            print(f"  {total} frames  |  {detected} with detections ({rate:.1f}%)  →  {out_csv}")
            summary_rows.append({
                "bag": name,
                "total_frames": total,
                "frames_with_detections": detected,
                "detection_rate_pct": f"{rate:.1f}",
                "status": "ok",
            })
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            summary_rows.append({
                "bag": name,
                "total_frames": 0,
                "frames_with_detections": 0,
                "detection_rate_pct": "0.0",
                "status": f"error: {exc}",
            })

    # ── Write summary ─────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bag", "total_frames", "frames_with_detections",
                        "detection_rate_pct", "status"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
