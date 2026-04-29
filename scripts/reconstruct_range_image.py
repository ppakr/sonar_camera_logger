#!/usr/bin/env python3
"""
reconstruct_range_image.py

Reads a ROS 2 / MCAP bag and adds /sonar_3d/range_image (sensor_msgs/Image,
32FC1, metres) derived from /sonar_3d/pointcloud_intensity.

All original topics are copied verbatim — zero data loss.

The reconstruction inverts the exact projection the sonar driver used:
    x = dist * cos(pitch) * cos(yaw)
    y = dist * cos(pitch) * sin(yaw)
    z = -dist * sin(pitch)
    pixel_x = (yaw   + fov_h/2) / fov_h * (width  - 1)
    pixel_y = (pitch + fov_v/2) / fov_v * (height - 1)

Grid dimensions (width × height) are read automatically from the first
/sonar_3d/strength_image in the bag. FOV can be auto-detected from the angular
extent of the first point cloud frame, or supplied explicitly.

Usage (single bag):
    python3 reconstruct_range_image.py \\
        --bag  /path/to/input_bag \\
        --output /path/to/output_bag

Usage (batch — one input bag per sub-directory):
    python3 reconstruct_range_image.py \\
        --bags_dir /path/to/processed_bags \\
        --output_dir /path/to/range_bags

Optional FOV override (degrees; default: auto-detect from data):
    --fov_h 60.0 --fov_v 45.0
"""

import argparse
import math
import os
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, PointCloud2


POINTCLOUD_TOPIC = "/sonar_3d/pointcloud_intensity"
STRENGTH_TOPIC = "/sonar_3d/strength_image"
RANGE_IMAGE_TOPIC = "/sonar_3d/range_image"


# ── Bag helpers ───────────────────────────────────────────────────────────────


def _open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    r = rosbag2_py.SequentialReader()
    r.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return r


# ── Grid-dimension detection ──────────────────────────────────────────────────


def detect_grid_dims(bag_path: str):
    """Return (width, height) from the first /sonar_3d/strength_image."""
    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if STRENGTH_TOPIC not in type_map:
        return None, None
    msg_cls = get_message(type_map[STRENGTH_TOPIC])
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == STRENGTH_TOPIC:
            msg = deserialize_message(data, msg_cls)
            return int(msg.width), int(msg.height)
    return None, None


# ── FOV detection ─────────────────────────────────────────────────────────────


def detect_fov(bag_path: str, width: int, height: int):
    """
    Estimate (fov_h_deg, fov_v_deg) from the angular extent of the first
    non-empty XYZI point cloud.

    The driver maps pixel (0, 0) to angle (-fov_h/2, -fov_v/2), so in theory
    fov = 2 * max(|angle|).  We add a half-pixel margin because the extreme
    pixels are at the boundary, not the centre.
    """
    reader = _open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if POINTCLOUD_TOPIC not in type_map:
        return None, None
    pc_cls = get_message(type_map[POINTCLOUD_TOPIC])

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != POINTCLOUD_TOPIC:
            continue
        msg = deserialize_message(data, pc_cls)
        x, y, z = _extract_xyz(msg)
        if len(x) < 10:
            continue

        dist = np.sqrt(x * x + y * y + z * z)
        valid = dist > 1e-6
        x, y, z, dist = x[valid], y[valid], z[valid], dist[valid]
        if len(x) < 10:
            continue

        yaw = np.arctan2(y, x)
        pitch = -np.arcsin(np.clip(z / dist, -1.0, 1.0))

        # half-pixel outward margin so pixel 0 ↔ angle -fov/2 is exact
        max_yaw = float(np.max(np.abs(yaw)))
        max_pitch = float(np.max(np.abs(pitch)))
        # Add one half-pixel margin so the estimated FOV matches the extreme pixels
        half_margin_h = (max_yaw / (width - 1)) * 0.5 if width > 1 else 0.0
        half_margin_v = (max_pitch / (height - 1)) * 0.5 if height > 1 else 0.0

        fov_h = 2.0 * (max_yaw + half_margin_h)
        fov_v = 2.0 * (max_pitch + half_margin_v)
        return math.degrees(fov_h), math.degrees(fov_v)

    return None, None


# ── Point-cloud parsing ───────────────────────────────────────────────────────


def _extract_xyz(pc_msg: PointCloud2):
    """
    Return (x, y, z) as float32 numpy arrays.

    The XYZI point cloud layout from sonar_driver.py:
        offset 0  : x  float32
        offset 4  : y  float32
        offset 8  : z  float32
        offset 12 : intensity uint8
    point_step may include trailing padding — use pc_msg.point_step to stride.
    """
    step = pc_msg.point_step
    raw = np.frombuffer(bytes(pc_msg.data), dtype=np.uint8)
    n = len(raw) // step
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rows = raw.reshape(n, step)
    x = np.ascontiguousarray(rows[:, 0:4]).view(np.float32).reshape(n)
    y = np.ascontiguousarray(rows[:, 4:8]).view(np.float32).reshape(n)
    z = np.ascontiguousarray(rows[:, 8:12]).view(np.float32).reshape(n)
    return x, y, z


# ── Core reconstruction ───────────────────────────────────────────────────────


def reconstruct_range_image(
    pc_msg: PointCloud2,
    width: int,
    height: int,
    fov_h_rad: float,
    fov_v_rad: float,
) -> np.ndarray:
    """
    Invert the sonar driver projection → HxW float32 range image (metres).
    Pixels with no sonar return remain 0.0.
    """
    x, y, z = _extract_xyz(pc_msg)
    if len(x) == 0:
        return np.zeros((height, width), dtype=np.float32)

    dist_sq = x * x + y * y + z * z
    valid = dist_sq > 1e-12
    x, y, z = x[valid], y[valid], z[valid]
    dist = np.sqrt(dist_sq[valid])

    yaw = np.arctan2(y, x)
    pitch = -np.arcsin(np.clip(z / dist, -1.0, 1.0))

    max_px = width - 1
    max_py = height - 1

    px = np.round((yaw + fov_h_rad / 2.0) / fov_h_rad * max_px).astype(np.int32)
    py = np.round((pitch + fov_v_rad / 2.0) / fov_v_rad * max_py).astype(np.int32)

    in_bounds = (px >= 0) & (px <= max_px) & (py >= 0) & (py <= max_py)
    px, py, dist = px[in_bounds], py[in_bounds], dist[in_bounds]

    range_img = np.zeros((height, width), dtype=np.float32)
    range_img[py, px] = dist
    return range_img


def _make_image_msg(arr: np.ndarray, header) -> Image:
    """Wrap HxW float32 numpy array in a sensor_msgs/Image (32FC1)."""
    msg = Image()
    msg.header = header
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = arr.shape[1] * 4  # 4 bytes per float32
    msg.data = arr.tobytes()
    return msg


# ── Main processing ───────────────────────────────────────────────────────────


def process_bag(
    bag_path: str,
    output_path: str,
    fov_h_deg: float | None,
    fov_v_deg: float | None,
) -> None:
    """
    Copy all messages from bag_path to output_path and append
    /sonar_3d/range_image for every /sonar_3d/pointcloud_intensity frame.
    """
    # --- Step 1: grid dimensions ---
    print("  Detecting grid dimensions... ", end="", flush=True)
    width, height = detect_grid_dims(bag_path)
    if width is None:
        print(f"\n  ERROR: {STRENGTH_TOPIC} not found — cannot determine grid size.")
        sys.exit(1)
    print(f"{width} × {height}")

    # --- Step 2: FOV (auto-detect or user-supplied) ---
    if fov_h_deg is None or fov_v_deg is None:
        print("  Auto-detecting FOV... ", end="", flush=True)
        fov_h_deg, fov_v_deg = detect_fov(bag_path, width, height)
        if fov_h_deg is None:
            print(
                f"\n  ERROR: {POINTCLOUD_TOPIC} not found — cannot auto-detect FOV. "
                "Use --fov_h / --fov_v."
            )
            sys.exit(1)
        print(f"{fov_h_deg:.2f}° H × {fov_v_deg:.2f}° V  (auto-detected)")
    else:
        print(f"  FOV: {fov_h_deg:.2f}° H × {fov_v_deg:.2f}° V  (user-supplied)")

    fov_h = math.radians(fov_h_deg)
    fov_v = math.radians(fov_v_deg)

    # --- Step 3: open reader + writer ---
    reader = _open_reader(bag_path)
    topic_meta = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_meta}

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    # Register all original topics
    for idx, tm in enumerate(topic_meta):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=idx,
                name=tm.name,
                type=tm.type,
                serialization_format="cdr",
            )
        )

    # Register the new range image topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=len(topic_meta),
            name=RANGE_IMAGE_TOPIC,
            type="sensor_msgs/msg/Image",
            serialization_format="cdr",
        )
    )

    pc_cls = get_message(type_map[POINTCLOUD_TOPIC]) if POINTCLOUD_TOPIC in type_map else None

    n_range = 0
    n_total = 0

    # --- Step 4: single-pass copy + reconstruct ---
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        n_total += 1

        # Always copy the original message verbatim
        writer.write(topic, data, t_ns)

        # For each XYZI point cloud frame, also produce the range image
        if topic == POINTCLOUD_TOPIC and pc_cls is not None:
            pc_msg = deserialize_message(data, pc_cls)
            range_arr = reconstruct_range_image(pc_msg, width, height, fov_h, fov_v)
            range_img_msg = _make_image_msg(range_arr, pc_msg.header)
            writer.write(
                RANGE_IMAGE_TOPIC,
                serialize_message(range_img_msg),
                t_ns,
            )
            n_range += 1
            if n_range % 100 == 0:
                print(f"    {n_range} range images reconstructed...", flush=True)

    print(f"  Done — {n_range} range images added ({n_total} original messages copied).")
    print(f"  Output: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Add /sonar_3d/range_image (32FC1, metres) to a rosbag. "
            "All original topics are kept unchanged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bag", metavar="DIR", help="Single input bag directory.")
    mode.add_argument(
        "--bags_dir",
        metavar="DIR",
        help="Batch mode: process every sub-directory as a bag.",
    )

    ap.add_argument(
        "--output",
        metavar="DIR",
        help="Output bag directory (single-bag mode). Must not exist.",
    )
    ap.add_argument(
        "--output_dir",
        metavar="DIR",
        help="Output directory for batch mode.",
    )
    ap.add_argument(
        "--fov_h",
        type=float,
        default=None,
        metavar="DEG",
        help="Horizontal FOV in degrees. Default: auto-detect from data.",
    )
    ap.add_argument(
        "--fov_v",
        type=float,
        default=None,
        metavar="DEG",
        help="Vertical FOV in degrees. Default: auto-detect from data.",
    )

    args = ap.parse_args()

    if args.bags_dir:
        # Batch mode
        if not args.output_dir:
            ap.error("--output_dir is required with --bags_dir")
        os.makedirs(args.output_dir, exist_ok=True)
        bags = sorted(
            d for d in os.listdir(args.bags_dir)
            if os.path.isdir(os.path.join(args.bags_dir, d))
        )
        if not bags:
            print(f"No sub-directories found in {args.bags_dir}")
            sys.exit(1)
        for name in bags:
            in_path = os.path.join(args.bags_dir, name)
            out_path = os.path.join(args.output_dir, name)
            if os.path.exists(out_path):
                print(f"[SKIP] {name}  (output already exists)")
                continue
            print(f"\n=== {name} ===")
            process_bag(in_path, out_path, args.fov_h, args.fov_v)
    else:
        # Single-bag mode
        if not args.output:
            ap.error("--output is required with --bag")
        if os.path.exists(args.output):
            print(f"ERROR: output path already exists: {args.output}")
            sys.exit(1)
        print(f"Input:  {args.bag}")
        process_bag(args.bag, args.output, args.fov_h, args.fov_v)


if __name__ == "__main__":
    main()
