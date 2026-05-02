#!/usr/bin/env python3
"""
build_tf_bags.py

For each bag in --bags_dir (original wl_wetlab_apr9 format), reads
/aruco/poses and computes the sonar_tag → object_tag TF using the
same logic and rapid-rotation filter as aruco_tf_publisher.py.

Outputs a new MCAP bag containing:
  • all original topics/messages copied verbatim
  • /tf_static : world → sonar_tag  and  object_tag → object_top  (once, static)
  • /tf        : sonar_tag → object_tag  (one message per valid /aruco/poses frame)

After this, run visualize_bags_rerun.py --save to produce .rrd files:

    python3 visualize_bags_rerun.py \\
        --bag <output_bag_dir> --save output.rrd

    # or batch all bags:
    python3 visualize_bags_rerun.py \\
        --bags_dir <output_bags_dir> --save all.rrd

Physical frame offsets (must match aruco_tf_publisher / apply_frame_corrections):
    sonar_head  is  [0.00,  0.05, -0.70] m from sonar_tag  in tag frame
    object_top is [0.00,  0.00, -0.15] m from object_tag in tag frame

Usage
-----
    source /home/aki/auv_ws/install/setup.bash
    python3 build_tf_bags.py \\
        --bags_dir  /media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9 \\
        --output_dir /media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_tf_bags

    # Optional: tighter rotation filter (default 30°)
    python3 build_tf_bags.py ... --max_rotation_jump_deg 20
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


# ── Physical offsets ──────────────────────────────────────────────────────────
# sonar_head position in sonar_tag frame  →  world → sonar_tag = -offset
SONAR_HEAD_OFFSET  = np.array([0.00,  0.05, -0.70])   # metres
# object_top position in object_tag frame
OBJECT_CTR_OFFSET  = np.array([0.00,  0.00, -0.15])   # metres

POSES_TOPIC  = "/aruco/poses"
TF_TOPIC     = "/tf"
TF_STATIC    = "/tf_static"
TF_TYPE      = "tf2_msgs/msg/TFMessage"


# ── Rotation helpers ──────────────────────────────────────────────────────────

def quat_to_matrix(x, y, z, w) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3×3 rotation matrix."""
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z),  s*(x*y - z*w),      s*(x*z + y*w)],
        [s*(x*y + z*w),      1 - s*(x*x + z*z),  s*(y*z - x*w)],
        [s*(x*z - y*w),      s*(y*z + x*w),      1 - s*(x*x + y*y)],
    ])


def matrix_to_quat(R) -> tuple:
    """3×3 rotation matrix → quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = float(np.copysign(x, R[2, 1] - R[1, 2]))
    y = float(np.copysign(y, R[0, 2] - R[2, 0]))
    z = float(np.copysign(z, R[1, 0] - R[0, 1]))
    return x, y, z, float(w)


def rotation_angle_between(R_prev, R_curr) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    R_diff = R_prev.T @ R_curr
    cos_angle = float(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cos_angle)


# ── TF message builders ───────────────────────────────────────────────────────

def make_tf_message(transforms_list):
    """Wrap a list of TransformStamped into a TFMessage."""
    from tf2_msgs.msg import TFMessage
    msg = TFMessage()
    msg.transforms = transforms_list
    return msg


def make_transform_stamped(parent, child, translation, qxyzw, t_ns: int):
    """Build a TransformStamped from raw components."""
    from geometry_msgs.msg import TransformStamped
    ts = TransformStamped()
    ts.header.stamp.sec     = int(t_ns // 1_000_000_000)
    ts.header.stamp.nanosec = int(t_ns %  1_000_000_000)
    ts.header.frame_id      = parent
    ts.child_frame_id       = child
    ts.transform.translation.x = float(translation[0])
    ts.transform.translation.y = float(translation[1])
    ts.transform.translation.z = float(translation[2])
    ts.transform.rotation.x = float(qxyzw[0])
    ts.transform.rotation.y = float(qxyzw[1])
    ts.transform.rotation.z = float(qxyzw[2])
    ts.transform.rotation.w = float(qxyzw[3])
    return ts


def identity_quat():
    return (0.0, 0.0, 0.0, 1.0)


# ── Static transforms ─────────────────────────────────────────────────────────

def build_static_tf_bytes(t_ns: int) -> bytes:
    """
    Two static transforms:
      world → sonar_tag     (sonar_head is the world origin)
      object_tag → object_top
    """
    from rclpy.serialization import serialize_message

    # world → sonar_tag:  sonar_tag origin in world frame = -sonar_head_offset
    t_world_sonar_tag = -SONAR_HEAD_OFFSET
    ts_w_st = make_transform_stamped(
        "world", "sonar_tag",
        t_world_sonar_tag, identity_quat(), t_ns,
    )

    # object_tag → object_top
    ts_ot_oc = make_transform_stamped(
        "object_tag", "object_top",
        OBJECT_CTR_OFFSET, identity_quat(), t_ns,
    )

    msg = make_tf_message([ts_w_st, ts_ot_oc])
    return serialize_message(msg)


# ── Single-bag processor ──────────────────────────────────────────────────────

def process_bag(input_path: str, output_path: str, max_jump_deg: float):
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from geometry_msgs.msg import PoseArray

    max_jump_rad = math.radians(max_jump_deg)

    # ── 1. Read metadata from source bag ──────────────────────────────────────
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    existing_topics = reader.get_all_topics_and_types()

    # ── 2. Set up writer with existing topics + /tf + /tf_static ──────────────
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    existing_topic_names = set()
    topic_id = 0
    for tm in existing_topics:
        writer.create_topic(rosbag2_py.TopicMetadata(
            topic_id, tm.name, tm.type, tm.serialization_format,
        ))
        existing_topic_names.add(tm.name)
        topic_id += 1

    for new_topic in (TF_TOPIC, TF_STATIC):
        if new_topic not in existing_topic_names:
            writer.create_topic(rosbag2_py.TopicMetadata(
                topic_id, new_topic, TF_TYPE, "cdr",
            ))
            topic_id += 1

    # ── 3. Stream messages, compute TF, write everything ──────────────────────
    prev_R_rel = None
    static_written = False
    n_tf = 0
    n_dropped = 0

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        # Write static TF once, anchored to the first message timestamp
        if not static_written:
            writer.write(TF_STATIC, build_static_tf_bytes(t_ns), t_ns)
            static_written = True

        # Copy original message verbatim
        writer.write(topic, data, t_ns)

        # Compute dynamic TF from /aruco/poses
        if topic != POSES_TOPIC:
            continue

        msg = deserialize_message(data, PoseArray)
        if len(msg.poses) < 2:
            continue

        p0 = msg.poses[0]   # sonar_tag  in camera frame
        p1 = msg.poses[1]   # object_tag in camera frame

        R0 = quat_to_matrix(p0.orientation.x, p0.orientation.y,
                             p0.orientation.z, p0.orientation.w)
        R1 = quat_to_matrix(p1.orientation.x, p1.orientation.y,
                             p1.orientation.z, p1.orientation.w)
        t0 = np.array([p0.position.x, p0.position.y, p0.position.z])
        t1 = np.array([p1.position.x, p1.position.y, p1.position.z])

        # T_sonar_tag_object_tag = T_cam_sonar_tag^{-1} * T_cam_object_tag
        R_rel = R0.T @ R1
        t_rel = R0.T @ (t1 - t0)

        # Rapid-rotation filter
        if prev_R_rel is not None:
            jump = rotation_angle_between(prev_R_rel, R_rel)
            if jump > max_jump_rad:
                n_dropped += 1
                print(f"    [drop] rotation jump {math.degrees(jump):.1f}° > "
                      f"{max_jump_deg:.1f}°  t={t_ns}")
                continue

        prev_R_rel = R_rel
        qx, qy, qz, qw = matrix_to_quat(R_rel)

        ts = make_transform_stamped(
            "sonar_tag", "object_tag",
            t_rel, (qx, qy, qz, qw), t_ns,
        )
        tf_bytes = serialize_message(make_tf_message([ts]))
        writer.write(TF_TOPIC, tf_bytes, t_ns)
        n_tf += 1

    del writer   # flush + close
    return n_tf, n_dropped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add TF transforms to original wetlab bags and save as new MCAP bags. "
            "Computes sonar_tag → object_tag from /aruco/poses (same logic as "
            "aruco_tf_publisher.py) with a rapid-rotation drop filter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9",
        help="Directory of original rosbag2 sub-folders.",
    )
    parser.add_argument(
        "--output_dir",
        default="/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_tf_bags",
        help="Where to write the new MCAP bags.",
    )
    parser.add_argument(
        "--max_rotation_jump_deg", type=float, default=30.0,
        help="Drop frames where the sonar_tag→object_tag rotation changes by "
             "more than this many degrees between consecutive detections.",
    )
    args = parser.parse_args()

    bags_dir   = Path(args.bags_dir)
    output_dir = Path(args.output_dir)

    if not bags_dir.exists():
        print(f"ERROR: bags_dir not found: {bags_dir}", file=sys.stderr)
        sys.exit(1)

    bags = sorted(
        p for p in bags_dir.iterdir()
        if p.is_dir() and (p / "metadata.yaml").exists()
    )
    if not bags:
        print(f"ERROR: no rosbag2 folders in {bags_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(bags)} bag(s)  →  output: {output_dir}\n")

    for bag in bags:
        out_path = output_dir / bag.name
        if out_path.exists():
            print(f"  [{bag.name}] already exists, skipping")
            continue
        print(f"  [{bag.name}] ...", end=" ", flush=True)
        try:
            n_tf, n_drop = process_bag(str(bag), str(out_path),
                                       args.max_rotation_jump_deg)
            print(f"ok  (tf_frames={n_tf}, dropped={n_drop})")
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone.  New bags in: {output_dir}")
    print()
    print("Next step — visualize one bag:")
    print(f"  python3 visualize_bags_rerun.py \\")
    print(f"      --bag {output_dir}/<bag_name> --save output.rrd")
    print()
    print("Or convert all bags at once:")
    print(f"  python3 visualize_bags_rerun.py \\")
    print(f"      --bags_dir {output_dir} --save wetlab_all.rrd")


if __name__ == "__main__":
    main()
