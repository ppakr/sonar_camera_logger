#!/usr/bin/env python3
"""
Verify corrected_poses by back-projecting positions onto the camera image.

For each frame where /aruco/corrected_poses is available:
  1. Recover the tag origin (t_tag = p_corrected - R @ offset)
  2. Project t_tag through the camera intrinsics → should land on the ArUco
     marker centre in /aruco/image_debug
  3. Project p_corrected → shows where the sonar head / object centre is
     relative to the image

Outputs
-------
  <output_dir>/verify_frame_NNNN.png   — annotated camera images

Usage
-----
    source /home/aki/auv_ws/install/setup.bash
    python3 verify_tf.py --bag /media/.../rosbag2_2026_04_09-14_34_35
    python3 verify_tf.py --bag ... --n_frames 8 --output_dir /tmp/tf_check
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Physical offsets (ENU, tag → component) ──────────────────────────────────
# Must match the values used when running apply_frame_corrections.py
OFFSETS = {
    "sonar_head":    np.array([0.00,  0.05, -0.70]),
    "object_top": np.array([0.00,  0.00, -0.15]),
}
# Ordering in corrected_poses: [sonar_head, object_top]
POSE_ORDER = ["sonar_head", "object_top"]

CALIB_PATH = (
    "/home/aki/auv_ws/src/sonar_camera_logger/config/calibration_parameters.yaml"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_calibration(yaml_path: str):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {yaml_path}")
    K = fs.getNode("cameraMatrix").mat()
    D = fs.getNode("distCoeffs").mat()
    fs.release()
    return K, D


def quat_wxyz_to_matrix(w, x, y, z) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation matrix."""
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z),  s*(x*y - z*w),      s*(x*z + y*w)     ],
        [s*(x*y + z*w),      1 - s*(x*x + z*z),  s*(y*z - x*w)     ],
        [s*(x*z - y*w),      s*(y*z + x*w),      1 - s*(x*x + y*y) ],
    ])


def project(pt3d: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """Project a 3-D camera-frame point to 2-D pixel coordinates."""
    if pt3d[2] <= 0:
        return None   # behind camera
    px = K[0, 0] * pt3d[0] / pt3d[2] + K[0, 2]
    py = K[1, 1] * pt3d[1] / pt3d[2] + K[1, 2]
    return int(round(px)), int(round(py))


def decode_image(msg) -> np.ndarray:
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    ch  = max(1, len(msg.data) // (msg.height * msg.width))
    img = arr.reshape(msg.height, msg.width, ch)
    if ch == 1:
        return cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
    return img.copy()


# ── Main verification ─────────────────────────────────────────────────────────

def verify_bag(bag_path: str, K: np.ndarray, n_frames: int, output_dir: str):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import Float64

    CAMERA_TOPIC = "/aruco/image_debug"
    POSES_TOPIC  = "/aruco/corrected_poses"
    DIST_TOPIC   = "/aruco/sonar_object_distance"

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=[CAMERA_TOPIC, POSES_TOPIC, DIST_TOPIC]
    ))

    # Buffer messages by timestamp
    cam_buf: dict[int, np.ndarray]         = {}
    pose_buf: dict[int, object]            = {}
    dist_buf: dict[int, float]             = {}

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == CAMERA_TOPIC:
            cam_buf[t_ns] = deserialize_message(data, Image)
        elif topic == POSES_TOPIC:
            pose_buf[t_ns] = deserialize_message(data, PoseArray)
        elif topic == DIST_TOPIC:
            dist_buf[t_ns] = deserialize_message(data, Float64).data

    print(f"  camera frames : {len(cam_buf)}")
    print(f"  pose frames   : {len(pose_buf)}")
    print(f"  dist frames   : {len(dist_buf)}")

    # Match pose timestamps to nearest camera frame
    cam_times  = np.array(sorted(cam_buf.keys()))
    pose_times = sorted(t for t, m in pose_buf.items() if len(m.poses) > 0)

    if not pose_times:
        print("  No corrected_poses with detected markers found.")
        print("  → The bag may predate the frame-correction pipeline.")
        return

    # Pick n_frames evenly spaced
    idx_pick = np.linspace(0, len(pose_times) - 1, min(n_frames, len(pose_times)),
                           dtype=int)
    selected = [pose_times[i] for i in idx_pick]

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0

    for t_pose in selected:
        pose_msg = pose_buf[t_pose]
        n_poses  = len(pose_msg.poses)

        # Nearest camera frame
        nearest_cam_t = cam_times[np.argmin(np.abs(cam_times - t_pose))]
        dt_ms = abs(nearest_cam_t - t_pose) / 1e6
        img = decode_image(cam_buf[nearest_cam_t])

        # Nearest distance measurement
        dist_times = np.array(sorted(dist_buf.keys()))
        measured_dist = -1.0
        if len(dist_times) > 0:
            nearest_dist_t = dist_times[np.argmin(np.abs(dist_times - t_pose))]
            measured_dist = dist_buf[nearest_dist_t]

        print(f"\n  Frame {frame_idx:03d}  t={t_pose}  n_poses={n_poses}"
              f"  cam_dt={dt_ms:.1f}ms  dist={measured_dist:.4f}m")

        # ── Per-pose analysis ──────────────────────────────────────────────────
        positions = []
        for i, pose in enumerate(pose_msg.poses):
            if i >= len(POSE_ORDER):
                break
            label  = POSE_ORDER[i]
            offset = OFFSETS[label]

            # Corrected component position in camera frame
            p = np.array([pose.position.x, pose.position.y, pose.position.z])
            o = pose.orientation
            w, x, y, z = o.w, o.x, o.y, o.z
            R = quat_wxyz_to_matrix(w, x, y, z)

            # Back-compute tag origin
            t_tag = p - R @ offset

            # Distance (depth from camera)
            tag_depth  = t_tag[2]
            comp_depth = p[2]

            print(f"    [{label}]")
            print(f"      tag origin   (camera frame) : {t_tag}")
            print(f"      component    (camera frame) : {p}")
            print(f"      tag depth    : {tag_depth:.3f} m")
            print(f"      component depth : {comp_depth:.3f} m")
            print(f"      offset in cam frame : {p - t_tag}")

            positions.append((label, t_tag, p, R, offset))

            # ── Project onto image ─────────────────────────────────────────────
            # Tag origin → should land near ArUco marker centre
            pix_tag  = project(t_tag, K)
            pix_comp = project(p, K)

            COLORS = {
                "sonar_head":    (80, 80, 255),   # red (BGR)
                "object_top": (80, 200, 80),   # green (BGR)
            }
            col = COLORS[label]

            if pix_tag:
                # Draw tag origin: cross-hair
                cv2.drawMarker(img, pix_tag, col, cv2.MARKER_CROSS, 30, 2)
                cv2.putText(img, f"{label[:3]}_tag  d={tag_depth:.2f}m",
                            (pix_tag[0] + 6, pix_tag[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            if pix_comp:
                # Draw component position: filled circle
                cv2.circle(img, pix_comp, 10, col, -1)
                cv2.putText(img, f"{label[:3]}  d={comp_depth:.2f}m",
                            (pix_comp[0] + 12, pix_comp[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            if pix_tag and pix_comp:
                cv2.line(img, pix_tag, pix_comp, col, 1)

        # ── Verify distance consistency ───────────────────────────────────────
        if len(positions) == 2:
            _, _, p_s, _, _ = positions[0]   # sonar_head
            _, _, p_o, _, _ = positions[1]   # object_top
            computed_dist = float(np.linalg.norm(p_s - p_o))
            print(f"    Distance check:")
            print(f"      Computed from positions : {computed_dist:.4f} m")
            print(f"      From dist topic         : {measured_dist:.4f} m")
            match = abs(computed_dist - measured_dist) < 0.001
            print(f"      Match                   : {'✓ OK' if match else '✗ MISMATCH'}")

            # Draw distance line on image
            for _, _, p_s2, _, _ in [positions[0]]:
                for _, _, p_o2, _, _ in [positions[1]]:
                    ps2d = project(p_s2, K)
                    po2d = project(p_o2, K)
                    if ps2d and po2d:
                        cv2.line(img, ps2d, po2d, (0, 220, 220), 2)
                        mid = ((ps2d[0]+po2d[0])//2, (ps2d[1]+po2d[1])//2)
                        cv2.putText(img, f"{computed_dist:.3f}m", mid,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)

        # ── Overlay info banner ───────────────────────────────────────────────
        banner_txt = [
            f"Frame {frame_idx:03d}   n_poses={n_poses}   cam_dt={dt_ms:.1f}ms",
            "Cross=tag origin (should align with ArUco marker)",
            "Circle=corrected component position",
        ]
        for bi, bt in enumerate(banner_txt):
            cv2.putText(img, bt, (6, 20 + bi * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)

        out_path = os.path.join(output_dir, f"verify_frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, img)
        print(f"    Saved → {out_path}")
        frame_idx += 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify corrected_poses TF by back-projecting positions onto "
            "the camera image. Tag origin cross-hairs should align with ArUco "
            "markers; component circles show sonar/object positions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bag", required=True,
                        help="Path to a rosbag2 directory (must be a final bag "
                             "with /aruco/corrected_poses).")
    parser.add_argument("--calibration", default=CALIB_PATH)
    parser.add_argument("--n_frames", type=int, default=6,
                        help="Number of frames to verify.")
    parser.add_argument("--output_dir", default="/tmp/tf_verify",
                        help="Where to write annotated PNG images.")
    args = parser.parse_args()

    if not Path(args.bag).is_dir():
        print(f"ERROR: bag not found: {args.bag}", file=sys.stderr)
        sys.exit(1)

    K, D = load_calibration(args.calibration)
    print(f"Camera matrix:\n{K}\n")

    print(f"Verifying bag: {args.bag}")
    verify_bag(args.bag, K, args.n_frames, args.output_dir)

    print(f"\nDone. Open images in {args.output_dir}/")
    print("What to look for:")
    print("  ✓ Cross-hairs (tag origins) overlap the ArUco markers in the image")
    print("  ✓ Circles (component positions) are offset by the expected distance")
    print("  ✓ Distance computed from positions matches the dist topic (printed above)")


if __name__ == "__main__":
    main()
