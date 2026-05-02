#!/usr/bin/env python3
"""
Visualize wet lab sonar bags with Rerun.

Logs per bag frame:
  sonar/pointcloud                — 3D point cloud coloured by intensity (hot colormap)
  sonar/strength_image            — 2D sonar strength bitmap (raw sensor output, uint8)
  sonar/range_image               — 2D sonar range image (float32, metres; zeros = no return)
  camera/image                    — ArUco debug image (camera view)
  aruco/sonar_tag                 — ArUco tag origin frame on the sonar mount (orange dot + axes)
  aruco/sonar_tag/offset_arrow    — Arrow from sonar tag → sonar transducer (0.70 m)
  aruco/sonar_head                — Corrected sonar transducer position + orientation axes (red)
  aruco/object_tag                — ArUco tag origin frame on the floater (light-green dot + axes)
  aruco/object_tag/offset_arrow   — Arrow from object tag → object centre (0.15 m)
  aruco/object_top             — Corrected object centre position + orientation axes (green)
  aruco/sonar_object_line         — Yellow line showing sonar-to-object distance
  metrics/sonar_object_distance   — Euclidean sonar-to-object distance over time (m)

Usage
-----
    source /home/aki/auv_ws/install/setup.bash

    # Open Rerun viewer live (spawns a window):
    python3 visualize_bags_rerun.py --bag <path_to_bag_dir>

    # Save to .rrd file and open later with `rerun file.rrd`:
    python3 visualize_bags_rerun.py --bag <path_to_bag_dir> --save output.rrd

    # Visualize ALL bags in a directory, coloured by material (needs labels CSV):
    python3 visualize_bags_rerun.py \
        --bags_dir /media/aki/2C76C6780AEDB4DB/wl_wetlab_apr9_final_bags \
        --labels ~/bag_previews/labels.csv \
        --save wetlab_all.rrd

    # Slow down playback (useful to inspect individual frames):
    python3 visualize_bags_rerun.py --bag <path> --every_nth 3

Topics consumed
---------------
  /sonar_3d/pointcloud_intensity  — PointCloud2 XYZI
  /sonar_3d/strength_image        — Image mono8 (sonar strength bitmap)
  /sonar_3d/range_image           — Image 32FC1 (reconstructed range, metres)
  /aruco/image_debug              — Image (camera)
  /tf_static                      — TFMessage  (world→sonar_tag, object_tag→object_top)
  /tf                             — TFMessage  (sonar_tag→object_tag, dynamic)
  /aruco/sonar_object_distance    — Float64  (-1 = not visible)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


# ── Colour palette for materials (RGBA uint8) ─────────────────────────────────
MAT_COLOURS = {
    "wood":    [139,  69,  19, 220],
    "metal":   [112, 128, 144, 220],
    "petg":    [ 32, 178, 170, 220],
    "unknown": [200, 200, 200, 180],
}

# Topics
SONAR_PC_TOPIC  = "/sonar_3d/pointcloud_intensity"
STRENGTH_TOPIC  = "/sonar_3d/strength_image"
RANGE_IMG_TOPIC = "/sonar_3d/range_image"
CAMERA_TOPIC    = "/aruco/image_debug"
TF_STATIC_TOPIC = "/tf_static"
TF_TOPIC        = "/tf"
DIST_TOPIC      = "/aruco/sonar_object_distance"

ALL_TOPICS = [SONAR_PC_TOPIC, STRENGTH_TOPIC, RANGE_IMG_TOPIC, CAMERA_TOPIC,
              TF_STATIC_TOPIC, TF_TOPIC, DIST_TOPIC]

# TF frame name → Rerun entity path.
# The path structure encodes the TF hierarchy so Rerun composes transforms
# automatically:  world → world/sonar_link → world/sonar_link/sonar_aruco → …
TF_ENTITY_PATHS = {
    "world":          "frames/world",
    "sonar_link":     "frames/world/sonar_link",
    "sonar_aruco":    "frames/world/sonar_link/sonar_aruco",
    "float_aruco":  "frames/world/sonar_link/sonar_aruco/float_aruco",
    "object_top":  "frames/world/sonar_link/sonar_aruco/float_aruco/object_top",
    # legacy names from earlier bags
    "sonar_tag":      "frames/world/sonar_tag",
    "object_tag":     "frames/world/sonar_tag/object_tag",
}
TF_AXIS_LENGTHS = {
    "world":          0.20,
    "sonar_link":     0.15,
    "sonar_aruco":    0.12,
    "float_aruco":  0.12,
    "object_top":  0.08,
    "sonar_tag":      0.12,
    "object_tag":     0.12,
}
TF_POINT_COLOURS = {
    "world":          [255, 220,  50, 255],   # yellow        — sonar head / world
    "sonar_link":     [255, 180,  20, 255],   # dark-yellow   — sonar transducer
    "sonar_aruco":    [255, 140,  60, 255],   # orange        — sonar ArUco tag
    "float_aruco":  [100, 220, 100, 255],   # lt-green      — floater ArUco tag
    "object_top":  [ 50, 200,  50, 255],   # green         — object centre
    "sonar_tag":      [255, 140,  60, 255],
    "object_tag":     [100, 220, 100, 255],
}
TF_POINT_RADII = {
    "world":          0.040,
    "sonar_link":     0.030,
    "sonar_aruco":    0.020,
    "float_aruco":  0.020,
    "object_top":  0.030,
    "sonar_tag":      0.020,
    "object_tag":     0.020,
}


# ── Message decoders ──────────────────────────────────────────────────────────

_PC2_DTYPE   = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
                5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
_PC2_NBYTES  = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}


def decode_pc2(msg) -> np.ndarray:
    """PointCloud2 → (N, 4) float32 [x, y, z, intensity].  Empty cloud → (0,4)."""
    n = msg.width * msg.height
    if n == 0 or not msg.data:
        return np.empty((0, 4), dtype=np.float32)

    step = msg.point_step
    raw  = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, step)
    fm   = {f.name: (f.offset,
                     _PC2_NBYTES.get(f.datatype, 4),
                     _PC2_DTYPE.get(f.datatype, np.float32))
            for f in msg.fields}

    def col(name):
        if name not in fm:
            return np.zeros(n, dtype=np.float32)
        off, nb, dt = fm[name]
        return np.frombuffer(
            np.ascontiguousarray(raw[:, off:off + nb]).tobytes(), dtype=dt
        ).astype(np.float32)

    pts = np.stack([col("x"), col("y"), col("z"), col("intensity")], axis=1)
    return pts[np.all(np.isfinite(pts[:, :3]), axis=1)]


def decode_image(msg) -> np.ndarray:
    """sensor_msgs/Image → (H, W, C) uint8."""
    # Avoid bytes(msg.data) copy — frombuffer works directly on bytes/array.array.
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    channels = max(1, len(msg.data) // (msg.height * msg.width))
    return arr.reshape(msg.height, msg.width, channels)


def decode_range_image(msg) -> np.ndarray:
    """sensor_msgs/Image (32FC1) → (H, W) float32 in metres. Zeros = no return."""
    arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    # Replace zeros with NaN so Rerun renders them as transparent (no false depth).
    out = arr.copy()
    out[out == 0.0] = np.nan
    return out


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


def intensity_to_rgba(intensity: np.ndarray) -> np.ndarray:
    """Map float intensity values to (N, 4) uint8 RGBA using a hot colormap."""
    if len(intensity) == 0:
        return np.empty((0, 4), dtype=np.uint8)
    lo, hi = float(np.nanmin(intensity)), float(np.nanmax(intensity))
    if hi <= lo:
        t = np.zeros_like(intensity)
    else:
        t = np.clip((intensity - lo) / (hi - lo), 0.0, 1.0)

    # "Hot" colormap: black → red → yellow → white
    r = np.clip(t * 3.0,           0, 1)
    g = np.clip(t * 3.0 - 1.0,     0, 1)
    b = np.clip(t * 3.0 - 2.0,     0, 1)
    a = np.ones_like(r)
    rgba = np.stack([r, g, b, a], axis=1)
    return (rgba * 255).astype(np.uint8)


# ── Blueprint ─────────────────────────────────────────────────────────────────

def make_blueprint() -> rrb.Blueprint:
    """Layout: 3-D view | camera / sonar images (strength + range) / distance plot."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Sonar Point Cloud + TF Frames",
                origin="/",
                contents=["sonar/**", "frames/**"],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Camera",
                    origin="camera/image",
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(
                        name="Sonar Strength Image",
                        origin="sonar/strength_image",
                    ),
                    rrb.Spatial2DView(
                        name="Sonar Range Image (m)",
                        origin="sonar/range_image",
                    ),
                ),
                rrb.TimeSeriesView(
                    name="Sonar-Object Distance (m)",
                    origin="metrics/",
                ),
            ),
            column_shares=[2, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )


# ── Per-bag logging ───────────────────────────────────────────────────────────

def log_bag(
    bag_path: str,
    recording_name: str,
    mat_colour: list[int],
    every_nth: int = 1,
    image_scale: float = 0.5,
):
    """Stream one bag into the active Rerun recording."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2, Image
    from std_msgs.msg import Float64

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=ALL_TOPICS))

    # Log static metadata label
    rr.log(
        f"bag_info/{recording_name}",
        rr.TextDocument(recording_name),
        static=True,
    )

    # Log colour legend for this bag's material
    rr.log(
        "sonar/pointcloud",
        rr.SeriesLines(colors=[mat_colour[:3]], names=[recording_name]),
        static=True,
    )

    def maybe_resize(frame: np.ndarray) -> np.ndarray:
        """Downscale image if image_scale < 1.0 to keep file size manageable."""
        if image_scale >= 1.0:
            return frame
        import cv2
        h, w = frame.shape[:2]
        return cv2.resize(frame, (max(1, int(w * image_scale)),
                                  max(1, int(h * image_scale))))

    sonar_frame_idx = 0
    range_frame_idx = 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        # All timestamps go onto the "ros_time" timeline (seconds since epoch)
        rr.set_time("ros_time", timestamp=t_ns * 1e-9)

        if topic == SONAR_PC_TOPIC:
            sonar_frame_idx += 1
            if sonar_frame_idx % every_nth != 0:
                continue
            msg  = deserialize_message(data, PointCloud2)
            pts  = decode_pc2(msg)
            if len(pts) > 0:
                colours = intensity_to_rgba(pts[:, 3])
                rr.log("sonar/pointcloud",
                       rr.Points3D(pts[:, :3], colors=colours, radii=0.003))

        elif topic == STRENGTH_TOPIC:
            msg   = deserialize_message(data, Image)
            frame = decode_image(msg)
            if frame.shape[2] == 1:
                frame = frame.squeeze(axis=2)
            rr.log("sonar/strength_image", rr.Image(maybe_resize(frame)))

        elif topic == RANGE_IMG_TOPIC:
            range_frame_idx += 1
            if range_frame_idx % every_nth != 0:
                continue
            msg = deserialize_message(data, Image)
            ri  = decode_range_image(msg)
            # DepthImage renders float metre data with a perceptual depth colormap.
            # NaN pixels (no sonar return) appear transparent.
            rr.log("sonar/range_image", rr.DepthImage(ri, meter=1.0))

        elif topic == CAMERA_TOPIC:
            msg   = deserialize_message(data, Image)
            frame = decode_image(msg)
            rr.log("camera/image", rr.Image(maybe_resize(frame)))

        elif topic in (TF_STATIC_TOPIC, TF_TOPIC):
            from tf2_msgs.msg import TFMessage
            msg      = deserialize_message(data, TFMessage)
            is_static = (topic == TF_STATIC_TOPIC)

            for ts in msg.transforms:
                child  = ts.child_frame_id
                path   = TF_ENTITY_PATHS.get(child)
                if path is None:
                    continue   # unknown frame — skip

                t  = np.array([ts.transform.translation.x,
                               ts.transform.translation.y,
                               ts.transform.translation.z])
                q  = ts.transform.rotation
                quat_xyzw = rr.datatypes.Quaternion(xyzw=[q.x, q.y, q.z, q.w])

                axis_len = TF_AXIS_LENGTHS.get(child, 0.10)
                colour   = TF_POINT_COLOURS.get(child, [200, 200, 200, 255])
                radius   = TF_POINT_RADII.get(child, 0.025)

                # Transform3D positions this entity relative to its parent path.
                # Because entity paths mirror the TF hierarchy (e.g.
                # frames/world/sonar_tag/object_tag), Rerun composes parent
                # transforms automatically — giving correct world-frame display.
                rr.log(
                    path,
                    rr.Transform3D(translation=t, quaternion=quat_xyzw),
                    rr.TransformAxes3D(axis_length=axis_len),
                    static=is_static,
                )
                # Point dot at the frame origin (logged in local frame → [0,0,0])
                rr.log(
                    f"{path}/origin",
                    rr.Points3D([[0.0, 0.0, 0.0]],
                                radii=radius,
                                colors=[colour],
                                labels=[child]),
                    static=is_static,
                )

        elif topic == DIST_TOPIC:
            msg = deserialize_message(data, Float64)
            if msg.data >= 0:   # -1.0 is sentinel for "not both visible"
                rr.log("metrics/sonar_object_distance",
                       rr.Scalars(msg.data))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize wet-lab sonar bags in Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--bag",
        help="Path to a single rosbag2 directory to visualize.",
    )
    src.add_argument(
        "--bags_dir",
        help="Directory of rosbag2 sub-folders (use with --labels).",
    )

    parser.add_argument(
        "--labels",
        default=None,
        help="CSV with columns: bag, shape, material, notes. "
             "Used to colour-code by material when --bags_dir is given.",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="FILE.rrd",
        help="Save recording to this .rrd file instead of spawning the viewer. "
             "Open later with:  rerun FILE.rrd",
    )
    parser.add_argument(
        "--every_nth", type=int, default=1,
        help="Log every Nth sonar frame (1 = all, 2 = half, etc.). "
             "Use >1 to reduce file size when saving.",
    )
    parser.add_argument(
        "--image_scale", type=float, default=0.5,
        help="Scale factor for camera and sonar strength images (0.5 = half size). "
             "Reduces .rrd file size significantly. Use 1.0 for full resolution.",
    )
    args = parser.parse_args()

    # ── Collect bags to process ───────────────────────────────────────────────
    if args.bag:
        bag_path = Path(args.bag)
        if not (bag_path / "metadata.yaml").exists():
            print(f"ERROR: not a valid bag directory: {bag_path}", file=sys.stderr)
            sys.exit(1)
        bags = [(bag_path, "unknown", MAT_COLOURS["unknown"])]

    else:  # --bags_dir
        bags_dir = Path(args.bags_dir)
        all_bags = sorted(
            p for p in bags_dir.iterdir()
            if p.is_dir() and (p / "metadata.yaml").exists()
        )
        if not all_bags:
            print(f"ERROR: no bags found in {bags_dir}", file=sys.stderr)
            sys.exit(1)

        # Load labels if provided
        label_map: dict[str, dict] = {}
        if args.labels:
            with open(args.labels, newline="") as f:
                for row in csv.DictReader(f):
                    label_map[row["bag"]] = row

        bags = []
        for p in all_bags:
            lbl  = label_map.get(p.name, {})
            mat  = lbl.get("material", "unknown").strip().lower()
            sh   = lbl.get("shape",    "").strip().lower()
            name = f"{sh}/{mat}" if sh else p.name
            col  = MAT_COLOURS.get(mat, MAT_COLOURS["unknown"])
            bags.append((p, name, col))

    # ── Initialise Rerun ──────────────────────────────────────────────────────
    rr.init("wl_wetlab_sonar", spawn=(args.save is None))
    rr.send_blueprint(make_blueprint())

    # Tell Rerun the scene uses FRD (Forward-Right-Down) coordinates so the
    # 3D view orients correctly (sonar_link is FRD: +x forward, +z down).
    rr.log("/", rr.ViewCoordinates.FRD, static=True)

    if args.save:
        rr.save(args.save)
        print(f"Recording to {args.save} ...")

    # ── Process bags ──────────────────────────────────────────────────────────
    print(f"Processing {len(bags)} bag(s)  "
          f"[every_nth={args.every_nth}, image_scale={args.image_scale}]")
    for bag_path, name, colour in bags:
        print(f"  [{bag_path.name}]  label={name} ...", flush=True)
        log_bag(str(bag_path), name, colour, args.every_nth, args.image_scale)
        print("  done")

    if args.save:
        print(f"\nSaved → {args.save}")
        print(f"Open with:  rerun {args.save}")
    else:
        print("\nStreaming complete. Close the Rerun viewer to exit.")


if __name__ == "__main__":
    main()
