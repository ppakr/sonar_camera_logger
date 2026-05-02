#!/usr/bin/env bash
# Full data pipeline for a single bag:
#   Step 1 — apply_frame_corrections.py   (re-runs ArUco solvePnP, rebuilds TF)
#   Step 2 — reconstruct_range_image.py   (adds /sonar_3d/range_image)
#   Step 3 — visualize_bags_rerun.py      (saves .rrd for Rerun)
#
# Usage:
#   bash pipeline_single_bag.sh <bag_name> [rrd_output_dir]
#
# Example:
#   bash pipeline_single_bag.sh 1_apr22_static_brick ~/rrd_apr22_range

set -eo pipefail

# ── Paths (edit as needed) ────────────────────────────────────────────────────
# RAW_DIR        : input bags (one subdir per recording session)
# TF_DIR         : post-step-1 staging (TF-corrected only, no range image yet)
# PROCESSED_DIR  : final fully processed bags (TF-corrected + range image)
RAW_DIR=/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr22_raw
TF_DIR=/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr22_tf_corrected
PROCESSED_DIR=/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr22_processed
CALIBRATION=~/auv_ws/src/sonar_camera_logger/config/microsoft_livecam_hd3000.yaml
SCRIPTS=~/auv_ws/src/sonar_camera_logger/scripts

# ── Arguments ─────────────────────────────────────────────────────────────────
BAG=${1:-}
RRD_DIR=${2:-~/rrd_apr22_range}

if [ -z "$BAG" ]; then
    echo "Usage: $0 <bag_name> [rrd_output_dir]"
    echo "Example: $0 1_apr22_static_brick ~/rrd_apr22_range"
    exit 1
fi

RAW_BAG="$RAW_DIR/$BAG"
TF_BAG="$TF_DIR/$BAG"
PROCESSED_BAG="$PROCESSED_DIR/$BAG"
RRD_FILE="$RRD_DIR/${BAG}.rrd"

if [ ! -d "$RAW_BAG" ]; then
    echo "ERROR: raw bag not found: $RAW_BAG"
    exit 1
fi

# ── Source ROS ────────────────────────────────────────────────────────────────
set +u
source ~/auv_ws/install/setup.bash
set -u

mkdir -p "$TF_DIR" "$PROCESSED_DIR" "$RRD_DIR"

# ── Step 1: Apply frame corrections ──────────────────────────────────────────
echo ""
echo "=== Step 1/3: Apply frame corrections ($BAG) ==="
if [ -d "$TF_BAG" ]; then
    echo "  Removing stale TF-corrected bag..."
    rm -rf "$TF_BAG"
fi
python3 "$SCRIPTS/apply_frame_corrections.py" \
    --bags_dir "$RAW_DIR" \
    --bag "$BAG" \
    --output_dir "$TF_DIR" \
    --calibration "$CALIBRATION"

# ── Step 2: Reconstruct range image ──────────────────────────────────────────
echo ""
echo "=== Step 2/3: Reconstruct range image ($BAG) ==="
if [ -d "$PROCESSED_BAG" ]; then
    echo "  Removing stale processed bag..."
    rm -rf "$PROCESSED_BAG"
fi
python3 "$SCRIPTS/reconstruct_range_image.py" \
    --bag "$TF_BAG" \
    --output "$PROCESSED_BAG"

# ── Step 3: Generate Rerun recording ─────────────────────────────────────────
echo ""
echo "=== Step 3/3: Generate Rerun recording ($BAG) ==="
if [ -f "$RRD_FILE" ]; then
    echo "  Removing stale .rrd..."
    rm -f "$RRD_FILE"
fi
python3 "$SCRIPTS/visualize_bags_rerun.py" \
    --bag "$PROCESSED_BAG" \
    --save "$RRD_FILE" \
    --every_nth 2 \
    --image_scale 0.5

echo ""
echo "Done! Open with:"
echo "  rerun $RRD_FILE"
