#!/usr/bin/env bash
# Batch-run pipeline_single_bag.sh over every bag in RAW_DIR.
#
# Skips bags whose final .rrd already exists, keeps going on per-bag failures,
# and prints an OK / SKIP / FAIL summary at the end.
#
# Usage:
#   bash pipeline_all_bags.sh [rrd_output_dir]
#
# Edit RAW_DIR below (or override on the command line via env) when switching
# sessions; the per-bag script picks up matching PROCESSED_DIR / RANGE_DIR
# from its own defaults.

set -eo pipefail

# ── Paths (edit as needed) ────────────────────────────────────────────────────
RAW_DIR=${RAW_DIR:-/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr22_raw}
RRD_OUT=${1:-${RRD_OUT:-$HOME/rrd_apr22_range}}
PIPELINE=$HOME/auv_ws/src/sonar_camera_logger/scripts/pipeline_single_bag.sh

if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: RAW_DIR not found: $RAW_DIR"
    exit 1
fi
if [ ! -x "$PIPELINE" ]; then
    echo "ERROR: pipeline script not found or not executable: $PIPELINE"
    exit 1
fi

mkdir -p "$RRD_OUT"

ok=0
skip=0
fail=0
failed=()

shopt -s nullglob
for bag_dir in "$RAW_DIR"/*/; do
    name=$(basename "$bag_dir")
    rrd="$RRD_OUT/${name}.rrd"

    if [ -f "$rrd" ]; then
        echo "[SKIP] $name (rrd already exists)"
        skip=$((skip + 1))
        continue
    fi

    echo ""
    echo "########################################"
    echo "### $name"
    echo "########################################"
    if bash "$PIPELINE" "$name" "$RRD_OUT"; then
        ok=$((ok + 1))
    else
        echo "[FAIL] $name (exit $?)"
        fail=$((fail + 1))
        failed+=("$name")
    fi
done

echo ""
echo "=== Batch summary ==="
echo "OK   : $ok"
echo "SKIP : $skip"
echo "FAIL : $fail"
if [ "$fail" -gt 0 ]; then
    printf '  failed: %s\n' "${failed[@]}"
    exit 1
fi
