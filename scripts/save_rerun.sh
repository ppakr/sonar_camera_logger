#!/usr/bin/env bash
set -eo pipefail

RANGE=/media/aki/2C76C6780AEDB4DB/wl_wetlab_apr22_range
VIZ=~/auv_ws/src/sonar_camera_logger/scripts/visualize_bags_rerun.py

# colcon setup scripts use unbound variables — disable -u around the source
set +u
source ~/auv_ws/install/setup.bash
set -u

for bag_dir in "$RANGE"/*/; do
  name=$(basename "$bag_dir")
  rrd="$RANGE/${name}.rrd"
  if [ -f "$rrd" ]; then
    echo "[SKIP] $name"
    continue
  fi
  echo "=== $name ==="
  python3 "$VIZ" --bag "$bag_dir" --save "$rrd" --every_nth 2 --image_scale 0.5
done
