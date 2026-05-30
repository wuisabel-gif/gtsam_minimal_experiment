#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/root/bags/barracuda_estimation_bag}"

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "Recording Barracuda estimation topics into: $OUTPUT_DIR"
echo "Press Ctrl+C to stop."

ros2 bag record \
  --output "$OUTPUT_DIR" \
  /barracuda/imu/data \
  /barracuda/depth \
  /barracuda/dvl/odometry \
  /tf \
  /tf_static
