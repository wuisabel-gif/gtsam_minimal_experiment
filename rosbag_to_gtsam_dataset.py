from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - intended for ROS2 runtime environments
    rosbag2_py = None
    deserialize_message = None
    get_message = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


DEFAULT_OUTPUT_PATH = Path("experiments/gtsam_minimal/data/jetson_rosbag_underwater.npz")


def quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def stamp_to_sec(msg: Any, fallback_nsec: int) -> float:
    header = getattr(msg, "header", None)
    if header is None:
        return float(fallback_nsec) * 1e-9
    stamp = header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def nearest_index(times: np.ndarray, value: float) -> int:
    idx = int(np.searchsorted(times, value, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before = times[idx - 1]
    after = times[idx]
    return idx - 1 if abs(value - before) <= abs(after - value) else idx


def read_messages(bag_path: Path, imu_topic: str, depth_topic: str, dvl_topic: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    if rosbag2_py is None or deserialize_message is None or get_message is None:
        raise RuntimeError(
            "rosbag2_py/rclpy serialization support is required. "
            f"Import failed with: {IMPORT_ERROR!r}"
        )

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    for topic in (imu_topic, depth_topic, dvl_topic):
        if topic not in topic_types:
            raise KeyError(f"Topic {topic!r} not found in rosbag {bag_path}")

    filter_topics = [imu_topic, depth_topic, dvl_topic]
    reader.set_filter(rosbag2_py.StorageFilter(topics=filter_topics))

    imu_type = get_message(topic_types[imu_topic])
    depth_type = get_message(topic_types[depth_topic])
    dvl_type = get_message(topic_types[dvl_topic])

    imu_messages: list[dict[str, Any]] = []
    depth_messages: list[dict[str, Any]] = []
    dvl_messages: list[dict[str, Any]] = []

    while reader.has_next():
        topic_name, serialized, bag_nsec = reader.read_next()
        if topic_name == imu_topic:
            msg = deserialize_message(serialized, imu_type)
            imu_messages.append({"stamp_sec": stamp_to_sec(msg, bag_nsec), "msg": msg})
        elif topic_name == depth_topic:
            msg = deserialize_message(serialized, depth_type)
            depth_messages.append({"stamp_sec": stamp_to_sec(msg, bag_nsec), "msg": msg})
        elif topic_name == dvl_topic:
            msg = deserialize_message(serialized, dvl_type)
            dvl_messages.append({"stamp_sec": stamp_to_sec(msg, bag_nsec), "msg": msg})

    return imu_messages, depth_messages, dvl_messages, topic_types[depth_topic]


def depth_to_meters(depth_type: str, msg: Any, args: argparse.Namespace) -> float:
    if depth_type == "sensor_msgs/msg/Range":
        return float(msg.range)
    if depth_type == "sensor_msgs/msg/FluidPressure":
        return float((msg.fluid_pressure - args.atmospheric_pressure_pa) / (args.fluid_density * args.gravity))
    raise ValueError(f"Unsupported depth topic type: {depth_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract /barracuda IMU, depth, and DVL topics from a ROS2 bag into a GTSAM .npz dataset."
    )
    parser.add_argument("bag", type=Path, help="Path to a ROS2 bag directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .npz dataset path.")
    parser.add_argument("--imu-topic", default="/barracuda/imu/data")
    parser.add_argument("--depth-topic", default="/barracuda/depth")
    parser.add_argument("--dvl-topic", default="/barracuda/dvl/odometry")
    parser.add_argument("--accel-noise-sigma", type=float, default=0.03)
    parser.add_argument("--gyro-noise-sigma", type=float, default=0.005)
    parser.add_argument("--dvl-noise-sigma", nargs=3, type=float, default=[0.04, 0.04, 0.05])
    parser.add_argument("--depth-noise-sigma", type=float, default=0.03)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--fluid-density", type=float, default=997.0)
    parser.add_argument("--atmospheric-pressure-pa", type=float, default=101325.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    imu_messages, depth_messages, dvl_messages, depth_type = read_messages(
        args.bag, args.imu_topic, args.depth_topic, args.dvl_topic
    )
    if not imu_messages or not depth_messages or not dvl_messages:
        raise RuntimeError("Bag must contain non-empty IMU, depth, and DVL topics.")

    imu_times = np.array([entry["stamp_sec"] for entry in imu_messages], dtype=float)
    measured_accel_body = np.array(
        [
            [
                float(entry["msg"].linear_acceleration.x),
                float(entry["msg"].linear_acceleration.y),
                float(entry["msg"].linear_acceleration.z),
            ]
            for entry in imu_messages
        ],
        dtype=float,
    )
    measured_gyro_body = np.array(
        [
            [
                float(entry["msg"].angular_velocity.x),
                float(entry["msg"].angular_velocity.y),
                float(entry["msg"].angular_velocity.z),
            ]
            for entry in imu_messages
        ],
        dtype=float,
    )

    depth_times = np.array([entry["stamp_sec"] for entry in depth_messages], dtype=float)
    depth_values = np.array([depth_to_meters(depth_type, entry["msg"], args) for entry in depth_messages], dtype=float)

    dvl_times = np.array([entry["stamp_sec"] for entry in dvl_messages], dtype=float)
    keyframe_indices = np.array([nearest_index(imu_times, stamp) for stamp in dvl_times], dtype=int)

    depth_match_indices = np.array([nearest_index(depth_times, stamp) for stamp in dvl_times], dtype=int)
    depth_measurements = depth_values[depth_match_indices]

    reference_positions = np.array(
        [
            [
                float(entry["msg"].pose.pose.position.x),
                float(entry["msg"].pose.pose.position.y),
                float(entry["msg"].pose.pose.position.z),
            ]
            for entry in dvl_messages
        ],
        dtype=float,
    )
    reference_velocities = np.array(
        [
            [
                float(entry["msg"].twist.twist.linear.x),
                float(entry["msg"].twist.twist.linear.y),
                float(entry["msg"].twist.twist.linear.z),
            ]
            for entry in dvl_messages
        ],
        dtype=float,
    )
    reference_yaw = np.array(
        [
            quat_xyzw_to_yaw(
                float(entry["msg"].pose.pose.orientation.x),
                float(entry["msg"].pose.pose.orientation.y),
                float(entry["msg"].pose.pose.orientation.z),
                float(entry["msg"].pose.pose.orientation.w),
            )
            for entry in dvl_messages
        ],
        dtype=float,
    )

    imu_dt = float(np.median(np.diff(imu_times))) if len(imu_times) > 1 else 0.05
    keyframe_dt = float(np.median(np.diff(dvl_times))) if len(dvl_times) > 1 else 1.0

    np.savez(
        args.output,
        times=imu_times,
        measured_accel_body=measured_accel_body,
        measured_gyro_body=measured_gyro_body,
        keyframe_indices=keyframe_indices,
        keyframe_times=dvl_times,
        dvl_velocity_world=reference_velocities,
        depth_measurements=depth_measurements,
        reference_positions=reference_positions,
        reference_velocities=reference_velocities,
        reference_yaw=reference_yaw,
        imu_dt=np.array([imu_dt], dtype=float),
        keyframe_dt=np.array([keyframe_dt], dtype=float),
        accel_noise_sigma=np.array([args.accel_noise_sigma], dtype=float),
        gyro_noise_sigma=np.array([args.gyro_noise_sigma], dtype=float),
        dvl_noise_sigma=np.array(args.dvl_noise_sigma, dtype=float),
        depth_noise_sigma=np.array([args.depth_noise_sigma], dtype=float),
        gravity_magnitude=np.array([args.gravity], dtype=float),
        source=np.array([f"rosbag:{args.bag.name}"]),
        depth_topic_type=np.array([depth_type]),
    )
    print(f"Wrote dataset to {args.output}")


if __name__ == "__main__":
    main()
