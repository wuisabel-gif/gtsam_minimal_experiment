from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import gtsam
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose2
from gtsam.symbol_shorthand import X


def add_system_python_paths() -> None:
    """
    Make Ubuntu/ROS system packages visible when the script is run from a venv.

    On the Jetson host, `rosbag2_py`, `rclpy`, and even `yaml` may live under
    `/usr/lib/python3/dist-packages`, while the user runs this script from a
    virtual environment created in the workspace. Adding the system dist-package
    paths keeps the host-side rosbag tooling usable without rebuilding the ROS
    Python packages inside the venv.
    """

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/local/lib/python{version}/dist-packages"),
        Path(f"/usr/lib/python{version}/dist-packages"),
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.append(candidate_str)


add_system_python_paths()

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


DEFAULT_OUTPUT_DIR = Path("experiments/gtsam_minimal/results_pose2_rosbag")


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


def pose_xy(poses: list[Pose2]) -> np.ndarray:
    return np.array([[pose.x(), pose.y()] for pose in poses], dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def save_factor_graph_diagram(output_path: Path, num_poses: int) -> None:
    fig, ax = plt.subplots(figsize=(max(8, num_poses * 0.5), 3.5))

    pose_y = 1.0
    factor_y = 0.0
    pose_x = np.arange(num_poses, dtype=float)

    ax.plot(0.0, factor_y, "s", color="tab:orange", markersize=12)
    ax.text(0.0, factor_y - 0.2, "Prior", ha="center", va="top", fontsize=9)
    ax.plot([0.0, 0.0], [factor_y, pose_y], color="0.55", linewidth=1.5)

    for x in pose_x:
        ax.plot(x, pose_y, "o", color="tab:blue", markersize=12)
        ax.text(x, pose_y + 0.18, f"x{int(x)}", ha="center", va="bottom", fontsize=9)

    for i in range(num_poses - 1):
        fx = i + 0.5
        ax.plot(fx, factor_y, "s", color="tab:orange", markersize=10)
        ax.text(fx, factor_y - 0.2, "Between", ha="center", va="top", fontsize=8)
        ax.plot([fx, i], [factor_y, pose_y], color="0.55", linewidth=1.4)
        ax.plot([fx, i + 1], [factor_y, pose_y], color="0.55", linewidth=1.4)

    ax.text(-0.7, pose_y, "Pose nodes", ha="right", va="center", fontsize=10)
    ax.text(-0.7, factor_y, "Factor nodes", ha="right", va="center", fontsize=10)
    ax.set_title("Pose2 Factor Graph from Rosbag Pose Topic")
    ax.set_xlim(-1.0, num_poses - 0.2)
    ax.set_ylim(-0.45, 1.45)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def read_pose_messages(bag_path: Path, pose_topic: str) -> tuple[list[dict[str, Any]], str]:
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
    if pose_topic not in topic_types:
        raise KeyError(f"Topic {pose_topic!r} not found in rosbag {bag_path}")

    pose_type_str = topic_types[pose_topic]
    pose_type = get_message(pose_type_str)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[pose_topic]))

    messages: list[dict[str, Any]] = []
    while reader.has_next():
        topic_name, serialized, bag_nsec = reader.read_next()
        if topic_name != pose_topic:
            continue
        msg = deserialize_message(serialized, pose_type)
        messages.append({"stamp_sec": stamp_to_sec(msg, bag_nsec), "msg": msg})

    return messages, pose_type_str


def downsample_messages(messages: list[dict[str, Any]], stride: int) -> list[dict[str, Any]]:
    stride = max(1, int(stride))
    return messages[::stride]


def pose2_from_message(msg: Any, topic_type: str) -> Pose2:
    if topic_type == "geometry_msgs/msg/PoseStamped":
        pose = msg.pose
    elif topic_type == "nav_msgs/msg/Odometry":
        pose = msg.pose.pose
    else:
        raise ValueError(f"Unsupported pose topic type: {topic_type}")

    yaw = quat_xyzw_to_yaw(
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    return Pose2(float(pose.position.x), float(pose.position.y), yaw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Pose2 factor graph from a rosbag pose topic such as /barracuda/zed_node/pose."
    )
    parser.add_argument("bag", type=Path, help="Path to a ROS2 bag directory.")
    parser.add_argument("--pose-topic", default="/barracuda/zed_node/pose")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=5, help="Keep every Nth pose message to limit graph size.")
    parser.add_argument("--trans-sigma", type=float, default=0.05)
    parser.add_argument("--rot-sigma", type=float, default=0.03)
    parser.add_argument("--init-jitter-xy", type=float, default=0.03)
    parser.add_argument("--init-jitter-yaw", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_messages, pose_type = read_pose_messages(args.bag, args.pose_topic)
    if not all_messages:
        raise RuntimeError(f"No messages found on {args.pose_topic}")

    messages = downsample_messages(all_messages, args.stride)
    poses = [pose2_from_message(entry["msg"], pose_type) for entry in messages]
    if len(poses) < 2:
        raise RuntimeError("Need at least two pose messages to build a pose graph.")

    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([args.trans_sigma, args.trans_sigma, args.rot_sigma], dtype=float)
    )
    between_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([args.trans_sigma, args.trans_sigma, args.rot_sigma], dtype=float)
    )

    graph.add(gtsam.PriorFactorPose2(X(0), poses[0], prior_noise))
    for i in range(len(poses) - 1):
        rel = poses[i].between(poses[i + 1])
        graph.add(gtsam.BetweenFactorPose2(X(i), X(i + 1), rel, between_noise))

    rng = np.random.default_rng(7)
    initial = gtsam.Values()
    raw_poses: list[Pose2] = []
    for i, pose in enumerate(poses):
        if i == 0:
            perturbed = pose
        else:
            perturbed = Pose2(
                pose.x() + float(rng.normal(0.0, args.init_jitter_xy)),
                pose.y() + float(rng.normal(0.0, args.init_jitter_xy)),
                pose.theta() + float(rng.normal(0.0, args.init_jitter_yaw)),
            )
        raw_poses.append(perturbed)
        initial.insert(X(i), perturbed)

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    optimized = [result.atPose2(X(i)) for i in range(len(poses))]

    reference_xy = pose_xy(poses)
    raw_xy = pose_xy(raw_poses)
    optimized_xy = pose_xy(optimized)

    metrics = {
        "source": f"rosbag:{args.bag.name}",
        "topic": args.pose_topic,
        "topic_type": pose_type,
        "num_messages_total": int(len(all_messages)),
        "num_poses_used": int(len(poses)),
        "stride": int(args.stride),
        "raw_rmse_to_reference": rmse(raw_xy, reference_xy),
        "optimized_rmse_to_reference": rmse(optimized_xy, reference_xy),
        "raw_final_error": float(np.linalg.norm(raw_xy[-1] - reference_xy[-1])),
        "optimized_final_error": float(np.linalg.norm(optimized_xy[-1] - reference_xy[-1])),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_factor_graph_diagram(args.output_dir / "factor_graph.png", len(poses))

    plt.figure(figsize=(6, 5))
    plt.plot(reference_xy[:, 0], reference_xy[:, 1], "k-o", label="reference pose topic")
    plt.plot(raw_xy[:, 0], raw_xy[:, 1], "r--o", label="perturbed initial")
    plt.plot(optimized_xy[:, 0], optimized_xy[:, 1], "g-o", label="optimized")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Pose2 Graph from Rosbag Pose Topic")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "trajectory_xy.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
