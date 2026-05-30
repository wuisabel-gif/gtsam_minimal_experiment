from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gtsam
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from gtsam.symbol_shorthand import X

from run_pose2_from_rosbag import (
    IMPORT_ERROR,
    add_system_python_paths,
    downsample_messages,
    read_pose_messages,
    stamp_to_sec,
)

add_system_python_paths()


DEFAULT_OUTPUT_DIR = Path("experiments/gtsam_minimal/results_pose3_rosbag")


def pose_xyz(poses: list[gtsam.Pose3]) -> np.ndarray:
    return np.array([[pose.x(), pose.y(), pose.z()] for pose in poses], dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def save_factor_graph_diagram(output_path: Path, num_poses: int) -> None:
    fig, ax = plt.subplots(figsize=(max(8, num_poses * 0.5), 4.0))

    pose_y = 1.0
    factor_y = 0.0
    pose_x = np.arange(num_poses, dtype=float)

    ax.plot(0.0, factor_y, "s", color="tab:orange", markersize=12)
    ax.text(0.0, factor_y - 0.2, "Prior3", ha="center", va="top", fontsize=9)
    ax.plot([0.0, 0.0], [factor_y, pose_y], color="0.55", linewidth=1.5)

    for x in pose_x:
        ax.plot(x, pose_y, "o", color="tab:blue", markersize=12)
        ax.text(x, pose_y + 0.18, f"X{int(x)}", ha="center", va="bottom", fontsize=9)

    for i in range(num_poses - 1):
        fx = i + 0.5
        ax.plot(fx, factor_y, "s", color="tab:orange", markersize=10)
        ax.text(fx, factor_y - 0.2, "Between3", ha="center", va="top", fontsize=8)
        ax.plot([fx, i], [factor_y, pose_y], color="0.55", linewidth=1.4)
        ax.plot([fx, i + 1], [factor_y, pose_y], color="0.55", linewidth=1.4)

    ax.text(-0.7, pose_y, "Pose3 nodes", ha="right", va="center", fontsize=10)
    ax.text(-0.7, factor_y, "Factor nodes", ha="right", va="center", fontsize=10)
    ax.set_title("Pose3 Factor Graph from Rosbag Pose Topic")
    ax.set_xlim(-1.0, num_poses - 0.2)
    ax.set_ylim(-0.45, 1.45)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def pose3_from_message(msg: Any, topic_type: str):
    if topic_type == "geometry_msgs/msg/PoseStamped":
        pose = msg.pose
    elif topic_type == "nav_msgs/msg/Odometry":
        pose = msg.pose.pose
    else:
        raise ValueError(f"Unsupported pose topic type: {topic_type}")

    rot = gtsam.Rot3.Quaternion(
        float(pose.orientation.w),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
    )
    point = gtsam.Point3(
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    return gtsam.Pose3(rot, point)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Pose3 factor graph from a rosbag pose topic such as /barracuda/zed_node/pose."
    )
    parser.add_argument("bag", type=Path, help="Path to a ROS2 bag directory.")
    parser.add_argument("--pose-topic", default="/barracuda/zed_node/pose")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=5, help="Keep every Nth pose message to limit graph size.")
    parser.add_argument("--trans-sigma", type=float, default=0.05)
    parser.add_argument("--rot-sigma", type=float, default=0.05)
    parser.add_argument("--init-jitter-xyz", type=float, default=0.03)
    parser.add_argument("--init-jitter-rot", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_messages, pose_type = read_pose_messages(args.bag, args.pose_topic)
    if not all_messages:
        raise RuntimeError(f"No messages found on {args.pose_topic}")

    messages = downsample_messages(all_messages, args.stride)
    poses = [pose3_from_message(entry["msg"], pose_type) for entry in messages]
    if len(poses) < 2:
        raise RuntimeError("Need at least two pose messages to build a pose graph.")

    graph = gtsam.NonlinearFactorGraph()
    sigmas = np.array(
        [
            args.rot_sigma,
            args.rot_sigma,
            args.rot_sigma,
            args.trans_sigma,
            args.trans_sigma,
            args.trans_sigma,
        ],
        dtype=float,
    )
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    between_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    graph.add(gtsam.PriorFactorPose3(X(0), poses[0], prior_noise))
    for i in range(len(poses) - 1):
        rel = poses[i].between(poses[i + 1])
        graph.add(gtsam.BetweenFactorPose3(X(i), X(i + 1), rel, between_noise))

    rng = np.random.default_rng(7)
    initial = gtsam.Values()
    raw_poses: list[gtsam.Pose3] = []
    for i, pose in enumerate(poses):
        if i == 0:
            perturbed = pose
        else:
            jitter_t = np.array(
                [
                    float(rng.normal(0.0, args.init_jitter_xyz)),
                    float(rng.normal(0.0, args.init_jitter_xyz)),
                    float(rng.normal(0.0, args.init_jitter_xyz)),
                ],
                dtype=float,
            )
            jitter_r = np.array(
                [
                    float(rng.normal(0.0, args.init_jitter_rot)),
                    float(rng.normal(0.0, args.init_jitter_rot)),
                    float(rng.normal(0.0, args.init_jitter_rot)),
                ],
                dtype=float,
            )
            perturb = gtsam.Pose3(gtsam.Rot3.RzRyRx(jitter_r), gtsam.Point3(*jitter_t))
            perturbed = pose.compose(perturb)
        raw_poses.append(perturbed)
        initial.insert(X(i), perturbed)

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    optimized = [result.atPose3(X(i)) for i in range(len(poses))]

    reference_xyz = pose_xyz(poses)
    raw_xyz = pose_xyz(raw_poses)
    optimized_xyz = pose_xyz(optimized)

    metrics = {
        "source": f"rosbag:{args.bag.name}",
        "topic": args.pose_topic,
        "topic_type": pose_type,
        "num_messages_total": int(len(all_messages)),
        "num_poses_used": int(len(poses)),
        "stride": int(args.stride),
        "raw_rmse_to_reference_xyz": rmse(raw_xyz, reference_xyz),
        "optimized_rmse_to_reference_xyz": rmse(optimized_xyz, reference_xyz),
        "raw_final_error_xyz": float(np.linalg.norm(raw_xyz[-1] - reference_xyz[-1])),
        "optimized_final_error_xyz": float(np.linalg.norm(optimized_xyz[-1] - reference_xyz[-1])),
        "reference_z_min": float(np.min(reference_xyz[:, 2])),
        "reference_z_max": float(np.max(reference_xyz[:, 2])),
        "optimized_z_min": float(np.min(optimized_xyz[:, 2])),
        "optimized_z_max": float(np.max(optimized_xyz[:, 2])),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_factor_graph_diagram(args.output_dir / "factor_graph.png", len(poses))

    plt.figure(figsize=(7, 5))
    plt.plot(reference_xyz[:, 0], reference_xyz[:, 2], "k-o", label="reference x-z")
    plt.plot(raw_xyz[:, 0], raw_xyz[:, 2], "r--o", label="perturbed initial x-z")
    plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 2], "g-o", label="optimized x-z")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Pose3 Graph from Rosbag Pose Topic (x-z)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "trajectory_xz.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(reference_xyz[:, 0], reference_xyz[:, 1], "k-o", label="reference x-y")
    plt.plot(raw_xyz[:, 0], raw_xyz[:, 1], "r--o", label="perturbed initial x-y")
    plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], "g-o", label="optimized x-y")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Pose3 Graph from Rosbag Pose Topic (x-y)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "trajectory_xy.png", dpi=150)
    plt.close()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(reference_xyz[:, 0], reference_xyz[:, 1], reference_xyz[:, 2], "k-o", label="reference")
    ax.plot(raw_xyz[:, 0], raw_xyz[:, 1], raw_xyz[:, 2], "r--o", label="perturbed initial")
    ax.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], optimized_xyz[:, 2], "g-o", label="optimized")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Pose3 Graph from Rosbag Pose Topic (3D)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory_xyz.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
