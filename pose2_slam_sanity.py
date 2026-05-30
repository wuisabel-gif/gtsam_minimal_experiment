from __future__ import annotations

import json
import math
from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose2
from gtsam.symbol_shorthand import X


OUTPUT_DIR = Path("experiments/gtsam_minimal/results_pose2_batch")


def build_ground_truth() -> list[Pose2]:
    """Closed square trajectory that returns to the start pose."""
    return [
        Pose2(0.0, 0.0, 0.0),
        Pose2(2.0, 0.0, 0.0),
        Pose2(2.0, 2.0, math.pi / 2.0),
        Pose2(0.0, 2.0, math.pi),
        Pose2(0.0, 0.0, -math.pi / 2.0),
        Pose2(0.0, 0.0, 0.0),
    ]


def between_sequence(poses: list[Pose2]) -> list[Pose2]:
    return [poses[i].between(poses[i + 1]) for i in range(len(poses) - 1)]


def integrate_measurements(measurements: list[Pose2]) -> list[Pose2]:
    poses = [Pose2(0.0, 0.0, 0.0)]
    for meas in measurements:
        poses.append(poses[-1].compose(meas))
    return poses


def pose_xy(poses: list[Pose2]) -> np.ndarray:
    return np.array([[pose.x(), pose.y()] for pose in poses], dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def save_factor_graph_diagram(output_path: Path, num_poses: int) -> None:
    """Draw a simple bipartite view of pose nodes and factor nodes."""
    fig, ax = plt.subplots(figsize=(9, 3.5))

    pose_y = 1.0
    factor_y = 0.0
    pose_x = np.arange(num_poses, dtype=float)

    # Prior + odometry factors + loop closure factor.
    factor_specs = [
        ("prior", -0.35, [0], "Prior"),
        ("odom01", 0.5, [0, 1], "Between"),
        ("odom12", 1.5, [1, 2], "Between"),
        ("odom23", 2.5, [2, 3], "Between"),
        ("odom34", 3.5, [3, 4], "Between"),
        ("odom45", 4.5, [4, 5], "Between"),
        ("loop", 2.5, [0, 5], "Loop"),
    ]

    for x in pose_x:
        ax.plot(x, pose_y, "o", color="tab:blue", markersize=14)
        ax.text(x, pose_y + 0.18, f"x{int(x)}", ha="center", va="bottom", fontsize=10)

    for _, fx, connected_poses, label in factor_specs:
        ax.plot(fx, factor_y, "s", color="tab:orange", markersize=12)
        ax.text(fx, factor_y - 0.2, label, ha="center", va="top", fontsize=9)
        for px in connected_poses:
            ax.plot([fx, px], [factor_y, pose_y], color="0.55", linewidth=1.5)

    ax.text(-0.7, pose_y, "Pose nodes", ha="right", va="center", fontsize=10)
    ax.text(-0.7, factor_y, "Factor nodes", ha="right", va="center", fontsize=10)
    ax.set_title("Pose2 Factor Graph Structure")
    ax.set_xlim(-1.0, num_poses - 0.2)
    ax.set_ylim(-0.45, 1.45)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = build_ground_truth()
    gt_xy = pose_xy(gt)

    # Slightly biased odometry to make loop closure correction visible.
    odom_measurements = [
        Pose2(2.08, 0.02, 0.01),
        Pose2(0.02, 1.95, math.pi / 2.0 - 0.03),
        Pose2(2.02, -0.03, math.pi / 2.0 + 0.02),
        Pose2(-0.04, 2.06, math.pi / 2.0 + 0.04),
        Pose2(0.0, 0.0, math.pi / 2.0 - 0.02),
    ]
    raw_poses = integrate_measurements(odom_measurements)
    raw_xy = pose_xy(raw_poses)

    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.03]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.12, 0.12, 0.08]))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.04]))

    graph.add(gtsam.PriorFactorPose2(X(0), gt[0], prior_noise))
    for i, rel in enumerate(odom_measurements):
        graph.add(gtsam.BetweenFactorPose2(X(i), X(i + 1), rel, odom_noise))

    # Loop closure: final pose should align with the start pose.
    graph.add(gtsam.BetweenFactorPose2(X(5), X(0), Pose2(0.0, 0.0, 0.0), loop_noise))

    initial = gtsam.Values()
    for i, pose in enumerate(raw_poses):
        initial.insert(X(i), pose)

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    optimized = [result.atPose2(X(i)) for i in range(len(raw_poses))]
    optimized_xy = pose_xy(optimized)

    metrics = {
        "baseline_rmse": rmse(raw_xy, gt_xy),
        "optimized_rmse": rmse(optimized_xy, gt_xy),
        "baseline_final_error": float(np.linalg.norm(raw_xy[-1] - gt_xy[-1])),
        "optimized_final_error": float(np.linalg.norm(optimized_xy[-1] - gt_xy[-1])),
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_factor_graph_diagram(OUTPUT_DIR / "factor_graph.png", len(raw_poses))

    plt.figure(figsize=(6, 5))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], "k-o", label="ground truth")
    plt.plot(raw_xy[:, 0], raw_xy[:, 1], "r--o", label="raw odometry")
    plt.plot(optimized_xy[:, 0], optimized_xy[:, 1], "g-o", label="optimized")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Pose2 SLAM Sanity Test")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_xy.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
