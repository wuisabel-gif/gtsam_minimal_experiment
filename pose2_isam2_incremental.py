from __future__ import annotations

import json
import math
from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose2
from gtsam.symbol_shorthand import X


OUTPUT_DIR = Path("experiments/gtsam_minimal/results_pose2_isam2")


def build_ground_truth() -> list[Pose2]:
    return [
        Pose2(0.0, 0.0, 0.0),
        Pose2(2.0, 0.0, 0.0),
        Pose2(2.0, 2.0, math.pi / 2.0),
        Pose2(0.0, 2.0, math.pi),
        Pose2(0.0, 0.0, -math.pi / 2.0),
        Pose2(0.0, 0.0, 0.0),
    ]


def integrate_measurements(measurements: list[Pose2]) -> list[Pose2]:
    poses = [Pose2(0.0, 0.0, 0.0)]
    for meas in measurements:
        poses.append(poses[-1].compose(meas))
    return poses


def pose_xy(poses: list[Pose2]) -> np.ndarray:
    return np.array([[pose.x(), pose.y()] for pose in poses], dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = build_ground_truth()
    gt_xy = pose_xy(gt)

    odom_measurements = [
        Pose2(2.08, 0.02, 0.01),
        Pose2(0.02, 1.95, math.pi / 2.0 - 0.03),
        Pose2(2.02, -0.03, math.pi / 2.0 + 0.02),
        Pose2(-0.04, 2.06, math.pi / 2.0 + 0.04),
        Pose2(0.0, 0.0, math.pi / 2.0 - 0.02),
    ]
    raw_poses = integrate_measurements(odom_measurements)
    raw_xy = pose_xy(raw_poses)

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.03]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.12, 0.12, 0.08]))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.04]))

    isam_params = gtsam.ISAM2Params()
    isam_params.setRelinearizeThreshold(0.01)
    isam = gtsam.ISAM2(isam_params)

    # Seed the graph with the prior.
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    graph.add(gtsam.PriorFactorPose2(X(0), gt[0], prior_noise))
    values.insert(X(0), raw_poses[0])
    isam.update(graph, values)

    for i, rel in enumerate(odom_measurements):
        step_graph = gtsam.NonlinearFactorGraph()
        step_values = gtsam.Values()
        step_graph.add(gtsam.BetweenFactorPose2(X(i), X(i + 1), rel, odom_noise))
        step_values.insert(X(i + 1), raw_poses[i + 1])
        isam.update(step_graph, step_values)

    loop_graph = gtsam.NonlinearFactorGraph()
    loop_graph.add(gtsam.BetweenFactorPose2(X(5), X(0), Pose2(0.0, 0.0, 0.0), loop_noise))
    isam.update(loop_graph, gtsam.Values())
    isam.update()

    result = isam.calculateEstimate()
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

    plt.figure(figsize=(6, 5))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], "k-o", label="ground truth")
    plt.plot(raw_xy[:, 0], raw_xy[:, 1], "r--o", label="raw odometry")
    plt.plot(optimized_xy[:, 0], optimized_xy[:, 1], "b-o", label="iSAM2")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Pose2 iSAM2 Incremental Test")
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
