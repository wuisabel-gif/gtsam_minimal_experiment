from __future__ import annotations

import argparse
import json
from pathlib import Path

import gtsam
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import B, V, X


DEFAULT_DATASET_PATH = Path("experiments/gtsam_minimal/data/synthetic_underwater_loop.npz")
DEFAULT_OUTPUT_DIR = Path("experiments/gtsam_minimal/results_underwater")


def pose_xyz(poses: list[gtsam.Pose3]) -> np.ndarray:
    return np.array([[p.x(), p.y(), p.z()] for p in poses], dtype=float)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def save_factor_graph_diagram(output_path: Path, num_keys: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x_coords = np.arange(num_keys, dtype=float)

    y_pose = 2.0
    y_vel = 1.0
    y_bias = 0.0
    y_depth = -1.0
    y_dvl = -2.0

    for i, x in enumerate(x_coords):
        ax.plot(x, y_pose, "o", color="tab:blue", markersize=12)
        ax.text(x, y_pose + 0.17, f"X{i}", ha="center", fontsize=9)
        ax.plot(x, y_vel, "o", color="tab:green", markersize=12)
        ax.text(x, y_vel + 0.17, f"V{i}", ha="center", fontsize=9)
        ax.plot(x, y_bias, "o", color="tab:purple", markersize=12)
        ax.text(x, y_bias + 0.17, f"B{i}", ha="center", fontsize=9)

        ax.plot(x, y_depth, "s", color="tab:orange", markersize=10)
        ax.text(x, y_depth - 0.18, "Depth", ha="center", va="top", fontsize=8)
        ax.plot([x, x], [y_depth, y_pose], color="0.6", linewidth=1.2)

        ax.plot(x, y_dvl, "s", color="tab:red", markersize=10)
        ax.text(x, y_dvl - 0.18, "DVL", ha="center", va="top", fontsize=8)
        ax.plot([x, x], [y_dvl, y_vel], color="0.6", linewidth=1.2)

    for i in range(num_keys - 1):
        mid_x = i + 0.5
        ax.plot(mid_x, 1.5, "s", color="0.25", markersize=10)
        ax.text(mid_x, 1.68, "IMU", ha="center", fontsize=8)
        for xx, yy in [(i, y_pose), (i + 1, y_pose), (i, y_vel), (i + 1, y_vel), (i, y_bias)]:
            ax.plot([mid_x, xx], [1.5, yy], color="0.45", linewidth=1.1)

        ax.plot(mid_x, -0.45, "s", color="tab:brown", markersize=8)
        ax.text(mid_x, -0.68, "Bias", ha="center", fontsize=8)
        ax.plot([mid_x, i], [-0.45, y_bias], color="0.55", linewidth=1.1)
        ax.plot([mid_x, i + 1], [-0.45, y_bias], color="0.55", linewidth=1.1)

    ax.text(-0.8, y_pose, "Pose", ha="right", va="center", fontsize=10)
    ax.text(-0.8, y_vel, "Velocity", ha="right", va="center", fontsize=10)
    ax.text(-0.8, y_bias, "Bias", ha="right", va="center", fontsize=10)
    ax.set_title("IMU + Depth + DVL Factor Graph Structure")
    ax.set_xlim(-1.0, num_keys - 0.2)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_reference_track(data: np.lib.npyio.NpzFile, keyframe_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if {"positions", "velocities", "yaw"}.issubset(data.files):
        return (
            np.asarray(data["positions"][keyframe_indices], dtype=float),
            np.asarray(data["velocities"][keyframe_indices], dtype=float),
            np.asarray(data["yaw"][keyframe_indices], dtype=float),
            "ground truth",
        )

    if {"reference_positions", "reference_velocities", "reference_yaw"}.issubset(data.files):
        return (
            np.asarray(data["reference_positions"], dtype=float),
            np.asarray(data["reference_velocities"], dtype=float),
            np.asarray(data["reference_yaw"], dtype=float),
            "reference odometry",
        )

    raise KeyError(
        "Dataset must contain either positions/velocities/yaw or "
        "reference_positions/reference_velocities/reference_yaw."
    )


def build_reference_poses(reference_positions: np.ndarray, reference_yaw: np.ndarray) -> list[gtsam.Pose3]:
    poses: list[gtsam.Pose3] = []
    for position, yaw in zip(reference_positions, reference_yaw):
        rot = gtsam.Rot3.Yaw(float(yaw))
        poses.append(gtsam.Pose3(rot, gtsam.Point3(*position)))
    return poses


def build_raw_track(
    times: np.ndarray,
    keyframe_indices: np.ndarray,
    measured_gyro_body: np.ndarray,
    dvl_velocity_world: np.ndarray,
    depth_measurements: np.ndarray,
    reference_positions: np.ndarray,
    reference_yaw: np.ndarray,
    imu_dt: float,
) -> tuple[list[gtsam.Pose3], np.ndarray]:
    raw_poses: list[gtsam.Pose3] = []
    raw_velocities = [dvl_velocity_world[0]]

    current_pos = reference_positions[0].copy()
    current_yaw = float(reference_yaw[0])

    for i, k in enumerate(keyframe_indices):
        if i == 0:
            current_pos[2] = -depth_measurements[0]
        else:
            dt = float(times[k] - times[keyframe_indices[i - 1]])
            current_pos = current_pos + dvl_velocity_world[i - 1] * dt
            current_pos[2] = -depth_measurements[i]
            gyro_slice = measured_gyro_body[keyframe_indices[i - 1] : k, 2]
            current_yaw += float(np.sum(gyro_slice) * imu_dt)
            raw_velocities.append(dvl_velocity_world[i])

        raw_pose = gtsam.Pose3(gtsam.Rot3.Yaw(current_yaw), gtsam.Point3(*current_pos))
        raw_poses.append(raw_pose)

    return raw_poses, np.asarray(raw_velocities, dtype=float)


def optimize_dataset(dataset_path: Path, output_dir: Path) -> dict[str, float | str | bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_path)

    times = np.asarray(data["times"], dtype=float)
    measured_accel_body = np.asarray(data["measured_accel_body"], dtype=float)
    measured_gyro_body = np.asarray(data["measured_gyro_body"], dtype=float)
    keyframe_indices = np.asarray(data["keyframe_indices"], dtype=int)
    keyframe_times = np.asarray(data["keyframe_times"], dtype=float)
    dvl_velocity_world = np.asarray(data["dvl_velocity_world"], dtype=float)
    depth_measurements = np.asarray(data["depth_measurements"], dtype=float)
    gravity = float(data["gravity_magnitude"][0])
    accel_sigma = float(data["accel_noise_sigma"][0])
    gyro_sigma = float(data["gyro_noise_sigma"][0])
    dvl_sigma = np.asarray(data["dvl_noise_sigma"], dtype=float)
    depth_sigma = float(data["depth_noise_sigma"][0])
    imu_dt = float(data["imu_dt"][0])
    source = str(data["source"][0]) if "source" in data.files else dataset_path.stem

    reference_positions, reference_velocities, reference_yaw, reference_label = load_reference_track(
        data, keyframe_indices
    )
    reference_poses = build_reference_poses(reference_positions, reference_yaw)
    raw_poses, raw_velocities_arr = build_raw_track(
        times,
        keyframe_indices,
        measured_gyro_body,
        dvl_velocity_world,
        depth_measurements,
        reference_positions,
        reference_yaw,
        imu_dt,
    )

    params = gtsam.PreintegrationParams.MakeSharedU(gravity)
    params.setAccelerometerCovariance(np.eye(3) * accel_sigma**2)
    params.setGyroscopeCovariance(np.eye(3) * gyro_sigma**2)
    params.setIntegrationCovariance(np.eye(3) * 1e-6)

    prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05, 0.03, 0.03, 0.03], dtype=float)
    )
    prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    prior_bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
    bias_between_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
    dvl_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(dvl_sigma, dtype=float))
    depth_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1000.0, 1000.0, depth_sigma], dtype=float))

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    zero_bias = gtsam.imuBias.ConstantBias()

    graph.add(gtsam.PriorFactorPose3(X(0), reference_poses[0], prior_pose_noise))
    graph.add(gtsam.PriorFactorVector(V(0), reference_velocities[0], prior_vel_noise))
    graph.add(gtsam.PriorFactorConstantBias(B(0), zero_bias, prior_bias_noise))

    num_keys = len(keyframe_indices)
    for i in range(num_keys):
        initial.insert(X(i), raw_poses[i])
        initial.insert(V(i), raw_velocities_arr[i])
        initial.insert(B(i), zero_bias)

        depth_point = gtsam.Point3(raw_poses[i].x(), raw_poses[i].y(), -depth_measurements[i])
        graph.add(gtsam.GPSFactor(X(i), depth_point, depth_noise))
        graph.add(gtsam.PriorFactorVector(V(i), dvl_velocity_world[i], dvl_noise))

    for i in range(num_keys - 1):
        pim = gtsam.PreintegratedImuMeasurements(params, zero_bias)
        start = int(keyframe_indices[i])
        stop = int(keyframe_indices[i + 1])
        for j in range(start, stop):
            pim.integrateMeasurement(measured_accel_body[j], measured_gyro_body[j], imu_dt)

        graph.add(gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), B(i), pim))
        graph.add(gtsam.BetweenFactorConstantBias(B(i), B(i + 1), zero_bias, bias_between_noise))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    optimized_poses = [result.atPose3(X(i)) for i in range(num_keys)]
    reference_xyz = pose_xyz(reference_poses)
    raw_xyz = pose_xyz(raw_poses)
    optimized_xyz = pose_xyz(optimized_poses)

    metrics: dict[str, float | str | bool] = {
        "source": source,
        "reference_label": reference_label,
        "num_keyframes": int(num_keys),
        "baseline_rmse": rmse(raw_xyz, reference_xyz),
        "optimized_rmse": rmse(optimized_xyz, reference_xyz),
        "baseline_final_error": float(np.linalg.norm(raw_xyz[-1] - reference_xyz[-1])),
        "optimized_final_error": float(np.linalg.norm(optimized_xyz[-1] - reference_xyz[-1])),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_factor_graph_diagram(output_dir / "factor_graph.png", num_keys)

    plt.figure(figsize=(6, 5))
    plt.plot(reference_xyz[:, 0], reference_xyz[:, 1], "k-o", label=reference_label)
    plt.plot(raw_xyz[:, 0], raw_xyz[:, 1], "r--o", label="raw")
    plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], "g-o", label="optimized")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Underwater GTSAM Prototype (XY)\n{source}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_xy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(keyframe_times, -reference_xyz[:, 2], "k-o", label=f"{reference_label} depth")
    plt.plot(keyframe_times, depth_measurements, "r--o", label="measured depth")
    plt.plot(keyframe_times, -optimized_xyz[:, 2], "g-o", label="optimized depth")
    plt.xlabel("time [s]")
    plt.ylabel("depth [m]")
    plt.title(f"Depth over Time\n{source}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "depth_vs_time.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {output_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the underwater GTSAM prototype on a dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to an .npz dataset with IMU/depth/DVL measurements.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and metrics should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimize_dataset(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
