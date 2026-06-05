from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "barracuda_estimation"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

if TYPE_CHECKING:
    from barracuda_estimation.gtsam_estimator import GtsamEstimator
    from barracuda_estimation.measurement_types import DepthSample, DvlSample, ImuSample


def load_estimator_types() -> tuple[type["GtsamEstimator"], type["DepthSample"], type["DvlSample"], type["ImuSample"]]:
    from barracuda_estimation.gtsam_estimator import GtsamEstimator
    from barracuda_estimation.measurement_types import DepthSample, DvlSample, ImuSample

    return GtsamEstimator, DepthSample, DvlSample, ImuSample


GtsamEstimator, DepthSample, DvlSample, ImuSample = load_estimator_types()


DATASET_PATH = REPO_ROOT / "experiments" / "gtsam_minimal" / "data" / "synthetic_underwater_loop.npz"
OUTPUT_DIR = REPO_ROOT / "experiments" / "gtsam_minimal" / "results_backend_harness"


def yaw_to_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def build_dead_reckoned_xy(
    times: np.ndarray,
    keyframe_indices: np.ndarray,
    dvl_velocity_world: np.ndarray,
    depth_measurements: np.ndarray,
) -> np.ndarray:
    keyframe_times = times[keyframe_indices]
    dead_reckoned = np.zeros((len(keyframe_indices), 3), dtype=float)
    dead_reckoned[0, 2] = -float(depth_measurements[0])

    for i in range(1, len(keyframe_indices)):
        dt = float(keyframe_times[i] - keyframe_times[i - 1])
        dead_reckoned[i, :2] = dead_reckoned[i - 1, :2] + dvl_velocity_world[i - 1, :2] * dt
        dead_reckoned[i, 2] = -float(depth_measurements[i])

    return dead_reckoned


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATASET_PATH)

    times = data["times"]
    positions = data["positions"]
    yaw = data["yaw"]
    measured_accel_body = data["measured_accel_body"]
    measured_gyro_body = data["measured_gyro_body"]
    keyframe_indices = data["keyframe_indices"]
    keyframe_times = data["keyframe_times"]
    dvl_velocity_world = data["dvl_velocity_world"]
    depth_measurements = data["depth_measurements"]

    gt_positions = positions[keyframe_indices]
    gt_yaw = yaw[keyframe_indices]
    dead_reckoned_positions = build_dead_reckoned_xy(
        times, keyframe_indices, dvl_velocity_world, depth_measurements
    )

    estimator = GtsamEstimator()
    estimates: list[np.ndarray] = []
    estimate_times: list[float] = []
    statuses: list[dict[str, object]] = []

    for key_i, keyframe_index in enumerate(keyframe_indices):
        start_idx = 0 if key_i == 0 else int(keyframe_indices[key_i - 1])
        stop_idx = int(keyframe_index) + 1
        for imu_idx in range(start_idx, stop_idx):
            estimator.add_imu(
                ImuSample(
                    stamp_sec=float(times[imu_idx]),
                    linear_accel=tuple(float(v) for v in measured_accel_body[imu_idx]),
                    angular_vel=tuple(float(v) for v in measured_gyro_body[imu_idx]),
                )
            )

        estimator.add_depth(
            DepthSample(
                stamp_sec=float(keyframe_times[key_i]),
                z_value=float(depth_measurements[key_i]),
                source="synthetic_range",
            )
        )
        estimator.add_dvl(
            DvlSample(
                stamp_sec=float(keyframe_times[key_i]),
                position_xyz=tuple(float(v) for v in dead_reckoned_positions[key_i]),
                velocity_xyz=tuple(float(v) for v in dvl_velocity_world[key_i]),
                orientation_xyzw=yaw_to_xyzw(float(gt_yaw[key_i])),
            )
        )

        status_before = estimator.status()
        estimate = estimator.step(float(keyframe_times[key_i]))
        status_after = estimator.status()
        statuses.append(
            {
                "time_sec": float(keyframe_times[key_i]),
                "status_before": status_before.__dict__,
                "status_after": status_after.__dict__,
                "produced_estimate": estimate is not None,
            }
        )
        if estimate is None:
            continue

        estimate_times.append(float(keyframe_times[key_i]))
        estimates.append(np.asarray(estimate.position_xyz, dtype=float))

    if not estimates:
        raise RuntimeError("Backend harness produced no estimates.")

    estimated_positions = np.vstack(estimates)
    matched_gt_positions = gt_positions[: len(estimated_positions)]
    matched_dead_reckoned = dead_reckoned_positions[: len(estimated_positions)]

    metrics = {
        "num_keyframes": int(len(keyframe_indices)),
        "num_estimates": int(len(estimated_positions)),
        "gtsam_available": bool(estimator.gtsam_available),
        "baseline_rmse": rmse(matched_dead_reckoned, matched_gt_positions),
        "estimated_rmse": rmse(estimated_positions, matched_gt_positions),
        "baseline_final_error": float(np.linalg.norm(matched_dead_reckoned[-1] - matched_gt_positions[-1])),
        "estimated_final_error": float(np.linalg.norm(estimated_positions[-1] - matched_gt_positions[-1])),
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (OUTPUT_DIR / "step_status.json").open("w", encoding="utf-8") as f:
        json.dump(statuses, f, indent=2)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], "k-o", label="ground truth")
    plt.plot(dead_reckoned_positions[:, 0], dead_reckoned_positions[:, 1], "r--o", label="dead reckoned")
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], "g-o", label="backend estimate")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("GtsamEstimator Backend Harness (XY)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_xy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(keyframe_times, -gt_positions[:, 2], "k-o", label="ground truth depth")
    plt.plot(keyframe_times, depth_measurements, "r--o", label="measured depth")
    plt.plot(estimate_times, -estimated_positions[:, 2], "g-o", label="backend estimate depth")
    plt.xlabel("time [s]")
    plt.ylabel("depth [m]")
    plt.title("GtsamEstimator Backend Harness (Depth)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "depth_vs_time.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
