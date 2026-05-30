from __future__ import annotations

from pathlib import Path

import numpy as np


OUTPUT_PATH = Path("experiments/gtsam_minimal/data/synthetic_underwater_loop.npz")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    duration = 20.0
    imu_dt = 0.05
    keyframe_dt = 1.0
    gravity = 9.81

    times = np.arange(0.0, duration + 1e-9, imu_dt)
    omega = 2.0 * np.pi / duration

    x = 3.0 * np.cos(omega * times)
    y = 2.0 * np.sin(omega * times)
    z = -1.5 - 0.35 * np.sin(2.0 * omega * times)
    positions = np.column_stack((x, y, z))

    vx = -3.0 * omega * np.sin(omega * times)
    vy = 2.0 * omega * np.cos(omega * times)
    vz = -0.35 * 2.0 * omega * np.cos(2.0 * omega * times)
    velocities = np.column_stack((vx, vy, vz))

    ax = -3.0 * omega**2 * np.cos(omega * times)
    ay = -2.0 * omega**2 * np.sin(omega * times)
    az = 0.35 * (2.0 * omega) ** 2 * np.sin(2.0 * omega * times)
    accelerations = np.column_stack((ax, ay, az))

    yaw = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(yaw, imu_dt)

    # Only yaw rotation for this simple prototype.
    specific_force_body = np.zeros_like(accelerations)
    gyro_body = np.zeros_like(accelerations)
    for i, psi in enumerate(yaw):
        c = np.cos(psi)
        s = np.sin(psi)
        rot_world_body = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        g_world = np.array([0.0, 0.0, -gravity])
        specific_force_body[i] = rot_world_body @ (accelerations[i] - g_world)
        gyro_body[i] = np.array([0.0, 0.0, yaw_rate[i]])

    rng = np.random.default_rng(7)
    accel_noise_sigma = 0.03
    gyro_noise_sigma = 0.005
    dvl_noise_sigma = np.array([0.04, 0.04, 0.05], dtype=float)
    depth_noise_sigma = 0.03

    measured_accel_body = specific_force_body + rng.normal(
        0.0, accel_noise_sigma, size=specific_force_body.shape
    )
    measured_gyro_body = gyro_body + rng.normal(
        0.0, gyro_noise_sigma, size=gyro_body.shape
    )

    keyframe_indices = np.arange(0, len(times), int(round(keyframe_dt / imu_dt)), dtype=int)
    if keyframe_indices[-1] != len(times) - 1:
        keyframe_indices = np.append(keyframe_indices, len(times) - 1)
    keyframe_times = times[keyframe_indices]

    dvl_velocity_world = velocities[keyframe_indices] + rng.normal(
        0.0, dvl_noise_sigma, size=(len(keyframe_indices), 3)
    )
    # Depth is positive downward.
    depth_measurements = -positions[keyframe_indices, 2] + rng.normal(
        0.0, depth_noise_sigma, size=len(keyframe_indices)
    )

    np.savez(
        OUTPUT_PATH,
        times=times,
        positions=positions,
        velocities=velocities,
        yaw=yaw,
        measured_accel_body=measured_accel_body,
        measured_gyro_body=measured_gyro_body,
        keyframe_indices=keyframe_indices,
        keyframe_times=keyframe_times,
        dvl_velocity_world=dvl_velocity_world,
        depth_measurements=depth_measurements,
        imu_dt=np.array([imu_dt]),
        keyframe_dt=np.array([keyframe_dt]),
        accel_noise_sigma=np.array([accel_noise_sigma]),
        gyro_noise_sigma=np.array([gyro_noise_sigma]),
        dvl_noise_sigma=dvl_noise_sigma,
        depth_noise_sigma=np.array([depth_noise_sigma]),
        gravity_magnitude=np.array([gravity]),
        source=np.array(["synthetic_underwater_loop"]),
    )
    print(f"Wrote dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
