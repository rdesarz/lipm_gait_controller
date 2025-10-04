import numpy as np
from matplotlib import pyplot as plt

from lipm_walking_controller.foot import compute_feet_path_and_poses

if __name__ == "__main__":
    # Parameters
    dt = 0.02  # Delta of time of the model simulation

    # Foot generation parameters
    t_ss = 1.0  # Single support phase time window
    t_ds = 0.2  # Double support phase time window
    lf_initial_pose = np.array([0.0, 0.1, 0.0])
    rf_initial_pose = np.array([0.0, -0.1, 0.0])
    n_steps = 5
    l_stride = 0.3
    max_height_foot = 0.2

    # Build feet path and poses
    t, lf_path, rf_path, steps_pose, _ = compute_feet_path_and_poses(
        rf_initial_pose,
        lf_initial_pose,
        n_steps,
        t_ss,
        t_ds,
        l_stride,
        dt,
        max_height_foot,
    )

    # Figure
    fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(12, 12))

    # Plot left and right feet trajectories on z-axis
    axes[0].plot(t, lf_path[:, 2], label="Left foot trajectory")
    axes[0].plot(t, rf_path[:, 2], label="Right foot trajectory")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("z pos [m]")
    axes[0].set_title("Feet trajectories on z-axis")

    # Plot left and right feet trajectories on x-axis
    axes[1].plot(t, lf_path[:, 0], label="Left foot trajectory")
    axes[1].plot(t, rf_path[:, 0], label="Right foot trajectory")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("x pos [m]")
    axes[1].set_title("Feet trajectories on x-axis")

    plt.show()
