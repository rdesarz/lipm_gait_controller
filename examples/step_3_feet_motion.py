import math

import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon

from lipm_walking_controller.controller import compute_zmp_ref
from lipm_walking_controller.foot import compute_feet_path_and_poses

if __name__ == "__main__":
    # Parameters
    dt = 0.005  # Delta of time of the model simulation
    g = 9.81  # Gravity
    zc = 0.814  # Height of the COM
    w = math.sqrt(g / zc)

    # Preview controller parameters
    t_preview = 1.6  # Time horizon used for the preview controller
    n_preview_steps = int(round(t_preview / dt))
    Qe = 1.0  # Cost on the integral error of the ZMP reference
    Qx = np.zeros(
        (3, 3)
    )  # Cost on the state vector variation. Zero by default as
    # we don't want to penalize strong variation.
    R = 1e-6  # Cost on the input command u(t)

    # ZMP reference parameters
    t_ss = 1.0  # Single support phase time window
    t_ds = 0.2  # Double support phase time window
    com_initial_pose = np.array([0.0, 0.0])
    lf_initial_pose = np.array([0.0, 0.1, 0.0])
    rf_initial_pose = np.array([0.0, -0.1, 0.0])
    foot_shape = Polygon(((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))
    n_steps = 10
    l_stride = 0.1
    max_height_foot = 0.2

    # Build ZMP reference to track
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

    zmp_ref = compute_zmp_ref(t, com_initial_pose, steps_pose, t_ss, t_ds)

    # Figure
    fig, axes = plt.subplots(2, 2, layout="constrained", figsize=(12, 8))

    axes[1, 1].plot(t, lf_path[:, 2], label="Left foot trajectory")
    axes[1, 1].plot(t, rf_path[:, 2], label="Right foot trajectory")
    axes[1, 1].axis("equal")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("t [s]")
    axes[1, 1].set_ylabel("z pos [m]")
    axes[1, 1].set_title("Feet trajectories")

    # Plot ZMP reference vs COM on the x axis
    axes[0, 1].plot(t, zmp_ref[:, 0], label="ZMP reference [x]")
    axes[0, 1].plot(t, lf_path[:, 0], label="Left foot trajectory")
    axes[0, 1].plot(t, rf_path[:, 0], label="Right foot trajectory")
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("t [s]")
    axes[0, 1].set_ylabel("x pos [m]")
    axes[0, 1].set_title("Feet and ZMP reference on z axis")

    # Plot ZMP reference vs COM on the y axis
    axes[0, 0].plot(t, zmp_ref[:, 1], label="ZMP reference [y]")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("y pos [m]")
    axes[0, 0].set_title("ZMP reference vs COM position on Y-axis")

    active_poly_patch = None

    ax_x = axes[0, 1]
    ax_y = axes[1, 1]
    for a in (ax_x, ax_y):
        a.grid(True)

    plt.show()
