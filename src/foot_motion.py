import math

import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio
from shapely import Polygon, Point, affinity, union
from shapely.ops import nearest_points
from scipy.linalg import solve_discrete_are


def compute_zmp_ref(com_initial_pose, steps, dt, ss_t, ds_t):
    T = int((len(steps) - 1) * (ss_t + ds_t) / dt + (ds_t + ss_t) / dt)

    t = np.arange(T) * dt
    zmp_ref = np.zeros([T, 2])

    # Step on the first foot
    mask = t < ds_t
    alpha = t[mask] / ds_t
    zmp_ref[mask, :] = (1 - alpha)[:, None] * com_initial_pose + alpha[:, None] * steps[0]

    # Alternate between foot
    for idx, (current_step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
        # Compute current time range
        t_start = ds_t + idx * (ss_t + ds_t)

        # Add single support phase
        zmp_ref[(t >= t_start) & (t < t_start + ss_t)] = current_step

        # Add double support phase
        mask = (t >= t_start + ss_t) & (t < t_start + ss_t + ds_t)
        alpha = (t[mask] - (t_start + ss_t)) / ds_t
        zmp_ref[mask, :] = (1 - alpha)[:, None] * current_step + alpha[:, None] * next_step

    # Last phase is single support at last foot pose
    mask = t >= ds_t + (len(steps) - 1) * (ss_t + ds_t)
    zmp_ref[mask, :] = steps[-1]

    return t, zmp_ref


def compute_double_support_polygon(current_foot_pose, next_foot_pose, foot_shape):
    curent_foot = affinity.translate(foot_shape, xoff=current_foot_pose[0], yoff=current_foot_pose[1])
    next_foot = affinity.translate(foot_shape, xoff=next_foot_pose[0], yoff=next_foot_pose[1])

    return union(curent_foot, next_foot).convex_hull


def compute_single_support_polygon(current_foot_pose, foot_shape):
    return affinity.translate(foot_shape, xoff=current_foot_pose[0], yoff=current_foot_pose[1])


def plot_steps(axes, steps_pose, step_shape):
    # Plot double support polygon
    for current_step, next_step in zip(steps_pose[:-1], steps_pose[1:]):
        support_polygon = compute_double_support_polygon(current_step, next_step, step_shape)

        x, y = support_polygon.exterior.xy
        axes.plot(x, y, color="blue")  # outline
        axes.fill(x, y, color="lightblue", alpha=0.5)  # filled polygon

    # Plot single support polygon
    for current_step in steps_pose:
        support_polygon = compute_single_support_polygon(current_step, step_shape)

        x, y = support_polygon.exterior.xy
        axes.plot(x, y, color="red")  # outline
        axes.fill(x, y, color="red", alpha=0.5)  # filled polygon


def get_active_polygon(k, dt, steps_pose, t_ss, t_ds, foot_shape):
    tk = k * dt
    t_step = t_ss + t_ds
    i = int(tk / t_step)
    i = min(i, len(steps_pose) - 2)
    t_in = tk - i * t_step
    if t_in < t_ss:
        return compute_single_support_polygon(steps_pose[i], foot_shape)
    elif tk >= (len(steps_pose) - 1) * t_step:
        return compute_single_support_polygon(steps_pose[-1], foot_shape)
    else:
        return compute_double_support_polygon(steps_pose[i], steps_pose[i + 1], foot_shape)


def clamp_to_polygon(u_xy, poly: Polygon):
    p = Point(u_xy[0], u_xy[1])
    if poly.contains(p):
        return u_xy

    # nearest point on boundary
    q = nearest_points(poly.exterior, p)[0]

    return np.array([q.x, q.y])


def compute_feet_path(rf_initial_pose, lf_initial_pose, n_steps, t_ss, t_ds, l_stride, dt, max_height_foot):
    # The sequence is the following:
    # Start with a double support phase to switch CoM on right foot
    # Then n_steps, for each step there is a single support phase and a double support phase. The length of the step is
    # given by l_stride.
    # At the last step, we add a single support step to join both feet at the same level and a double support step to
    # place the CoM in the middle of the feet

    total_time = t_ds + (n_steps + 1) * (t_ss + t_ds)
    N = int(total_time / dt)
    t = np.arange(N) * dt

    # Initialize path
    rf_path = np.zeros([N, 3])
    lf_path = np.zeros([N, 3])

    # Switch of the CoM to the right foot implies both feet stays at the same position
    mask = t < t_ds
    rf_path[mask, :] = rf_initial_pose
    lf_path[mask, :] = lf_initial_pose

    # Compute motion of left foot
    mask = (t >= t_ds) & (t < t_ss + t_ds)
    theta = (t[mask] - t_ds) * math.pi / t_ss
    lf_path[mask, 0] = np.sin(theta) * max_height_foot

    return t, lf_path, rf_path

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
        (3, 3))  # Cost on the state vector variation. Zero by default as we don't want to penalize strong variation.
    R = 1e-6  # Cost on the input command u(t)

    # ZMP reference parameters
    t_ss = 1.0  # Single support phase time window
    t_ds = 0.2  # Double support phase time window
    com_initial_pose = np.array([0.0, 0.0])
    lf_initial_pose = np.array([0.0, 0.1, 0.0])
    rf_initial_pose = np.array([0.0, -0.1, 0.0])
    foot_shape = Polygon(((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))
    steps_pose = np.array([[0.0, -0.1],
                           [0.3, 0.1],
                           [0.6, -0.1],
                           [0.9, 0.1]])
    n_steps = 3
    l_stride = 0.3
    max_height_foot = 0.2

    # Build ZMP reference to track
    # t, zmp_ref = compute_zmp_ref(com_initial_pose, steps_pose, dt, t_ss, t_ds)

    t, lf_path, rf_path = compute_feet_path(rf_initial_pose, lf_initial_pose, n_steps, t_ss, t_ds, l_stride, dt, max_height_foot)

    # Figure
    fig, axes = plt.subplots(2, 2, layout="constrained", figsize=(12,8))

    axes[0, 0].plot(t, lf_path[:, 2], label='Left foot trajectory')
    axes[0, 0].plot(t, rf_path[:, 2], label='Right foot trajectory')
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("t [s]")
    axes[0, 0].set_ylabel("z pos [m]")
    axes[0, 0].set_title("Feet trajectories")

    # Plot ZMP reference vs COM on the x axis
    # axes[0, 1].plot(t, zmp_ref[:, 0], label='ZMP reference [x]')
    axes[0, 1].plot(t, lf_path[:, 0], label='Left foot trajectory')
    axes[0, 1].plot(t, rf_path[:, 0], label='Right foot trajectory')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("t [s]")
    axes[0, 1].set_ylabel("x pos [m]")
    axes[0, 1].set_title("Feet and ZMP reference on z axis")

    # Plot ZMP reference vs COM on the y axis
    # axes[1, 1].plot(t, zmp_ref[:, 1], label='ZMP reference [y]')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("y pos [m]")
    axes[1, 1].set_title("ZMP reference vs COM position on Y-axis")

    active_poly_patch = None

    ax_x = axes[0, 1]
    ax_y = axes[1, 1]
    for a in (ax_x, ax_y):
        a.grid(True)

    plt.show()


