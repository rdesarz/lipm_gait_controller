import math

import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon, Point, affinity, union
from shapely.ops import nearest_points
from scipy.linalg import solve_discrete_are


def compute_zmp_ref(com_initial_pose, steps, dt, ss_t, ds_t):
    T = int((len(steps) - 1) * (ss_t + ds_t) / dt + ds_t / dt)

    t = np.arange(T) * dt
    zmp_ref = np.zeros([T, 2])

    # Step on the first foot
    mask = t < ds_t
    alpha = t[mask] / ds_t
    zmp_ref[mask, :] = (1 - alpha)[:, None] * com_initial_pose + alpha[:, None] * steps[0]

    for idx, (current_step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
        # Compute current time range
        t_start = ds_t + idx * (ss_t + ds_t)

        # Add single support phase
        zmp_ref[(t >= t_start) & (t < t_start + ss_t)] = current_step

        # Add double support phase
        mask = (t >= t_start + ss_t) & (t < t_start + ss_t + ds_t)
        alpha = (t[mask] - (t_start + ss_t)) / ds_t
        zmp_ref[mask, :] = (1 - alpha)[:, None] * current_step + alpha[:, None] * next_step

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
    else:
        return compute_double_support_polygon(steps_pose[i], steps_pose[i + 1], foot_shape)


def clamp_to_polygon(u_xy, poly: Polygon):
    p = Point(u_xy[0], u_xy[1])
    if poly.contains(p):
        return u_xy

    # nearest point on boundary
    q = nearest_points(poly.exterior, p)[0]

    return np.array([q.x, q.y])


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
    foot_shape = Polygon(((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))
    steps_pose = np.array([[0.0, -0.1],
                           [0.3, 0.1],
                           [0.6, -0.1],
                           [0.9, 0.1]])

    # Applied force parameters
    m = 60.0  # kg
    Fx, Fy = 0.0, 0.0  # N
    tau = 0.3  # s duration
    n_push = int(tau / dt)
    k_push = int((t_ss + 0.5 * t_ds) / dt)  # mid-DS of first pair

    # Build ZMP reference to track
    t, zmp_ref = compute_zmp_ref(com_initial_pose, steps_pose, dt, t_ss, t_ds)

    # Discrete cart-table model with jerk input
    A = np.array([[1.0, dt, 0.5 * dt * dt],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]], dtype=float)
    B = np.array([[dt ** 3 / 6.0],
                  [dt ** 2 / 2.0],
                  [dt]], dtype=float)
    C = np.array([[1.0, 0.0, -zc / g]], dtype=float)  # 1x3

    A1 = np.block([[np.eye(1), C @ A],
                   [np.zeros((3, 1)), A]])  # 4x4
    B1 = np.vstack((C @ B, B))  # 4x1
    I1 = np.vstack((np.array([1]), np.zeros((3, 1))))  # 4x1
    F = np.vstack((C @ A, A))

    Q = np.block([[Qe, np.zeros((1, 3))],
                  [np.zeros((3, 1)), Qx]])  # 4x4

    # Compute K by solving Ricatti equation
    K = solve_discrete_are(A1, B1, Q, R)

    # Compute Gi and Gx
    Gi = float(np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ I1)
    Gx = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ F

    # Compute Gd
    Ac = A1 - B1 @ np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ A1
    X1 = Ac.T @ K @ I1
    X = X1
    Gd = np.zeros((n_preview_steps - 1))
    Gd[0] = -Gi
    for l in range(n_preview_steps - 2):
        Gd[l + 1] = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ X
        X = Ac.T @ X

    T = len(t)
    u = np.zeros((T, 2))

    zmp_padded = np.vstack([
        zmp_ref,
        np.repeat(zmp_ref[-1][None, :], n_preview_steps, axis=0)
    ])

    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    y0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    x = np.zeros((len(zmp_ref) + 1, 4), dtype=float)
    y = np.zeros((len(zmp_ref) + 1, 4), dtype=float)
    x[0] = x0
    y[0] = y0

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 8))

    # TL: XY live
    ax_live_plot = axes[0, 0]
    ax_live_plot.axis('equal')
    ax_live_plot.grid(True)
    ax_live_plot.legend()
    ax_live_plot.set_xlabel("x pos [m]")
    ax_live_plot.set_ylabel("y pos [m]")
    ax_live_plot.set_title("ZMP / CoM trajectories")

    # Plot ZMP reference vs COM on the x axis
    axes[0, 1].plot(t, zmp_ref[:, 0], marker='.', label='ZMP reference [x]')
    com_ref_x_line, = axes[0, 1].plot([], [], marker='.', label='COM [x]')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("t [s]")
    axes[0, 1].set_ylabel("x pos [m]")
    axes[0, 1].set_title("ZMP reference vs COM position on X-axis")

    # Plot ZMP reference vs COM on the y axis
    axes[1, 1].plot(t, zmp_ref[:, 1], marker='.', label='ZMP reference [y]')
    com_ref_y_line, = axes[1, 1].plot([], [], marker='.', label='COM [y]')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("y pos [m]")
    axes[1, 1].set_title("ZMP reference vs COM position on Y-axis")

    # Plot preview controller gains
    axes[1, 0].plot(np.arange(1, n_preview_steps) * dt, Gd, marker='.', label='Preview gains [y]')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("preview gain [-]")
    axes[1, 0].set_title("Preview Gains")

    # after creating ax_xy
    info = ax_live_plot.text(
        0.05, 0.92, "", transform=ax_live_plot.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8)
    )

    # Static ZMP ref path for context
    zmp_path_line, = ax_live_plot.plot(zmp_ref[:, 0], zmp_ref[:, 1], linestyle='-', label='ZMP ref', alpha=1.0)
    com_path_line, = ax_live_plot.plot([], [], linestyle='-', label='CoM', color='red')

    active_poly_patch = None

    ax_live_plot.legend()

    # TR, BR, BL static axes (filled post-sim)
    ax_x = axes[0, 1]
    ax_y = axes[1, 1]
    ax_gain = axes[1, 0]
    for a in (ax_x, ax_y, ax_gain):
        a.grid(True)

    # Simulate + draw
    zmp_xy_hist = []
    com_xy_hist = []

    update_frequency = 0.02
    draw_every = max(1, int(update_frequency / dt))

    for k in range(T):
        # Apply the requested perturbation
        if k_push <= k < k_push + n_push:
            x[k, 1] += (Fx / m) * dt
            y[k, 1] += (Fy / m) * dt

        # Get zmp ref horizon
        zmp_ref_horizon = zmp_padded[k + 1:k + n_preview_steps]

        # Compute uk
        u[k, 0] = -Gi * x[k, 0] - Gx @ x[k, 1:] + Gd.T @ zmp_ref_horizon[:, 0]
        u[k, 1] = -Gi * y[k, 0] - Gx @ y[k, 1:] + Gd.T @ zmp_ref_horizon[:, 1]

        # Compute integrated error
        x[k + 1, 0] = x[k, 0] + (C @ x[k, 1:] - zmp_ref[k, 0])
        y[k + 1, 0] = y[k, 0] + (C @ y[k, 1:] - zmp_ref[k, 1])

        x[k + 1, 1:] = (A @ x[k, 1:] + B.ravel() * u[k, 0])
        y[k + 1, 1:] = (A @ y[k, 1:] + B.ravel() * u[k, 1])

        # record current points
        com_xy_hist.append([x[k + 1, 1], y[k + 1, 1]])
        zmp_xy_hist.append([zmp_ref[k, 0], zmp_ref[k, 1]])

        # draw at cadence
        com_arr = np.asarray(com_xy_hist)
        zmp_arr = np.asarray(zmp_xy_hist)

        if k % draw_every == 0:
            com_path_line.set_data(com_arr[:, 0], com_arr[:, 1])
            com_ref_x_line.set_data(t[0:k + 1], com_arr[:, 0])
            com_ref_y_line.set_data(t[0:k + 1], com_arr[:, 1])

            poly = get_active_polygon(k, dt, steps_pose, t_ss, t_ds, foot_shape)
            if active_poly_patch is not None:
                active_poly_patch.remove()
                active_poly_patch = None
            px, py = poly.exterior.xy
            active_poly_patch = ax_live_plot.fill(px, py, color='green', alpha=0.25)[0]
            info.set_text(f"t={k * dt:.2f}s")

            plt.pause(update_frequency)

    plt.show()
