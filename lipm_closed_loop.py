import math

import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon, Point, affinity, union
from shapely.ops import nearest_points


def compute_lipm_dynamic_model(w, dt):
    c = math.cosh(w * dt)
    s = math.sinh(w * dt)
    A = np.array([[c, s / w],
                  [w * s, c]], dtype=float)
    B = np.array([1.0 - c, -w * s], dtype=float)  # shape (2,)

    return A, B


def compute_zmp_ref(steps, dt, ss_t, ds_t):
    T = int((len(steps) - 1) * (ss_t + ds_t) / dt)

    t = np.arange(T) * dt
    zmp_ref = np.zeros([T, 2])

    for idx, (current_step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
        # Compute current time range
        t_start = idx * (ss_t + ds_t)

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


def get_active_polygon(k, dt, steps_pose, t_ss, t_ds):
    tk = k * dt
    t_step = t_ss + t_ds

    i = int(tk / t_step)  # step index
    i = min(i, len(steps_pose) - 2)
    t_in = tk - i * t_step
    if t_in < t_ss:  # single support on step i
        return compute_single_support_polygon(steps_pose[i], foot_shape)
    else:  # double support between i and i+1
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
    dt = 0.01  # delta
    g = 9.81  # Gravity
    zc = 0.8  # Height of the COM
    w = math.sqrt(g / zc)
    a = math.exp(w * dt)  # CP open-loop pole
    rho = 0.98  # desired closed-loop pole
    K = (rho - a) / (1 - a)  # ≈ 1.1 for your params
    t_ss = 2.0  # single support phase time window
    t_ds = 5.0  # double support phase time window
    foot_shape = Polygon(((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))
    steps_pose = np.array([[0.0, 0.0],
                           [0.5, 0.4],
                           [1.0, 0.0],
                           [1.5, 0.4]])

    m = 60.0  # kg
    Fx, Fy = 0.0, 0.0  # N
    tau = 0.3  # s duration
    n_push = int(tau / dt)
    k_push = int((t_ss + 0.5 * t_ds) / dt)  # mid-DS of first pair

    t, zmp_ref = compute_zmp_ref(steps_pose, dt, t_ss, t_ds)
    T = len(t)

    A, B = compute_lipm_dynamic_model(w, dt)

    x0 = np.array([0.0, 0.0], dtype=float)  # [x, xdot]
    y0 = np.array([0.0, 0.0], dtype=float)  # [y, ydot]
    x = np.zeros((len(zmp_ref) + 1, 2), dtype=float)
    y = np.zeros((len(zmp_ref) + 1, 2), dtype=float)
    x[0] = x0
    y[0] = y0

    # Initialize capture point position
    cp = np.zeros((T + 1, 2))
    cp[0, 0] = x[0, 0] + x[0, 1] / w
    cp[0, 1] = y[0, 0] + y[0, 1] / w

    u = np.zeros((T, 2))

    # Compute the trajectory of the COM
    for k, (t, (ux_ref, uy_ref)) in enumerate(zip(t, zmp_ref)):
        # Apply perturbation if needed
        if k_push <= k < k_push + n_push:
            x[k, 1] += (Fx / m) * dt
            y[k, 1] += (Fy / m) * dt

        # Compute u from capture point and zmp_ref
        u[k, 0] = ux_ref + K * (cp[k, 0] - ux_ref)
        u[k, 1] = uy_ref + K * (cp[k, 1] - uy_ref)

        # Clamp to active support polygon
        poly = get_active_polygon(k, dt, steps_pose, t_ss, t_ds)
        u[k] = clamp_to_polygon(u[k], poly)

        # Compute COM position
        x[k + 1] = A @ x[k] + B * u[k, 0]
        y[k + 1] = A @ y[k] + B * u[k, 1]

        # Compute next capture point position
        cp[k + 1, 0] = x[k + 1, 0] + x[k + 1, 1] / w
        cp[k + 1, 1] = y[k + 1, 0] + y[k + 1, 1] / w

    # Plot
    fig, axes = plt.subplots()
    axes.plot(zmp_ref[:, 0], zmp_ref[:, 1], marker='.', label='ZMP')
    # axes.plot(cp[1:, 0], cp[1:, 1], label='CP', color='green')  # skip x[0] if zmp_ref starts at t>0
    axes.plot(x[1:, 0], y[1:, 0], label='CoM', color='red')
    # axes.plot(u[1:, 0], u[1:, 1], label='u', color='b')

    plot_steps(axes, steps_pose, foot_shape)
    axes.axis('equal')
    axes.grid(True)
    axes.legend()

    err_com = np.linalg.norm(np.c_[x[1:, 0], y[1:, 0]] - zmp_ref, axis=1)
    err_cp = np.linalg.norm(cp[1:] - u, axis=1)
    print("RMS CoM–ZMP:", np.sqrt(np.mean(err_com ** 2)))
    print("max |CP−u|:", np.max(err_cp))

    plt.show()
