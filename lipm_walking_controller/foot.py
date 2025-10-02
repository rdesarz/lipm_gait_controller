import math
import numpy as np
from shapely import Polygon, Point, affinity, union
from shapely.ops import nearest_points


def clamp_to_polygon(u_xy, poly: Polygon):
    p = Point(u_xy[0], u_xy[1])
    if poly.contains(p):
        return u_xy

    # nearest point on boundary
    q = nearest_points(poly.exterior, p)[0]

    return np.array([q.x, q.y])


def compute_double_support_polygon(current_foot_pose, next_foot_pose, foot_shape):
    curent_foot = affinity.translate(
        foot_shape, xoff=current_foot_pose[0], yoff=current_foot_pose[1]
    )
    next_foot = affinity.translate(
        foot_shape, xoff=next_foot_pose[0], yoff=next_foot_pose[1]
    )

    return union(curent_foot, next_foot).convex_hull


def compute_single_support_polygon(current_foot_pose, foot_shape):
    return affinity.translate(
        foot_shape, xoff=current_foot_pose[0], yoff=current_foot_pose[1]
    )


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
        return compute_double_support_polygon(
            steps_pose[i], steps_pose[i + 1], foot_shape
        )


def compute_feet_path_and_poses(
    rf_initial_pose, lf_initial_pose, n_steps, t_ss, t_ds, l_stride, dt, max_height_foot
):
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
    rf_path = np.ones([N, 3]) * rf_initial_pose
    lf_path = np.ones([N, 3]) * lf_initial_pose
    phases = np.ones(N)

    # Switch of the CoM to the right foot implies both feet stays at the same position
    mask = t < t_ds
    rf_path[mask, :] = rf_initial_pose
    lf_path[mask, :] = lf_initial_pose

    steps_pose = np.zeros((n_steps + 2, 2))
    steps_pose[0] = rf_initial_pose[0:2]
    for i in range(1, n_steps + 1):
        sign = -1.0 if i % 2 == 0 else 1.0
        steps_pose[i] = np.array([i * l_stride, sign * math.fabs(lf_initial_pose[1])])

    # Add a last step to have both feet at the same level
    steps_pose[-1] = steps_pose[-2]
    steps_pose[-1][1] = steps_pose[-1][1] * -1.0

    # Compute motion of left foot
    for k in range(0, n_steps, 2):
        t_begin = t_ds + k * (t_ss + t_ds)
        t_end = t_ds + k * (t_ss + t_ds) + t_ss
        mask = (t >= t_begin) & (t < t_end)
        sub_time = t[mask] - (t_ds + k * (t_ss + t_ds))

        # Compute motion on z-axis
        theta = sub_time * math.pi / t_ss
        lf_path[mask, 2] += np.sin(theta) * max_height_foot
        phases[mask] = -1.0

        # Compute motion on x-axis
        if k == 0:
            alpha = sub_time / t_ss
            lf_path[mask, 0] = (1 - alpha) * lf_initial_pose[0] + alpha * steps_pose[
                k + 1
            ][0]
        else:
            alpha = sub_time / t_ss
            lf_path[mask, 0] = (1 - alpha) * steps_pose[k - 1][0] + alpha * steps_pose[
                k + 1
            ][0]

        # # Add constant part till the next step
        t_begin = t_ds + k * (t_ss + t_ds) + t_ss
        t_end = total_time
        mask = (t >= t_begin) & (t < t_end)
        lf_path[mask, 0] = steps_pose[k + 1][0]

    # Compute motion of right foot
    for k in range(1, n_steps + 1, 2):
        t_begin = t_ds + k * (t_ss + t_ds)
        t_end = t_ds + k * (t_ss + t_ds) + t_ss
        mask = (t > t_begin) & (t < t_end)
        sub_time = t[mask] - (t_ds + k * (t_ss + t_ds))

        # Compute motion on z-axis
        theta = sub_time * math.pi / t_ss
        rf_path[mask, 2] += np.sin(theta) * max_height_foot
        phases[mask] = 1.0

        # Compute motion on x-axis
        if k == 1:
            alpha = sub_time / t_ss
            rf_path[mask, 0] = (1 - alpha) * rf_initial_pose[0] + alpha * steps_pose[
                k + 1
            ][0]
        else:
            alpha = sub_time / t_ss
            rf_path[mask, 0] = (1 - alpha) * steps_pose[k - 1][0] + alpha * steps_pose[
                k + 1
            ][0]

        # # Add constant part till the next step
        t_begin = t_ds + k * (t_ss + t_ds) + t_ss
        t_end = total_time
        mask = (t >= t_begin) & (t < t_end)
        rf_path[mask, 0] = steps_pose[k + 1][0]

    return t, lf_path, rf_path, steps_pose, phases
