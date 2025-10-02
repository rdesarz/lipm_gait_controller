import math
from time import sleep, clock_gettime
import meshcat
import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.transformations as tf

from lipm_walking_controller.controller import initialize_preview_control, compute_zmp_ref
from lipm_walking_controller.foot import compute_feet_path_and_poses, get_active_polygon
from lipm_walking_controller.inverse_kinematic import qp_inverse_kinematics, QPParams
from lipm_walking_controller.model import Talos

if __name__ == "__main__":
    # Parameters
    dt = 0.02  # Delta of time of the model simulation
    g = 9.81  # Gravity
    zc = 0.89  # Height of the COM
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

    foot_shape = Polygon(((0.11, 0.05), (0.11, -0.05), (-0.11, -0.05), (-0.11, 0.05)))
    n_steps = 20
    l_stride = 0.3
    max_height_foot = 0.05

    enable_live_plot = False

    A, B, C, Gd, Gx, Gi = initialize_preview_control(dt, zc, g, Qe, Qx, R, n_preview_steps)

    # Initialize the model position
    talos = Talos(path_to_model="~/projects")

    # Initialize visualizer
    viz = MeshcatVisualizer(talos.model, talos.geom, talos.vis)
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except Exception:
        viz = None

    oMf_rf0 = talos.data.oMf[talos.right_foot_id].copy()
    oMf_lf0 = talos.data.oMf[talos.left_foot_id].copy()

    lf_initial_pose = oMf_lf0.translation
    rf_initial_pose = oMf_rf0.translation
    com_initial_pose = pin.centerOfMass(talos.model, talos.data, talos.q)

    # Build ZMP reference to track
    t, lf_path, rf_path, steps_pose, phases = compute_feet_path_and_poses(rf_initial_pose, lf_initial_pose, n_steps,
                                                                          t_ss, t_ds,
                                                                          l_stride, dt, max_height_foot)

    zmp_ref = compute_zmp_ref(t, com_initial_pose[0:2], steps_pose, t_ss, t_ds)

    T = len(t)
    u = np.zeros((T, 2))

    zmp_padded = np.vstack([
        zmp_ref,
        np.repeat(zmp_ref[-1][None, :], n_preview_steps, axis=0)
    ])

    x0 = np.array([0.0, com_initial_pose[0], 0.0, 0.0], dtype=float)
    y0 = np.array([0.0, com_initial_pose[1], 0.0, 0.0], dtype=float)
    x = np.zeros((len(zmp_ref) + 1, 4), dtype=float)
    y = np.zeros((len(zmp_ref) + 1, 4), dtype=float)
    x[0] = x0
    y[0] = y0

    # Figure
    fig, axes = plt.subplots(3, 2, layout="constrained", figsize=(12, 8))

    ax_live_plot = axes[0, 0]
    ax_live_plot.grid(True)
    ax_live_plot.legend()
    ax_live_plot.set_xlabel("x pos [m]")
    ax_live_plot.set_ylabel("y pos [m]")
    ax_live_plot.set_title("ZMP / CoM trajectories")

    # Plot ZMP reference vs COM on the x axis
    axes[0, 1].plot(t, zmp_ref[:, 0], label='ZMP reference [x]')
    com_ref_x_line, = axes[0, 1].plot([], [], label='COM [x]')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("t [s]")
    axes[0, 1].set_ylabel("x pos [m]")
    axes[0, 1].set_title("ZMP reference vs COM position on X-axis")

    # Plot ZMP reference vs COM on the y axis
    axes[1, 1].plot(t, zmp_ref[:, 1], label='ZMP reference [y]')
    com_ref_y_line, = axes[1, 1].plot([], [], label='COM [y]')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("y pos [m]")
    axes[1, 1].set_title("ZMP reference vs COM position on Y-axis")

    axes[1, 0].plot(t, phases, marker='.', label='Phases [y]')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("Phases [-]")
    axes[1, 0].set_title("Phases ")

    lf_x_traj_line, = axes[2, 0].plot(t, lf_path[:, 0], label='Left foot trajectory')
    rf_x_traj_line, = axes[2, 0].plot(t, rf_path[:, 0], label='Right foot trajectory')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    axes[2, 0].set_xlabel("t [s]")
    axes[2, 0].set_ylabel("z pos [m]")
    axes[2, 0].set_title("Feet trajectories on x axis")

    lf_z_traj_line, = axes[2, 1].plot(t, lf_path[:, 2], label='Left foot trajectory')
    rf_z_traj_line, = axes[2, 1].plot(t, rf_path[:, 2], label='Right foot trajectory')
    axes[2, 1].grid(True)
    axes[2, 1].legend()
    axes[2, 1].set_xlabel("t [s]")
    axes[2, 1].set_ylabel("x pos [m]")
    axes[2, 1].set_title("Feet trajectories on z axis")

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

    ax_x = axes[0, 1]
    ax_y = axes[1, 1]
    for a in (ax_x, ax_y):
        a.grid(True)

    sleep(0.5)

    # Simulate
    zmp_xy_hist = []
    com_xy_hist = []

    update_frequency = 0.02
    draw_every = max(1, int(update_frequency / dt))

    frames = []
    for k in range(T):
        start = clock_gettime(0)

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

        # Plot
        com_xy_hist.append([x[k + 1, 1], y[k + 1, 1]])
        zmp_xy_hist.append([zmp_ref[k, 0], zmp_ref[k, 1]])

        com_arr = np.asarray(com_xy_hist)
        zmp_arr = np.asarray(zmp_xy_hist)

        com_target = np.array([x[k, 1], y[k, 1], lf_initial_pose[2] + zc])

        params = QPParams(fixed_foot_frame=talos.right_foot_id, moving_foot_frame=talos.left_foot_id,
                          torso_frame=talos.torso_id, model=talos.model,
                          data=talos.data, w_torso=10.0, w_com=10.0, w_mf=10.0, w_ff=1000.0, mu=1e-5, dt=dt)
        if phases[k] < 0.0:
            params.fixed_foot_frame = talos.right_foot_id
            params.moving_foot_frame = talos.left_foot_id

            oMf_lf = pin.SE3(oMf_lf0.rotation, lf_path[k])
            q_new, dq = qp_inverse_kinematics(talos.q, com_target, oMf_lf, params)
            talos.q = q_new
        else:
            params.fixed_foot_frame = talos.left_foot_id
            params.moving_foot_frame = talos.right_foot_id

            oMf_rf = pin.SE3(oMf_rf0.rotation, rf_path[k])
            q_new, dq = qp_inverse_kinematics(talos.q, com_target, oMf_rf, params)
            talos.q = q_new

        pin.forwardKinematics(talos.model, talos.data, talos.q)
        pin.updateFramePlacements(talos.model, talos.data)

        com = pin.centerOfMass(talos.model, talos.data, talos.q)

        n = viz.viewer[f"world/com_traj/pt_{k:05d}"]
        n.set_object(meshcat.geometry.Sphere(0.01), meshcat.geometry.MeshLambertMaterial(color=0xff0000))
        n.set_transform(tf.translation_matrix(com))

        if viz:
            viz.display(talos.q)

        # Compute the remaining time to render in real time the visualization
        stop = clock_gettime(0)
        elapsed_dt = stop - start
        remaining_dt = dt - elapsed_dt
        sleep(max(0.0, remaining_dt))

        if k % draw_every == 0 and enable_live_plot:
            com_path_line.set_data(com_arr[:, 0], com_arr[:, 1])
            com_ref_x_line.set_data(t[0:k + 1], com_arr[:, 0])
            com_ref_y_line.set_data(t[0:k + 1], com_arr[:, 1])

            poly = get_active_polygon(k, dt, steps_pose, t_ss, t_ds, foot_shape)
            if active_poly_patch is not None:
                active_poly_patch.remove()
                active_poly_patch = None
            px, py = poly.exterior.xy
            active_poly_patch = ax_live_plot.fill(px, py, color='green', alpha=0.25, label='Support polygon')[0]
            ax_live_plot.legend()
            info.set_text(f"t={k * dt:.2f}s")

            rf_z_traj_line.set_data(t[0:k + 1], rf_path[0:k + 1, 2])
            lf_z_traj_line.set_data(t[0:k + 1], lf_path[0:k + 1, 2])

            rf_x_traj_line.set_data(t[0:k + 1], rf_path[0:k + 1, 0])
            lf_x_traj_line.set_data(t[0:k + 1], lf_path[0:k + 1, 0])

            plt.pause(0.001)

            # Uncomment to save the plot
            # fig.canvas.draw()
            # frame = np.asarray(fig.canvas.buffer_rgba())
            # frames.append(frame.copy())

    # Uncomment to save the plot
    # imageio.mimsave("img/traj.gif", frames[2:], fps=int(1/update_frequency * 2), loop=10) # Save at frame divided by 2

    if not enable_live_plot:
        com_arr = np.asarray(com_xy_hist)

        com_path_line.set_data(com_arr[:, 0], com_arr[:, 1])
        com_ref_x_line.set_data(t, com_arr[:, 0])
        com_ref_y_line.set_data(t, com_arr[:, 1])

        plt.show()
