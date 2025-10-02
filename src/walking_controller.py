import math, os, sys
from dataclasses import dataclass
from typing import Any
from time import sleep, clock_gettime

import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon, Point, affinity, union
from shapely.ops import nearest_points
from scipy.linalg import solve_discrete_are
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from qpsolvers import solve_qp


def compute_zmp_ref(t, com_initial_pose, steps, ss_t, ds_t):
    T = len(t)
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

    return zmp_ref


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


@dataclass
class QPParams:
    fixed_foot_frame: str
    moving_foot_frame: str
    torso_frame: str
    model: pin.Model
    data: Any
    w_torso: float
    w_com: float
    w_mf: float
    w_ff: float
    mu: float
    dt: float


def se3_task_error_and_jacobian(model, data, q, frame_id, M_des):
    # Pose of frame i in world; LOCAL frame convention (right differentiation)
    oMi = data.oMf[frame_id]  # requires updateFramePlacements()
    iMd = oMi.actInv(M_des)  # ^i M_d  = oMi^{-1} * oMdes
    e6 = pin.log(iMd).vector  # right-invariant pose error in LOCAL frame

    # Geometric Jacobian in LOCAL frame
    Jb = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)

    # Right Jacobian of the log map (Pinocchio’s Jlog6)
    Jl = -pin.Jlog6(iMd.inverse())  # maps LOCAL spatial vel -> d(log) in se(3)

    # Task Jacobian
    Jtask = Jl @ Jb  # minus sign per right-invariant residual

    return e6, Jtask


def qp_inverse_kinematics(q, com_target, oMf_target, params: QPParams):
    pin.forwardKinematics(params.model, params.data, q)
    pin.updateFramePlacements(params.model, params.data)

    nv = params.model.nv

    # -------- CoM task (Euclidean) --------
    com = pin.centerOfMass(params.model, params.data, q)
    Jcom = pin.jacobianCenterOfMass(params.model, params.data, q)  # (3,nv)
    e_com = com_target - com

    # -------- Fixed-foot hard contact (6D) --------
    # Drive residual to zero at velocity level: J_ff * dq = -Kc * e_ff
    e_ff, J_ff = se3_task_error_and_jacobian(
        params.model, params.data, q, params.fixed_foot_frame,
        params.data.oMf[params.fixed_foot_frame].copy()  # hold current pose
    )

    # -------- Moving-foot soft pose task (6D) --------
    e_mf, J_mf = se3_task_error_and_jacobian(
        params.model, params.data, q, params.moving_foot_frame, oMf_target
    )

    # -------- Torso roll/pitch soft task --------
    # Select angular part of e_torso and corresponding Jacobian rows
    e_torso6, J_torso6 = se3_task_error_and_jacobian(
        params.model, params.data, q, params.torso_frame,
        # keep current yaw: project desired as same yaw in world
        pin.SE3(params.data.oMf[params.torso_frame].rotation,  # overwritten right below
                params.data.oMf[params.torso_frame].translation)
    )
    # Replace desired yaw by current yaw -> zero yaw error implicitly
    # Keep only roll,pitch components of angular error (indices 0,1) and rows (0,1) of J
    S = np.zeros((3, 6))
    S[0, 3] = 1.0
    S[1, 4] = 1.0
    S[2, 5] = 1.0
    e_torso = S @ e_torso6
    J_torso = S @ J_torso6

    # -------- Quadratic cost --------
    H = ((Jcom.T @ (np.eye(3) * params.w_com) @ Jcom)
         + (J_torso.T @ (np.eye(3) * params.w_torso) @ J_torso)
         + (J_ff.T @ (np.eye(6) * params.w_ff) @ J_ff)
         + (J_mf.T @ (np.eye(6) * params.w_mf) @ J_mf)
         + np.eye(nv) * params.mu)

    g = ((- Jcom.T @ (np.eye(3) * params.w_com) @ e_com)
         + (J_torso.T @ (np.eye(3) * params.w_torso) @ e_torso)
         + (J_mf.T @ (np.eye(6) * params.w_mf) @ e_mf)
         + (J_ff.T @ (np.eye(6) * params.w_ff) @ e_ff))

    dq = solve_qp(P=H, q=g, solver="osqp")

    q_next = pin.integrate(params.model, q, dq)

    return q_next, dq


def compute_feet_path_and_poses(rf_initial_pose, lf_initial_pose, n_steps, t_ss, t_ds, l_stride, dt, max_height_foot):
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
            lf_path[mask, 0] = (1 - alpha) * lf_initial_pose[0] + alpha * steps_pose[k + 1][0]
        else:
            alpha = sub_time / t_ss
            lf_path[mask, 0] = (1 - alpha) * steps_pose[k - 1][0] + alpha * steps_pose[k + 1][0]

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
            rf_path[mask, 0] = (1 - alpha) * rf_initial_pose[0] + alpha * steps_pose[k + 1][0]
        else:
            alpha = sub_time / t_ss
            rf_path[mask, 0] = (1 - alpha) * steps_pose[k - 1][0] + alpha * steps_pose[k + 1][0]

        # # Add constant part till the next step
        t_begin = t_ds + k * (t_ss + t_ds) + t_ss
        t_end = total_time
        mask = (t >= t_begin) & (t < t_end)
        rf_path[mask, 0] = steps_pose[k + 1][0]

    return t, lf_path, rf_path, steps_pose, phases


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
    n_steps = 5
    l_stride = 0.3
    max_height_foot = 0.05

    enable_live_plot = False

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

    # Simulate
    zmp_xy_hist = []
    com_xy_hist = []

    update_frequency = 0.02
    draw_every = max(1, int(update_frequency / dt))


    def set_joint(q, joint_name, val):
        jid = full_model.getJointId(joint_name)
        if jid > 0 and full_model.joints[jid].nq == 1:
            q[full_model.joints[jid].idx_q] = val


    # Load full model
    PKG_PARENT = os.path.expanduser(os.environ.get("PKG_PARENT", "~/projects"))
    URDF = os.path.join(PKG_PARENT, "talos_data/urdf/talos_full.urdf")

    if not os.path.isfile(URDF):
        print(f"URDF not found: {URDF}\nSet PKG_PARENT or clone talos_data.", file=sys.stderr)
        sys.exit(1)

    full_model, full_col_model, full_vis_model = pin.buildModelsFromUrdf(URDF, PKG_PARENT, pin.JointModelFreeFlyer())

    # List joints
    for j_id, j_name in enumerate(full_model.names):
        print(j_id, j_name, full_model.joints[j_id].shortname())

    # List frames
    for i, frame in enumerate(full_model.frames):
        print(i, frame.name, frame.parent, frame.type)

    # Initialize the model position
    q = pin.neutral(full_model)

    # Position the arms
    set_joint(q, "leg_left_4_joint", 0.0)
    set_joint(q, "leg_right_4_joint", 0.0)
    set_joint(q, "arm_right_4_joint", -1.2)
    set_joint(q, "arm_left_4_joint", -1.2)

    # We lock joints of the upper body
    joints_to_lock = [i for i in range(14, 48)]

    # Build reduced model
    red_model, red_geom = pin.buildReducedModel(full_model, full_col_model, joints_to_lock, q)
    _, red_vis = pin.buildReducedModel(full_model, full_vis_model, joints_to_lock, q)
    red_data = red_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(red_model, red_geom, red_vis)
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except Exception:
        viz = None

    LF = red_model.getFrameId("left_sole_link")
    RF = red_model.getFrameId("right_sole_link")
    TORSO = red_model.getFrameId("torso_1_link")

    # Initialize reduced model
    q = pin.neutral(red_model)

    # Initialize legs position
    set_joint(q, "leg_left_1_joint", 0.0)
    set_joint(q, "leg_left_2_joint", 0.0)
    set_joint(q, "leg_left_3_joint", -0.5)
    set_joint(q, "leg_left_4_joint", 1.0)
    set_joint(q, "leg_left_5_joint", -0.6)

    set_joint(q, "leg_right_1_joint", 0.0)
    set_joint(q, "leg_right_2_joint", 0.0)
    set_joint(q, "leg_right_3_joint", -0.5)
    set_joint(q, "leg_right_4_joint", 1.0)
    set_joint(q, "leg_right_5_joint", -0.6)

    pin.forwardKinematics(red_model, red_data, q)
    pin.updateFramePlacements(red_model, red_data)
    oMf_rf0 = red_data.oMf[RF].copy()
    oMf_lf0 = red_data.oMf[LF].copy()

    lf_initial_pose = oMf_lf0.translation
    rf_initial_pose = oMf_rf0.translation
    com_initial_pose = pin.centerOfMass(red_model, red_data, q)

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

    sleep(2.0)

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

        params = QPParams(fixed_foot_frame=RF, moving_foot_frame=LF, torso_frame=TORSO, model=red_model,
                          data=red_data, w_torso=10.0, w_com=10.0, w_mf=10.0, w_ff=1000.0, mu=1e-5, dt=dt)
        if phases[k] < 0.0:
            params.fixed_foot_frame = RF
            params.moving_foot_frame = LF

            oMf_lf = pin.SE3(oMf_lf0.rotation, lf_path[k])
            q_new, dq = qp_inverse_kinematics(q, com_target, oMf_lf, params)
            q = q_new
        else:
            params.fixed_foot_frame = LF
            params.moving_foot_frame = RF

            oMf_rf = pin.SE3(oMf_rf0.rotation, rf_path[k])
            q_new, dq = qp_inverse_kinematics(q, com_target, oMf_rf, params)
            q = q_new

        pin.forwardKinematics(red_model, red_data, q)
        pin.updateFramePlacements(red_model, red_data)
        com_final = pin.centerOfMass(red_model, red_data, q)
        lf_final = red_data.oMf[LF].translation
        if viz:
            viz.display(q)

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

            plt.pause(update_frequency)

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
