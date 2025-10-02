import numpy as np

from scipy.linalg import solve_discrete_are


def compute_zmp_ref(t, com_initial_pose, steps, ss_t, ds_t):
    T = len(t)
    zmp_ref = np.zeros([T, 2])

    # Step on the first foot
    mask = t < ds_t
    alpha = t[mask] / ds_t
    zmp_ref[mask, :] = (1 - alpha)[:, None] * com_initial_pose + alpha[:, None] * steps[
        0
    ]

    # Alternate between foot
    for idx, (current_step, next_step) in enumerate(zip(steps[:-1], steps[1:])):
        # Compute current time range
        t_start = ds_t + idx * (ss_t + ds_t)

        # Add single support phase
        zmp_ref[(t >= t_start) & (t < t_start + ss_t)] = current_step

        # Add double support phase
        mask = (t >= t_start + ss_t) & (t < t_start + ss_t + ds_t)
        alpha = (t[mask] - (t_start + ss_t)) / ds_t
        zmp_ref[mask, :] = (1 - alpha)[:, None] * current_step + alpha[
            :, None
        ] * next_step

    # Last phase is single support at last foot pose
    mask = t >= ds_t + (len(steps) - 1) * (ss_t + ds_t)
    zmp_ref[mask, :] = steps[-1]

    return zmp_ref


def initialize_preview_control(dt, zc, g, Qe, Qx, R, n_preview_steps):
    # Discrete cart-table model with jerk input
    A = np.array(
        [[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=float
    )
    B = np.array([[dt**3 / 6.0], [dt**2 / 2.0], [dt]], dtype=float)
    C = np.array([[1.0, 0.0, -zc / g]], dtype=float)  # 1x3

    A1 = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])  # 4x4
    B1 = np.vstack((C @ B, B))  # 4x1
    I1 = np.vstack((np.array([1]), np.zeros((3, 1))))  # 4x1
    F = np.vstack((C @ A, A))

    Q = np.block([[Qe, np.zeros((1, 3))], [np.zeros((3, 1)), Qx]])  # 4x4

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

    return A, B, C, Gd, Gx, Gi
