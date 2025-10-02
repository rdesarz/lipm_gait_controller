import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are


# Preview gain computation required for implementation of # Kajita's LIPM preview control.
# The detail of the implementation can be found in Katayama et al. (1985)
if __name__ == "__main__":
    # Params
    T = 0.005  # 5 ms
    zc = 0.814  # COM height [m]
    Qe = 1.0
    Qx = np.zeros((3, 3))
    R = 1e-6
    preview_time = 1.6
    NL = int(round(preview_time / T))
    g = 9.81

    # Discrete cart-table model with jerk input
    A = np.array([[1.0, T, 0.5 * T * T], [0.0, 1.0, T], [0.0, 0.0, 1.0]], dtype=float)
    B = np.array([[T**3 / 6.0], [T**2 / 2.0], [T]], dtype=float)
    C = np.array([[1.0, 0.0, -zc / g]], dtype=float)  # 1x3

    # Augment with integral of ZMP error
    A1 = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])  # 4x4
    B1 = np.vstack((C @ B, B))  # 4x1
    I1 = np.vstack((np.array([1]), np.zeros((3, 1))))  # 4x1
    F = np.vstack((C @ A, A))

    Q = np.block([[Qe, np.zeros((1, 3))], [np.zeros((3, 1)), Qx]])  # 4x4

    # Compute K from Ricatti
    K = solve_discrete_are(A1, B1, Q, R)

    # Compute Gi and Gx
    Gi = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ I1
    Gx = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ F

    # Compute Gd
    Ac = A1 - B1 @ np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ K @ A1
    X1 = Ac.T @ K @ I1
    X = X1
    Gd = np.zeros((NL, 1))
    Gd[0, 0] = np.array([-Gi])
    for l in range(NL - 1):
        Gd[l + 1] = np.linalg.inv(R + B1.T @ K @ B1) @ B1.T @ X
        X = Ac.T @ X

    print(f"T={T}s  zc={zc}m  preview={NL} steps (~{preview_time}s)")
    print("Gi =", Gi)
    print("Gx =", Gx)
    print("First 10 Gp:", Gd[:10])

    # Plot preview gains
    plt.figure()
    plt.plot(np.arange(1, NL + 1), Gd)
    plt.xlabel("Preview step j")
    plt.ylabel("Gp(j)")
    plt.title("Kajita Preview Gains Gp(j)")
    plt.tight_layout()
    plt.show()

    # Plot state-feedback gains
    plt.figure()
    labels = ["x", "xdot", "xddot"]
    plt.bar(labels, Gx.flatten())
    plt.xlabel("State component")
    plt.ylabel("Gain value")
    plt.title("State Feedback Gains Gx")
    plt.tight_layout()
    plt.show()
