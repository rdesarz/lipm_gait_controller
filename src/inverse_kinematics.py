import os, pinocchio as pin
import sys
from time import sleep

import numpy as np
from pinocchio.visualize import MeshcatVisualizer

# -------------------- Config --------------------
# Parent dir that contains "talos_data/"
PKG_PARENT = os.path.expanduser(os.environ.get("PKG_PARENT", "~/projects"))
URDF = os.path.join(PKG_PARENT, "talos-data/urdf/talos_full.urdf")

# Targets in world frame [meters]
COM_TARGET = np.array([0.0, 0.0, -0.19305])  # set your desired CoM
LF_POS_TARGET = np.array([0.5, 0.2, -1.08305])  # set your desired left-foot position

# Solver params
ITERS = 200
DAMP = 1e-6
TOL_DQ = 1e-8
W_RF = 100.0   # keep right foot fixed (6D)
W_LF = 10.0    # left-foot position (3D)
W_COM = 3.0    # CoM (3D)
RF = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

# -------------------- Model & Viz --------------------
if not os.path.isfile(URDF):
    print(f"URDF not found: {URDF}\nSet PKG_PARENT or clone talos_data.", file=sys.stderr)
    sys.exit(1)

model, col_model, vis_model = pin.buildModelsFromUrdf(URDF, PKG_PARENT, pin.JointModelFreeFlyer())
data = model.createData()

viz = MeshcatVisualizer(model, col_model, vis_model)
try:
    viz.initViewer(open=True)
    viz.loadViewerModel()
except Exception:
    viz = None

# Frames
LF = model.getFrameId("left_sole_link")
RFID = model.getFrameId("right_sole_link")

# -------------------- Seed --------------------
q = pin.neutral(model)
# mild knee flexion to avoid singular straight legs
names = [j.shortname() for j in model.joints]


def set_joint(q, name, val):
    jid = names.index(name) if name in names else -1
    if jid > 0:
        idx = model.joints[jid].idx_q
        if model.joints[jid].nq == 1:
            q[idx] = val


set_joint(q, "leg_left_4_joint", 0.20)  # approximate knees
set_joint(q, "leg_right_4_joint", 0.20)

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
oMf_rf0 = data.oMf[RFID].copy()  # anchor pose for right foot

# -------------------- IK Loop --------------------
def ik_step(q):
    # FK and Jacobians
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    com = pin.centerOfMass(model, data, q)  # (3,)
    pin.computeCentroidalMap(model, data, q)
    Jcom = pin.jacobianCenterOfMass(model, data, q)  # (3,nv)

    oMf_lf = data.oMf[LF]
    oMf_rf = data.oMf[RFID]
    J_lf = pin.computeFrameJacobian(model, data, q, LF, RF)  # (6,nv)
    J_rf = pin.computeFrameJacobian(model, data, q, RFID, RF)  # (6,nv)

    # Errors
    e_rf6 = pin.log(oMf_rf.inverse() * oMf_rf0)  # keep RF at initial pose
    e_lf3 = (LF_POS_TARGET - oMf_lf.translation)  # only position
    e_com = (COM_TARGET - com)

    # Stack tasks
    J = np.vstack([
        np.sqrt(W_RF) * J_rf,  # 6 x nv
        np.sqrt(W_LF) * J_lf[:3, :],  # 3 x nv
        np.sqrt(W_COM) * Jcom  # 3 x nv
    ])
    e = np.hstack([
        np.sqrt(W_RF) * e_rf6,
        np.sqrt(W_LF) * e_lf3,
        np.sqrt(W_COM) * e_com
    ])

    # Damped least squares
    H = J.T @ J + DAMP * np.eye(model.nv)
    g = J.T @ e
    dq = np.linalg.solve(H, g)

    q_next = pin.integrate(model, q, dq)
    return q_next, dq


for it in range(ITERS):
    q_new, dq = ik_step(q)
    q = q_new
    if viz:
        viz.display(q)
    if np.linalg.norm(dq) < TOL_DQ:
        break

# -------------------- Report --------------------
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
com_final = pin.centerOfMass(model, data, q)
lf_final = data.oMf[LF].translation
rf_final = data.oMf[RFID]

print(f"Iterations: {it + 1}")
print(f"CoM final     : {com_final}  | err: {COM_TARGET - com_final}")
print(f"Left foot pos : {lf_final}    | err: {LF_POS_TARGET - lf_final}")
print("Right foot pose kept (6D error close to zero).")
if viz:
    viz.display(q)
    sleep(1.0)
