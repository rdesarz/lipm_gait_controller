#!/usr/bin/env python3
# HQP IK: hard contact on right foot, soft CoM + left foot position

import os, sys, numpy as np, pinocchio as pin
from time import sleep
from pinocchio.visualize import MeshcatVisualizer
from qpsolvers import solve_qp  # pip install qpsolvers osqp

# -------------------- Config --------------------
PKG_PARENT = os.path.expanduser(os.environ.get("PKG_PARENT", "~/projects"))
URDF = os.path.join(PKG_PARENT, "talos_data/urdf/talos_full.urdf")  # note: underscore

# World-frame targets
COM_TARGET = np.array([0.0, 0.0, -0.19305])
LF_POS_TARGET = np.array([0.5, 0.2, -1.08305])

# Solver params
ITERS = 200
TOL_DQ = 1e-8
W_LF = 10.0
W_COM = 3.0
MU = 1e-6
DQ_LIM = 0.5  # rad/iter (box limits)
RFREF = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

# -------------------- Model & Viz --------------------
if not os.path.isfile(URDF):
    print(f"URDF not found: {URDF}\nSet PKG_PARENT or clone talos_data.", file=sys.stderr);
    sys.exit(1)

model, col_model, vis_model = pin.buildModelsFromUrdf(URDF, PKG_PARENT, pin.JointModelFreeFlyer())
data = model.createData()
viz = MeshcatVisualizer(model, col_model, vis_model)
try:
    viz.initViewer(open=True)
    viz.loadViewerModel()
except Exception:
    viz = None

LF = model.getFrameId("left_sole_link")
RF = model.getFrameId("right_sole_link")

# -------------------- Seed --------------------
q = pin.neutral(model)


# light knee flex to avoid singular straight legs
def set_joint(q, joint_name, val):
    jid = model.getJointId(joint_name)
    if jid > 0 and model.joints[jid].nq == 1:
        q[model.joints[jid].idx_q] = val


set_joint(q, "leg_left_4_joint", 0.20)
set_joint(q, "leg_right_4_joint", 0.20)

pin.forwardKinematics(model, data, q);
pin.updateFramePlacements(model, data)
oMf_rf0 = data.oMf[RF].copy()


# -------------------- HQP step --------------------
def hqp_step(q):
    # Compute forward kinematics for the current configuration
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Compute center of mass position as well as its Jacobian
    com = pin.centerOfMass(model, data, q)
    pin.computeCentroidalMap(model, data, q)  # ensure Jcom validity
    Jcom = pin.jacobianCenterOfMass(model, data, q)  # (3,nv)

    # Compute pose and Jacobian of the left and right foot
    oMf_lf = data.oMf[LF]
    oMf_rf = data.oMf[RF]
    J_lf = pin.computeFrameJacobian(model, data, q, LF, RFREF)[:3, :]  # (3,nv)
    J_rf = pin.computeFrameJacobian(model, data, q, RF, RFREF)  # (6,nv)

    # Compute errors of positionning
    e_lf = LF_POS_TARGET - oMf_lf.translation # We want the left foot to match the desired target
    e_com = COM_TARGET - com # We want the CoM to match the desired target
    e_rf6 = pin.log(oMf_rf.inverse() * oMf_rf0).vector  # target zero
    b_eq = -e_rf6  # J_rf dq = -e_rf6

    nv = model.nv

    # Quadratic cost
    H = (W_LF * (J_lf.T @ J_lf)) + (W_COM * (Jcom.T @ Jcom)) + MU * np.eye(nv)
    g = -(W_LF * (J_lf.T @ e_lf) + W_COM * (Jcom.T @ e_com))

    # Equality: hard contact
    Aeq, beq = J_rf, b_eq

    # Velocity box limits via inequalities
    dqmax = DQ_LIM * np.ones(nv)
    dqmin = -DQ_LIM * np.ones(nv)
    A_ineq = np.vstack([np.eye(nv), -np.eye(nv)])
    ub = np.hstack([dqmax, -dqmin])
    # qpsolvers expects either (G,h) with x <= h or (lb <= A x <= ub). We'll use A and ub with no lb.
    dq = solve_qp(P=H, q=g, G=A_ineq, h=ub, A=Aeq, b=beq, solver="osqp")  # returns None if infeasible

    if dq is None:
        # fallback: least-squares with equality via normal equations
        # solve [H  Aeq^T; Aeq 0][dq; λ] = [ -g; beq ]
        KKT = np.block([[H, Aeq.T], [Aeq, np.zeros((Aeq.shape[0], Aeq.shape[0]))]])
        rhs = np.hstack([-g, beq])
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        dq = sol[:nv]

    q_next = pin.integrate(model, q, dq)
    return q_next, dq


# -------------------- Loop --------------------
for it in range(ITERS):
    q_new, dq = hqp_step(q)
    q = q_new
    if viz: viz.display(q)
    if np.linalg.norm(dq) < TOL_DQ: break

# -------------------- Report --------------------
pin.forwardKinematics(model, data, q);
pin.updateFramePlacements(model, data)
com_final = pin.centerOfMass(model, data, q)
lf_final = data.oMf[LF].translation
print(f"Iterations: {it + 1}")
print(f"CoM final     : {com_final}  | err: {COM_TARGET - com_final}")
print(f"Left foot pos : {lf_final}    | err: {LF_POS_TARGET - lf_final}")
if viz:
    viz.display(q)
    sleep(1.0)
