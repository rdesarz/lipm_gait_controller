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
COM_TARGET = np.array([-0.02404194,  0.00122989, -0.150])

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
    print(f"URDF not found: {URDF}\nSet PKG_PARENT or clone talos_data.", file=sys.stderr)
    sys.exit(1)

full_model, full_col_model, full_vis_model = pin.buildModelsFromUrdf(URDF, PKG_PARENT, pin.JointModelFreeFlyer())

for j_id, j_name in enumerate(full_model.names):
    print(j_id, j_name, full_model.joints[j_id].shortname())

# -------------------- Seed --------------------
q = pin.neutral(full_model)

# light knee flex to avoid singular straight legs
def set_joint(q, joint_name, val):
    jid = full_model.getJointId(joint_name)
    if jid > 0 and full_model.joints[jid].nq == 1:
        q[full_model.joints[jid].idx_q] = val

set_joint(q, "leg_left_4_joint", 0.0)
set_joint(q, "leg_right_4_joint", 0.0)
set_joint(q, "arm_right_4_joint", -1.2)
set_joint(q, "arm_left_4_joint", -1.2)

joints_to_lock = [i for i in range(14, 48)]

# Build reduced model
red_model, red_geom = pin.buildReducedModel(full_model, full_col_model, joints_to_lock, q)
_, red_vis = pin.buildReducedModel(full_model, full_vis_model, joints_to_lock, q)

red_data = red_model.createData()

viz = MeshcatVisualizer(red_model, red_geom, red_vis)
try:
    viz.initViewer(open=True)
    viz.loadViewerModel()
except Exception:
    viz = None

LF = red_model.getFrameId("left_sole_link")
RF = red_model.getFrameId("right_sole_link")

q = pin.neutral(red_model)

pin.forwardKinematics(red_model, red_data, q)
pin.updateFramePlacements(red_model, red_data)
oMf_rf0 = red_data.oMf[RF].copy()

LF_POS_TARGET = red_data.oMf[LF].translation
print(pin.centerOfMass(red_model, red_data, q))

# -------------------- HQP step -------------------
def hqp_step(q):
    # Compute forward kinematics for the current configuration
    pin.forwardKinematics(red_model, red_data, q)
    pin.updateFramePlacements(red_model, red_data)

    # Compute center of mass position as well as its Jacobian
    com = pin.centerOfMass(red_model, red_data, q)
    pin.computeCentroidalMap(red_model, red_data, q)  # ensure Jcom validity
    Jcom = pin.jacobianCenterOfMass(red_model, red_data, q)  # (3,nv)

    # Compute pose and Jacobian of the left and right foot
    oMf_lf = red_data.oMf[LF]
    oMf_rf = red_data.oMf[RF]
    J_lf = pin.computeFrameJacobian(red_model, red_data, q, LF, RFREF)[:3, :]  # (3,nv)
    J_rf = pin.computeFrameJacobian(red_model, red_data, q, RF, RFREF)  # (6,nv)

    # Compute errors of positionning
    e_lf = LF_POS_TARGET - oMf_lf.translation  # We want the left foot to match the desired target
    e_com = COM_TARGET - com  # We want the CoM to match the desired target
    e_rf6 = pin.log(oMf_rf.inverse() * oMf_rf0).vector
    b_eq = -e_rf6  # J_rf dq = -e_rf6

    nv = red_model.nv

    # Quadratic cost
    H = (W_LF * (J_lf.T @ J_lf)) + (W_COM * (Jcom.T @ Jcom)) + MU * np.eye(nv)
    g = -(W_LF * (J_lf.T @ e_lf) + W_COM * (Jcom.T @ e_com))

    # Equality: hard contact
    Aeq, beq = J_rf, b_eq

    # qpsolvers expects either (G,h) with x <= h or (lb <= A x <= ub). We'll use A and ub with no lb.
    dq = solve_qp(P=H, q=g, A=Aeq, b=beq, solver="osqp")  # returns None if infeasible

    q_next = pin.integrate(red_model, q, dq)

    return q_next, dq


# Single step of the QP
q_new, dq = hqp_step(q)
q = q_new

# -------------------- Report --------------------
pin.forwardKinematics(red_model, red_data, q)
pin.updateFramePlacements(red_model, red_data)
com_final = pin.centerOfMass(red_model, red_data, q)
lf_final = red_data.oMf[LF].translation
print(f"CoM final     : {com_final}  | err: {COM_TARGET - com_final}")
print(f"Left foot pos : {lf_final}    | err: {LF_POS_TARGET - lf_final}")
if viz:
    viz.display(q)
    sleep(1.0)
