import os, sys
import numpy as np
import pinocchio as pin
from time import sleep
from pinocchio.visualize import MeshcatVisualizer
from qpsolvers import solve_qp

from scipy.spatial.transform import Rotation


def set_joint(q, joint_name, val):
    jid = full_model.getJointId(joint_name)
    if jid > 0 and full_model.joints[jid].nq == 1:
        q[full_model.joints[jid].idx_q] = val


# Solver params
ITERS = 200
TOL_DQ = 1e-8
W_COM = 3.0
W_TORSO = 10.0  # tune
MU = 1e-6
DQ_LIM = 0.5
RFREF = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

# Load full model
PKG_PARENT = os.path.expanduser(os.environ.get("PKG_PARENT", "~/projects"))
URDF = os.path.join(PKG_PARENT, "talos_data/urdf/talos_full.urdf")

if not os.path.isfile(URDF):
    print(
        f"URDF not found: {URDF}\nSet PKG_PARENT or clone talos_data.", file=sys.stderr
    )
    sys.exit(1)

full_model, full_col_model, full_vis_model = pin.buildModelsFromUrdf(
    URDF, PKG_PARENT, pin.JointModelFreeFlyer()
)

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
red_model, red_geom = pin.buildReducedModel(
    full_model, full_col_model, joints_to_lock, q
)
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

print(f"Initial center of mass position: {pin.centerOfMass(red_model, red_data, q)}")
print(f"Initial left foot position: {red_data.oMf[LF].translation}")


def apply_qp(q, com_target):
    pin.forwardKinematics(red_model, red_data, q)
    pin.updateFramePlacements(red_model, red_data)

    com = pin.centerOfMass(red_model, red_data, q)
    Jcom = pin.jacobianCenterOfMass(red_model, red_data, q)

    RF_REF = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    J_lf = pin.computeFrameJacobian(red_model, red_data, q, LF, RF_REF)
    J_rf = pin.computeFrameJacobian(red_model, red_data, q, RF, RF_REF)  # (6,nv)

    # --- torso upright constraint (roll=pitch=0, yaw free) ---
    R = red_data.oMf[TORSO].rotation
    roll, pitch, yaw = Rotation.from_matrix(R).as_euler("xyz", degrees=False)
    Rdes = Rotation.from_euler("xyz", [0, 0, yaw]).as_matrix()
    e_rot = pin.log3(R.T @ Rdes)  # so3 error (3,)

    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # select roll,pitch only

    J_torso6 = pin.computeFrameJacobian(red_model, red_data, q, TORSO, RF_REF)
    J_torso_ang = J_torso6[3:6, :]  # (3,nv)
    A_torso = S @ J_torso_ang  # (2,nv)

    # Compute errors
    e_com = com_target - com
    e_rf6 = pin.log(red_data.oMf[RF].inverse() * oMf_rf0).vector
    e_lf6 = pin.log(red_data.oMf[LF].inverse() * oMf_lf0).vector

    # Compute cost
    nv = red_model.nv
    H = MU * np.eye(nv) + W_COM * (Jcom.T @ Jcom) + W_TORSO * (A_torso.T @ A_torso)
    g = -W_COM * (Jcom.T @ e_com) - W_TORSO * (A_torso.T @ (S @ e_rot))

    # Compute equality constraints
    Aeq = np.vstack([J_rf, J_lf])
    beq = np.hstack([e_rf6, e_lf6])

    # Solve QP
    dq = solve_qp(P=H, q=g, A=Aeq, b=beq, solver="osqp")

    # Integrate to get next q
    q_next = pin.integrate(red_model, q, dq)

    return q_next, dq


# Implement a squat motion
com_target = pin.centerOfMass(red_model, red_data, q)

for l in range(10):
    for k in range(20):
        com_target[2] = com_target[2] - 0.01
        q_new, dq = apply_qp(q, com_target)
        q = q_new

        pin.forwardKinematics(red_model, red_data, q)
        pin.updateFramePlacements(red_model, red_data)
        com_final = pin.centerOfMass(red_model, red_data, q)
        lf_final = red_data.oMf[LF].translation
        print(f"CoM final     : {com_final}  | err: {com_target - com_final}")
        if viz:
            viz.display(q)
            sleep(0.05)

    for k in range(20):
        com_target[2] = com_target[2] + 0.01
        q_new, dq = apply_qp(q, com_target)
        q = q_new

        pin.forwardKinematics(red_model, red_data, q)
        pin.updateFramePlacements(red_model, red_data)
        com_final = pin.centerOfMass(red_model, red_data, q)
        lf_final = red_data.oMf[LF].translation
        print(f"CoM final     : {com_final}  | err: {com_target - com_final}")
        if viz:
            viz.display(q)
            sleep(0.05)
