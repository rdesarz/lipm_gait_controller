# pip install pin==3.1.0 pinocchio numpy
# Optional: pip install meshcat
# git clone https://github.com/stack-of-tasks/talos-data.git

import os, time, numpy as np, pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# --------- User inputs ----------
TALOS_REPO = "/home/rdesarz/projects/talos_data"
URDF_PATH  = os.path.join(TALOS_REPO, "urdf", "talos_full.urdf")

# Either provide trajectories (T×3) or single targets (3,)
# Example placeholders:
com_traj    = np.array([[0.0, 0.0, 0.0],
                        [0.02, 0.00, 0.0],
                        [0.04, 0.01, 0.0],
                        [0.06, 0.01, 0.0],
                        [0.08, 0.00, 0.0]])
lf_pos_traj = np.repeat(np.array([[0.0, 0.1, 0.0]]), len(com_traj), axis=0)

dt        = 0.05          # viz step
Kp_com    = 30.0
Kp_foot   = 200.0
w_com     = 10.0
w_foot    = 1e5
reg       = 1e-6
ik_dt     = 0.005
ik_iters  = 100
ik_tol    = 1e-5

# --------- Load robot ----------
model, cmodel, vmodel = pin.buildModelsFromUrdf(
    URDF_PATH, package_dirs=["/home/rdesarz/projects"], root_joint=pin.JointModelFreeFlyer()
)
data = model.createData()
nv   = model.nv

def find_frame_id(model, candidates):
    names = {f.name: i for i,f in enumerate(model.frames)}
    for n in candidates:
        if n in names: return names[n]
    nlow = [(f.name.lower(), i) for i,f in enumerate(model.frames)]
    for n in candidates:
        nl = n.lower()
        for name,i in nlow:
            if nl in name: return i
    raise RuntimeError(f"Missing frame: {candidates}")

fid_lf   = find_frame_id(model, ["leg_left_6_link","left_sole_link","left_foot","left_ankle"])
fid_base = find_frame_id(model, ["base_link","root_link","pelvis"])

def J_com(q):
    pin.centerOfMass(model, data, q)
    pin.ccrba(model, data, q, np.zeros(nv))
    return data.Jcom.copy()

def J_frame_local(q, fid):
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    return pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL).copy()

def se3_err(Tcur, Tdes):
    return pin.log6(Tcur.inverse() * Tdes).vector

# --------- IK routine (left-foot stance) ----------
def solve_left_stance_ik(q_init, com_des, lf_pos_des, Rlf_des=None):
    q = q_init.copy()
    # fix left-foot orientation to initial if not provided
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    if Rlf_des is None:
        Rlf_des = data.oMf[fid_lf].rotation.copy()
    Tlf_des = pin.SE3(Rlf_des, lf_pos_des.copy())

    for _ in range(ik_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        com = pin.centerOfMass(model, data, q).copy()
        Tlf = data.oMf[fid_lf].copy()

        Jc  = J_com(q)                          # 3×nv
        e_c = com_des - com                     # 3
        v_c = Kp_com * e_c

        Jf  = J_frame_local(q, fid_lf)          # 6×nv
        e_f = se3_err(Tlf, Tlf_des)             # 6
        v_f = Kp_foot * e_f

        A = np.vstack((np.sqrt(w_com)*Jc, np.sqrt(w_foot)*Jf))
        b = np.hstack((np.sqrt(w_com)*v_c,  np.sqrt(w_foot)*v_f))
        H = A.T @ A + reg*np.eye(nv)
        g = -A.T @ b
        dq = np.linalg.solve(H, -g)

        q[:nv] += ik_dt * dq
        if max(np.linalg.norm(e_c), np.linalg.norm(e_f)) < ik_tol:
            break
    return q

# --------- Build a whole trajectory ----------
q0 = pin.neutral(model)
pin.forwardKinematics(model, data, q0); pin.updateFramePlacements(model, data)
Rlf0 = data.oMf[fid_lf].rotation.copy()

Qs = []
q = q0.copy()
for t in range(len(com_traj)):
    q = solve_left_stance_ik(q, com_traj[t], lf_pos_traj[t], Rlf_des=Rlf0)
    Qs.append(q.copy())
Qs = np.stack(Qs)

# --------- Visualize in MeshCat ----------
viz = MeshcatVisualizer(model, cmodel, vmodel)
viz.initViewer(open=True)
viz.loadViewerModel()

while True:
    for q in Qs:
        viz.display(q); time.sleep(dt)
    for q in reversed(Qs):
        viz.display(q); time.sleep(dt)