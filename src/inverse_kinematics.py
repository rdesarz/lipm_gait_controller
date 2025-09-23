import os, pinocchio as pin
from time import sleep

from pinocchio.visualize import MeshcatVisualizer

PKG_PARENT = "/home/rdesarz/projects/"  # parent dir containing talos_data/
URDF = os.path.join(PKG_PARENT, "talos-data/urdf/talos_full.urdf")

model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF, PKG_PARENT, pin.JointModelFreeFlyer())
data = model.createData()

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True); viz.loadViewerModel()

q = pin.neutral(model)  # or your previous configuration
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
left_foot  = model.getFrameId("left_sole_link")
right_foot = model.getFrameId("right_sole_link")

viz.display(q)

sleep(4.0)