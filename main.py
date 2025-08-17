import math
import time

import pybullet as p
import pybullet_data

# connect to GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load ground plane
p.loadURDF("plane.urdf")

# load humanoid (from PyBullet’s data)
humanoid = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 5], p.getQuaternionFromEuler([math.pi / 2.0, 0, 0]))

# set gravity
p.setGravity(0, 0, -9.81)

dt = 1.0 / 240.0

# run simulation
while True:
    p.stepSimulation()
    time.sleep(dt)  # 240 Hz sim step
