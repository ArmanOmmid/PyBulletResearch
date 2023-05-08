
import numpy as np
import pybullet as p
import pybullet_data
import time

# Setpu simulation
p.connect(p.GUI) # Gives us visualations of our simulation if we want it. Use p.DIRECT if we don't want to render graphics. 
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0) # Not a real time simulation

# load assets 
p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1]) # Instantiate with xyz coordinates and nijk orientation (quaternion). Defaults are [0, 0, 0] and [0, 0, 0, 1]
targid = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase = True) # Robotic Arm. Fixed Base means that the base of the object does not move (but is still physical)
obj_of_focus = targid

p.getNumJoints(targid)
jointid = 4
jlower = p.getJointInfo(targid, jointid)[8] # Create joints for Robotic Arm
jupper = p.getJointInfo(targid, jointid)[9]

for step in range(500):
    joint_two_targ = np.random.uniform(jlower, jupper)
    joint_four_targ = np.random.uniform(jlower, jupper)
    p.setJointMotorControlArray(targid, [2, 4], p.POSITION_CONTROL, targetPositions = [joint_two_targ, joint_four_targ])
    focus_position, focus_orientation = p.getBasePositionAndOrientation(targid)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focus_position) # Follow the robotic arm 
    p.stepSimulation()
    time.sleep(1./240.)
