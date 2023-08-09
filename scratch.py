import mujoco as mj
import mujoco.viewer as mj_viewer
import time
from math import pi
from grasp_simulator.utils import set_camera_overview
from grasp_simulator.manipulated_object import ManipulatedObject
from grasp_simulator.ur_controller import UrController
from grasp_simulator.grasp_simulator import GraspSimulator


simulator = GraspSimulator(launch_viewer=True, real_time=True)
simulator.verbose = 1
for i in range(200):
    simulator.step_simulation()
simulator.try_grasp([0, -0.65, 1.0], [0, pi/2, 0])
simulator.run_inifinitely()

# TODO:
# - other objects
# - shorter timestep 1/500 -> 1/32