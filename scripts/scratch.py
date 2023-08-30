from math import pi
import numpy as np
from grasp_simulator.grasp_simulator import GraspSimulator
import matplotlib.pyplot as plt
from grasp_simulator.utils import plot_depth_image, plot_lidar_im_depth_tri


simulator = GraspSimulator(launch_viewer=False, real_time=False)
simulator.reset()
simulator.simulate_seconds(0.2)  # best to give it a bit of time to settle


simulator.simulate_seconds(30)  # leave window open for 30 more seconds
