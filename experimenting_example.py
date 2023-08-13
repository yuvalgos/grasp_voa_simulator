from math import pi
import numpy as np
from grasp_simulator.grasp_simulator import GraspSimulator
import tqdm
from tqdm.contrib import itertools


# try to grasp at grid of positions x = [-0.05, 0.05], y = [-0.05, 0.05], z = [0.03, 0.1]
# generate a list of all coordinates:
x = np.linspace(-0.05, 0.05, 3)
y = np.linspace(-0.05, 0.05, 3)
z = np.array([0.03, 0.1])
grasping_positions = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

# for each position try to grasp from different directions:
rx = np.linspace(-pi, pi, 5)
ry = np.linspace(-pi, pi, 5)
rz = np.linspace(-pi, pi, 5)
grasping_orientations = np.array(np.meshgrid(rx, ry, rz)).T.reshape(-1, 3)

print(f"trying to grasp at {len(grasping_positions)} positions and {len(grasping_orientations)} orientations, total"
      f"of {len(grasping_positions) * len(grasping_orientations)} grasps")


simulator = GraspSimulator(launch_viewer=False, real_time=False)

# will take less than an hour on crappy laptop:
for grasp_position, grasp_orientation in itertools.product(grasping_positions, grasping_orientations):
        result = simulator.try_grasp(grasp_position, grasp_orientation)

# TODO: iterate over stable poses, and add noise to x,y,rz in each pose
# TODO: choose less grasp orientation. need to look manually only on the relevant one's
# TODO: now a grasp takes about a second. maybe we can change waiting time for each step in the simulator
#       to speed it up a bit.
# TODO: do the experiments concurrently on different processes
