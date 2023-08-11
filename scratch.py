from math import pi
from grasp_simulator.grasp_simulator import GraspSimulator
import matplotlib.pyplot as plt


simulator = GraspSimulator(launch_viewer=True, real_time=True)
# simulator = GraspSimulator(launch_viewer=False, real_time=False)
# Note: without viewer, the simulation is about ~20x faster
simulator.verbose = 1

# sensor = simulator.sensor
# sens = sensor.render()
# sens[sens > 1] = 1
# plt.imshow(sens)
# plt.colorbar()
# plt.show()


# object starts at default position, grasp the front
simulator.reset()
simulator.simulate_seconds(0.5)  # best to give it a bit of time to settle
res = simulator.try_grasp([0, 0, 0.1], [pi, 0, -pi/2])
print("--------grasp result: ", res)

# now grasp from the side
simulator.reset()
simulator.simulate_seconds(0.5)
res = simulator.try_grasp([0, 0, 0.1], [pi, 0, 0])
print("--------grasp result: ", res)

# object laying down, grasp from above
simulator.reset(object_orientation=[0, pi/2, 0], object_position_offset=[0, 0, 0.1])
simulator.simulate_seconds(0.5)
res = simulator.try_grasp([0, 0, 0.02], [0, pi/2, 0])
print("--------grasp result: ", res)

# object laying down, grasp high from the front and miss
simulator.reset(object_orientation=[0, pi/2, 0], object_position_offset=[0, 0, 0.1])
simulator.simulate_seconds(0.5)
res = simulator.try_grasp([0, 0, 0.2], [pi, 0, -pi/2])
print("--------grasp result: ", res)


print("grasp result: ", res)

simulator.simulate_seconds(30)
# simulator.run_inifinitely()

# TODO:
# - other objects