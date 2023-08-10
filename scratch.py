from math import pi
from grasp_simulator.grasp_simulator import GraspSimulator
import matplotlib.pyplot as plt


simulator = GraspSimulator(launch_viewer=True, real_time=True)
# simulator = GraspSimulator(launch_viewer=False, real_time=False)
simulator.verbose = 1
simulator.simulate_seconds(1)

# sensor = simulator.sensor
# sens = sensor.render()
# sens[sens > 1] = 1
# plt.imshow(sens)
# plt.colorbar()
# plt.show()


res = simulator.try_grasp([0, -0.65, 1.25], [pi, 0, -pi/2])

simulator.reset()
simulator.simulate_seconds(0.5)
res = simulator.try_grasp([0, -0.65, 1.22], [pi, 0, -pi/2])

simulator.reset()
simulator.simulate_seconds(0.5)
res = simulator.try_grasp([0, -0.65, 1.25], [pi, 0, 0])

print("grasp result: ", res)

simulator.simulate_seconds(30)
# simulator.run_inifinitely()

# TODO:
# - other objects