from math import pi
from grasp_simulator.grasp_simulator import GraspSimulator
import matplotlib.pyplot as plt


simulator = GraspSimulator(launch_viewer=True, real_time=True)
simulator.verbose = 1

simulator.reset()
simulator.simulate_seconds(0.5)  # best to give it a bit of time to settle

sensor = simulator.sensor
sens = sensor.render()
sens[sens > 1] = 1
plt.imshow(sens)
plt.colorbar()
plt.show()

simulator.simulate_seconds(30)  # leave window open for 30 more seconds
simulator.run_infinitely()  # if you want to play with the viewer

# TODO:
# - other objects