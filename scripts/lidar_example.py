import numpy as np
from grasp_simulator.grasp_simulator import GraspSimulator
from grasp_simulator.utils import plot_depth_image, plot_lidar_im_depth_tri


simulator = GraspSimulator(launch_viewer=False, real_time=False)
simulator.reset()
simulator.simulate_seconds(0.2)  # best to give it a bit of time to settle

# lidar is used through simulator. sensors
sensors = simulator.sensors

# this method gets a lidar output that is form the side of the object, at a given height.
# The lidar is "placed" 25 cm away from the table origin.
side_lidar_out = sensors.get_side_lidar_output(height=0.05)
plot_depth_image(np.expand_dims(side_lidar_out, 0), name="side lidar")
# output is array of size 45, for 45 degrees of the lidar.

# for debugging, you can also get images from a camera that is placed at the same location as the lidar with
# get_images=True. you will get lidar output, image, and depth image.
lidar, im, depth_im = sensors.get_side_lidar_output(height=0.05, get_images=True)
plot_lidar_im_depth_tri(np.expand_dims(side_lidar_out, 0), im, depth_im, name="side lidar")

# there is also a predefined front lidar which is placed 25 cm front of the object.
lidar, im, depth_im = sensors.get_front_lidar_output(height=0.05, get_images=True)
plot_lidar_im_depth_tri(np.expand_dims(lidar, 0), im, depth_im, name="front lidar")

# different height
lidar, im, depth_im = sensors.get_side_lidar_output(height=0.15, get_images=True)
plot_lidar_im_depth_tri(np.expand_dims(lidar, 0), im, depth_im, name="side lidar height=15cm")

# custom camera settings:
# you can get lidar (and images) from any pose of camera you want using get_lidar_output_by_parameters.
lidar, im, depth_im = sensors.get_lidar_output_by_parameters(lookat=[0.05, 0, 0.1],
                                                             distance=0.4,
                                                             elevation=-45,
                                                             azimuth=45,
                                                             get_images=True)
plot_lidar_im_depth_tri(np.expand_dims(lidar, 0), im, depth_im, name="custom camera", depth_threshold=1)


simulator.simulate_seconds(30)  # leave window open for 30 more seconds
