import numpy as np
import yaml

from grasp_simulator.grasp_simulator import GraspSimulator
from generated_lidar import extract_lidar_readings
from grasp_simulator.utils import plot_depth_image, plot_lidar_im_depth_tri

if __name__ == '__main__':
    mesh_file = '../data/objects/mug/mug.obj'
    poses_file = '../config/poses/mug_poses.yaml'
    with open(poses_file, "r") as yaml_file:
        # Parse the YAML content into a Python dictionary
        poses = yaml.safe_load(yaml_file)
    lidar_dists = [0.15, 0.20, 0.25, 0.30]

    simulator = GraspSimulator(launch_viewer=True, real_time=True)
    for dist in lidar_dists:
        for pose in poses.keys():
            simulator.reset(object_position_offset=poses[pose]['obj_pos'],
                            object_orientation=poses[pose]['obj_orientation'], lidar_dist=dist)
            simulator.simulate_seconds(0.2)
            sensors = simulator.sensors
            lidar, im, depth_im = sensors.get_front_lidar_output(height=0.05, get_images=True)
            plot_lidar_im_depth_tri(np.expand_dims(lidar, 0), im, depth_im, name="front lidar")
            _, _, reading = extract_lidar_readings(obj_file_path=mesh_file, pose=poses[pose], lidar_height=0.0,
                                                   lidar_dist=dist)
