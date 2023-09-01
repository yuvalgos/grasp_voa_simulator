import numpy as np
import yaml
import csv

from grasp_simulator.grasp_simulator import GraspSimulator
from generated_lidar import extract_lidar_readings
from grasp_simulator.utils import plot_depth_image, plot_lidar_im_depth_tri
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mesh_file = '../data/objects/mug/mug.obj'
    poses_file = '../config/poses/mug_poses.yaml'
    obj_name = 'mug'
    debug = False
    with open(poses_file, "r") as yaml_file:
        # Parse the YAML content into a Python dictionary
        poses = yaml.safe_load(yaml_file)
    lidar_dists = [0.15, 0.20, 0.25, 0.30]

    simulator = GraspSimulator(launch_viewer=False, real_time=False)
    for dist in lidar_dists:
        for pose in poses.keys():
            simulator.reset(object_position_offset=poses[pose]['obj_pos'],
                            object_orientation=poses[pose]['obj_orientation'], lidar_dist=dist)
            simulator.simulate_seconds(0.2)
            sensors = simulator.sensors
            lidar, im, depth_im = sensors.get_front_lidar_output(height=0.063, get_images=True)
            # plot_lidar_im_depth_tri(np.expand_dims(lidar, 0), im, depth_im, name="front lidar")
            inter_points, _, reading = extract_lidar_readings(obj_file_path=mesh_file, pose=poses[pose],
                                                              lidar_height=0.05, lidar_dist=dist)

            both_readings = []
            points_2D = []
            test = []
            for i in range(len(lidar)):
                if debug and lidar[i] < 0.5 and 90 + 22 - i in reading.keys():
                    points_2D.append(
                        [lidar[i] * np.cos(np.radians(90 + 22 - i)), lidar[i] * np.sin(np.radians(90 + 22 - i))])
                    test.append([reading[90 + 22 - i] * np.cos(np.radians(90 + 22 - i)),
                                 reading[90 + 22 - i] * np.sin(np.radians(90 + 22 - i))])
                if 90 + 22 - i in reading.keys():
                    both_readings.append((90 + 22 - i, lidar[i], reading[90 + 22 - i]))

            with open('../results/obs_func_exp/' + obj_name + '/' + pose + '_' + str(int(dist*10)) + '.csv', 'w',
                      newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(both_readings)

            if debug:
                z_plane = 0.0
                points_2D = np.array(test)
                test = np.array(inter_points)
                test_3d = np.column_stack((test, np.full(test.shape[0], z_plane)))
                points_3d = np.column_stack((points_2D, np.full(points_2D.shape[0], z_plane)))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b')
                ax.scatter(test_3d[:, 0], test_3d[:, 1], test_3d[:, 2], c='r')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()
