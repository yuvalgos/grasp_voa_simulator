import numpy as np
import yaml
import matplotlib.pyplot as plt
import csv

from generated_lidar import extract_lidar_readings
from grasp_simulator.grasp_simulator import GraspSimulator

debug = True


def degree2pixel(degree):
    if degree < 180:
        idx = 22 - degree
    else:
        idx = 22 + (360 - degree)
    return idx


def pixel2degree(pixel):
    if pixel <= 22:
        degree = 22 - pixel
    else:
        degree = 360 - (pixel - 22)
    return degree


def different_pov(mesh_file, dist, scale, poses, obj, xml_file):
    simulator = GraspSimulator(launch_viewer=True, real_time=True, obj_file=xml_file, lidar_dist=dist)
    # samples, readings = sample_lidar(self.center, self.radius, self.handle_len, self.handle_angle)
    for pose_id, pose in poses.items():
        simulator.reset(object_position_offset=pose['obj_pos'],
                        object_orientation=pose['obj_orientation'], lidar_dist=dist)
        simulator.simulate_seconds(0.2)
        sensors = simulator.sensors
        generated = {}
        inter = {}
        for q in range(1, 5):
            samples, _, readings = extract_lidar_readings(obj_file_path=mesh_file, pose=pose,
                                                          lidar_height=0.05, lidar_dist=dist,
                                                          scale=scale, q=q)
            generated[q] = readings
            inter[q] = samples

        recorded = {}
        hh = 0.0682
        recorded[1], x0, y0 = sensors.get_left_lidar_output(height=hh, get_images=True)
        recorded[2], x1, y1 = sensors.get_front_lidar_output(height=hh, get_images=True)
        recorded[3], x2, y2 = sensors.get_side_lidar_output(height=hh, get_images=True)
        recorded[4], x3, y3 = sensors.get_back_lidar_output(height=hh, get_images=True)

        for q in range(1, 5):
            lidar = recorded[q]
            both_readings = []
            points_2D = []
            test = []
            for i in range(len(lidar)):
                degree = pixel2degree(i)
                if lidar[i] < 0.5 and degree in generated[q].keys():
                    points_2D.append(
                        [lidar[i] * np.cos(np.radians(degree)), lidar[i] * np.sin(np.radians(degree))])
                    test.append([generated[q][degree] * np.cos(np.radians(degree)),
                                 generated[q][degree] * np.sin(np.radians(degree))])
                if degree in generated[q].keys():
                    both_readings.append((degree, lidar[i], generated[q][degree]))

            with open('../results/different_pov/' + obj + '/' + str(q) + '_' + str(pose_id) + '.csv', 'w',
                      newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(both_readings)

            if debug:
                z_plane = 0.0
                points_2D = np.array(points_2D)
                test = np.array(inter[q])
                test_3d = np.column_stack((test, np.full(test.shape[0], z_plane)))
                points_3d = np.column_stack((points_2D, np.full(points_2D.shape[0], z_plane)))
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r')
                ax.scatter(test_3d[:, 0], test_3d[:, 1], test_3d[:, 2], c='b')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()


if __name__ == '__main__':
    obj = 'mug'
    mesh_file = '../data/objects/' + obj + '/' + obj + '.obj'
    dist = 0.2
    scale = 0.01
    xml_file = '../data/world_' + obj + '.xml'
    with open('../config/poses/' + obj + '_poses.yaml', 'r') as yaml_file:
        poses = yaml.safe_load(yaml_file)
    different_pov(mesh_file=mesh_file, dist=dist, scale=scale, poses=poses, obj=obj, xml_file=xml_file)
