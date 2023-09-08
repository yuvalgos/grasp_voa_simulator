import numpy as np
import random
from tqdm.contrib import itertools
import yaml
import csv

from grasp_simulator.grasp_simulator import GraspSimulator

if __name__ == '__main__':
    grasps_file = '../config/grasps/mug_grasps.yaml'
    with open(grasps_file, "r") as yaml_file:
        # Parse the YAML content into a Python dictionary
        grasps = yaml.safe_load(yaml_file)

    poses_file = '../config/poses/mug_poses.yaml'
    with open(poses_file, "r") as yaml_file:
        # Parse the YAML content into a Python dictionary
        poses = yaml.safe_load(yaml_file)

    trails_per_pair = 100
    simulator = GraspSimulator(launch_viewer=False, real_time=False, obj_file='../data/world_expo.xml')
    rows = [[''] + list(poses.keys())]
    # will take less than an hour on crappy laptop:
    row = []
    max_translation_noise = 0.01  # Maximum translation noise in meters
    max_rotation_noise = float(np.radians(2.0))
    for grasp, pose in itertools.product(grasps.keys(), poses.keys()):
        counter = 0
        for _ in range(trails_per_pair):
            translation_noise = np.array([random.uniform(-max_translation_noise, max_translation_noise) for _ in range(3)])
            rotation_noise = np.array([random.uniform(-max_rotation_noise, max_rotation_noise) for _ in range(3)])
            noisy_pos = np.array(poses[pose]['obj_pos']) + translation_noise
            noisy_ort = poses[pose]['obj_orientation'] + rotation_noise
            simulator.reset(object_position_offset=noisy_pos,
                            object_orientation=noisy_ort)
            result = simulator.try_grasp(grasps[grasp]['grasp_pos'], grasps[grasp]['grasp_orientation'])
            counter += 1 if result else 0
        row.append(str(counter))
        if len(row) == len(poses):
            rows.append([grasp] + row)
            row = []

    with open('../results/grasp_score/mug_' + str(max_translation_noise) + '.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)
