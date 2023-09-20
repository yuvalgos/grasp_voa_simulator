import yaml
import os
import numpy as np
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import csv


def show_ob(ob):
    v = ob
    if not isinstance(ob, np.ndarray):
        v = ob2vec(ob)
    X = []
    Y = []
    for i in range(v.shape[0]):
        if v[i] == 1:
            continue
        X.append(v[i] * np.cos(np.radians(i)))
        Y.append(v[i] * np.sin(np.radians(i)))
    plt.scatter(X, Y)
    plt.show()


def read_data(readings_file, sensor_id, pose_id):
    file_name = sensor_id[1] + '_' + pose_id + '.csv'
    source_path = os.path.join(readings_file, file_name)
    df = pd.read_csv(source_path, header=None)
    real_reading = np.ones((360,))
    generated_reading = np.ones((360,))
    for index, row in df.iterrows():
        degrees = int(row[0])
        if float(row[1]) > 1.:
            continue
        real_reading[degrees] = row[1]
        generated_reading[degrees] = row[2]
    return real_reading, generated_reading


def read_grasp_score(obj, noise=0.):
    file = '../results/grasp_score/' + obj + '/' + obj
    if noise == 0:
        file += '.csv'
    else:
        file += '_' + str(noise) + '.csv'

    grasp_score = {}
    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Read the header row to get column names
        column_names = next(csv_reader)[1:]

        # Iterate through the remaining rows
        for row in csv_reader:
            # Use the first column (row[0]) as the key for the outer dictionary
            key = 'G' + str(int(row[0][-1]) - 1)

            # Create an inner dictionary for this row, using column names as keys
            inner_dict = {column_names[i]: float(value) / 100 for i, value in enumerate(row[1:])}

            # Add the inner dictionary to the data_dict using the key
            grasp_score[key] = inner_dict

        return grasp_score


def ob2vec(ob):
    v = np.ones((360,))
    for degree in ob.keys():
        v[degree] = ob[degree]
    return v


def similarity(ob1, ob2):
    v1 = ob2vec(ob1)
    v2 = ob2vec(ob2)
    norm = np.linalg.norm(v1 - v2)
    return float(np.exp(-norm))


class GaussianSim:
    def __init__(self, std=1):
        self.name = 'gaussian_similarity'
        self.std = std
        self.id = 3

    def __call__(self, ob1, ob2):
        v1 = ob1
        v2 = ob2
        if not isinstance(ob1, np.ndarray):
            v1 = ob2vec(ob1)
        if not isinstance(ob2, np.ndarray):
            v2 = ob2vec(ob2)
        norm = np.linalg.norm(v1 - v2)
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-norm ** 2 / (2 * self.std ** 2))


class NormSim:
    def __init__(self):
        self.name = 'norm_similarity'
        self.id = 2

    def __call__(self, ob1, ob2):
        v1 = ob1
        v2 = ob2
        if not isinstance(ob1, np.ndarray):
            v1 = ob2vec(ob1)
        if not isinstance(ob2, np.ndarray):
            v2 = ob2vec(ob2)
        norm = np.linalg.norm(v1 - v2)
        return float(np.exp(-norm))


class DeterministicSim:
    def __init__(self, err=0.008):
        self.name = 'deterministic_similarity'
        self.err = err
        self.id = 1

    def __call__(self, ob1, ob2):
        v1 = ob1
        v2 = ob2
        if not isinstance(ob1, np.ndarray):
            v1 = ob2vec(ob1)
        if not isinstance(ob2, np.ndarray):
            v2 = ob2vec(ob2)
        is_in_range = (v1 - v2 >= -self.err) & (v1 - v2 <= self.err)
        return np.all(is_in_range).astype(float)


class VOA:
    def __init__(self, obj, noise, readings_file):  # , poses_file, mesh_file):
        self.readings_file = readings_file
        self.obj = obj
        self.noise = noise
        self.grasp_score = read_grasp_score(obj, noise)
        self.grasps = list(self.grasp_score.keys())
        self.obj_poses = list(self.grasp_score[self.grasps[0]].keys())
        self.sensor_configs = ['Q1', 'Q2', 'Q3', 'Q4']
        self.belief = {}
        for pose in self.obj_poses:
            self.belief[pose] = 1 / len(self.obj_poses)

        self.init_grasp_star = None
        self.init_q_star = 0.0
        for grasp in self.grasps:
            q = 0
            for p_k in self.obj_poses:
                q += self.belief[p_k] * self.grasp_score[grasp][p_k]
            if self.init_q_star < q:
                self.init_q_star = q
                self.init_grasp_star = grasp

        self.generated_readings = {}
        for sensor_id, pose_id in product(self.sensor_configs, self.obj_poses):
            if not sensor_id in self.generated_readings.keys():
                self.generated_readings[sensor_id] = {}
            _, self.generated_readings[sensor_id][pose_id] = read_data(self.readings_file, sensor_id, pose_id)
            # show_ob(self.generated_readings[sensor_id][pose_id])

    def voa_belief_update(self, sensor_id, p_i, sim_function):
        belief = {}
        sensor_id_generate_readings = self.generated_readings[sensor_id]
        normalization = 0
        for p_j in self.obj_poses:
            belief[p_j] = sim_function(sensor_id_generate_readings[p_i], sensor_id_generate_readings[p_j]) * \
                          self.belief[p_j]
            normalization += belief[p_j]
        for p in belief.keys():
            belief[p] /= normalization
        return belief

    def belief_update_by_ob(self, sim_function, ob, sensor_id):
        belief = {}
        sensor_id_generate_readings = self.generated_readings[sensor_id]
        normalization = 0
        for p_j in self.obj_poses:
            belief[p_j] = sim_function(ob, sensor_id_generate_readings[p_j]) * self.belief[p_j]
            normalization += belief[p_j]
        if normalization == 0:
            norm = NormSim()
            max_value = max(norm(ob, sensor_id_generate_readings[k]) for k in belief.keys())
            max_keys = [k for k in belief.keys() if max_value - norm(ob, sensor_id_generate_readings[k]) < 0.015]
            for k in max_keys:
                belief[k] = 1.0 / len(max_keys)
            return belief
        for p in belief.keys():
            belief[p] /= normalization
        return belief

    def __call__(self, sensor_id, sim_function):
        voa = -self.init_q_star
        for p_i in self.obj_poses:
            new_belief = self.voa_belief_update(sensor_id, p_i, sim_function)
            _, q_star = self.best_grasp(new_belief)
            voa += self.belief[p_i] * q_star
        return voa

    def best_grasp(self, belief):
        q_star = 0
        best_grasp = None
        for grasp in self.grasps:
            q = 0
            for p_k in self.obj_poses:
                q += belief[p_k] * self.grasp_score[grasp][p_k]
            if q > q_star:
                q_star = q
                best_grasp = grasp
        return best_grasp, q_star

    def results_table1(self, sensors, sim_function):
        csv_table = [[''], [''], ['true'], ['∅']]
        for p in self.obj_poses:
            csv_table[0] += [id_correcting(p), id_correcting(p), id_correcting(p), id_correcting(p)]
            csv_table[1] += ['x₉*', 'γ*(β\')', 'xˆ₉*', 'γ*(βˆ\')']
        table = {'': {}, 'ground_truth': {}}
        for pose in self.obj_poses:
            table[''][pose] = (self.init_grasp_star, self.init_q_star, self.init_grasp_star, self.init_q_star)
            s_star = 0.
            g_star = None
            for grasp in self.grasps:
                gs = self.grasp_score[grasp][pose]
                if gs > s_star:
                    s_star = gs
                    g_star = grasp
            table['ground_truth'][pose] = (g_star, s_star, g_star, s_star)
            csv_table[2] += [id_correcting(g_star), s_star, id_correcting(g_star), s_star]
            csv_table[3] += [id_correcting(self.init_grasp_star), self.init_q_star, id_correcting(self.init_grasp_star),
                             self.init_q_star]
        for i, x_s in enumerate(sensors):
            table[x_s] = {}
            csv_table += [[x_s]]
            for pose in self.obj_poses:
                real, gen = read_data(self.readings_file, x_s, pose)
                beta = self.belief_update_by_ob(sim_function, real, x_s)
                beta_hat = self.belief_update_by_ob(sim_function, gen, x_s)
                x_star, q_star = self.best_grasp(beta)
                x_star_hat, q_star_hat = self.best_grasp(beta_hat)
                table[x_s][pose] = (x_star, q_star, x_star_hat, q_star_hat)
                csv_table[i + 4] += [id_correcting(x_star), q_star, id_correcting(x_star_hat), q_star_hat]
        with open('../results/tables/' + self.obj + '/' + str(self.noise) + '_' + sim_function.name + '_table.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in csv_table:
                csv_writer.writerow(row)
        return table

    def plot_example(self):
        file_name = '1_pose_4.csv'
        source_path = os.path.join(self.readings_file, file_name)
        df = pd.read_csv(source_path, header=None)
        real_reading = {}
        generated_reading = {}
        for index, row in df.iterrows():
            degrees = int(row[0])
            real_reading[degrees] = row[1]
            generated_reading[degrees] = row[2]
        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        for i in real_reading.keys():
            X1.append(real_reading[i] * np.cos(np.radians(i)))
            X2.append(generated_reading[i] * np.cos(np.radians(i)))
            Y1.append(real_reading[i] * np.sin(np.radians(i)))
            Y2.append(generated_reading[i] * np.sin(np.radians(i)))
        plt.scatter(X2, Y2, c='b', s=50)
        plt.scatter(X1, Y1, c='r', s=50)  # 's' controls the size of the points
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.title('Real Data vs. Synthetic Data')
        # plt.legend()
        plt.show()


class AEVD:
    def __init__(self, obj, noise, readings_file):  # , poses_file, mesh_file):
        self.readings_file = readings_file
        self.grasp_score = read_grasp_score(obj, noise)
        self.grasps = list(self.grasp_score.keys())
        self.obj_poses = list(self.grasp_score[self.grasps[0]].keys())
        self.sensor_configs = ['Q1', 'Q2', 'Q3', 'Q4']
        self.belief = {}
        for pose in self.obj_poses:
            self.belief[pose] = 1 / len(self.obj_poses)

        self.init_q_star = 0
        self.init_grasp_star = ''
        for grasp in self.grasps:
            q = 0
            for p_k in self.obj_poses:
                q += self.belief[p_k] * self.grasp_score[grasp][p_k]
            if self.init_q_star < q:
                self.init_q_star = q
                self.init_grasp_star = grasp

        self.generated_readings = {}
        self.real_readings = {}
        for sensor_id, pose_id in product(self.sensor_configs, self.obj_poses):
            if not sensor_id in self.generated_readings.keys():
                self.generated_readings[sensor_id] = {}
                self.real_readings[sensor_id] = {}
            self.real_readings[sensor_id][pose_id], self.generated_readings[sensor_id][pose_id] = read_data(
                self.readings_file, sensor_id, pose_id)

    def belief_update(self, sensor_id, ob, similarity):
        belief = {}
        sensor_id_generate_readings = self.generated_readings[sensor_id]
        normalization = 0
        for p_j in self.obj_poses:
            belief[p_j] = similarity(ob, sensor_id_generate_readings[p_j]) * self.belief[p_j]
            normalization += belief[p_j]
        if normalization == 0:
            norm = NormSim()
            max_value = max(norm(ob, sensor_id_generate_readings[k]) for k in belief.keys())
            max_keys = [k for k in belief.keys() if max_value - norm(ob, sensor_id_generate_readings[k]) < 0.015]
            for k in max_keys:
                belief[k] = 1.0 / len(max_keys)
            return belief
        for p in belief.keys():
            belief[p] /= normalization
        return belief

    def __call__(self, sensor_id, sim_function):
        aevd = -self.init_q_star
        for p_i in self.obj_poses:
            new_belief = self.belief_update(sensor_id, self.real_readings[sensor_id][p_i], sim_function)
            q_star = 0
            for grasp in self.grasps:
                q = 0
                for p_k in self.obj_poses:
                    q += new_belief[p_k] * self.grasp_score[grasp][p_k]
                q_star = max(q, q_star)
            aevd += q_star * self.belief[p_i]
        return aevd


def int2Roman(num):
    if num == 1:
        return 'I'
    if num == 2:
        return 'II'
    if num == 3:
        return 'III'
    if num == 4:
        return 'IV'


def analysis1(obj, noise):
    voa_calc = VOA(obj=obj, noise=noise, readings_file='../results/different_pov/' + obj)

    sims = [DeterministicSim(), GaussianSim(), NormSim()]
    colors1 = ['blue', 'red', 'green']
    colors2 = ['dodgerblue', 'tomato', 'lime']
    sensors = ['Q1', 'Q2', 'Q3', 'Q4']
    aevd_calc = AEVD(obj=obj, noise=noise, readings_file='../results/different_pov/' + obj)

    voa_points = {}
    aevd_points = {}
    for sensor in sensors:
        voa_points[sensor] = []
        aevd_points[sensor] = []
    for sim_function, sensor_id in product(sims, sensors):
        voa_points[sensor_id].append(voa_calc(sensor_id, sim_function))
        aevd_points[sensor_id].append(aevd_calc(sensor_id, sim_function))
    voa_y_lines = {}
    aevd_y_lines = {}
    for i, sim in enumerate(sims):
        voa_y_lines[sim] = []
        aevd_y_lines[sim] = []
        for sensor in sensors:
            voa_y_lines[sim].append(voa_points[sensor][i])
            aevd_y_lines[sim].append(aevd_points[sensor][i])
    for i, category in enumerate(sensors):
        x_values = [int2Roman(int(category[1]))] * len(sims)
        plt.scatter(x_values, voa_points[category], marker='o', color=colors1, s=1500)
        plt.scatter(x_values, aevd_points[category], marker='^', color=colors2, s=2000)
    # for i, sim in enumerate(sims):
    #     plt.plot(ssensors, voa_y_lines[sim], color=colors1[i], linestyle='-', linewidth=2)
    #     plt.plot(ssensors, aevd_y_lines[sim], color=colors2[i], linestyle='-', linewidth=2)

    plt.xlabel('Sensor Config', fontsize=24)
    handles = []
    handles.append(plt.Line2D([0], [0], marker='o', color='b', label=f'VOA, metric 1', markersize=20))
    handles.append(plt.Line2D([0], [0], marker='^', color='dodgerblue', label=f'Executed, metric 1', markersize=20))
    handles.append(plt.Line2D([0], [0], marker='o', color='r', label=f'VOA, metric 2', markersize=20))
    handles.append(plt.Line2D([0], [0], marker='^', color='tomato', label=f'Executed, metric 2', markersize=20))
    handles.append(plt.Line2D([0], [0], marker='o', color='g', label=f'VOA, metric 3', markersize=20))
    handles.append(plt.Line2D([0], [0], marker='^', color='lime', label=f'Executed, metric 3', markersize=20))
    plt.legend(handles=handles, loc='lower right', ncol=1, fontsize='xx-large')

    # Rotate x-axis labels for better readability (optional)
    plt.xticks(rotation=45, fontsize=24)
    plt.tight_layout()
    plt.gcf().set_size_inches(14, 10)
    plt.savefig('../results/scatter_graph/' + obj + '/n_' + str(noise) + '.png')
    plt.show()


def id_correcting(base_id):
    if base_id is None:
        return None
    return base_id[:-1] + str(int(base_id[-1]) - 1)


def table(obj, noise):
    voa_calc = VOA(obj=obj, noise=noise, readings_file='../results/different_pov/' + obj)
    sims = [DeterministicSim(), GaussianSim(), NormSim()]
    sensors = ['Q1', 'Q2', 'Q3', 'Q4']
    for sim in sims:
        voa_calc.results_table1(sensors, sim)


def base(obj, noise, test_num):
    voa_calc = VOA(obj=obj, noise=noise, readings_file='../results/different_pov/' + obj)
    aevd_calc = AEVD(obj=obj, noise=noise, readings_file='../results/different_pov/' + obj)
    sims = [DeterministicSim(), GaussianSim(), NormSim()]
    sensors = ['Q1', 'Q2', 'Q3', 'Q4']
    if test_num == 1:
        for sim, sensor in product(sims, sensors):
            print(sim.name, sensor, voa_calc(sensor, sim), aevd_calc(sensor, sim))
    if test_num == 2:
        fig, axes = plt.subplots(4, 3, figsize=(12, 9))
        i = 0
        for sensor, sim in product(sensors, sims):
            file_name = sensor[1] + '_1.csv'
            source_path = os.path.join('../results/' + obj + '/different_pov', file_name)
            df = pd.read_csv(source_path, header=None)
            real_reading = {}
            for index, row in df.iterrows():
                degrees = int(row[0])
                real_reading[degrees] = row[1]
            ax = axes.flat[i]
            belief = aevd_calc.belief_update(sensor, real_reading, sim)
            values = np.zeros((1, 4))
            for pose in belief.keys():
                values[0, int(pose[1]) - 1] = belief[pose]
            ax.imshow(values, cmap='GnBu', vmin=0, vmax=1)
            for j in range(4):
                ax.text(j, 0, "{:.4f}".format(belief['P' + str(j + 1)]), ha='center', va='center', color='black')
            title = 'Sensor config ' + sensor[1], 'τ' + str(sim.id)
            ax.set_title(title, fontsize=16)
            ax.axis('off')
            i += 1

        # Adjust spacing between subplots (optional)
        plt.tight_layout()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    obj = 'mug'
    noise = 0.05
    analysis1(obj, noise)
    table(obj, noise)
