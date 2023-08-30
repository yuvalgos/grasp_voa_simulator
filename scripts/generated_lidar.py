import numpy as np
import math
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)


def visualize_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection from the vertices and faces
    mesh = [vertices[face] for face in faces]
    ax.add_collection3d(Poly3DCollection(mesh, alpha=0.6, facecolors='cyan', linewidths=1, edgecolors='r'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def find_intersecting_triangles(vertices, faces, h):
    intersecting_triangles = []

    for face in faces:
        vertices_of_face = vertices[face]
        z_values = vertices_of_face[:, 2]

        # Check if the vertices of the triangle cross the plane at height 'h'
        above_h = np.any(z_values >= h)
        below_h = np.any(z_values <= h)

        if above_h and below_h:
            intersecting_triangles.append(face)

    return intersecting_triangles


def intersection_plane_segment(point1, point2, h):
    if (point1[2] < h and point2[2] < h) or (point1[2] > h and point2[2] > h):
        # Both endpoints are on the same side of the plane, no intersection.
        return None

    # Calculate the intersection point using linear interpolation.
    t = (h - point1[2]) / (point2[2] - point1[2])
    x = point1[0] + t * (point2[0] - point1[0])
    y = point1[1] + t * (point2[1] - point1[1])
    return np.array([x, y])


def extract_contour(vertices, triangle_vertices, h):
    intersection_segments = []
    whole_points = []
    for i, face in enumerate(triangle_vertices):
        triangle = vertices[face]
        triangle_intersection_points = []

        for j in range(3):
            p1, p2 = triangle[j], triangle[(j + 1) % 3]
            if (p1[2] < h < p2[2]) or (p1[2] > h > p2[2]):
                intersection_point = intersection_plane_segment(p1, p2, h)
                triangle_intersection_points.append(intersection_point)
                whole_points.append(intersection_point)
        if len(triangle_intersection_points) > 1:
            intersection_segments.append(triangle_intersection_points)
    return np.array(intersection_segments), np.array(whole_points)


def transformation(tx, ty, tz, angle_x, angle_y, angle_z):
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])
    rotation_matrix_y = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ])
    rotation_matrix_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ])

    combined_rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    translation_vector = np.array([tx, ty, tz])

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = combined_rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


def closest_intersection(segments, m):
    closest_point = (float('inf'), float('inf'))

    # Iterate through the segments
    for segment in segments:
        x1 = segment[0, 0]
        y1 = segment[0, 1]
        x2 = segment[1, 0]
        y2 = segment[1, 1]

        slope_2 = (y2 - y1) / (x2 - x1)
        x = (y1 - slope_2 * x1) / (m - slope_2)
        y = m * x

        # Check if the intersection point is within the segment
        if x1 <= x <= x2 or x2 <= x <= x1:
            distance_to_origin = math.sqrt(x ** 2 + y ** 2)
            current_point = (x, y)

            # Update the closest intersection point if this one is closer
            if distance_to_origin < math.sqrt(closest_point[0] ** 2 + closest_point[1] ** 2):
                closest_point = current_point

    return closest_point


def extract_lidar_readings(obj_file_path, pose=None, pose_file=None, lidar_height=0.05, lidar_dist=0.25, scale=0.01):
    vertices, faces = load_obj(obj_file_path)
    vertices *= scale
    desired_pose = pose
    if pose_file is not None:
        with open(pose_file, 'r') as yaml_file:
            desired_pose = yaml.load(yaml_file, Loader=yaml.FullLoader)['pose_3']
    tx, ty, tz = 0., 0., 0.
    # ty -= lidar_dist
    rx, ry, rz = desired_pose['obj_orientation']
    trans = transformation(tx, ty, tz, rx, ry, rz)
    vertices = np.dot(np.hstack((vertices, np.ones((vertices.shape[0], 1)))), trans)[:, :3]
    visualize_mesh(vertices, faces)
    h = lidar_height

    intersecting_triangles = np.array(find_intersecting_triangles(vertices, faces, h))
    contour, points = extract_contour(vertices, intersecting_triangles, h)

    inter_points = []
    readings = {}
    for i in range(360):
        pnt = closest_intersection(contour, np.tan(i))
        if pnt[0] != np.inf and pnt[1] != np.inf:
            inter_points.append(pnt)
            idx = i if i <= 180 else i - 360
            readings[idx] = (np.sqrt(pnt[1] ** 2 + pnt[0] ** 2))
    return np.array(inter_points), points, readings


if __name__ == "__main__":
    mesh_file = '../data/objects/mug/mug.obj'
    poses_file = '../config/poses/mug_poses.yaml'
    inter_points, points, readings = extract_lidar_readings(mesh_file, pose_file=poses_file, lidar_height=0.03)
    # plt.scatter(inter_points[:, 0], inter_points[:, 1])
    # plt.show()

    # # Visualize the mesh with intersecting triangles
    # # visualize_mesh(vertices, intersecting_triangles)
    # # Choose a constant z value for the plane
    z_plane = 0.0  # You can set this to any value you like

    # Create a 3D representation of the points by appending the constant z value
    # points_3d = np.column_stack((points, np.full(points.shape[0], z_plane)))
    inter_points_3d = np.column_stack((inter_points, np.full(inter_points.shape[0], z_plane)))

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter the points on the plane
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    ax.scatter(inter_points_3d[:, 0], inter_points_3d[:, 1], inter_points_3d[:, 2])

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
