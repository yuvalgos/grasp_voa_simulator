from math import pi
import mujoco as mj
import numpy as np
from mujoco import MjvOption, MjvCamera
from grasp_simulator.const_and_config import TABLE_ORIGIN


class Sensor:
    def __init__(self, model, data, lidar_dist):
        self.model = model
        self.data = data

        # I would have wanted one line of 45 pixels, but there is a problem when rendering depth images with aspect
        # ratio != 1. So I had to make it 45x45 and we will just use the middle line.
        # if you change fov which is 45 degrees you needto change the height and width as well.
        self.lidar_renderer = mj.Renderer(model, 45, 45)

        self.good_camera_renderer = mj.Renderer(model, 256, 256)  # to visualize lidar point of view (for debug)
        self.good_camera_depth_renderer = mj.Renderer(model, 256, 256)  # to visualize lidar point of view (for debug)

        self.lidar_renderer.enable_depth_rendering()
        self.good_camera_depth_renderer.enable_depth_rendering()

        self.side_lidar_camera = MjvCamera()
        self.side_lidar_camera.lookat = TABLE_ORIGIN
        self.side_lidar_camera.distance = lidar_dist
        self.side_lidar_camera.elevation = 0
        self.side_lidar_camera.azimuth = 180

        self.front_lidar_camera = MjvCamera()
        self.front_lidar_camera.lookat = TABLE_ORIGIN
        self.front_lidar_camera.distance = lidar_dist
        self.front_lidar_camera.elevation = 0
        self.front_lidar_camera.azimuth = 90

        self.left_lidar_camera = MjvCamera()
        self.left_lidar_camera.lookat = TABLE_ORIGIN
        self.left_lidar_camera.distance = lidar_dist
        self.left_lidar_camera.elevation = 0
        self.left_lidar_camera.azimuth = 0

        self.back_lidar_camera = MjvCamera()
        self.back_lidar_camera.lookat = TABLE_ORIGIN
        self.back_lidar_camera.distance = lidar_dist
        self.back_lidar_camera.elevation = 0
        self.back_lidar_camera.azimuth = 270

    def reset(self, lidar_dist):
        self.side_lidar_camera.distance = lidar_dist
        self.front_lidar_camera.distance = lidar_dist

    def get_side_lidar_output(self, height=0.1, get_images=False):
        self.side_lidar_camera.lookat = TABLE_ORIGIN + [0, 0, height]
        return self.get_lidar_output_by_camera(self.side_lidar_camera, get_images)

    def get_front_lidar_output(self, height=0.1, get_images=False):
        self.front_lidar_camera.lookat = TABLE_ORIGIN + [0, 0, height]
        return self.get_lidar_output_by_camera(self.front_lidar_camera, get_images)

    def get_left_lidar_output(self, height=0.1, get_images=False):
        self.left_lidar_camera.lookat = TABLE_ORIGIN + [0, 0, height]
        return self.get_lidar_output_by_camera(self.left_lidar_camera, get_images)

    def get_back_lidar_output(self, height=0.1, get_images=False):
        self.back_lidar_camera.lookat = TABLE_ORIGIN + [0, 0, height]
        return self.get_lidar_output_by_camera(self.back_lidar_camera, get_images)

    def get_lidar_output_by_camera(self, camera: MjvCamera, get_images=False):
        self.lidar_renderer.update_scene(self.data, camera=camera)
        lidar_out = self.lidar_renderer.render()
        # cut the middle line
        lidar_out = lidar_out[self.lidar_renderer.height // 2, :]

        if not get_images:
            return lidar_out

        self.good_camera_renderer.update_scene(self.data, camera=camera)
        self.good_camera_depth_renderer.update_scene(self.data, camera=camera)
        good_camera_out = self.good_camera_renderer.render()
        good_camera_depth_out = self.good_camera_depth_renderer.render()

        return lidar_out, good_camera_out, good_camera_depth_out

    def get_lidar_output_by_parameters(self, lookat, distance, elevation, azimuth, get_images=False):
        camera = MjvCamera()
        camera.lookat = TABLE_ORIGIN + np.array(lookat)
        camera.distance = distance
        camera.elevation = elevation
        camera.azimuth = azimuth
        return self.get_lidar_output_by_camera(camera, get_images)
