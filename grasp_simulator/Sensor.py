from math import pi
import mujoco as mj
from mujoco import MjvOption, MjvCamera
from grasp_simulator.const_and_config import TABLE_ORIGIN


class Sensor:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.renderer = mj.Renderer(model, 45, 45)

    def render(self):
        height = 0.1

        camera = MjvCamera()
        camera.lookat = TABLE_ORIGIN + [0, 0, height]
        camera.distance = 0.25
        camera.elevation = 0
        camera.azimuth = 180

        # if you want to change field of view do it here, but after rendering you have to return to the old value (45)
        # and also note that you have to change resolution of renderer, so it will be one pixel per angle like the lab.
        # self.model.vis.global_.fovy = 60

        # option = MjvOption() # if you want to change the rendering options do it here
        # option.flags[mj.mjtVisFlag.mjVIS...] = ???


        self.renderer.update_scene(self.data, camera=camera)
        self.renderer.enable_depth_rendering()
        return self.renderer.render()
