import mujoco as mj
from mujoco import MjvOption


class Sensor:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.renderer = mj.Renderer(model, 1, 100)

    def render(self):
        self.renderer.update_scene(self.data, camera='object_side')
        # TODO, find somehow to set max distance for performance or just do it manually
        self.renderer.enable_depth_rendering()
        return self.renderer.render()
