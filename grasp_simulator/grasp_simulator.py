import numpy as np
import scipy
from grasp_simulator.utils import set_camera_overview
from grasp_simulator.manipulated_object import ManipulatedObject
from grasp_simulator.ur_controller import UrController
import mujoco as mj
import mujoco.viewer as mj_viewer


class GraspSimulator:
    def __init__(self, launch_viewer=True):
        self.model = mj.MjModel.from_xml_path("./data/world_mug.xml")
        self.data = mj.MjData(self.model)

        self.m_obj = ManipulatedObject(self.model, self.data)
        self.controller = UrController(self.model, self.data)

        # mj.mj_resetData(model, data)
        mj.mj_forward(self.model, self.data)

        self.viewer = None
        if launch_viewer:
            self.viewer = mj_viewer.launch(self.model, self.data)
            set_camera_overview(self.viewer)

        # just put the object in a nice position for starting
        self.m_obj.set_position([0, -0.6, 1.2])
        self.m_obj.set_orientation_euler([0, 0, 1.5])

    def try_grasp(self, ee_pos, ee_orientation) -> bool:
        ''' try a grasp with the current object pose and a given grasp parameters, return whether successful'''
        
        initial_obj_pos = self.m_obj.get_position()
        
        self.move_ee_to_pre_grasp(ee_pos, ee_orientation)
        self.move_from_pre_grasp_to_grasp_pose()
        self.controller.close_gripper()
        self.pick_up()
        
        final_obj_pos = self.m_obj.get_position()
        
        return self.check_grasp_success(initial_obj_pos, final_obj_pos)

    def move_ee_to_pre_grasp(self, ee_pos, ee_orientation):
        # pre grasp is 15cm back from the object in end effector coordinates
        # get orientation rotation matrix:
        R = scipy.spatial.transform.Rotation.from_euler('xyz', ee_orientation).as_matrix()
        # get pre grasp position in world coordinates:
        pre_grasp_pos = ee_pos - R @ np.array([0.15, 0, 0])

        self.controller.set_control_input_with_ik(pre_grasp_pos, ee_orientation)


    def move_from_pre_grasp_to_grasp_pose(self):
        pass

    def pick_up(self):
        pass

    def check_grasp_success(self, initial_obj_pos, final_obj_pos) -> bool:
        pass
        

