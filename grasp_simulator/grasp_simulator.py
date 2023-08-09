import time

import numpy as np
import scipy
from grasp_simulator.utils import set_camera_overview
from grasp_simulator.manipulated_object import ManipulatedObject
from grasp_simulator.ur_controller import UrController
import mujoco as mj
import mujoco.viewer as mj_viewer


class GraspSimulator:
    def __init__(self, launch_viewer=True, real_time=False):
        """
        :param launch_viewer: whether to launch the mujoco viewer
        :param real_time: whether to run the simulation in real time for visualisation or as fast as possible
        """
        self.real_time = real_time

        self.model = mj.MjModel.from_xml_path("./data/world_mug.xml")
        self.data = mj.MjData(self.model)

        self.m_obj = ManipulatedObject(self.model, self.data)
        self.controller = UrController(self.model, self.data)

        # mj.mj_resetData(model, data)
        mj.mj_forward(self.model, self.data)

        self.viewer = None
        if launch_viewer:
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            set_camera_overview(self.viewer)

        # just put the object in a nice position for starting
        self.m_obj.set_position([0, -0.65, 1.2])
        self.m_obj.set_orientation_euler([0, 0, 1.5])

        self.verbose = 0

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

        success = self.move_ee_to_pose(pre_grasp_pos, ee_orientation)
        if self.verbose>0:
            print("move_ee_to_pre_grasp success: ", success)

    def move_from_pre_grasp_to_grasp_pose(self):
        pass

    def pick_up(self):
        pass

    def check_grasp_success(self, initial_obj_pos, final_obj_pos) -> bool:
        pass

    def move_ee_to_pose(self, ee_pos, ee_orientation, max_time=5, pos_max_err=0.2,  max_vel=0.05) -> bool:
        """
        :param ee_pos: target position in world coordinates
        :param ee_orientation: target orientation in euler angles
        :param max_time: maximum time in simulation to move there
        :param pos_max_err: maximum acceptable error in each axis to consider the target reached
        :param max_vel: maximum velocity in each joint to consider end of motion. we only check position arrival,
            not orientation we want to make sure the robot is not moving anymore
        """
        self.controller.set_control_input_with_ik(ee_pos, ee_orientation)
        max_iter = int(max_time / self.model.opt.timestep)
        for i in range(max_iter):
            self.step_simulation()
            if self.controller.is_position_reached(ee_pos, pos_max_err, max_vel):
                return True
        return False

    def step_simulation(self):
        step_start = time.time()

        mj.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

        if self.real_time:
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def run_inifinitely(self):
        while True:
            self.step_simulation()



        

