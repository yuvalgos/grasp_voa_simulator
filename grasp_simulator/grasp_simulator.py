import time
from math import pi
import numpy as np
import scipy
from grasp_simulator.utils import set_camera_overview
from grasp_simulator.manipulated_object import ManipulatedObject
from grasp_simulator.ur_controller import UrController
from grasp_simulator.Sensor import Sensor
import mujoco as mj
import mujoco.viewer as mj_viewer


"""
Grasping simulator
main useable method is try_grasp(position, orientation). 
This method will try to grasp the object at the given position and orientation and pick it up.
returns True if successful, a.k.a. the object height changed. 

grasp procidure:
1. move to pre-grasp pose which is the grasp pose with a 15 cm backwards offset from the object
2. move to grasp pose
3. close gripper
4. pick up and wait for a second
5. check if object height is different from initial height by enough to consider the grasp successful

world coordinates in mujoco is not reflected to the user, instead, for simplicity, the center of the object's
table is considered the origin of the world (for controlling the robot arm and the object initial position).
"""

TABLE_ORIGIN = np.array([0, -0.65, 1.115])
def table2world(position):
    return position + TABLE_ORIGIN


OBJECT_START_POSITION = table2world(np.array([0, 0, 0.25]))  # just a default
OBJECT_START_ORIENTATION = [0, 0, 0]  # just a default orientation

PRE_GRASP_DISTANCE = 0.15

PICKUP_POINT = table2world(np.array([0, 0, 0.4]))
PICKUP_ORIENTATION = [0, pi/4, 0]

MIN_HEIGHT_DIFF_FOR_SUCCESS = 0.10

class GraspSimulator:
    def __init__(self, launch_viewer=True, real_time=False):
        """
        :param launch_viewer: whether to launch the mujoco viewer
        :param real_time: whether to run the simulation in real time for visualisation or as fast as possible
        """
        self.real_time = real_time

        self.model = mj.MjModel.from_xml_path("./data/world_sprayflask.xml")
        self.data = mj.MjData(self.model)

        self.m_obj = ManipulatedObject(self.model, self.data)
        self.controller = UrController(self.model, self.data)
        self.sensor = Sensor(self.model, self.data)

        mj.mj_forward(self.model, self.data)

        self.viewer = None
        if launch_viewer:
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            set_camera_overview(self.viewer)

        # just put the object in a nice position for starting
        self.m_obj.set_position(OBJECT_START_POSITION)
        self.m_obj.set_orientation_euler(OBJECT_START_ORIENTATION)

        self.verbose = 0

        self.reset()

    def reset(self, object_orientation=None, object_position_offset=None):
        if object_orientation is None:
            object_orientation = OBJECT_START_ORIENTATION
        position = OBJECT_START_POSITION if object_position_offset is None else table2world(object_position_offset)

        mj.mj_resetData(self.model, self.data)
        self.m_obj.set_position(position)
        self.m_obj.set_orientation_euler(object_orientation)
        self.controller.reset()

    def try_grasp(self, ee_pos_table, ee_orientation) -> bool:
        ''' try a grasp with the current object pose and a given grasp parameters, return whether successful'''
        ee_pos = table2world(ee_pos_table)

        start_time = time.time()
        initial_obj_pos = self.m_obj.get_position().copy()

        # TODO: handle failure in each of these steps
        self.move_ee_to_pre_grasp(ee_pos, ee_orientation)
        self.move_from_pre_grasp_to_grasp_pose(ee_pos, ee_orientation)
        self.close_gripper()
        self.pick_up()
        self.simulate_seconds(1)
        
        final_obj_pos = self.m_obj.get_position()

        if self.verbose>0:
            print("initial_obj_pos: ", initial_obj_pos)
            print("final_obj_pos: ", final_obj_pos)
            print("total simulation time: ", self.data.time)
            print("wall clock time: ", time.time()-start_time)
        
        return self.check_grasp_success(initial_obj_pos, final_obj_pos)

    def move_ee_to_pre_grasp(self, ee_pos, ee_orientation):
        # pre grasp is 15cm back from the object in end effector coordinates
        # get orientation rotation matrix:
        R = scipy.spatial.transform.Rotation.from_euler('xyz', ee_orientation).as_matrix()
        # get pre grasp position in world coordinates:
        pre_grasp_pos = ee_pos - R @ np.array([PRE_GRASP_DISTANCE, 0, 0])

        success = self.move_ee_to_pose(pre_grasp_pos, ee_orientation, max_time=3)
        if self.verbose > 0:
            print("move_ee_to_pre_grasp success: ", success)

    def move_from_pre_grasp_to_grasp_pose(self, ee_pos, ee_orientation):
        success = self.move_ee_to_pose(ee_pos, ee_orientation, max_time=1.5)
        if self.verbose > 0:
            print("move from pre grasp to grasp success: ", success)

    def pick_up(self):
        success = self.move_ee_to_pose(PICKUP_POINT, PICKUP_ORIENTATION, max_time=2)
        if self.verbose > 0:
            print("pick up movement success: ", success, "(arm, not necessarily object!)")

    def check_grasp_success(self, initial_obj_pos, final_obj_pos) -> bool:
        return final_obj_pos[2] - initial_obj_pos[2] > MIN_HEIGHT_DIFF_FOR_SUCCESS

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

    def close_gripper(self, max_time=0.5):
        ''' close gripper but also simulate wait for it to close '''
        self.controller.close_gripper()
        max_iter = int(max_time / self.model.opt.timestep)
        for i in range(max_iter):
            self.step_simulation()

    def step_simulation(self):
        step_start = time.time()

        mj.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

        if self.real_time:
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def simulate_seconds(self, seconds):
        max_iter = int(seconds / self.model.opt.timestep)
        for i in range(max_iter):
            self.step_simulation()

    def run_infinitely(self):
        while True:
            self.step_simulation()

