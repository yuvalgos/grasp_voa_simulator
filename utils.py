import mujoco as mj
import scipy
import ikpy.chain
from mujoco import viewer as mj_viewer
from pyquaternion import Quaternion
import numpy as np


def set_camera_overview(viewer: mj_viewer.Handle):
    viewer.cam.azimuth = 120.0
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -45.0
    viewer.cam.fixedcamid = -1
    viewer.cam.lookat[:] = [0.0, -0.3, 0.7]
    viewer.cam.trackbodyid = -1
    viewer.cam.type = 0


def plot_ik_res(joint_state, target, chain_ik):
    import matplotlib.pyplot as plt
    import ikpy.utils.plot as plot_utils

    fig, ax = plot_utils.init_3d_figure()
    chain_ik.plot(joint_state, ax, target=target)
    plt.show()


INITIAL_JOINT_POSITION = [-1.5 ,-1.5, 0.5, -1.5, -1.5, 0]
END_EFFECTOR_TRANSLATION = [.12, .0, 0]  # for some reason x is depth. maybe I don't know robotics conventions


class UrController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.robot_control_input = data.ctrl[0:6]
        self.gripper_control_input = data.ctrl[6]

        # set starting position manually (not by control to be at a better position for starting (avoid collisions):
        shoulder_jntadr = model.body('shoulder_link').jntadr[0]
        self.data.qpos[shoulder_jntadr:shoulder_jntadr+6] = INITIAL_JOINT_POSITION

        self.chain_ik = ikpy.chain.Chain.from_urdf_file("./data/ur5_gripper.urdf",
                                                        active_links_mask=[False,] + [True]*6 + [False, ]*2,
                                                        # first is base, 2 last are gripper and EE /\
                                                        last_link_vector=END_EFFECTOR_TRANSLATION)

        self.open_gripper()

    def set_robot_control_input(self, control_input):
        assert len(control_input) == 6
        self.data.ctrl[0:6] = control_input

    def set_control_input_with_ik(self, position, orientation3d): #
        # get transformation matrix from orientation3d and add position
        orientation_matrix = scipy.spatial.transform.Rotation.from_euler('xyz', orientation3d).as_matrix()
        transformation_matrix = np.zeros((4,4))
        transformation_matrix[0:3, 0:3] = orientation_matrix
        transformation_matrix[0:3, 3] = position
        transformation_matrix[3, 3] = 1

        joint_position = self.chain_ik.inverse_kinematics_frame(transformation_matrix,
                                                                orientation_mode='all')

        self.set_robot_control_input(joint_position[1:7])

    def close_gripper(self):
        self.data.ctrl[6] = -1.0

    def open_gripper(self):
        self.data.ctrl[6] = 0.2


class ManipulatedObject:
    '''
     represent, query and manually manipulate the manipulated object
     assuming it's body name is 'manipulated_object'
    '''
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.jntadr = model.body('manipulated_object').jntadr[0]

    def set_pose(self, pose):
        assert len(pose) == 7
        self.data.qpos[self.jntadr:self.jntadr + 7] = pose

    def set_position(self, position):
        assert len(position) == 3
        self.data.qpos[self.jntadr:self.jntadr + 3] = position

    def zero_velocities(self):
        self.data.qvel[self.jntadr:self.jntadr + 7] = [0.0, ] * 7

    def set_orientation_quat(self, orientation):
        assert len(orientation) == 4
        self.data.qpos[self.jntadr + 3:self.jntadr + 7] = orientation

    def set_orientation_euler(self, orientation):
        assert len(orientation) == 3
         # use scipy to convert euler to quat
        orientation_quat = scipy.spatial.transform.Rotation.from_euler('xyz', orientation).as_quat()
        self.data.qpos[self.jntadr + 3:self.jntadr + 7] = orientation_quat

    def get_orientation_euler(self):
        return scipy.spatial.transform.Rotation.from_quat(self.data.qpos[self.jntadr + 3:self.jntadr + 7]).as_euler('xyz')

    def get_orientation_quat(self):
        return self.data.qpos[self.jntadr + 3:self.jntadr + 7]

    def get_position(self):
        return self.data.qpos[self.jntadr:self.jntadr + 3]

    def get_pose(self):
        return self.data.qpos[self.jntadr:self.jntadr + 7]
