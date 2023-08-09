import ikpy.chain
import mujoco as mj
import numpy as np
import scipy


INITIAL_JOINT_POSITION = [-1.5 , -1.5, 0.5, -1.5, -1.5, 0]
END_EFFECTOR_TRANSLATION = np.array([.12, .0, 0])  # for some reason x is depth. maybe I don't know robotics conventions


class UrController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.robot_control_input = data.ctrl[0:6]
        self.gripper_control_input = data.ctrl[6]

        # set starting position manually (not by control to be at a better position for starting (avoid collisions):
        self.shoulder_jntadr = model.body('shoulder_link').jntadr[0]
        self.data.qpos[self.shoulder_jntadr:self.shoulder_jntadr + 6] = INITIAL_JOINT_POSITION
        # set control to same value as well:
        self.set_robot_control_input(INITIAL_JOINT_POSITION)

        self.chain_ik = ikpy.chain.Chain.from_urdf_file("./data/ur5_gripper.urdf",
                                                        active_links_mask=[False,] + [True]*6 + [False, ]*2,
                                                        # first is base, 2 last are gripper and EE /\
                                                        last_link_vector=END_EFFECTOR_TRANSLATION)

        mj.mj_forward(model, data)  # make sure that data is up-to-date for next line
        self.base_link_pos = self.data.body('base_link').xpos

        self.ee_data = self.data.body('ee_link')

        self.open_gripper()

    def set_robot_control_input(self, control_input):
        assert len(control_input) == 6
        self.data.ctrl[0:6] = control_input

    def set_control_input_with_ik(self, position, orientation3d):
        position = self.world2robot_coord(position.copy())
        # get transformation matrix from orientation3d and add position
        orientation_matrix = scipy.spatial.transform.Rotation.from_euler('xyz', orientation3d).as_matrix()
        transformation_matrix = np.zeros((4,4))
        transformation_matrix[0:3, 0:3] = orientation_matrix
        transformation_matrix[0:3, 3] = position
        transformation_matrix[3, 3] = 1

        joint_position = self.chain_ik.inverse_kinematics_frame(transformation_matrix,
                                                                orientation_mode='all')

        # TODO: add forwoard kinematics computation to validate success
        # TODO: some simple collision avoidance with the table? maybe just change the table distance and height

        self.set_robot_control_input(joint_position[1:7])

    def close_gripper(self):
        self.data.ctrl[6] = -1.0

    def open_gripper(self):
        self.data.ctrl[6] = 0.4

    def world2robot_coord(self, position):
        return position - self.base_link_pos

    def robot2world_coord(self, position):
        return position + self.base_link_pos

    def get_ee_pose(self):
        position = self.ee_data.xpos.copy() - END_EFFECTOR_TRANSLATION
        orientation = scipy.spatial.transform.Rotation.from_quat(self.ee_data.xquat.copy()).as_euler('xyz')

        raise NotImplementedError("don't use get ee pose as there is a problem with the orientation matching from the urdf"
                                  "usedfor ik and the one used for the mujoco model. couldn't fix that yet.")

        return position, orientation

    def get_ee_position(self):
        return self.ee_data.xpos.copy() + END_EFFECTOR_TRANSLATION

    def is_position_reached(self, ee_position, pos_max_err, max_vel):
        '''
        generally, we would like to check that we are close enough in position and in orientation,
        but there is a problem that need to be fixed with oreintation: there is mismatch between inverse
        kinematics and mujoco model. so for now we only check position and that the robot doesn't move.
        '''

        close_enough = np.all(np.abs(self.get_ee_position() - ee_position) < pos_max_err)
        joint_vel = self.data.qvel[self.shoulder_jntadr:self.shoulder_jntadr + 6]
        not_moving = np.all(joint_vel < max_vel)
        return close_enough and not_moving

