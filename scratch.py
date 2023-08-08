import mujoco as mj
import mujoco.viewer as mj_viewer
import time
from math import pi
from utils import set_camera_overview, ManipulatedObject, UrController


model = mj.MjModel.from_xml_path("./data/world_mug.xml")
data = mj.MjData(model)

# mj_viewer.launch(model, data) # blocks user code but opens interactive viewer
# mujoco.mj_resetData(model, data)
# mujoco.mj_forward(model, data)

mobj = ManipulatedObject(model, data)
mobj.set_position([0,-0.6,0.95])
mobj.set_orientation_euler([0, 0, 1.5])

controller = UrController(model, data)

with mj.viewer.launch_passive(model, data) as viewer:
    set_camera_overview(viewer)

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    # controller.set_robot_control_input([-1.5 ,-0.3, 0, 0, 0, -1])
    controller.set_control_input_with_ik([0.0, -0.6, 0.95], [pi, 0, -pi])

    while viewer.is_running() and time.time() - start < 500:
        step_start = time.time()

        # if (time.time() - start)> 1:
            # data.qpos[:] = [-pi / 2, ] * 6 + [0.0, ] * 2
            # move the joints with motor actuation


            #TODO set maximum torque/velocity that is lower

        # if (time.time() - start)> 17:
        #     controller.close_gripper()
        # if (time.time() - start)> 26:
        #     controller.open_gripper()


        mj.mj_step(model, data)
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)