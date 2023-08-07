import mujoco as mj
import mujoco.viewer as mj_viewer
import time
from math import pi
from utils import set_camera_overview, ManipulatedObject


model = mj.MjModel.from_xml_path("./data/world_mug.xml")
data = mj.MjData(model)

# mj_viewer.launch(model, data) # blocks user code but opens interactive viewer
# mujoco.mj_resetData(model, data)
# mujoco.mj_forward(model, data)

mobj = ManipulatedObject(model, data)
mobj.set_position([0,-0.6,1])

with mj.viewer.launch_passive(model, data) as viewer:
    set_camera_overview(viewer)

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 500:
        step_start = time.time()

        if (time.time() - start)>3:
            # data.qpos[:] = [-pi / 2, ] * 6 + [0.0, ] * 2
            # move the joints with motor actuation
            data.ctrl[:] = [-pi / 2, ] * 6 + [0.0, ]
            #TODO finish the xml's first and edit the OAI cup's xml
            #TODO: use ikpy here
            #TODO set maximum torque/velocity that is lower
        print("data.qpos", data.qpos)

        mj.mj_step(model, data)
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)