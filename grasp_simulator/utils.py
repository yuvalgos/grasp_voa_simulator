from mujoco import viewer as mj_viewer


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


