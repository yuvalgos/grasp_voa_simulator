from mujoco import viewer as mj_viewer
import matplotlib.pyplot as plt


def plot_lidar_im_depth_tri(lidar, im, depth_im, depth_threshold=0.5, name=None):
    depth_im[depth_im > depth_threshold] = depth_threshold
    lidar[lidar > depth_threshold] = depth_threshold

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(lidar)
    axs[0].set_title("lidar")
    axs[1].imshow(im)
    axs[1].set_title("image")
    axs[2].imshow(depth_im)
    axs[2].set_title("depth image")
    if name is not None:
        fig.suptitle(name)
    plt.tight_layout()
    plt.show()

def plot_depth_image(im, threshold=0.5, name=None):
    im[im > threshold] = threshold
    plt.imshow(im)
    if name is not None:
        plt.title(name)
    plt.colorbar()
    plt.show()

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


