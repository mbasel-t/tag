import multiprocessing


def main(_gpu_id):

    import genesis as gs
    import numpy as np
    from TerrainTest import TerrainManager
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## CAMERA SETTING ##########################
    #            (set true if recording, false otherwise)

    recording = False


    ########################## init ##########################
    # get current gpu
    torch.cuda.set_device(_gpu_id)
    gpu_id = torch.cuda.current_device()
    print("gpu_id:", gpu_id, "/", torch.cuda.device_count())
    # quit()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
    gs.init(backend=gs.gpu, logger_verbose_time=True)
    # print(str(gs.device))
    # quit()

    ########################## create a scene ##########################
    scene = gs.Scene(
        show_viewer = False,
        viewer_options = gs.options.ViewerOptions(
            res           = (320, 240),
            camera_pos    = (3.5, 0.0, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 40,
            max_FPS       = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True,
            world_frame_size = 1.0,
            show_link_frame  = False,
            show_cameras     = False,
            plane_reflection = True,
            ambient_light    = (0.1, 0.1, 0.1),
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    ########################## plane & terrain ##########################
    # plane = scene.add_entity(
    #     gs.morphs.Plane(),
    # )
    
    n, size, z_off = 2, 5.0, 10
    subterrain_types = ["flat_terrain", "random_uniform_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain"]

    test_terrain = TerrainManager(
        n, size, z_off, 0.05, 0.01, subterrain_types
    )

    terrain = scene.add_entity(
        test_terrain.terrain()
    )

    terrain = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0.5, 0.0, z_off+0.5)
        )
    )

    terrain = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(-0.5, 0.0, z_off+0.5)
        )
    )

    ########################## put the camera ##########################

    if recording:
        cam = scene.add_camera(
            res    = (640, 480),
            pos    = (3.5, 0.0, 2.5+z_off),
            lookat = (0, 0, 0.5+z_off),
            fov    = 30,
            GUI    = False,
        )

    ########################## build ##########################
    B = 22000 # current max
    scene.build(n_envs=B, env_spacing=(4.0, 4.0))

    if recording: cam.start_recording()

    for i in range(100):
        scene.step()
        if recording:
            cam.set_pose(
                pos    = (10.0 * np.sin(i / 100), 10.0 * np.cos(i / 100), 2+z_off),
                lookat = (0, 0, 0.5+z_off),
            )
            cam.render()

    if recording: cam.stop_recording(save_to_filename='video_multiGPU_' + str(gpu_id) + '.mp4', fps=60)


def run(gpu_id, func):
    # Set environment args
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TI_VISIBLE_DEVICE"] = str(gpu_id)
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    # main script
    func(0)

## chat code ##
def spawn_processes():
    num_gpus = 4
    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=run, args=(i, main))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    spawn_processes()