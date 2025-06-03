from dataclasses import dataclass, field
from typing import Any, Tuple

import genesis as gs
import numpy as np

import genesis.utils.misc as mu
from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils

def default(x):
    return field(default_factory=lambda: x)

class TerrainManager:
    def __init__(self, n:int, size:float, z_offset:float, horizontal_scale:float, vertical_scale:float, subterrain_types:list):
        self.n = n
        self.size = size
        self.z_offset = z_offset
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.subterrain_types = subterrain_types

        pos = -1/2 * size * n
        self.pos = (pos, pos, z_offset)
        self.generate_terrain()

    def generate_terrain(self):
        st_types_2d_arr = [[None for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.n):
                random_terrain = self.subterrain_types[np.random.randint(len(self.subterrain_types))] # grab a random subterrain from the list of subterrains

                st_types_2d_arr[i][j] = random_terrain

        self.result = WalledTerrain(
            n_subterrains = (self.n, self.n),
            subterrain_size = (self.size, self.size),
            horizontal_scale = self.horizontal_scale,
            vertical_scale = self.vertical_scale,
            pos = self.pos,
            subterrain_types = st_types_2d_arr
        )
        return self.terrain()

    def add_walls(
        self,
        n:int=1,
        max_n:int=-1,
        size:int=10,
        max_size:int=-1,
        horizontal_only:bool=False,
        vertical_only:bool=False
    ): 
        if (not (horizontal_only and vertical_only)) and (not size < 0) and (not n < 0):
            self.result.randomize_walls(
                n=n,
                max_n=max_n,
                size=size,
                max_size=max_size,
                horizontal_only=horizontal_only,
                vertical_only=vertical_only
            )
    
    def terrain(self):
        return self.result.terrain()

class WalledTerrain:
    def __init__(self,
        randomize: bool = False,  # whether to randomize the terrain
        n_subterrains: Tuple[int, int] = (3, 3),  # number of subterrains in x and y directions
        subterrain_size: Tuple[float, float] = (12.0, 12.0),  # meter
        horizontal_scale: float = 0.25,  # meter size of each cell in the subterrain
        vertical_scale: float = 0.005,  # meter height of each step in the subterrain
        pos: Tuple[float, float, float] = (0, 0, 0),
        subterrain_types: Any = "flat_terrain"
    ):
        self.randomize = randomize
        self.n_subterrains = n_subterrains
        self.subterrain_size = subterrain_size
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.pos = pos
        self.subterrain_types = subterrain_types
        self._post_init()

    def _post_init(self):
        ################## Validate args ##################
        supported_subterrain_types = [
            "flat_terrain",
            "fractal_terrain",
            "random_uniform_terrain",
            "sloped_terrain",
            "pyramid_sloped_terrain",
            "discrete_obstacles_terrain",
            "wave_terrain",
            "stairs_terrain",
            "pyramid_stairs_terrain",
            "stepping_stones_terrain",
        ]

        if isinstance(self.subterrain_types, str):
            subterrain_types = []
            for i in range(self.n_subterrains[0]):
                row = []
                for j in range(self.n_subterrains[1]):
                    row.append(self.subterrain_types)
                subterrain_types.append(row)
            self.subterrain_types = subterrain_types
        else:
            if np.array(self.subterrain_types).shape != (self.n_subterrains[0], self.n_subterrains[1]):
                gs.raise_exception(
                    "`subterrain_types` should be either a string or a 2D list of strings with the same shape as `n_subterrains`."
                )

        for row in self.subterrain_types:
            for subterrain_type in row:
                if subterrain_type not in supported_subterrain_types:
                    gs.raise_exception(
                        f"Unsupported subterrain type: {subterrain_type}, should be one of {supported_subterrain_types}"
                    )

        if not mu.is_approx_multiple(self.subterrain_size[0], self.horizontal_scale) or not mu.is_approx_multiple(
            self.subterrain_size[1], self.horizontal_scale
        ):
            gs.raise_exception("`subterrain_size` should be divisible by `horizontal_scale`.")


        ################## Initialization ##################
        self._init_walls()
        self._make_heightfield()
        self._add_walls()
        self._generate_terrain()

    def _init_walls(self):
        subterrain_rows = int(self.subterrain_size[0] / self.horizontal_scale)
        subterrain_cols = int(self.subterrain_size[1] / self.horizontal_scale)

        self.walls = np.zeros(
            np.array(self.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

    def _make_heightfield(self):
        subterrain_rows = int(self.subterrain_size[0] / self.horizontal_scale)
        subterrain_cols = int(self.subterrain_size[1] / self.horizontal_scale)

        heightfield = np.zeros(
            np.array(self.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

        for i in range(self.n_subterrains[0]):
            for j in range(self.n_subterrains[1]):
                subterrain_type = self.subterrain_types[i][j]

                new_subterrain = isaacgym_terrain_utils.SubTerrain(
                    width=subterrain_rows,
                    length=subterrain_cols,
                    vertical_scale= self.vertical_scale,
                    horizontal_scale= self.horizontal_scale,
                )
                if not self.randomize:
                    saved_state = np.random.get_state()
                    np.random.seed(0)

                if subterrain_type == "flat_terrain":
                    subterrain_height_field = np.zeros((subterrain_rows, subterrain_cols), dtype=np.int16)

                elif subterrain_type == "fractal_terrain":
                    subterrain_height_field = fractal_terrain(new_subterrain, levels=8, scale=5.0).height_field_raw

                elif subterrain_type == "random_uniform_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.random_uniform_terrain(
                        new_subterrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.1,
                        downsampled_scale=0.5,
                    ).height_field_raw

                elif subterrain_type == "sloped_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.sloped_terrain(
                        new_subterrain,
                        slope=-0.5,
                    ).height_field_raw

                elif subterrain_type == "pyramid_sloped_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_sloped_terrain(
                        new_subterrain,
                        slope=-0.1,
                    ).height_field_raw

                elif subterrain_type == "discrete_obstacles_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.discrete_obstacles_terrain(
                        new_subterrain,
                        max_height=0.05,
                        min_size=1.0,
                        max_size=5.0,
                        num_rects=20,
                    ).height_field_raw

                elif subterrain_type == "wave_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.wave_terrain(
                        new_subterrain,
                        num_waves=2.0,
                        amplitude=0.1,
                    ).height_field_raw

                elif subterrain_type == "stairs_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stairs_terrain(
                        new_subterrain,
                        step_width=0.75,
                        step_height=-0.1,
                    ).height_field_raw

                elif subterrain_type == "pyramid_stairs_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_stairs_terrain(
                        new_subterrain,
                        step_width=0.75,
                        step_height=-0.1,
                    ).height_field_raw

                elif subterrain_type == "stepping_stones_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stepping_stones_terrain(
                        new_subterrain,
                        stone_size=1.0,
                        stone_distance=0.25,
                        max_height=0.2,
                        platform_size=0.0,
                    ).height_field_raw

                else:
                    gs.raise_exception(f"Unsupported subterrain type: {subterrain_type}")

                if not self.randomize:
                    np.random.set_state(saved_state)

                heightfield[
                    i * subterrain_rows : (i + 1) * subterrain_rows, j * subterrain_cols : (j + 1) * subterrain_cols
                ] = subterrain_height_field

        self.base_heightfield = heightfield
    
    def _add_walls(self):
        subterrain_rows = int(self.subterrain_size[0] / self.horizontal_scale)
        subterrain_cols = int(self.subterrain_size[1] / self.horizontal_scale)

        heightfield = np.zeros(
            np.array(self.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

        for x in range(len(heightfield)):
            for y in range(len(heightfield[x])):
                heightfield[x,y] = self.base_heightfield[x,y] + self.walls[x,y]

        self.heightfield = heightfield

    def _generate_terrain(self):
        self.result = gs.morphs.Terrain(
            horizontal_scale= self.horizontal_scale,
            vertical_scale= self.vertical_scale,
            pos= self.pos,
            height_field= self.heightfield
        )

    def randomize_walls(self, n, max_n, size, max_size, horizontal_only, vertical_only):
        # TODO: add randomization to ts

        # sample
        for x in range(len(self.walls)):
            self.walls[x,0] = 70

        self._add_walls()
        self._generate_terrain()

    def terrain(self):
        return self.result

    def phf(self): #debug
        for row in self.heightfield:
            print(row)
        quit()



################### SAMPLE CODE ###################
def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer = False,
        viewer_options = gs.options.ViewerOptions(
            res           = (640, 480),
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

    plane = scene.add_entity(
        gs.morphs.Plane()
    )

    n, size, z_off = 2, 5.0, 10
    # terrain = scene.add_entity(
    #     gs.morphs.Terrain(
    #         n_subterrains=(n,n),
    #         subterrain_size=(size, size),
    #         horizontal_scale=0.05,
    #         vertical_scale=0.01,
    #         pos=(-size, -size, z_off),
    #         subterrain_types = [
    #             ["flat_terrain", "random_uniform_terrain"],
    #             ["pyramid_sloped_terrain", "discrete_obstacles_terrain"]
    #         ]
    #     )
    # )

    subterrain_types = ["flat_terrain", "random_uniform_terrain", "pyramid_sloped_terrain", "discrete_obstacles_terrain"]

    test_terrain = TerrainManager(
        n, size, z_off, 0.05, 0.01, subterrain_types
    )

    test_terrain.add_walls()

    terrain = scene.add_entity(
        test_terrain.terrain()
    )

    cam = scene.add_camera(
        res    = (640, 480),
        pos    = (3.5, 0.0, 2.5+z_off),
        lookat = (0, 0, 0.5+z_off),
        fov    = 30,
        GUI    = False,
    )

    scene.build()

    # render rgb, depth, segmentation, and normal
    # rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

    cam.start_recording()
    import numpy as np

    for i in range(400):
        scene.step()
        cam.set_pose(
            pos    = (10.0 * np.sin(i / 100), 10.0 * np.cos(i / 100), 2+z_off),
            lookat = (0, 0, 0.5+z_off),
        )
        cam.render()
    cam.stop_recording(save_to_filename='video_tt.mp4', fps=60)

if __name__ == "__main__":
    main()