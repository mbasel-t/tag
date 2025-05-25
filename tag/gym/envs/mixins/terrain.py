class TerrainMixin:
    def validate(self):
        # TODO this should go somewhere else ? like terrain config
        if self.cfg.terrain.mesh_type not in ["heightfield"]:
            self.cfg.terrain.curriculum = False

    def _init_terrain(self):
        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == "plane":
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type == "heightfield":
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type == "heightfield":
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0  # give a small margin(1.0m)
            self.terrain_x_range[1] = (
                self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            )
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = (
                self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
            )
        elif self.cfg.terrain.mesh_type == "plane":  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length / 2 + 1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length / 2 - 1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length / 2 + 1  # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length / 2 - 1

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()
        height_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points)
        height_points[0, :, 0] += self.base_pos[0, 0]
        height_points[0, :, 1] += self.base_pos[0, 1]
        height_points[0, :, 2] = self.measured_heights[0, :]
        # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        self.scene.draw_debug_spheres(
            height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7)
        )  # only draw for the first env

    def _update_terrain_curriculum(self, env_ids):
        if not self.cfg.terrain.curriculum:
            return
