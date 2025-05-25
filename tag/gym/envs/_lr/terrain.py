def _create_heightfield(self):
    """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
    self.terrain = self.scene.add_entity(
        gs.morphs.Terrain(
            pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
            horizontal_scale=self.cfg.terrain.horizontal_scale,
            vertical_scale=self.cfg.terrain.vertical_scale,
            height_field=self.utils_terrain.height_field_raw,
        )
    )
    self.height_samples = (
        torch.tensor(self.utils_terrain.heightsamples)
        .view(self.utils_terrain.tot_rows, self.utils_terrain.tot_cols)
        .to(self.device)
    )


def _get_env_origins(self):
    """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
    Otherwise create a grid.
    """
    if self.cfg.terrain.mesh_type in ["heightfield"]:
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # put robots at the origins defined by the terrain
        max_init_level = self.cfg.terrain.max_init_terrain_level
        if not self.cfg.terrain.curriculum:
            max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            (self.num_envs / self.cfg.terrain.num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.utils_terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
    else:
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
        # restrict envs to a square of [plane_length/2, plane_length/2]
        spacing = min(
            (self.cfg.terrain.plane_length / 2) / (num_rows - 1),
            (self.cfg.terrain.plane_length / 2) / (num_cols - 1),
        )
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0
        self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
        self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4


def _init_height_points(self):
    """Returns points at which the height measurments are sampled (in base frame)

    Returns:
        [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
    """
    y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
    x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(x, y)

    self.num_height_points = grid_x.numel()
    points = torch.zeros(
        self.num_envs,
        self.num_height_points,
        3,
        device=self.device,
        requires_grad=False,
    )
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points


def _get_heights(self, env_ids=None):
    """Samples heights of the terrain at required points around each robot.
        The points are offset by the base's position and rotated by the base's yaw

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    Raises:
        NameError: [description]

    Returns:
        [type]: [description]
    """
    if self.cfg.terrain.mesh_type == "plane":
        return torch.zeros(
            self.num_envs,
            self.num_height_points,
            device=self.device,
            requires_grad=False,
        )
    elif self.cfg.terrain.mesh_type == "none":
        raise NameError("Can't measure height with terrain mesh type 'none'")

    if env_ids:
        points = quat_apply_yaw(
            self.base_quat[env_ids].repeat(1, self.num_height_points),
            self.height_points[env_ids],
        ) + (self.base_pos[env_ids, :3]).unsqueeze(1)
    else:
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
            self.base_pos[:, :3]
        ).unsqueeze(1)

    points += self.cfg.terrain.border_size
    points = (points / self.cfg.terrain.horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
    py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

    heights1 = self.height_samples[px, py]
    heights2 = self.height_samples[px + 1, py]
    heights3 = self.height_samples[px, py + 1]
    heights = torch.min(heights1, heights2)
    heights = torch.min(heights, heights3)

    return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale
