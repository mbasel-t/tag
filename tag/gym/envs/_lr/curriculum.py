def _push_robots(self):
    """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
    if self.push_interval_s > 0 and not self.debug:
        max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        # in Genesis, base link also has DOF, it's 6DOF if not fixed.
        dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
        push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
        dofs_vel[:, :2] += push_vel
        self.robot.set_dofs_velocity(dofs_vel)


def _update_terrain_curriculum(self, env_ids):
    """Implements the game-inspired curriculum.

    Args:
        env_ids (List[int]): ids of environments being reset
    """
    # Implement Terrain curriculum
    if not self.init_done:
        # don't change on initial reset
        return
    distance = torch.norm(self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terains
    move_up = distance > self.utils_terrain.env_length / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    # Robots that solve the last level are sent to a random one
    self.terrain_levels[env_ids] = torch.where(
        self.terrain_levels[env_ids] >= self.max_terrain_level,
        torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
        torch.clip(self.terrain_levels[env_ids], 0),
    )  # (the minumum level is zero)
    self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


def update_command_curriculum(self, env_ids):
    """Implements a curriculum of increasing commands

    Args:
        env_ids (List[int]): ids of environments being reset
    """
    # If the tracking reward is above 80% of the maximum, increase the range of commands
    if (
        torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
        > 0.8 * self.reward_scales["tracking_lin_vel"]
    ):
        self.command_ranges["lin_vel_x"][0] = np.clip(
            self.command_ranges["lin_vel_x"][0] - 0.5,
            -self.cfg.commands.max_curriculum,
            0.0,
        )
        self.command_ranges["lin_vel_x"][1] = np.clip(
            self.command_ranges["lin_vel_x"][1] + 0.5,
            0.0,
            self.cfg.commands.max_curriculum,
        )
