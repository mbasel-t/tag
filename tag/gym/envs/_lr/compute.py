def compute_observations(self):
    """Computes observations"""
    self.obs_buf = torch.cat(
        (
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity,  # 3
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # num_dofs
            self.dof_vel * self.obs_scales.dof_vel,  # num_dofs
            self.actions,  # num_actions
        ),
        dim=-1,
    )

    self.maybe_compute_heights()

    # add noise if needed
    self.obs_buf = self.noise.inject(self.obs_buf)


def maybe_compute_heights(self):
    # add perceptive inputs if not blind
    if self.cfg.terrain.measure_heights:
        heights = (
            torch.clip(
                self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)


def _compute_torques(self, actions):
    # control_type = 'P'
    actions_scaled = actions * self.cfg.control.action_scale
    torques = (
        self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
        - self.batched_d_gains * self.dof_vel
    )
    return torques


def _compute_target_dof_pos(self, actions):
    # control_type = 'P'
    actions_scaled = actions * self.cfg.control.action_scale
    target_dof_pos = actions_scaled + self.default_dof_pos

    return target_dof_pos
