def compute_reward(self):
    """Compute rewards
    Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
    adds each terms to the episode sums and to the total reward
    """
    self.rew_buf[:] = 0.0
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]
        self.rew_buf += rew
        self.episode_sums[name] += rew
    if self.cfg.rewards.only_positive_rewards:
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
    # add termination reward after clipping
    if "termination" in self.reward_scales:
        rew = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew
        self.episode_sums["termination"] += rew


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

    # add noise if needed
    if self.add_noise:
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    if self.num_privileged_obs is not None:
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.last_actions,
            ),
            dim=-1,
        )


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
