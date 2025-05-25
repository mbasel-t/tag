import torch


def check_base_pos_out_of_bound(self):
    """Check if the base position is out of the terrain bounds"""
    x_out_of_bound = (self.base_pos[:, 0] >= self.terrain_x_range[1]) | (self.base_pos[:, 0] <= self.terrain_x_range[0])
    y_out_of_bound = (self.base_pos[:, 1] >= self.terrain_y_range[1]) | (self.base_pos[:, 1] <= self.terrain_y_range[0])
    out_of_bound_buf = x_out_of_bound | y_out_of_bound
    envs_idx = out_of_bound_buf.nonzero(as_tuple=False).flatten()
    # reset base position to initial position
    self.base_pos[envs_idx] = self.base_init_pos
    self.base_pos[envs_idx] += self.env_origins[envs_idx]
    self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)


def check_termination(self):
    """Check if environments need to be reset"""
    self.reset_buf = torch.any(
        torch.norm(self.link_contact_forces[:, self.termination_indices, :], dim=-1) > 1.0,
        dim=1,
    )
    self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
    self.reset_buf |= self.time_out_buf
