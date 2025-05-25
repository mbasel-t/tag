import genesis as gs
import torch


class ObservationNoise:
    """Handler for observation noise."""

    def __init__(self, cfg, obs_scales, obs_dim, device, measure_heights=False):
        self._cfg = cfg
        self.add_noise = cfg.add_noise
        ns = cfg.noise_scales
        nl = cfg.noise_level
        self._vec = torch.zeros(obs_dim, device=device, dtype=gs.tc_float)
        self._vec[:3] = ns.lin_vel * nl * obs_scales.lin_vel
        self._vec[3:6] = ns.ang_vel * nl * obs_scales.ang_vel
        self._vec[6:9] = ns.gravity * nl
        self._vec[9:12] = 0.0
        self._vec[12:24] = ns.dof_pos * nl * obs_scales.dof_pos
        self._vec[24:36] = ns.dof_vel * nl * obs_scales.dof_vel
        self._vec[36:48] = 0.0
        if measure_heights:
            self._vec[48:235] = ns.height_measurements * nl * obs_scales.height_measurements

    @property
    def scale_vec(self):
        return self._vec

    def inject(self, obs: torch.Tensor) -> torch.Tensor:
        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self._vec
        return obs
