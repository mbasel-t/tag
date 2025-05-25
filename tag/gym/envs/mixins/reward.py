from dataclasses import asdict, dataclass

import genesis as gs
import torch

from tag.gym.base.config import defaultcls


@dataclass
class RewardScales:
    # limitation
    dof_pos_limits: float = -10.0
    collision: float = -1.0

    # command tracking
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5

    # smooth
    lin_vel_z: float = -2.0
    base_height: float = -1.0
    ang_vel_xy: float = -0.05
    orientation: float = -1.0
    dof_vel: float = -5.0e-4
    dof_acc: float = -2.0e-7
    action_rate: float = -0.01
    torques: float = -2.0e-4

    # gait
    feet_air_time: float = 1.0
    # similar_to_default: float = -0.05


@dataclass
class RewardConfig:
    pass


@dataclass
class DenseReward(RewardConfig):
    soft_dof_pos_limit: float = 0.9
    base_height_target: float = 0.36
    # feet_height_target: float = 0.075
    tracking_sigma: float = 0.25

    scales: RewardScales = defaultcls(RewardScales)


@dataclass
class WalkReward(DenseReward):
    pass


@dataclass
class SparseReward(RewardConfig):
    pass


class RewardMixin:
    def _float(shape):
        return torch.zeros(shape, device=gs.device, dtype=gs.tc_float)

    def _init_reward(self):
        assert getattr(self, "R", None) is not None, "Reward config not set"
        assert isinstance(self.R, RewardConfig), "Reward config not set *properly*"

        scales = asdict(self.R.scale)
        # remove zero scales + multiply non-zero ones by self.dt
        self.reward_scales = {k: scale * self.dt for k, scale in scales.items() if scale != 0}
        self.reward_functions = {k: getattr(self, f"_reward_{k}") for k in self.reward_scales.keys()}
        self.episode_sums = {k: _float((self.n_envs,)) for k in self.reward_scales.keys()}

        self.rew_buf = _float((self.n_envs,))

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        if getattr(self, "measured_heights", None) is not None:
            base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
            rew = torch.square(base_height - self.cfg.rewards.base_height_target)
            return rew

        # no terrain
        return torch.square(self.base_pos[:, 2] - self.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0 * (torch.norm(self.link_contact_forces[:, self.penalized_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)


def _prepare_reward_function(self):
    """Prepares a list of reward functions, whcih will be called to compute the total reward.
    Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
    """

    # remove zero scales + multiply non-zero ones by dt
    for key in list(self.reward_scales.keys()):
        scale = self.reward_scales[key]
        if scale == 0:
            self.reward_scales.pop(key)
        else:
            self.reward_scales[key] *= self.dt

    # prepare list of functions
    self.reward_functions = []
    self.reward_names = []
    for name, scale in self.reward_scales.items():
        if name == "termination":
            continue
        self.reward_names.append(name)
        name = "_reward_" + name
        self.reward_functions.append(getattr(self, name))

    # reward episode sums
    self.episode_sums = {
        name: torch.zeros(
            self.num_envs,
            dtype=gs.tc_float,
            device=self.device,
            requires_grad=False,
        )
        for name in self.reward_scales.keys()
    }


"""
reward_functions = {
    "_reward_lin_vel_z": _reward_lin_vel_z,
    "_reward_ang_vel_xy": _reward_ang_vel_xy,
    "_reward_orientation": _reward_orientation,
    "_reward_base_height": _reward_base_height,
    "_reward_torques": _reward_torques,
    "_reward_dof_vel": _reward_dof_vel,
    "_reward_dof_acc": _reward_dof_acc,
    "_reward_action_rate": _reward_action_rate,
    "_reward_collision": _reward_collision,
    "_reward_termination": _reward_termination,
    # limits
    "_reward_dof_pos_limits": _reward_dof_pos_limits,
    # '_reward_dof_vel_limits': _reward_dof_vel_limits,
    "_reward_torque_limits": _reward_torque_limits,
    # command tracking
    "_reward_tracking_lin_vel": _reward_tracking_lin_vel,
    "_reward_tracking_ang_vel": _reward_tracking_ang_vel,
    # gait
    "_reward_feet_air_time": _reward_feet_air_time,
    "_reward_stand_still": _reward_stand_still,
    "_reward_dof_close_to_default": _reward_dof_close_to_default,
}
"""
