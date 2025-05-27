from dataclasses import asdict, dataclass

import genesis as gs
import torch

from tag.utils import defaultcls


@dataclass
class RewardScales:
    """limitation"""

    dof_pos_limits: float = -10.0  # TODO add support
    # collision: float = -1.0 # TODO add support
    termination: float = -1  # early termination penalty

    """ command tracking """
    tracking_lin_vel: float = 5.0
    tracking_ang_vel: float = 0.5

    """ smooth """
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.0
    # torques: float = -2.0e-4 # TODO add support
    dof_vel: float = -5.0e-4
    dof_acc: float = -2.0e-7
    base_height: float = -1.0
    action_rate: float = -0.01

    """ gait """
    feet_air_time: float = 3.0  # Encourage long swing steps. However, not high clearances.
    feet_stumble = -0.0
    feet_slip = -0.1  # Penalizing foot slipping on the ground.
    stand_still = -0.0
    similar_to_default: float = -0.05


@dataclass
class RewardConfig:
    pass


@dataclass
class DenseReward(RewardConfig):
    only_positive_rewards: bool = True  # True  # clip 0,inf

    soft_dof_pos_limit: float = 0.9
    soft_dof_vel_limit: float = 1.0
    soft_torque_limit: float = 1.0

    base_height_target: float = 0.36
    over_height_ok: bool = True  # if True, base height can be higher than target, no penalty

    # feet_height_target: float = 0.075
    tracking_sigma: float = 0.25  # tracking reward = exp(-error^2/sigma)

    termination_if_roll_greater_than: float = 0.8
    termination_if_pitch_greater_than: float = 0.8
    termination_if_height_lower_than: float = 0.2

    scales: RewardScales = defaultcls(RewardScales)


@dataclass
class WalkReward(DenseReward):
    pass


@dataclass
class SparseReward(RewardConfig):
    pass


def _float(shape):
    return torch.zeros(shape, device=gs.device, dtype=gs.tc_float)


class RewardMixin:
    def _init_reward(self):
        assert getattr(self.cfg, "rewards", None) is not None, "Reward config not set"
        assert isinstance(self.cfg.rewards, RewardConfig), "Reward config not set *properly*"

        scales = asdict(self.cfg.rewards.scales)
        # remove zero scales + multiply non-zero ones by self.dt
        self.reward_scales = {k: scale * self.dt for k, scale in scales.items() if scale != 0}
        self.reward_functions = {k: getattr(self, f"_reward_{k}") for k in self.reward_scales.keys()}
        self.episode_sums = {k: _float((self.n_envs,)) for k in self.reward_scales.keys()}

        self.rew_buf = _float((self.n_envs,))

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]

            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    #
    # ------------ reward functions----------------
    #

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
        tgt = self.cfg.rewards.base_height_target
        if getattr(self, "measured_heights", None) is not None:
            base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        else:  # no terrain
            base_height = self.base_pos[:, 2]

        diff = tgt - base_height  # if base is too high, diff<0, maybe clip
        if self.cfg.rewards.over_height_ok:
            diff = torch.clip(diff, min=0.0)
        rew = torch.square(diff)
        return rew

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
        # episode ended but not truncated
        return self.reset_buf * ~self.checks["truncate"]

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

    def _reward_feet_slip(self, pipeline_state) -> torch.Tensor:
        """from brax barkour
        Penalize large feet velocity for feet that are in contact with the ground.
        """

        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)

        # foot velocity in world frame, not in local frame

        # get velocities at feet which are offset from lower legs
        # pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        # feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # offset = base.Transform.create(pos=feet_offset)
        # foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        # foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        return torch.sum(torch.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))


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
