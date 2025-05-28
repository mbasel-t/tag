from dataclasses import dataclass
import math

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from rich.pretty import pprint
import torch

from tag.gym.envs.mixins.reward import RewardMixin, WalkReward
from tag.gym.envs.robotic import Go2EnvConfig, RobotEnv
from tag.protocols import Wraps, _Env
from tag.utils import default, defaultcls


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


@dataclass
class CommandConfig:
    num_commands: int = 3
    lin_vel_x_range: list[float] = default([-0.5, 1.0])
    lin_vel_y_range: list[float] = default([-0.1, 0.1])
    ang_vel_range: list[float] = default([-0.2, 0.2])


DEFAULT = CommandConfig()
OVERFIT = CommandConfig(
    num_commands=3,
    lin_vel_x_range=[0.05, 0.05],
    lin_vel_y_range=[0.0, 0.0],
    ang_vel_range=[0.0, 0.0],
)


@dataclass
class WalkEnvConfig(Go2EnvConfig):
    command: CommandConfig = default(OVERFIT)
    rewards: WalkReward = defaultcls(WalkReward)
    auto_reset: bool = True


class RSLWrapper(_Env, Wraps):
    def __init__(self, env: _Env):
        super().__init__(env)
        self.env = env
        self.num_envs = self.cfg.sim.num_envs  # for rsl rl # TODO make some env conversion

        self.num_obs = self.observe().shape
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None


class SpaceClipWrapper(_Env, Wraps):
    def __init__(self, env: _Env, clip_actions: float):
        super().__init__(env)
        self.env = env
        self.clip_actions = clip_actions

    def step(self, actions):
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        return self.env.step(actions)


class Walk(RobotEnv, RewardMixin):
    def __init__(self, cfg: WalkEnvConfig, env_cfg, obs_cfg):
        super().__init__(cfg)

        if self.cfg.cam.follow:
            self.cam_follow(self.robot.robot)

        # RSL RL
        self.num_envs = self.cfg.sim.num_envs
        self.num_obs = 48
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = self.cfg.sim.dt
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg

        self.obs_scales = obs_cfg["obs_scales"]

        # add robot
        self.base_init_pos = torch.tensor(self.cfg.robot.state.pos, device=gs.device)
        self.base_init_quat = torch.tensor(self.cfg.robot.state.quat, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.feet_indices = self.robot.feet

        self._init_reward()
        self._init_buffers()

    def build(self):
        super().build()
        self._init_robot_param()

    def _resample_commands(self, envs_idx):
        # low,high = torch.Tensor(
        # [
        # self.cfg.command.lin_vel_x_range,
        # self.cfg.command.lin_vel_y_range,
        # self.cfg.command.ang_vel_range,
        # ],
        # device=gs.device,
        # dtype=gs.tc_float,
        # ).T
        # cmd = (high-low)*torch.rand(size=(len(envs_idx), 3), device=gs.device)+ low
        # return cmd

        self.commands[envs_idx, 0] = gs_rand_float(*self.cfg.command.lin_vel_x_range, (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.cfg.command.lin_vel_y_range, (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.cfg.command.ang_vel_range, (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.robot.control_dofs_position(target_dof_pos, self.robot.dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.robot.get_pos()
        self.base_quat[:] = self.robot.robot.get_quat()

        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,  # change to tile for clarity
                self.base_quat,
            ),
            rpy=True,
            degrees=False,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.robot.get_dofs_position(self.robot.dofs)
        self.dof_vel[:] = self.robot.robot.get_dofs_velocity(self.robot.dofs)

        self.link_contact_forces = self.robot.contact_forces

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        # check termination and reset
        self.checks = {
            "truncate": self.episode_length_buf > self.max_episode_length,
            "pitch": torch.abs(self.base_euler[:, 1]) > self.cfg.rewards.termination_if_pitch_greater_than,
            "roll": torch.abs(self.base_euler[:, 0]) > self.cfg.rewards.termination_if_roll_greater_than,
            "height": self.base_pos[:, 2] < self.cfg.rewards.termination_if_height_lower_than,
        }
        # pprint({k: v.sum().item() for k, v in self.checks.items()})
        self.reset_buf = self.checks["truncate"] | self.checks["pitch"] | self.checks["roll"] | self.checks["height"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if self.cfg.auto_reset:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.compute_reward()
        self.render()

        # scales = self.obs_scales | {"cmd": self.commands_scale}
        # self.obs = self.robot.observe(self.inv_base_init_quat, self.global_gravity, scales, self.commands, self.default_dof_pos, self.actions)
        # self.obf_buf = self.obs
        # self.buf = {'obs': self.obs, 'rew': self.rew_buf, 'reset': self.reset_buf}

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions * self.env_cfg["action_scale"],  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        # pprint({'rew': self.rew_buf, 'reset': self.reset_buf})
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.robot.dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.B, device=gs.device))
        return self.obs_buf, None

    #
    #
    #

    def _init_robot_param(self):
        self.idxs = self.robot.indices
        print(self.robot.link_names)
        pprint(self.idxs)

        assert len(self.idxs["terminate"]) > 0
        assert len(self.idxs["feet"]) > 0

        # NOTE(mhyatt) can we remove unused ?
        # self.feet_link_indices_world_frame = [i + 1 for i in self.feet_indices]

        soft = self.cfg.rewards.soft_dof_pos_limit
        self.dof_pos_limits = self.robot.pos_limits(soft=soft)
        self.torque_limits = self.robot.torque_limits()

        # contact gait
        self.last_contacts = torch.zeros((self.num_envs, len(self.idxs["feet"])), device=self.device, dtype=gs.tc_int)
        self.link_contact_forces = self.robot.contact_forces
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.idxs["feet"])),
            device=self.device,
            dtype=gs.tc_float,
        )

    def _init_buffers(self):
        def _float(shape):
            return torch.zeros(shape, device=gs.device, dtype=gs.tc_float)

        def _int(shape):
            return torch.zeros(shape, device=gs.device, dtype=gs.tc_int)

        self.base_lin_vel = _float((self.B, 3))
        self.base_ang_vel = _float((self.B, 3))

        self.projected_gravity = _float((self.B, 3))
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.B, 1)

        self.obs_buf = _float((self.B, self.num_obs))
        self.rew_buf = _float((self.B,))
        self.reset_buf = _int((self.B,))
        self.episode_length_buf = _int((self.B,))
        self.commands = _float((self.B, self.cfg.command.num_commands))

        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.actions = _float((self.B, self.num_actions))
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = _float((self.B, 3))
        self.base_quat = _float((self.B, 4))

        self.default_dof_pos = torch.tensor(
            [self.robot.cfg.state.joints[name] for name in self.robot.cfg.dof_names],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
