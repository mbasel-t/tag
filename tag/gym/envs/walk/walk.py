import math

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from rich.pretty import pprint
import torch

from tag.gym.base.env import BaseEnv
from tag.gym.envs.mixins.cam import CameraMixin
from tag.gym.envs.mixins.reward import RewardMixin, WalkReward
from tag.gym.robots.go2 import Go2Robot


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Walk(BaseEnv, CameraMixin, RewardMixin):
    def __init__(
        self,
        cfg,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
    ):
        super().__init__(cfg)
        self.num_envs = self.n_envs  # for rsl rl # TODO make some env conversion
        self.auto_reset = cfg.auto_reset

        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]

        self._init_scene()
        self._setup_camera()

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        pprint(self.base_init_pos.shape)

        self.robot = Go2Robot(
            self.scene,
            self.cfg.robot,
            self.n_envs,
        )

        self.build()
        self.set_camera(lookat=(0, 0, 0))

        # PD control parameters
        self.robot.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.robot.dofs)
        self.robot.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.robot.dofs)

        self.cfg.rewards = WalkReward()
        self._init_reward()
        self._init_buffers()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

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
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.robot.get_dofs_position(self.robot.dofs)
        self.dof_vel[:] = self.robot.robot.get_dofs_velocity(self.robot.dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = {
            "truncate": self.episode_length_buf > self.max_episode_length,
            "pitch": torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"],
            "roll": torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"],
        }
        self.reset_buf = self.reset_buf["truncate"] | self.reset_buf["pitch"] | self.reset_buf["roll"]

        pprint({"n_reset": self.reset_buf})

        # self.reset_buf = self.episode_length_buf > self.max_episode_length
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if self.auto_reset:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.render()

        # scales = self.obs_scales | {"cmd": self.commands_scale}
        # self.obs = self.robot.observe(self.inv_base_init_quat, self.global_gravity, scales, self.commands, self.default_dof_pos, self.actions)
        # self.obf_buf = self.obs
        # self.buf = {'obs': self.obs, 'rew': self.rew_buf, 'reset': self.reset_buf}

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        # pprint({'rew': self.rew_buf, 'reset': self.reset_buf})
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras  # v2.2.4

    def get_observations(self):
        # NOTE(mhyatt) not needed for rsl-rl v1.0.2
        # self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf  # , self.extras

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
        self.reset_idx(torch.arange(self.n_envs, device=gs.device))
        return self.obs_buf, None

    #
    #
    #

    def _init_buffers(self):
        def _float(shape):
            return torch.zeros(shape, device=gs.device, dtype=gs.tc_float)

        def _int(shape):
            return torch.zeros(shape, device=gs.device, dtype=gs.tc_int)

        self.base_lin_vel = _float((self.n_envs, 3))
        self.base_ang_vel = _float((self.n_envs, 3))

        self.projected_gravity = _float((self.n_envs, 3))
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.n_envs, 1)

        self.obs_buf = _float((self.n_envs, self.num_obs))
        self.rew_buf = _float((self.n_envs,))
        self.reset_buf = _int((self.n_envs,))
        self.episode_length_buf = _int((self.n_envs,))
        self.commands = _float((self.n_envs, self.num_commands))

        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.actions = _float((self.n_envs, self.num_actions))
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = _float((self.n_envs, 3))
        self.base_quat = _float((self.n_envs, 4))

        self.default_dof_pos = torch.tensor(
            [self.robot.cfg.state.joints[name] for name in self.robot.cfg.dof_names],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
