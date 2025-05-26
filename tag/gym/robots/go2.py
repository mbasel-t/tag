from dataclasses import dataclass
from typing import Dict, Tuple

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
from gymnasium import spaces
import numpy as np
import torch

from tag.gym.base.config import MJCF, URDF, Control, InitState, RobotConfig, default
from tag.gym.robots.robot import Robot
from tag.names import BASE, MENAGERIE

local_dofs = [6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]
GO2MJCF = MJCF(file=str(MENAGERIE / "unitree_go2" / "go2.xml"), local_dofs=local_dofs)
GO2URDF = URDF(file=str(BASE / "other/resources/go2/urdf" / "go2.urdf"), local_dofs=local_dofs)


@dataclass
class State:
    pass


@dataclass
class RobotState(State):
    pass


@dataclass
class Go2RobotState(RobotState):
    base_pos: torch.Tensor
    base_quat: torch.Tensor
    base_velo: torch.Tensor
    base_ang: torch.Tensor

    link_pos: torch.Tensor
    link_quat: torch.Tensor
    link_vel: torch.Tensor
    link_links_ang: torch.Tensor


@dataclass
class Go2Config(RobotConfig):
    control: Control = default(Control(kp=40.0, kd=2.0))
    asset: MJCF | URDF = default(GO2URDF)

    state: InitState = default(
        InitState(
            joints={
                "FL_hip_joint": 0.1,
                "RL_hip_joint": 0.1,
                "FR_hip_joint": -0.1,
                "RR_hip_joint": -0.1,
                "FL_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0,
                "FR_thigh_joint": 0.8,
                "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5,
                "RL_calf_joint": -1.5,
                "FR_calf_joint": -1.5,
                "RR_calf_joint": -1.5,
            },
            pos=[0.0, 0.0, 0.42],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
    )

    foot_name: list[str] = default(["foot"])
    penalize_contacts_on: list[str] = default(["thigh", "calf"])
    terminate_after_contacts_on: list[str] = default(["base"])
    links_to_keep: list[str] = default(["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    self_collisions: bool = True

    @property
    def dof_names(self):
        return list(self.state.joints.keys())


class Go2Robot(Robot):
    def __init__(self, scene: gs.Scene, cfg: Go2Config, n_envs: int, color: Tuple | None = None):
        # TODO: Figure out spaces without passing n_envs through
        self.cfg = cfg

        # TODO move to self.cfg.robot.asset.create(scene) API
        # or add self.cfg.robot.create(scene)

        self.robot = scene.add_entity(
            gs.morphs.URDF(
                file=cfg.asset.file,
                pos=cfg.state.pos,
                quat=cfg.state.quat,
                # merge_fixed_links= False # needed for feet else links to keep
                links_to_keep=cfg.links_to_keep,
            ),
            surface=gs.surfaces.Default(color=color),
        )

        # TODO(dle): Add correct min and max values
        self.observation_space = spaces.Dict(
            {
                "base_pos": spaces.Box(-np.inf, np.inf, shape=(n_envs, 3), dtype=np.float32),
                "base_quat": spaces.Box(-np.inf, np.inf, shape=(n_envs, 4), dtype=np.float32),
                "base_velo": spaces.Box(-np.inf, np.inf, shape=(n_envs, 3), dtype=np.float32),
                "base_ang": spaces.Box(-np.inf, np.inf, shape=(n_envs, 3), dtype=np.float32),
                "link_pos": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 3), dtype=np.float32),
                "link_quat": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 4), dtype=np.float32),
                "link_vel": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 3), dtype=np.float32),
                "link_links_ang": spaces.Box(-np.inf, np.inf, shape=(n_envs, 12, 3), dtype=np.float32),
                # NOTE(dle): Requires Current Genesis Branch
                # "link_acc": spaces.Box(-np.inf, np.inf, shape=(12, 3), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,  # TODO(dle): Find Correct Joint Ranges
            high=1.0,
            shape=(n_envs, len(cfg.asset.local_dofs)),
            dtype=np.float32,
        )

    def reset_idx(self, idx):
        # TODO
        pass

    def act(self, action: torch.Tensor, mode: str = "position"):
        # FEATURE: Velocity/Force if needed
        # NOTE(dle): dofs_idx_local should import from Go2Config, needs to be fixed.
        if mode == "position":
            self.robot.control_dofs_position(
                position=action,
                dofs_idx_local=np.array([6, 8, 7, 9, 10, 12, 11, 13, 14, 16, 15, 17]),
            )

    def observe_state(self) -> Dict:
        obs = {
            "base_pos": self.robot.get_pos(),
            "base_quat": self.robot.get_quat(),
            "base_velo": self.robot.get_vel(),
            "base_ang": self.robot.get_ang(),
            "link_pos": self.robot.get_links_pos(),
            "link_quat": self.robot.get_links_quat(),
            "link_vel": self.robot.get_links_vel(),
            "link_links_ang": self.robot.get_links_ang(),
            # NOTE(dle): Requires Current Genesis Branch
            # "link_acc": self.robot.get_links_acc(),
            # NOTE(dle): Requires Current Genesis Branch
            # "link_force": self.robot.get_links_net_contact_force()
        }
        return obs

    # FEATURE
    def randomize(self, cfg):
        pass

    def compute_observations(self) -> Dict:
        return self.observe_state()

    @property
    def dofs(self):
        # NOTE(mhyatt) new genesis API prefers dofs vs dof, which returns list
        names = sum(
            [self.robot.get_joint(name).dofs_idx_local for name in self.cfg.dof_names],
            [],
        )
        return names

    @property
    def feet(self):
        return self.find_link_indices(self.cfg.foot_name)

    @property
    def indices(self):
        idxs = {
            "terminate": self.find_link_indices(self.cfg.terminate_after_contacts_on),
            "penalize": self.find_link_indices(self.cfg.penalize_contacts_on),
            "links": self.find_link_indices(self.cfg.links_to_keep),
            "feet": self.feet,
        }
        return idxs

    def pos_limits(self, soft: int | None = None):
        lim = torch.stack(self.robot.get_dofs_limit(self.dofs), dim=1)
        if soft is not None:
            lim = self.soften_limits(lim, soft)
        return lim

    def torque_limits(self):
        return self.robot.get_dofs_force_range(self.dofs)[1]

    @property
    def contact_forces(self):
        link_contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=gs.device,
            dtype=gs.tc_float,
        )
        return link_contact_forces

    def soften_limits(self, lim: np.ndarray, soft: int | None = None):
        m = (lim[:, 0] + lim[:, 1]) / 2
        r = lim[:, 1] - lim[:, 0]

        factor = 0.5 * r * soft
        lim[:, 0] = m - factor
        lim[:, 1] = m + factor
        return lim

    def observe(
        self,
        inv_base_init_quat,
        global_gravity,
        scales,
        commands,
        default_dof_pos,
        actions,
    ) -> None:
        # TODO(mhyatt) remove need for passed args
        # better handling of internal state

        # TODO(codex) clean for readability

        self.buf = {}
        base_pos = self.robot.get_pos()
        base_quat = self.robot.get_quat()

        base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(base_quat) * inv_base_init_quat,
                base_quat,
            ),
            rpy=True,
            degrees=True,
        )

        inv_base_quat = inv_quat(base_quat)
        base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        projected_gravity = transform_by_quat(global_gravity, inv_base_quat)
        dof_pos = self.robot.get_dofs_position(self.dofs)
        dof_vel = self.robot.get_dofs_velocity(self.dofs)

        # compute observations
        obs = torch.cat(
            [
                base_ang_vel * scales["ang_vel"],  # 3
                projected_gravity,  # 3
                commands * scales["cmd"],  # 3
                (dof_pos - default_dof_pos) * scales["dof_pos"],  # 12
                dof_vel * scales["dof_vel"],  # 12
                actions,  # 12
            ],
            axis=-1,
        )
        return obs

    def reset(self, envs_idx):
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
        self.base_pos[envs_idx] = self.cfg.state.pos.reshape(1, -1)
        self.base_quat[envs_idx] = self.cfg.state.quat.reshape(1, -1)
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

        # TODO observe state
