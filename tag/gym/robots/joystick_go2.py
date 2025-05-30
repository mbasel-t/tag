from dataclasses import dataclass

import genesis as gs
import numpy as np
import torch
from flax.traverse_util import flatten_dict
from genesis.utils.geom import (inv_quat, quat_to_xyz, transform_by_quat,
                                transform_quat_by_quat)
from gymnasium import spaces
from rich.pretty import pprint

from tag.names import BASE
from tag.protocols import Wraps
from tag.utils import default, defaultcls, space2spec

from .go2 import Go2Robot
from .robot import Robot


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


class JoyStickGo2(Wraps):
    """Wrapper that maps joystick commands to Go2 joint actions using a policy."""

    def __init__(
        self,
        robot: Go2Robot,
        cmd: CommandConfig,
        scales,
        path: str = BASE / "joy.ts.pt",
    ):
        self.robot = robot
        self.cmd = cmd
        self.scales = scales
        self.policy = torch.jit.load(path).to(gs.device)
        self.action = None

        # required = ["a", "b", "c"]
        # spec = flatten_dict(space2spec(robot.observation_space))
        # msg = f"Expected observation space to contain {required}, but got {spec.keys()}"
        # assert all([r in spec for r in required]), msg

        # Expect joystick commands in (x, y, z) linear velocity and (roll, pitch, yaw)

    @property
    def action_space(self):
        low = np.array(
            [
                self.cmd.lin_vel_x_range[0],
                self.cmd.lin_vel_y_range[0],
                self.cmd.ang_vel_range[0],
            ],
        )
        high = np.array(
            [
                self.cmd.lin_vel_x_range[1],
                self.cmd.lin_vel_y_range[1],
                self.cmd.ang_vel_range[1],
            ],
        )
        space = spaces.Box(
            low=low,
            high=high,
            shape=(self.cmd.num_commands,),
            dtype=np.float64,
        )
        return space

    @property
    def wrapped(self):
        return self.robot

    def act(self, command: torch.Tensor):
        _obs = self._obs_for_joy(command)
        rel_action = self.policy(_obs)
        abs_action = rel_action * self.scales.action + self.cfg.state.dof_pos
        self.robot.act(abs_action, mode="position")
        self.action = rel_action

    def __getattr__(self, item):
        return getattr(self.robot, item)

    def _obs_for_joy(self, command):
        """resamples from robot obs to get what we need for the joystick"""

        command = torch.as_tensor(command, device=gs.device, dtype=gs.tc_float)
        obs = flatten_dict(self.robot.observe(), sep=".")

        base_lin_vel = transform_by_quat(obs["base.vel"], self.inv_quat)
        base_ang_vel = transform_by_quat(obs["base.ang"], self.inv_quat)
        dof_pos, dof_vel = obs["dof.pos"], obs["dof.vel"]

        action = self.action if self.action is not None else torch.zeros_like(dof_pos)
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        ).repeat(self.B, 1)
        projected_gravity = transform_by_quat(self.global_gravity, self.inv_quat)

        _obs = {
            # scaled base velocity transformed wrt current base quat
            "lin_vel": base_lin_vel,
            # scaled ang velocity transformed wrt current base quat
            "ang_vel": base_ang_vel,
            # wrt current base quat
            "projected_gravity": projected_gravity,
            "cmd": command,
            # relative to init
            "dof_pos": dof_pos - self.cfg.state.dof_pos,
            # abs velocity
            "dof_vel": dof_vel,
            # scaled action (relative to default dof pos)
            "action": action,
        }

        for k in (_scales := self.scales.expand()).keys():
            print(k)
            scale = _scales.get(k, 1.0)
            print(type(_obs[k]), type(scale))
            _obs[k] *= scale

        _obs_flat = torch.cat(list(_obs.values()), axis=-1)
        return _obs_flat
