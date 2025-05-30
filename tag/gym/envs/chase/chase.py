from dataclasses import dataclass
from typing import Tuple

import genesis as gs
import numpy as np
import torch
from gymnasium.spaces import Dict

from tag.gym.base.config import Task
from tag.gym.envs.robotic import MultiGo2EnvConfig, RobotEnv
from tag.gym.robots.joystick_go2 import OVERFIT, CommandConfig, JoyStickGo2
from tag.gym.robots.multi import MultiRobot
from tag.utils import default, defaultcls

from ..world import WorldEnv


@dataclass
class Scales:
    """observation and action scales"""

    action: float = 0.25
    obs: dict[str, float] = default(
        {
            # "num_obs": 60,
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
    )

    def expand(self):
        cmd_scale = torch.Tensor(
            [
                self.obs["lin_vel"],
                self.obs["lin_vel"],
                self.obs["ang_vel"],
            ],
        ).to(gs.device)
        return self.obs | {"action": self.action, "cmd": cmd_scale}


@dataclass
class ChaseEnvConfig(MultiGo2EnvConfig):
    task: Task = default(Task())
    joy: bool = True
    # Default joystick controller, should be in joy wrapper config
    cmd: CommandConfig = default(OVERFIT)
    scales: Scales = defaultcls(Scales)


class Chase(RobotEnv):
    """Simple two-robot chase environment."""

    def __init__(self, cfg: ChaseEnvConfig):
        """Create a new environment instance."""
        # super().__init__(cfg) # TODO doesnt play nice since multirobot needs create
        WorldEnv.__init__(self, cfg)
        self.cfg: ChaseEnvConfig = cfg

        # TODO(mbt): Implement Color System
        self.robot = MultiRobot(
            self.scene,  # enable cfg create API
            self.cfg.robot,
            self.cfg.n_robots,  # NOTE(mhyatt) this should be in a multirobot config
        )
        if self.cfg.joy:  # manually wrap joy go2
            for k, r in self.robot.robots.items():
                self.robot.robots[k] = JoyStickGo2(
                    r, cmd=self.cfg.cmd, scales=self.cfg.scales
                )

    def set_control_gains(self):
        # TODO: Implement Method - Should this method be in another class?
        # NOTE(mhyatt) make it as method of robot called on their init
        pass

    def step(self, actions: dict) -> Tuple[dict, None, None, None, None]:
        """Advance the simulation by one step."""

        obs = super().step(actions)

        # TODO: Properly Implement Step Method - Actions, Updates, etc.
        # TODO(dle): Implement Dummy Reward System
        # reward = self.get_reward()

        reward, term, trunc, info = 0.0, False, False, {}
        return obs, reward, term, trunc, info

    """
    @property
    def observation_space(self):
        return  Dict(
            {
                # TODO: This needs to be properly tiled, solving by passing n_envs through robot
                "robots": self.robot.observation_space,
                "terrain": Dict({}),
                "obstacles": Dict({}),
            }
        )
    """
