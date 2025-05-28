from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces
import numpy as np
import torch

from tag.gym.robots import RobotTyp
from tag.gym.robots.go2 import Go2Config
from tag.gym.robots.robot import Robot, RobotConfig
from tag.utils import defaultcls

from .world import WorldEnv, WorldEnvConfig


@dataclass
class RobotEnvConfig(WorldEnvConfig):
    """Configuration for the robotic environment."""

    robot: RobotConfig = defaultcls(RobotConfig)  # usually Go2


@dataclass
class Go2EnvConfig(RobotEnvConfig):
    robot: Go2Config = defaultcls(Go2Config)


@dataclass
class MultiGo2EnvConfig(Go2EnvConfig):
    robot: RobotTyp = defaultcls(Go2Config)
    n_robots: int = 2


class RobotEnv(WorldEnv):
    """Environment for robotic tasks."""

    def __init__(self, cfg: RobotEnvConfig):
        super().__init__(cfg)
        self.robot: Robot = cfg.robot.create(self.scene)
        # self.robot.reset(np.arange(self.B).tolist()) # add to build sequence?

        # Initialize the observation and action spaces
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Dict({})

    def pre_step(self):
        pass

    def step(self, actions: dict):
        self.robot.act(actions)
        super().step()
        obs = self.observe()
        self.post_step()  # cleanup and state mgt
        return obs

    def post_step(self):
        pass

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset the environment state."""
        self.robot.reset(np.arange(self.B).tolist())
        self.scene.step()  # prime with 1 step
        return self.observe(), {}

    def _reset_idx(self, envs_idx: list[int]):
        """Reset specific environments by index."""
        self.robot.reset(envs_idx)

    def observe(self) -> dict[str, torch.Tensor]:
        """Collect observations from robots and environment."""
        self._obs = {"robot": self.robot.observe()} | super().observe()
        return self._obs
