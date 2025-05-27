# from ._lr.env import LeggedRobot as LR
# from chase.chase_env import Chase
from .base import BaseEnvConfig, BaseEnv
from .world import WorldEnvConfig, WorldEnv
from .robotic import RobotEnvConfig, RobotEnv

from typing import Union

ops = (WorldEnvConfig, BaseEnvConfig, RobotEnvConfig)
EnvTyp = Union[*ops]
