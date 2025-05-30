from typing import Union
from .base import BaseEnvConfig, BaseEnv
from .world import WorldEnvConfig, WorldEnv
from .robotic import RobotEnvConfig, RobotEnv

#
from .chase.chase import Chase
from .walk.walk import Walk


ops = (WorldEnvConfig, BaseEnvConfig, RobotEnvConfig)
EnvTyp = Union[*ops]
