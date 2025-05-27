from .robot import Robot,RobotConfig
from .go2 import Go2Config,Go2Robot
from typing import Union

ops = (RobotConfig,Go2Config)
RobotTyp = Union[*ops]
