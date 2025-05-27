from dataclasses import dataclass

from tag.gym.base.config import Task
from tag.gym.envs.robotic import RobotEnvConfig
from tag.gym.robots import RobotTyp
from tag.gym.robots.go2 import Go2Config
from tag.utils import default, defaultcls


@dataclass
class Go2EnvConfig(RobotEnvConfig):
    robot: Go2Config = defaultcls(Go2Config)


@dataclass
class MultiGo2EnvConfig(Go2EnvConfig):
    robot: RobotTyp = defaultcls(Go2Config)
    n_robots: int = 2


# --- Task Environments ---


@dataclass
class ChaseEnvConfig(MultiGo2EnvConfig):
    task: Task = default(Task())
