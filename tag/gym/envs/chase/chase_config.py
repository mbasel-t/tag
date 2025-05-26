from dataclasses import dataclass

from tag.gym.base.config import EnvConfig, Sim, Solver, Task, Terrain, Viewer, Vis
from tag.gym.robots.go2 import Go2Config
from tag.utils import default, defaultcls


@dataclass
class Go2EnvConfig(EnvConfig):
    terrain: Terrain = default(Terrain())
    viewer: Viewer = default(Viewer())
    vis: Vis = default(Vis())
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())
    robot: Go2Config = defaultcls(Go2Config)
    n_robots: int = 1


# --- Task Environments ---


class ChaseEnvConfig(Go2EnvConfig):
    task: Task = default(Task())


# IMPLEMENT: Configurations for Tasks/Rewards/Observations
