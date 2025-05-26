from dataclasses import dataclass
from enum import Enum

from tag.gym.envs.mixins.terrain import Terrain
from tag.utils import default, defaultcls

# TODO: Paintable/Randomized Terrain Implementation


@dataclass
class Viewer:
    """controls viewer settings"""

    headless: bool = True  # show to display? y/n

    @property
    def show_viewer(self):
        return not self.headless

    @property
    def gui(self):
        return not self.headless


class Resolution(Enum):
    """camera resolution"""

    K4 = (3840, 2160)
    # QHD = (2560, 1440)
    HD = (1920, 1080)
    # FHD = (1440, 1080)
    P720 = (1280, 720)
    P480 = (640, 480)
    P240 = (426, 240)
    P144 = (256, 144)


@dataclass
class Vis:
    """controls visual observations"""

    visualized: bool = True  # TODO need better name

    # TODO these should be in viewer
    # TODO rename this class Camera?
    show_world_frame: bool = True
    n_rendered_envs: int = 1
    env_spacing: list[float] = default([2.5, 2.5])

    resolution: Resolution = Resolution.P480  # camera resolution
    pos: list[float] = default([10.0, 0.0, 6.0])  # camera position
    lookat: list[float] = default([11.0, 5.0, 3.0])  # camera lookat position
    fov: float = 40.0  # field of view

    @property
    def res(self):
        return self.resolution.value


@dataclass
class Solver:
    collision: bool = True
    joint_limit: bool = True
    dt: float = 0.02  # 50hz robot step


@dataclass
class Sim:
    dt: int = 0.01  # 100hz sim step physics
    num_envs: int = 1
    substeps: int = 1


@dataclass
class Task:
    num_actions: int = 12  # arb
    episode_length: int = 120  # for testing
    max_episode_length: int = 1000
    num_obs: int = 20  # arb
    num_privileged_obs: int = 10  # arb


# Robot Configs


@dataclass
class InitState:
    joints: dict[str, float]  # default pose

    pos: list[float] = default([0.0, 0.0, 1.0])  # spawn position
    quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # spawn orientation

    # randomize_angle: bool = False  # DR - Initial Angle Spawn
    # angle_range: list[float] = default([0.0, 0.0])  # min/max angle randomization

    # TODO(dle) random stiffness dampening


@dataclass
class State:
    base_pos: list[float] = default([0.0, 0.0, 0.42])  # base link position
    base_quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # base link orientation
    lin_vel: list[float] = default([0.0, 0.0, 0.0])  # base link linear velocity
    ang_vel: list[float] = default([0.0, 0.0, 0.0])  # base link angular velocity


@dataclass
class Control:
    kp: float = 1.0
    kd: float = 1.0
    control_type: str = "P"
    action_scale: float = 0.5  # TODO(dle) add example in docstring
    decimation: float = 4
    latency: bool = False


import genesis as gs


@dataclass
class Asset:
    file: str
    local_dofs: list[int]

    pos: list[float] = default([0.0, 0.0, 0.0])
    color: list[float] | None = None

    def create(self, scene):
        return scene.add_entity(
            self._morph(
                file=self.file,
                pos=self.pos,
            ),
            surface=gs.surfaces.Default(color=self.color),
        )


class URDF(Asset):
    pass

    @property
    def _morph(self):
        return gs.morphs.URDF


class MJCF(Asset):
    pass

    @property
    def _morph(self):
        return gs.morphs.MJCF


@dataclass
class RobotConfig:
    state: InitState = defaultcls(InitState)
    state: State = default(State())
    control: Control = default(Control())
    asset: Asset = defaultcls(Asset)


# Environment Config Class
# TODO: Env config does not neccessarily need a robot.
#       Robot's joints must be known in order to config
@dataclass
class EnvConfig:
    terrain: Terrain = default(Terrain())
    viewer: Viewer = default(Viewer())
    vis: Vis = default(Vis())
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())

    def __post__init__(self):
        if self.sim.num_envs < 1:
            raise ValueError("num_envs must be greater than 0")
        if self.sim.num_envs < self.vis.n_rendered_envs:
            raise ValueError("n_rendered_envs must be less than or equal to num_envs")


# IMPLEMENT: Configurations for Tasks/Rewards/Observations
