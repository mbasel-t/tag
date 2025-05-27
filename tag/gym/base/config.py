from dataclasses import dataclass
from enum import Enum

import genesis as gs

from tag.utils import default


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

    show_world_frame: bool = True
    n_rendered_envs: int = 1
    env_spacing: list[float] = default([2.5, 2.5])


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
    episode_length: int = 120  # for testing
    max_episode_length: int = 1000


@dataclass
class State:
    pos: list[float] = default([0.0, 0.0, 0.42])  # base link position
    quat: list[float] = default([1.0, 0.0, 0.0, 0.0])  # base link orientation
    vel: list[float] = default([0.0, 0.0, 0.0])  # base link linear velocity
    ang: list[float] = default([0.0, 0.0, 0.0])  # base link angular velocity

    # TODO(dle) random stiffness dampening


@dataclass
class Control:
    kp: float = 1.0
    kd: float = 1.0
    control_type: str = "P"
    action_scale: float = 0.5  # TODO(dle) add example in docstring
    decimation: float = 4
    latency: bool = False


@dataclass
class Asset:
    file: str
    # local_dofs: list[int] # NOTE(mhyatt) why is this here

    color: list[float] | None = None

    def create(self, scene, pos, quat, **kwargs):
        return scene.add_entity(
            self._morph(file=self.file, pos=pos, quat=quat, **kwargs),
            surface=gs.surfaces.Default(color=self.color),
        )


class URDF(Asset):
    pass

    @property
    def _morph(self):
        return gs.morphs.URDF


class MJCF(Asset):
    # NOTE(mhyatt) MJCF doesnt have links_to_keep
    pass

    @property
    def _morph(self):
        return gs.morphs.MJCF
