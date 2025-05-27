from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools

from genesis.engine.entities.rigid_entity import RigidEntity
from gymnasium import spaces

from tag.gym.base.config import Asset, Control, State
from tag.protocols import Wraps, _Robot
from tag.utils import defaultcls

_counter = itertools.count()


@dataclass
class RobotState(State):
    pass


@dataclass
class RobotConfig(ABC):
    asset: Asset = defaultcls(Asset)
    state: RobotState = defaultcls(RobotState)
    control: Control = defaultcls(Control)

    def _create(self, scene) -> RigidEntity:
        """create robot asset"""
        return self.asset.create(scene, pos=self.state.pos, quat=self.state.quat)

    @abstractmethod
    def create(self, scene) -> "Robot":
        """Create a robot instance."""
        return Robot(scene, self)


class Robot(_Robot, Wraps):
    REGISTRY: dict[str, "Robot"] = {}

    def __init__(self, scene, cfg: RobotConfig):
        """Initialize the robot with a scene and configuration."""
        self.cfg = cfg
        self.robot = cfg._create(scene)

        # Initialize observation and action spaces
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Dict({})

        # Register the robot
        self.register()

    #
    # Utilities
    #

    @property
    def B(self):
        return self._solver.n_envs

    @property
    def name(self) -> str:
        """Return the name of the robot."""
        return self.idx  # from Entity

        # if getattr(self, "_name", None) is None:
        # self._name = uuid.uuid4().hex
        # self._uid = str(next(_counter))
        # self._morph = self.__class__.__name__

    def register(self) -> None:
        """Register the robot obj in the registry."""
        if self.name in Robot.REGISTRY:
            raise ValueError(f"Robot with name {self.name} already registered.")
        Robot.REGISTRY[self.name] = self

    def find_link_indices(self, names):
        """Find the indices of the links in the robot."""
        return [
            link.idx - self.robot.link_start for link in self.robot.links if any(name in link.name for name in names)
        ]

    @property
    def link_names(self):
        return [link.name for link in self.robot.links]
