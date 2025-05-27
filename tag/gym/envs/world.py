from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from gymnasium import spaces

from tag.utils import defaultcls

from .base import BaseEnv, BaseEnvConfig
from .mixins import CameraMixin, DomainRandMixin, TerrainMixin
from .mixins.cam import Cam
from .mixins.terrain import Terrain


@dataclass
class WorldEnvConfig(BaseEnvConfig):
    terrain: Terrain = defaultcls(Terrain)
    cam: Cam = defaultcls(Cam)


class WorldEnv(BaseEnv, TerrainMixin, CameraMixin, DomainRandMixin):
    """Environment without robots for sensor-only tasks."""

    def __init__(self, cfg: WorldEnvConfig):
        super().__init__(cfg)

        self._init_terrain()  # TODO uncomment when terrain is ready
        self._setup_camera()

        # TODO inherit camera obs space
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Dict({})
        self._obs: dict = {}

    def reset(self) -> Tuple[dict, None]:
        return self.observe(), None

    def observe(self) -> Tuple[dict, dict]:
        self._obs = self.cam.render()
        return {"imgs": self._obs}
