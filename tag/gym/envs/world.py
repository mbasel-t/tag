"""Minimal env for terrain and camera tests."""

from __future__ import annotations

from typing import Tuple

from gymnasium import spaces

from tag.gym.base.config import EnvConfig
from tag.gym.base.env import BaseEnv
from tag.gym.envs.terrain_mixin import TerrainEnvMixin


class WorldEnv(BaseEnv, TerrainEnvMixin):
    """Environment without robots for sensor-only tasks."""

    def __init__(self, cfg: EnvConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self._init_scene()
        self._init_terrain()

        self.cam = None
        if self.cfg.vis.visualized:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.0, 0.0, 2.0),
                lookat=(0.0, 0.0, 0.0),
                fov=60,
                GUI=False,
            )

        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Dict({})
        self._obs: dict = {}

    def step(self, actions: dict | None = None) -> Tuple[dict, None, None, None, None]:
        self.scene.step()
        if self.cam is not None:
            self.cam.render()
        return self._obs, None, None, None, None

    def reset(self) -> Tuple[dict, None]:
        return {}, None

    def get_observations(self) -> Tuple[dict, dict]:
        return self._obs, {}

