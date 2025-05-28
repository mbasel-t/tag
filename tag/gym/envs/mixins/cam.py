"""Simple camera utilities for environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import genesis as gs
import numpy as np

from tag.names import BASE
from tag.utils import default


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
class Cam:
    """controls visual observations"""

    enable: bool = True

    resolution: Resolution = Resolution.P480  # resolution
    pos: list[float] = default([2.0, 0.0, 2.5])  # position
    lookat: list[float] = default([0.0, 0.0, 0.5])  # lookat position
    fov: float = 40.0  # field of view
    follow: bool = False  # follow entity

    @property
    def res(self):
        return self.resolution.value


class CameraMixin:
    """Mixin adding a floating camera and basic controls."""

    cam: gs.Camera | None
    _recording: bool
    _recorded_frames: list

    def _setup_camera(self) -> None:
        """Create the floating camera from ``self.cfg.viewer``."""
        if self.cfg.cam.enable:
            self.cam = self.scene.add_camera(
                res=self.cfg.cam.res,
                pos=np.array(self.cfg.cam.pos),
                lookat=np.array(self.cfg.cam.lookat),
                fov=self.cfg.cam.fov,
                GUI=not self.cfg.viewer.gui,
            )
            self._recording = False
            self._recorded_frames = []

    def set_camera(
        self,
        pos: Optional[Tuple[float]] = None,
        lookat: Optional[Tuple[float]] = None,
    ):
        """Move the floating camera."""
        if self.cam is None:
            return
        assert pos is not None or lookat is not None, "Either pos or lookat must be provided"
        self.cam.set_pose(pos=pos, lookat=lookat)

    def render(self, *args, **kwargs):
        if self.cfg.cam.enable:
            imgs = self.cam.render(*args, **kwargs)
            return imgs

    def record_visualization(self, fname: str = None) -> None:
        """Finalize and save camera recordings, if any."""
        if self.cam is None:
            print("cam is None")
            return

        if self.cfg.cam.enable and hasattr(self, "cam"):
            dir = BASE / "mp4"
            dir.mkdir(parents=True, exist_ok=True)
            fname = dir / f"{fname if fname else 'video'}.mp4"
            self.cam.stop_recording(save_to_filename=fname, fps=60)

    def follow(self, entity, fixed_axis: tuple[float] | None = (None, 1.5, 1.0)):
        """Follow an entity with the camera."""
        if self.cam is None:
            return
        self.cam.follow_entity(entity, fixed_axis=fixed_axis)
