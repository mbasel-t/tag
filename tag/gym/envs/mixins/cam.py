"""Simple camera utilities for environments."""

from __future__ import annotations

from typing import Tuple

import genesis as gs
import numpy as np


class CameraMixin:
    """Mixin adding a floating camera and basic controls."""

    cam: gs.Camera | None
    _recording: bool
    _recorded_frames: list

    def _setup_camera(self) -> None:
        """Create the floating camera from ``self.cfg.viewer``."""
        if self.cfg.vis.visualized:
            self.cam = self.scene.add_camera(
                res=self.cfg.vis.res,
                pos=np.array(self.cfg.vis.pos),
                lookat=np.array(self.cfg.vis.lookat),
                fov=self.cfg.vis.fov,
                GUI=not self.cfg.viewer.gui,
            )
            self._recording = False
            self._recorded_frames = []

    def set_camera(
        self, pos: Optional[Tuple[float, float, float]] = None, lookat: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Move the floating camera."""
        if self.cam is None:
            return
        assert pos is not None or lookat is not None, "Either pos or lookat must be provided"
        self.cam.set_pose(pos=pos, lookat=lookat)

    def render(self, *args, **kwargs):
        if self.cfg.vis.visualized:
            imgs = self.cam.render(*args, **kwargs)
            return imgs
