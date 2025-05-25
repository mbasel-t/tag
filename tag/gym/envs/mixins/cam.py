"""Simple camera utilities for environments."""

from __future__ import annotations

from typing import Tuple

import genesis as gs
import numpy as np


class CameraMixin:
    """Mixin adding a floating camera and basic controls."""

    floating_camera: gs.Camera | None
    _recording: bool
    _recorded_frames: list

    def _setup_camera(self) -> None:
        """Create the floating camera from ``self.cfg.viewer``."""
        self.floating_camera = self.scene.add_camera(
            res=self.cfg.vis.res,
            pos=np.array(self.cfg.vis.pos),
            lookat=np.array(self.cfg.vis.lookat),
            fov=self.cfg.vis.fov,
            GUI=not self.cfg.viewer.gui,
        )
        self._recording = False
        self._recorded_frames = []

    def set_camera(self, pos: Tuple[float, float, float], lookat: Tuple[float, float, float]) -> None:
        """Move the floating camera."""
        if self.floating_camera is None:
            return
        self.floating_camera.set_pose(pos=pos, lookat=lookat)
