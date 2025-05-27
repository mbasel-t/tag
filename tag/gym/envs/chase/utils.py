"""Helper utilities for the chase environment."""

from __future__ import annotations

import genesis as gs

from tag.gym.robots.go2 import Go2Config
from tag.gym.robots.multi import MultiRobot


def create_robots(scene: gs.Scene, cfg: Go2Config) -> MultiRobot:
    """Instantiate the pair of Go2 robots for the chase task."""
    return MultiRobot(scene, cfg, ["r1", "r2"])
