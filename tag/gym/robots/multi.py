from copy import deepcopy
from typing import Dict

import genesis as gs
from gymnasium import spaces

from tag.gym.robots.go2 import Go2Config, Go2Robot
from tag.gym.robots.robot import Robot

from .utils import tile_xyz

# rgba
RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0, 1.0]
BLUE = [0.0, 0.0, 1.0, 1.0]
ORANGE = [1.0, 0.5, 0.0, 1.0]
NA = None


class MultiRobot(Robot):
    """Version 1"""

    def __init__(self, scene: gs.Scene, cfg: Go2Config, n: list[str], colors=None):
        # super().__init__(scene, cfg) # cant super init because no multirobot config
        self.cfg = cfg
        self._init_robots(scene, n, colors)

    def _init_robots(self, scene, n, colors=None):
        # FEATURE: Need settings for distance apart, where to spawn, etc.
        # NOTE(dle): Need to add something to define the distance between them properly

        init_pos_map = tile_xyz(n, self.cfg.state.pos[2])
        colors = [[RED, BLUE][i % 2] for i in range(n)] if colors is None else colors

        self.robots = {}
        for i in range(n):
            _cfg = deepcopy(self.cfg)
            _cfg.state.pos = init_pos_map[i]
            _cfg.asset.color = colors[i]

            robot: Go2Robot = _cfg.create(scene)
            self.robots[robot.name] = robot

            # NOTE(dle): Spaces Temp Fix

        # NOTE(dle): If we don't tile, torch automatically tiles repeats, which is bad because we are getting different data
        # TODO: These need to be tiled n_envs x dofs

    def __iter__(self):
        return iter(self.robots.values())

    def wrapped(self):
        raise NotImplementedError("you cant unwrap a multirobot")

    def act(self, actions: Dict, mode: str = "position"):
        for k, robot in self.robots.items():
            robot.act(action=actions[k])

    def observe(self) -> Dict:
        return {k: robot.observe() for k, robot in self.robots.items()}

    def reset(self, envs_idx: list[int]) -> Dict:
        """Reset all robots in the multi-robot environment."""
        for robot in self.robots.values():
            robot.reset(envs_idx)

    @property
    def action_space(self):
        """Return the action space of the multi-robot."""
        return spaces.Dict({k: r.action_space for k, r in self.robots.items()})

    @property
    def observation_space(self):
        """Return the observation space of the multi-robot."""
        return spaces.Dict({k: r.observation_space for k, r in self.robots.items()})
