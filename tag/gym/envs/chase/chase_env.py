from dataclasses import dataclass
from typing import Tuple

from gymnasium.spaces import Dict

from tag.gym.base.config import Task
from tag.gym.envs.robotic import MultiGo2EnvConfig, RobotEnv
from tag.gym.robots.multi import MultiRobot
from tag.utils import default

from ..world import WorldEnv


@dataclass
class ChaseEnvConfig(MultiGo2EnvConfig):
    task: Task = default(Task())


class Chase(RobotEnv):
    """Simple two-robot chase environment."""

    def __init__(self, cfg: ChaseEnvConfig):
        """Create a new environment instance."""
        # super().__init__(cfg) # TODO doesnt play nice since multirobot needs create
        WorldEnv.__init__(self, cfg)
        self.cfg: ChaseEnvConfig = cfg

        # TODO(mbt): Implement Color System
        self.robot = MultiRobot(
            self.scene,  # enable cfg create API
            self.cfg.robot,
            self.cfg.n_robots,  # NOTE(mhyatt) this should be in a multirobot config
        )

        self._init_spaces()

    def set_control_gains(self):
        # TODO: Implement Method - Should this method be in another class?
        # NOTE(mhyatt) make it as method of robot called on their init
        pass

    def step(self, actions: dict) -> Tuple[dict, None, None, None, None]:
        """Advance the simulation by one step."""

        obs = super().step(actions)

        # TODO: Properly Implement Step Method - Actions, Updates, etc.
        # TODO(dle): Implement Dummy Reward System
        # reward = self.get_reward()

        reward, term, trunc, info = 0.0, False, False, {}
        return obs, reward, term, trunc, info

    def _init_spaces(self):
        """Define observation and action spaces."""
        self.observation_space = Dict(
            {
                # TODO: This needs to be properly tiled, solving by passing n_envs through robot
                "robots": self.robot.observation_space,
                "terrain": Dict({}),
                "obstacles": Dict({}),
            }
        )

        self.action_space = self.robot.action_space
