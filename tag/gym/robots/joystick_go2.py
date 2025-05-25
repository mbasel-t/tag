from gymnasium import spaces
import numpy as np
import torch

from .go2 import Go2Robot
from .robot import Robot


class JoyStickGo2(Robot):
    """Wrapper that maps joystick commands to Go2 joint actions using a policy."""

    def __init__(self, robot: Go2Robot, policy_path: str):
        self.robot = robot
        self.policy = torch.jit.load(policy_path)

        self.observation_space = robot.observation_space
        # Expect joystick commands in (x, y, z) linear velocity and (roll, pitch, yaw)
        n_envs = robot.action_space.shape[0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_envs, 6),
            dtype=np.float32,
        )

    def reset(self):
        return self.robot.reset()

    def act(self, action: torch.Tensor, mode: str = "position"):
        # Map joystick command through policy to joint commands
        joint_action = self.policy(action)
        self.robot.act(joint_action, mode=mode)

    def compute_observations(self):
        return self.robot.observe_state()

    def __getattr__(self, item):
        return getattr(self.robot, item)
