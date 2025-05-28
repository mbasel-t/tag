from dataclasses import dataclass
from pathlib import Path

import genesis as gs
from go2_eval import load_configs
from rsl_rl.runners import OnPolicyRunner
import torch
import tyro

from tag.gym.envs.walk.walk import Walk, WalkEnvConfig
from tag.names import BASE
from tag.utils import defaultcls


@dataclass
class Runtime:
    pass


class _EvalConfig(Runtime):
    pass


class _TrainConfig(Runtime):
    pass


@dataclass
class Config:
    path: str  # path to dir with .pt files
    ckpt: int

    env: WalkEnvConfig = defaultcls(WalkEnvConfig)

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise ValueError(f"Path {self.path} does not exist.")


def main(cfg: Config):
    # check_rsl_rl()
    gs.init()

    env_cfg, obs_cfg, command_cfg, train_cfg = load_configs(cfg.path)
    env = Walk(
        cfg.env,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, cfg.path, device=gs.device)
    resume_path = cfg.path / f"model_{cfg.ckpt}.pt"
    runner.load(str(resume_path))
    pi = runner.alg.actor_critic.actor

    tspi = torch.jit.script(pi)
    tspi.save(BASE / "policy.ts.pt")


if __name__ == "__main__":
    main(tyro.cli(Config))
