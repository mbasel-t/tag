from dataclasses import dataclass
import json
from pathlib import Path

import genesis as gs
from go2_train import Config
from rsl_rl.runners import OnPolicyRunner
import torch
from tqdm import tqdm
import tyro

from tag.gym.envs.walk.walk import Walk
from tag.names import BASE


def load_configs(log_dir):
    with open(log_dir / "configs.json", "r") as f:
        _cfgs = json.load(f)

    env_cfg, obs_cfg,  train_cfg = (
        _cfgs["env_cfg"],
        _cfgs["obs_cfg"],
        # _cfgs["command_cfg"],
        _cfgs["train_cfg"],
    )
    return env_cfg, obs_cfg,  train_cfg


@dataclass
class EvalConfig(Config):
    ckpt: int = 100  # the checkpoint to load
    auto_reset: bool = False
    use_ts: bool = False
    path: str | Path = None


def main(cfg: EvalConfig):
    # check_rsl_rl()
    gs.init()

    log_dir = Path(cfg.path)
    env_cfg, obs_cfg,  train_cfg = load_configs(log_dir)
    env = Walk(
        cfg,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
    )

    if cfg.use_ts:
        pi = torch.jit.load(str(BASE / "policy.ts.pt")).to(gs.device)
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
        resume_path = log_dir / f"model_{cfg.ckpt}.pt"
        runner.load(str(resume_path))
        pi = runner.get_inference_policy(device=gs.device)

    env.build()
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in tqdm(range(len(env)), desc="Eval..."):
            actions = pi(obs)
            obs, rews, dones, infos = env.step(actions)

    env.record_visualization("go2_eval")


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
