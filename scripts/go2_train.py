from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import shutil

import genesis as gs
from rich.pretty import pprint
from rsl_rl.runners import OnPolicyRunner
import tyro

from tag.gym.envs.walk.walk import Walk, WalkEnvConfig

# from tag.gym.base.config import Control


def check_rsl_rl():
    try:
        try:
            if metadata.version("rsl-rl"):
                raise ImportError
        except metadata.PackageNotFoundError:
            if metadata.version("rsl-rl-lib") != "2.2.4":
                raise ImportError
    except (metadata.PackageNotFoundError, ImportError) as e:
        raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "logger": "wandb",
            "wandb_project": "tag_walk",
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # base pose
        # "base_init_pos": [0.0, 0.0, 0.42],
        # "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }

    obs_cfg = {
        "num_obs": 60,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }

    return env_cfg, obs_cfg


def save_configs(log_dir, cfg, env_cfg, obs_cfg, train_cfg):
    if log_dir.exists():
        shutil.rmtree(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    # with open(log_dir / "config.json", "w") as f:
    # json.dump(asdict(cfg), f, indent=4)

    with open(log_dir / "configs.json", "w") as f:
        json.dump(
            {
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "train_cfg": train_cfg,
            },
            f,
            indent=4,
        )


@dataclass
class Config(WalkEnvConfig):
    exp_name: str = "go2-walking"
    train_steps: int = 101
    auto_reset: bool = True

    # robot: Go2Config = default(
    # Go2Config(state=
    # pos=[0.0, 0.0, 0.42],
    # quat=[1.0, 0.0, 0.0, 0.0],
    # # control=Control(kp=20.0, kd=0.5),  # from dict config
    # )
    # )

    def __post_init__(self):
        # super().__post_init__()
        self.train_steps += 1
        self.vis.pos = (2.0, 0.0, 2.5)


def main(cfg: Config):
    check_rsl_rl()
    gs.init(logging_level="warning")

    pprint(cfg)

    env_cfg, obs_cfg = get_cfgs()
    train_cfg = get_train_cfg(cfg.exp_name, cfg.train_steps)

    log_dir = Path(f"logs/{cfg.exp_name}")
    save_configs(log_dir, cfg, env_cfg, obs_cfg, train_cfg)

    env = Walk(
        cfg,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=cfg.train_steps, init_at_random_ep_len=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
