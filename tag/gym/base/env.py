from abc import abstractmethod
from typing import Dict, Tuple, Union

import genesis as gs
import torch

from tag.gym.base.config import EnvConfig
from tag.names import BASE

# TODO add somewhere
# num_privileged_obs: int
# max_episode_length: int | torch.Tensor
# episode len is in task cfg


class BaseEnv:
    def __init__(self, cfg: EnvConfig):
        """Initialize common environment attributes."""

        self.cfg: EnvConfig = cfg
        self.device = gs.gpu
        self.n_envs = cfg.sim.num_envs
        self.n_rendered = cfg.vis.n_rendered_envs
        self.env_spacing = cfg.vis.env_spacing

        # TODO task not defined
        # task_cfg = getattr(cfg, "task", Task())
        # self.num_obs = task_cfg.num_obs
        # self.num_privileged_obs = task_cfg.num_privileged_obs
        # self.max_episode_length = task_cfg.max_episode_length

        self._max_steps = int(1e3)
        # self._init_buffers()

    def __len__(self) -> int:
        """Return the number of environments."""
        return self._max_steps

    def _init_scene(self) -> gs.Scene:
        self.scene = gs.Scene(
            show_viewer=not self.cfg.viewer.headless,
            vis_options=gs.options.VisOptions(
                show_world_frame=self.cfg.vis.show_world_frame,
                n_rendered_envs=self.cfg.vis.n_rendered_envs,
            ),
            # physics
            sim_options=gs.options.SimOptions(dt=self.cfg.sim.dt, substeps=self.cfg.sim.substeps),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=self.cfg.solver.joint_limit,
                dt=self.cfg.solver.dt,  # TODO rename solver.dt to control.dt ?
            ),
        )

    def build(self):
        self.scene.build(
            n_envs=self.n_envs,
            env_spacing=self.env_spacing if self.n_rendered > 1 else [0, 0],
        )
        if self.cam is not None:
            self.cam.start_recording()

    def step(self, actions: torch.Tensor) -> None: ...

    def reset(self) -> None: ...

    @abstractmethod
    def get_observations(self) -> Tuple[torch.Tensor, Dict]: ...

    @abstractmethod
    def get_privileged_obs(self) -> Union[torch.Tensor, Dict]: ...

    def observation_space(self) -> Tuple[torch.Tensor, Dict]: ...

    def action_space(self) -> Tuple[torch.Tensor, Dict]: ...

    def record_visualization(self, fname: str = None) -> None:
        """Finalize and save camera recordings, if any."""
        if getattr(self.cfg.vis, "visualized", False) and hasattr(self, "cam"):
            dir = BASE / "mp4"
            dir.mkdir(parents=True, exist_ok=True)
            fname = dir / f'{fname if fname else "video"}.mp4'
            self.cam.stop_recording(save_to_filename=fname, fps=60)
