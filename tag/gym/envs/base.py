from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import genesis as gs
import jax
import torch

from tag.gym.base.config import Sim, Solver, Viewer, Vis
from tag.protocols import Wraps, _Env
from tag.utils import default, defaultcls


@dataclass
class BaseEnvConfig:
    viewer: Viewer = default(Viewer())  # TODO combine with base
    vis: Vis = default(Vis())
    solver: Solver = default(Solver())
    sim: Sim = default(Sim())

    def __post__init__(self):
        if self.sim.num_envs < 1:
            raise ValueError("num_envs must be greater than 0")
        if self.sim.num_envs < self.vis.n_rendered_envs:
            raise ValueError("n_rendered_envs must be less than or equal to num_envs")


class BaseEnv(_Env):
    def __init__(self, cfg: BaseEnvConfig):
        """Initialize common environment attributes."""

        self.cfg: BaseEnvConfig = cfg
        self.B = cfg.sim.num_envs  # shorthand
        self.device = gs.gpu
        self.cam = None

        self._max_steps = int(1e3)
        self._init_scene()

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
            n_envs=self.cfg.sim.num_envs,
            env_spacing=self.cfg.vis.env_spacing if self.cfg.vis.n_rendered_envs > 1 else [0, 0],
        )
        if self.cam is not None:
            self.cam.start_recording()  # TODO move to the camera mixin

    def step(self) -> None:
        self.scene.step()

    def reset(self) -> None: ...

    def observation_space(self) -> Tuple[torch.Tensor, Dict]: ...

    def action_space(self) -> Tuple[torch.Tensor, Dict]: ...


Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


@dataclass
class Step:  # (base.Base):
    """Environment step for training and inference."""

    state: Optional[Observation]
    obs: Observation
    reward: jax.Array
    done: Dict[str, jax.Array]
    metrics: Dict[str, jax.Array] = defaultcls(dict)
    info: Dict[str, Any] = defaultcls(dict)


class EnvWrapper(_Env, Wraps):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: _Env):
        self._env = env

    def reset(self, rng):
        return self._env.reset(rng)

    def step(self, *args, **kwargs):
        return self._env.step(*args, **kwargs)

    def wrapped(self) -> _Env:
        """Return the wrapped environment."""
        return self._env
