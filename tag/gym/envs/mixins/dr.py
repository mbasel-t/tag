"""Helpers for simulation domain randomization."""

from __future__ import annotations

from dataclasses import dataclass

import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import torch

from tag.utils import default


@dataclass
class DR:
    """pass empty list to turn off randomization"""

    friction: list[float] = default([0.2, 1.7])
    mass: list[float] = default([-1.0, 1.0])
    com: list[float] = default([-0.01, 0.01])  # center of mass displacement

    push_robots: bool = True
    push_interval_s: int = 15
    max_push_vel_xy: float = 1.0

    simulate_action_latency: bool = False  # 1 step delay


class DomainRandMixin:
    """Mixin providing common domain randomizations."""

    def find_rigid_solver(self) -> RigidSolver:
        """Return the ``RigidSolver`` used by ``self.scene``."""
        for solver in self.scene.sim.solvers:
            if isinstance(solver, RigidSolver):
                self.rigid_solver = solver
                return solver
        raise RuntimeError("RigidSolver not found")

    def _randomize_friction(self, envs=None) -> None:
        """Randomize friction of all geoms."""
        min_f, max_f = self.cfg.dr.friction
        solver = self.rigid_solver
        ratios = gs.rand((len(envs), 1), dtype=float).repeat(1, solver.n_geoms) * (max_f - min_f) + min_f
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), envs)

    def _randomize_base_mass(self, envs=None) -> None:
        """Randomize base link mass."""
        min_m, max_m = self.cfg.dr.mass
        added_mass = gs.rand((len(envs), 1), dtype=float) * (max_m - min_m) + min_m
        self.rigid_solver.set_links_mass_shift(added_mass, [1], envs)

    def _randomize_com_displacement(self, envs) -> None:
        """Randomize base COM displacement."""
        min_d, max_d = self.cfg.dr.com
        com_shift = gs.rand((len(envs), 1, 3), dtype=float) * (max_d - min_d) + min_d
        self.rigid_solver.set_links_COM_shift(com_shift, [1], envs)

    def dr(self, envs=None) -> None:
        """Apply enabled randomizations."""

        if envs is None:
            envs = torch.arange(self.n_envs, device=self.device)
        if self.cfg.dr.friction:
            self._randomize_friction(envs)
        if self.cfg.dr.mass:
            self._randomize_base_mass(envs)
        if self.cfg.dr.com:
            self._randomize_com_displacement(envs)

    def _push(self, body):
        """Random pushes the body (robot). Emulates an impulse by setting a randomized base velocity."""
        if self.push_interval_s > 0 and not self.debug:
            max_push_vel_xy = self.cfg.dr.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = body.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            body.set_dofs_velocity(dofs_vel)
