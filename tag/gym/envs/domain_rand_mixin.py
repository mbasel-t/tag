"""Helpers for simulation domain randomization."""

from __future__ import annotations

import torch
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


class DomainRandMixin:
    """Mixin providing common domain randomizations."""

    def _randomize_friction(self, env_ids=None) -> None:
        """Randomize friction of all geoms."""
        min_f, max_f = self.cfg.domain_rand.friction_range
        solver = self.rigid_solver
        ratios = (
            gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms)
            * (max_f - min_f)
            + min_f
        )
        solver.set_geoms_friction_ratio(
            ratios, torch.arange(0, solver.n_geoms), env_ids
        )

    def _randomize_base_mass(self, env_ids=None) -> None:
        """Randomize base link mass."""
        min_m, max_m = self.cfg.domain_rand.added_mass_range
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_m - min_m) + min_m
        self.rigid_solver.set_links_mass_shift(added_mass, [1], env_ids)

    def _randomize_com_displacement(self, env_ids) -> None:
        """Randomize base COM displacement."""
        min_d, max_d = self.cfg.domain_rand.com_displacement_range
        com_shift = (
            gs.rand((len(env_ids), 1, 3), dtype=float) * (max_d - min_d) + min_d
        )
        self.rigid_solver.set_links_COM_shift(com_shift, [1], env_ids)

    def find_rigid_solver(self) -> RigidSolver:
        """Return the ``RigidSolver`` used by ``self.scene``."""
        for solver in self.scene.sim.solvers:
            if isinstance(solver, RigidSolver):
                self.rigid_solver = solver
                return solver
        raise RuntimeError("RigidSolver not found")

    def dr(self, env_ids=None) -> None:
        """Apply enabled randomizations."""
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        if getattr(self.cfg.domain_rand, "randomize_friction", False):
            self._randomize_friction(env_ids)
        if getattr(self.cfg.domain_rand, "randomize_base_mass", False):
            self._randomize_base_mass(env_ids)
        if getattr(self.cfg.domain_rand, "randomize_com_displacement", False):
            self._randomize_com_displacement(env_ids)
