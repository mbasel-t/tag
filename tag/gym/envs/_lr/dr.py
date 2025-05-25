import genesis as gs


def _randomize_friction(self, env_ids=None):
    """Randomize friction of all links"""
    min_friction, max_friction = self.cfg.domain_rand.friction_range

    solver = self.rigid_solver

    ratios = (
        gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction) + min_friction
    )
    solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)


def _randomize_base_mass(self, env_ids=None):
    """Randomize base mass"""
    min_mass, max_mass = self.cfg.domain_rand.added_mass_range
    base_link_id = 1
    added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
    self.rigid_solver.set_links_mass_shift(
        added_mass,
        [
            base_link_id,
        ],
        env_ids,
    )


def _randomize_com_displacement(self, env_ids):
    min_displacement, max_displacement = self.cfg.domain_rand.com_displacement_range
    base_link_id = 1

    com_displacement = (
        gs.rand((len(env_ids), 1, 3), dtype=float) * (max_displacement - min_displacement) + min_displacement
    )
    # com_displacement[:, :, 0] -= 0.02

    self.rigid_solver.set_links_COM_shift(
        com_displacement,
        [
            base_link_id,
        ],
        env_ids,
    )
