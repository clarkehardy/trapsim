"""trapsim.physics.langevin  –  Thermal kick paired with any damping force.

Fluctuation–dissipation theorem: for a particle of mass m at temperature T
that loses energy at rate γ, the equilibrium Maxwell–Boltzmann velocity
distribution is maintained by random kicks of variance

    σ² = (k_B T / m) · (1 − exp(−2γ Δt))    per Cartesian component

applied after each accepted integration step.  γ here is the *total*
damping rate from all `Physics.damping_rate` providers, which the
integrator passes via `env.total_damping_rate`.
"""

from __future__ import annotations

import math
import numpy as np

from .base import Physics

KB_J = 1.38065e-23   # J / K


class Langevin(Physics):
    """Stochastic kick that thermalises the particle to `temperature_k`.

    Parameters
    ----------
    temperature_k : float
        Effective bath temperature [K].  Usually the same as the gas
        temperature used for EpsteinDrag.
    """

    def __init__(self, temperature_k: float):
        self.T = float(temperature_k)

    def kick(self, dt_us, t_us, pos_mm, vel_mm_us, env):
        gamma = env.total_damping_rate    # 1/µs
        if gamma * dt_us <= 1e-12:
            return np.zeros(3)
        m_kg = env.particle["mass_kg"]
        # variance in (m/s)²; convert to (mm/µs)² by × 1e-6 → σ × 1e-3
        var_mps2 = (KB_J * self.T / m_kg) * (1.0 - math.exp(-2.0 * gamma * dt_us))
        sigma_mm_us = math.sqrt(var_mps2) * 1e-3
        return sigma_mm_us * env.rng.standard_normal(3)
