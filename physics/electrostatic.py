"""trapsim.physics.electrostatic  –  q·E/m from interpolated electrode fields."""

from __future__ import annotations

import numpy as np

from .base import Physics


class Electrostatic(Physics):
    """Coulomb acceleration on a single charged particle.

    Reads E(pos) from `env.field(pos)` (units V/mm) and returns
    q·E/m in mm/µs².  Conversion:
        a [m/s²]    = q [C] · E [V/m]    / m [kg]
        E [V/m]     = 1000 · E [V/mm]
        a [mm/µs²]  = a [m/s²] · 1e-3    (1 m/s² = 1e-3 mm/µs²)
    Net factor:  a [mm/µs²] = q [C] · E [V/mm] / m [kg]
    """

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        Ex, Ey, Ez = env.field(pos_mm)
        q_C  = env.particle["charge_C"]
        m_kg = env.particle["mass_kg"]
        scale = q_C / m_kg
        return np.array([Ex * scale, Ey * scale, Ez * scale])
