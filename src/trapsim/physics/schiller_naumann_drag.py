"""trapsim.physics.schiller_naumann_drag  –  Schiller-Naumann drag.

Valid for a rigid sphere from Stokes flow through the inertial regime
(Re ~ 0 to ~800).  Reduces to Stokes drag as Re → 0.

    C_D  = (24 / Re) × (1 + 0.15 · Re^0.687)
    F    = ½ ρ_gas π r² v² C_D(Re)    [N, directed opposite to velocity]

Because drag is nonlinear in speed, it cannot be expressed as a linear
damping coefficient and is implemented via accel() rather than
damping_rate().  Langevin thermal noise, if used, will not see this drag
in env.total_damping_rate — but for particles large enough to need
Schiller-Naumann (r ≳ 1 µm at atmospheric pressure), thermal fluctuations
are negligible and Langevin can be omitted.
"""

from __future__ import annotations

import numpy as np

from .base import Physics


class SchillerNaumannDrag(Physics):
    """Drag for a sphere in the inertial regime (Re up to ~800).

    Parameters
    ----------
    rho_gas_kg_m3 : float
        Gas density [kg/m³].  Air at 25 °C, 1 atm: 1.184.
    eta_pa_s : float
        Dynamic viscosity [Pa·s].  Air at 25 °C: 1.85e-5.
    """

    def __init__(self, rho_gas_kg_m3: float, eta_pa_s: float):
        self.rho = float(rho_gas_kg_m3)
        self.eta = float(eta_pa_s)

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        r_m  = env.particle["radius_m"]
        m_kg = env.particle["mass_kg"]

        v_mps  = vel_mm_us * 1e3          # mm/µs → m/s
        speed  = float(np.linalg.norm(v_mps))
        if speed < 1e-15:
            return np.zeros(3)

        Re  = 2.0 * r_m * self.rho * speed / self.eta
        C_D = (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)

        F_mag = 0.5 * self.rho * np.pi * r_m ** 2 * speed ** 2 * C_D  # N
        a_mps2 = F_mag / m_kg                                           # m/s²
        # Convert m/s² → mm/µs²: × 1e3 (m→mm) / (1e6)² (s²→µs²) = × 1e-9
        return -(a_mps2 * 1e-9) * (v_mps / speed)  # opposite to v
