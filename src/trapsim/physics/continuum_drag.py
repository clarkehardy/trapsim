"""trapsim.physics.continuum_drag  –  Auto-switching continuum drag.

Uses Schiller-Naumann drag (via accel) when Re > re_crossover, and Stokes
drag (via damping_rate) when Re <= re_crossover.  This means:

  - High-Re (free fall): correct inertial drag, no damping_rate contribution
    → Langevin is effectively off (env.total_damping_rate ≈ 0).
  - Low-Re (trapped): Stokes damping_rate feeds Langevin correctly via the
    fluctuation-dissipation theorem.

The two regimes match to within ~15% at Re = 1 (the default crossover),
since C_D = (24/Re)(1 + 0.15 Re^0.687) → 1.15 × Stokes at Re = 1.
"""

from __future__ import annotations

import math
import numpy as np

from .base import Physics


class ContinuumDrag(Physics):
    """Drag for a sphere across the full continuum regime.

    Schiller-Naumann at Re > `re_crossover`, Stokes at Re <= `re_crossover`.
    Pair with ``Langevin`` for thermal noise: it activates automatically in
    the Stokes regime and is negligible in the inertial regime.

    Parameters
    ----------
    rho_gas_kg_m3 : float
        Gas density [kg/m³].  Air at 25 °C, 1 atm: 1.184.
    eta_pa_s : float
        Dynamic viscosity [Pa·s].  Air at 25 °C: 1.85e-5.
    re_crossover : float
        Reynolds number below which the model switches from
        Schiller-Naumann to Stokes.  Default 1.0.
    """

    def __init__(self, rho_gas_kg_m3: float, eta_pa_s: float,
                 re_crossover: float = 1.0):
        self.rho  = float(rho_gas_kg_m3)
        self.eta  = float(eta_pa_s)
        self.re_c = float(re_crossover)

    def _re(self, r_m: float, speed_mps: float) -> float:
        return 2.0 * r_m * self.rho * speed_mps / self.eta

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        r_m  = env.particle["radius_m"]
        m_kg = env.particle["mass_kg"]
        v_mps  = vel_mm_us * 1e3
        speed  = float(np.linalg.norm(v_mps))
        if speed < 1e-15 or self._re(r_m, speed) <= self.re_c:
            return np.zeros(3)
        Re  = self._re(r_m, speed)
        C_D = (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)
        F   = 0.5 * self.rho * np.pi * r_m ** 2 * speed ** 2 * C_D
        return -(F / m_kg * 1e-9) * (v_mps / speed)  # mm/µs²

    def damping_rate(self, t_us, pos_mm, vel_mm_us, env):
        r_m  = env.particle["radius_m"]
        m_kg = env.particle["mass_kg"]
        v_mps  = vel_mm_us * 1e3
        speed  = float(np.linalg.norm(v_mps))
        if speed > 1e-15 and self._re(r_m, speed) > self.re_c:
            return 0.0
        return (6.0 * math.pi * self.eta * r_m / m_kg) * 1e-6  # µs⁻¹
