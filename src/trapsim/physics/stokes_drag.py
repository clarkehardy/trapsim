"""trapsim.physics.stokes_drag  –  Stokes (viscous) drag.

Valid in the continuum, low-Reynolds-number regime (Kn << 1, Re << 1).
Rule of thumb: use when Re < 0.1; acceptable up to Re ~ 1.

F_drag = -6π η r v
γ      = 6π η r / m   [s⁻¹]

The integrator applies γ as an exact exponential factor on velocity each
step; do NOT override accel() for drag or it will be double-counted.
"""

from __future__ import annotations

import math

from .base import Physics


class StokesDrag(Physics):
    """Linear (Stokes) drag for a sphere in the continuum regime.

    Parameters
    ----------
    eta_pa_s : float
        Dynamic viscosity of the surrounding gas [Pa·s].
        Air at 25 °C: 1.85e-5.  Air at 20 °C: 1.81e-5.
    """

    def __init__(self, eta_pa_s: float):
        self.eta = float(eta_pa_s)

    def damping_rate(self, t_us, pos_mm, vel_mm_us, env):
        r_m  = env.particle["radius_m"]
        m_kg = env.particle["mass_kg"]
        # 6πηr/m [s⁻¹] converted to [µs⁻¹]
        return (6.0 * math.pi * self.eta * r_m / m_kg) * 1e-6