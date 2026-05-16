"""trapsim.physics.epstein_drag  –  Free-molecular (Epstein) drag.

Valid in the Kn >> 1 regime (background-gas mean free path much larger
than the particle radius), which is the case for sub-µm spheres at sub-bar
pressures.

γ = (8π/3) · r² · P / (m · c̄)        [s⁻¹]
c̄ = sqrt(8 k_B T / (π · M_gas))     mean thermal speed [m/s]

The integrator applies γ as an exact exponential factor on velocity each
step; this Physics also exposes a tiny acceleration `-γ·v` so that the
RK4/5 error estimator sees the drag (otherwise it would always pick a
needlessly small dt to "resolve" the unmodelled drag transient).  Actually
we do NOT add -γ·v to accel because that double-counts when combined with
the exact factor — instead we report `damping_rate` and the integrator
handles the velocity decay exactly.
"""

from __future__ import annotations

import math
from typing import Optional

from .base import Physics

KB_J   = 1.38065e-23            # J / K
AMU_KG = 1.66054e-27            # kg per amu


class EpsteinDrag(Physics):
    """Free-molecular drag in a background gas.

    Parameters
    ----------
    pressure_pa : float
        Baseline gas pressure [Pa].
    temperature_k : float
        Gas temperature [K].
    gas_mass_amu : float
        Molar mass of the gas [amu] (N₂ = 28.0).
    pressure_ramp : dict, optional
        Linear pressure ramp.  Keys:
          - "trigger" : str   — trigger name that fires the ramp
          - "p_final_pa" : float
          - "duration_us" : float
        Before the trigger fires the pressure is `pressure_pa`.  Between
        t_fire and t_fire + duration_us the pressure ramps linearly to
        `p_final_pa`, then stays there.
    scale : float
        Multiply the drag rate by this factor (1.0 = real drag,
        0.0 = drag off).  Useful for sensitivity studies.
    """

    def __init__(self, pressure_pa: float, temperature_k: float,
                 gas_mass_amu: float = 28.0,
                 pressure_ramp: Optional[dict] = None,
                 scale: float = 1.0):
        self.p_baseline = float(pressure_pa)
        self.T          = float(temperature_k)
        self.M_kg       = float(gas_mass_amu) * AMU_KG
        self.scale      = float(scale)
        self.ramp       = pressure_ramp     # validated on first call
        self.c_bar      = math.sqrt(8.0 * KB_J * self.T / (math.pi * self.M_kg))
        # _per_pa_per_kg_per_r2: convert P[Pa] · r[m]² / m[kg] → γ[1/µs]
        # (8π/3) · 1e-6 (s → µs)
        self._coeff = (8.0 * math.pi / 3.0) / self.c_bar * 1e-6

    # ─── public helpers ────────────────────────────────────────────────
    def pressure_at(self, t_us: float, trigger_state: dict) -> float:
        if self.ramp is None:
            return self.p_baseline
        t_fire = trigger_state.get(self.ramp["trigger"])
        if t_fire is None:
            return self.p_baseline
        elapsed = t_us - t_fire
        if elapsed <= 0.0:
            return self.p_baseline
        if elapsed >= self.ramp["duration_us"]:
            return float(self.ramp["p_final_pa"])
        frac = elapsed / self.ramp["duration_us"]
        return self.p_baseline + frac * (self.ramp["p_final_pa"] - self.p_baseline)

    # ─── Physics hook ──────────────────────────────────────────────────
    def damping_rate(self, t_us, pos_mm, vel_mm_us, env):
        r_m  = env.particle["radius_m"]
        m_kg = env.particle["mass_kg"]
        P    = self.pressure_at(t_us, env.trigger_state)
        return self.scale * self._coeff * r_m * r_m / m_kg * P
