"""trapsim.physics  –  Pluggable force / damping / noise modules.

Each module subclasses `Physics` and overrides one or more hooks:

    accel(t_us, pos_mm, vel_mm_us, env)
        Deterministic acceleration [mm/µs²] summed into the RK4/5 RHS.
    damping_rate(t_us, pos_mm, vel_mm_us, env)
        Linear damping coefficient γ [1/µs].  All providers are summed
        into a total γ that the integrator applies via the exact
        exponential factor `v ← exp(-γ·dt) · v` after each accepted step.
        Langevin uses the same total γ for fluctuation-dissipation noise.
    kick(dt_us, t_us, pos_mm, vel_mm_us, env)
        Stochastic Δv [mm/µs] applied after each accepted step.

`env` exposes:
    env.particle           dict with 'mass_kg', 'charge_C', 'radius_m'
    env.voltages           dict {electrode_name: V_now}
    env.field(pos_mm)      returns (Ex, Ey, Ez) in V/mm at pos
    env.trigger_state      per-particle dict {trigger_name: t_fire_us or None}
    env.total_damping_rate γ_total summed across physics (read inside kick)
    env.rng                numpy Generator for stochastic methods
"""

from .base import Physics
from .electrostatic import Electrostatic
from .gravity import Gravity
from .epstein_drag import EpsteinDrag
from .continuum_drag import ContinuumDrag
from .langevin import Langevin
from .magnetic import Magnetic, BFieldSource, UniformField, ScalarPotentialPA

__all__ = [
    "Physics",
    "Electrostatic",
    "Gravity",
    "EpsteinDrag",
    "ContinuumDrag",
    "Langevin",
    "Magnetic",
    "BFieldSource",
    "UniformField",
    "ScalarPotentialPA",
]
