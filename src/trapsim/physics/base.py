"""trapsim.physics.base  –  Physics interface."""

from __future__ import annotations

import numpy as np

ZERO3 = np.zeros(3)


class Physics:
    """Base class for force/damping/noise modules.

    Override any subset of the three hooks below.  The integrator calls
    `accel` inside each RK4/5 stage; `damping_rate` once per accepted
    step (summed across all physics); and `kick` once per accepted step
    (each physics contributes independently).

    Units: position in mm, velocity in mm/µs, acceleration in mm/µs²,
    damping rate in 1/µs, time in µs.
    """

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        """Return a length-3 ndarray of acceleration [mm/µs²]."""
        return ZERO3

    def damping_rate(self, t_us, pos_mm, vel_mm_us, env):
        """Return scalar γ [1/µs] (linear-damping coefficient)."""
        return 0.0

    def kick(self, dt_us, t_us, pos_mm, vel_mm_us, env):
        """Return a length-3 ndarray of Δv [mm/µs]."""
        return ZERO3
