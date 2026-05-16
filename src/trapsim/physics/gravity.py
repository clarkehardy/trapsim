"""trapsim.physics.gravity  –  Uniform gravitational acceleration."""

from __future__ import annotations

import numpy as np

from .base import Physics

# Standard gravity in mm/µs²: 9.81 m/s² · 1e-3 = 9.81e-9
G_STANDARD_MM_US2 = 9.81e-9

_AXIS_VEC = {
    "+x": np.array([ 1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([ 0.0, 1.0, 0.0]),
    "-y": np.array([ 0.0,-1.0, 0.0]),
    "+z": np.array([ 0.0, 0.0, 1.0]),
    "-z": np.array([ 0.0, 0.0,-1.0]),
}


class Gravity(Physics):
    """Constant gravitational acceleration in a chosen direction.

    Parameters
    ----------
    g_mm_us2 : float
        Magnitude of g in mm/µs².  Default 9.81e-9 (Earth surface).
    axis : str
        One of "+x", "-x", "+y", "-y", "+z", "-z".  Default "-y".
    """

    def __init__(self, g_mm_us2: float = G_STANDARD_MM_US2, axis: str = "-y"):
        if axis not in _AXIS_VEC:
            raise ValueError(f"axis must be one of {list(_AXIS_VEC)}, got {axis!r}")
        self._a = float(g_mm_us2) * _AXIS_VEC[axis]

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        return self._a
