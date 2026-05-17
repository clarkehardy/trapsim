"""trapsim.physics.magnetic  –  Lorentz force from a pluggable B-field source.

The `Magnetic` plugin computes a = (q/m) (v × B) and adds it to the integrator.
The `B(t, pos)` field comes from any subclass of `BFieldSource`:

    UniformField(B_T)             constant background
    ScalarPotentialPA(path)       B = -∇ψ from a magfield.pa file (output of
                                  trapsim refine in magnetic mode)

Adding more sources (Biot-Savart loops, imported grids, ...) only needs a
new BFieldSource subclass; Magnetic.accel() is unchanged.

Unit handling (matches `Electrostatic`):
    v [mm/µs] = 1e3 × v [m/s]
    B [T] is SI.
    a [mm/µs²] = (q [C] / m [kg]) × (v_mm_us × B_T) × 1e-6
"""

from __future__ import annotations

import os
import numpy as np

from .base import Physics


class BFieldSource:
    """Return B(t, pos) in Tesla.  Override `B()`."""

    def B(self, t_us, pos_mm, env):
        raise NotImplementedError


class UniformField(BFieldSource):
    """Spatially constant B [Tesla]."""

    def __init__(self, B_T):
        self._B = np.asarray(B_T, dtype=np.float64)
        if self._B.shape != (3,):
            raise ValueError(
                f"UniformField B_T must be a length-3 vector; got shape "
                f"{self._B.shape}")

    def B(self, t_us, pos_mm, env):
        return self._B


class ScalarPotentialPA(BFieldSource):
    """B = -∇ψ from a magnetic-potential PA file (`magfield.pa`).

    ψ is stored in units of T·mm (the solver convention — see
    `voxelize.build_magnetic_source` for derivation), so the gradient
    against grid coords in mm gives B directly in Tesla.
    """

    def __init__(self, path: str):
        from ..io.pa import read_magfield   # avoid circular import at module load
        psi, NX, NY, NZ, dx = read_magfield(os.fspath(path))
        self._psi = psi             # (NZ, NY, NX)
        self._NX  = int(NX)
        self._NY  = int(NY)
        self._NZ  = int(NZ)
        self._dx  = float(dx)
        self._wo  = None            # (wox, woy, woz), cached on first call

    def _trilinear_grad(self, px, py, pz):
        """Return (∂ψ/∂x, ∂ψ/∂y, ∂ψ/∂z) at grid-index coords px,py,pz."""
        NX, NY, NZ = self._NX, self._NY, self._NZ
        dx  = self._dx
        psi = self._psi

        fx = max(0.0, min(px, NX - 1.0001))
        fy = max(0.0, min(py, NY - 1.0001))
        fz = max(0.0, min(pz, NZ - 1.0001))
        i0 = int(fx); wx = fx - i0; i0 = min(i0, NX - 2)
        j0 = int(fy); wy = fy - j0; j0 = min(j0, NY - 2)
        k0 = int(fz); wz = fz - k0; k0 = min(k0, NZ - 2)
        ox, oy, oz = 1.0 - wx, 1.0 - wy, 1.0 - wz

        c000 = psi[k0,   j0,   i0  ]
        c100 = psi[k0,   j0,   i0+1]
        c010 = psi[k0,   j0+1, i0  ]
        c110 = psi[k0,   j0+1, i0+1]
        c001 = psi[k0+1, j0,   i0  ]
        c101 = psi[k0+1, j0,   i0+1]
        c011 = psi[k0+1, j0+1, i0  ]
        c111 = psi[k0+1, j0+1, i0+1]

        dpsi_dx = (oy*oz*(c100-c000) + wy*oz*(c110-c010) +
                   oy*wz*(c101-c001) + wy*wz*(c111-c011)) / dx
        dpsi_dy = (ox*oz*(c010-c000) + wx*oz*(c110-c100) +
                   ox*wz*(c011-c001) + wx*wz*(c111-c101)) / dx
        dpsi_dz = (ox*oy*(c001-c000) + wx*oy*(c101-c100) +
                   ox*wy*(c011-c010) + wx*wy*(c111-c110)) / dx
        return dpsi_dx, dpsi_dy, dpsi_dz

    def B(self, t_us, pos_mm, env):
        if self._wo is None:
            self._wo = env._world_offset
        wox, woy, woz = self._wo
        dx = self._dx
        px = (pos_mm[0] - wox) / dx
        py = (pos_mm[1] - woy) / dx
        pz = (pos_mm[2] - woz) / dx
        gx, gy, gz = self._trilinear_grad(px, py, pz)
        return np.array([-gx, -gy, -gz])


class Magnetic(Physics):
    """Lorentz acceleration on a single charged particle.

    `source` must be a `BFieldSource` (UniformField, ScalarPotentialPA, ...).
    """

    def __init__(self, source: BFieldSource):
        if not isinstance(source, BFieldSource):
            raise TypeError(
                f"Magnetic source must subclass BFieldSource; got {type(source)}")
        self.source = source

    def accel(self, t_us, pos_mm, vel_mm_us, env):
        B = self.source.B(t_us, pos_mm, env)
        q_C  = env.particle["charge_C"]
        m_kg = env.particle["mass_kg"]
        scale = q_C / m_kg * 1e-6
        return scale * np.cross(vel_mm_us, B)
