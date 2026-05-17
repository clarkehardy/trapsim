"""Smoke tests for trapsim.physics.magnetic — Lorentz force unit handling
and ScalarPotentialPA gradient."""

import math
import os
import struct

import numpy as np
import pytest

from trapsim.physics.magnetic import (
    Magnetic, BFieldSource, UniformField, ScalarPotentialPA)


class FakeEnv:
    """Minimal env object for the Magnetic plugin."""
    def __init__(self, particle, world_offset=(0.0, 0.0, 0.0)):
        self.particle = particle
        self._world_offset = world_offset


PARTICLE = {
    "charge_C": 1.602176634e-17,    # 100 elementary charges
    "mass_kg":  5.3e-18,            # ~166 nm silica nanosphere
}


# ── UniformField ──────────────────────────────────────────────────────────────

class TestUniformField:
    def test_returns_configured_vector(self):
        src = UniformField([0.1, 0.0, 0.0])
        env = FakeEnv(PARTICLE)
        B = src.B(0.0, np.array([1.0, 2.0, 3.0]), env)
        np.testing.assert_allclose(B, [0.1, 0.0, 0.0])

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            UniformField([0.1, 0.2])


# ── Magnetic plugin: unit conversion ──────────────────────────────────────────

class TestMagneticLorentz:
    def test_perpendicular_motion(self):
        """v = v_z ẑ, B = B_y ŷ ⇒ F = q(v × B) = q v_z B_y (ẑ × ŷ) = -q v_z B_y x̂
        a [mm/µs²] = (q/m) v_mm_us B_T · 1e-6"""
        v_mm_us = 5.0
        B_T = 0.1
        src = UniformField([0.0, B_T, 0.0])
        mag = Magnetic(source=src)
        env = FakeEnv(PARTICLE)
        a = mag.accel(0.0, np.zeros(3),
                      np.array([0.0, 0.0, v_mm_us]), env)
        expected_ax = -(PARTICLE["charge_C"] / PARTICLE["mass_kg"]
                        * v_mm_us * B_T * 1e-6)
        np.testing.assert_allclose(a, [expected_ax, 0.0, 0.0], rtol=1e-12)

    def test_parallel_motion_zero_force(self):
        """v ∥ B ⇒ v × B = 0."""
        src = UniformField([0.0, 0.0, 0.1])
        mag = Magnetic(source=src)
        env = FakeEnv(PARTICLE)
        a = mag.accel(0.0, np.zeros(3), np.array([0.0, 0.0, 5.0]), env)
        np.testing.assert_allclose(a, [0.0, 0.0, 0.0], atol=1e-30)

    def test_zero_velocity(self):
        src = UniformField([0.1, 0.2, 0.3])
        mag = Magnetic(source=src)
        env = FakeEnv(PARTICLE)
        a = mag.accel(0.0, np.zeros(3), np.zeros(3), env)
        np.testing.assert_allclose(a, [0.0, 0.0, 0.0])

    def test_rejects_non_bfieldsource(self):
        with pytest.raises(TypeError):
            Magnetic(source=lambda t, pos, env: np.zeros(3))


# ── ScalarPotentialPA: gradient of a known ψ field ────────────────────────────
#
# Construct a tiny magfield.pa whose ψ varies linearly with z:
#   ψ(z) = α · (z - z0)
# Then B = -∇ψ should be (0, 0, -α) everywhere in the grid interior.

def _write_magfield_pa(path, NX, NY, NZ, dx, psi_values):
    """Write a magnetic-mode PA file: header + psi*SCALE_REF, no sentinels."""
    SCALE_REF = 100000.0
    with open(path, "wb") as f:
        f.write(struct.pack("<i", -2))
        f.write(struct.pack("<i",  1))
        f.write(struct.pack("<d", SCALE_REF))
        f.write(struct.pack("<i", NX))
        f.write(struct.pack("<i", NY))
        f.write(struct.pack("<i", NZ))
        f.write(struct.pack("<i", 1600))
        f.write(struct.pack("<d", dx))
        f.write(struct.pack("<d", dx))
        f.write(struct.pack("<d", dx))
        (psi_values * SCALE_REF).astype("<f8").tofile(f)


class TestScalarPotentialPA:
    def test_linear_psi_gives_uniform_B(self, tmp_path):
        NX, NY, NZ = 10, 12, 14
        dx = 0.5                # mm
        wox, woy, woz = 0.0, 0.0, 0.0
        alpha = 0.03            # T (gradient slope = alpha → B_z = -alpha)

        # ψ(x,y,z) = α·z (z is the slowest axis)
        z = np.arange(NZ) * dx
        psi = np.broadcast_to(
            alpha * z[:, None, None],
            (NZ, NY, NX),
        ).copy()

        pa_path = tmp_path / "magfield.pa"
        _write_magfield_pa(str(pa_path), NX, NY, NZ, dx, psi)

        src = ScalarPotentialPA(str(pa_path))
        env = FakeEnv(PARTICLE, world_offset=(wox, woy, woz))

        # Sample a few interior points; B should be (0, 0, -alpha) everywhere.
        for px, py, pz in [
            (1.0, 1.0, 1.0),
            (1.25, 2.5, 3.75),
            (3.5, 4.5, 5.25),
        ]:
            B = src.B(0.0, np.array([px, py, pz]), env)
            np.testing.assert_allclose(B, [0.0, 0.0, -alpha],
                                       atol=1e-12,
                                       err_msg=f"at ({px},{py},{pz})")

    def test_rejects_electrode_pa(self, tmp_path):
        """read_magfield should fail loudly on an electrode-style PA file
        (one containing 'this electrode' sentinel values)."""
        NX, NY, NZ = 4, 4, 4
        dx = 1.0
        # An electrode PA has values ≥ 1.5·SCALE_REF at electrode voxels
        SCALE_REF = 100000.0
        raw = np.zeros(NX * NY * NZ, dtype="<f8")
        raw[0] = 2.0 * SCALE_REF + 1  # one fake electrode voxel
        with open(tmp_path / "bad.pa", "wb") as f:
            f.write(struct.pack("<i", -2))
            f.write(struct.pack("<i",  1))
            f.write(struct.pack("<d", SCALE_REF))
            f.write(struct.pack("<i", NX))
            f.write(struct.pack("<i", NY))
            f.write(struct.pack("<i", NZ))
            f.write(struct.pack("<i", 1600))
            f.write(struct.pack("<d", dx))
            f.write(struct.pack("<d", dx))
            f.write(struct.pack("<d", dx))
            raw.tofile(f)
        with pytest.raises(IOError, match="electrode"):
            ScalarPotentialPA(str(tmp_path / "bad.pa"))
