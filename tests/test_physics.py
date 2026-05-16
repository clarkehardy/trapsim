"""Smoke tests for trapsim.physics — EpsteinDrag, Gravity, Langevin."""

import math

import numpy as np
import pytest

from trapsim.physics.epstein_drag import EpsteinDrag, KB_J, AMU_KG
from trapsim.physics.gravity import Gravity, G_STANDARD_MM_US2
from trapsim.physics.langevin import Langevin


# ── Test fixtures ──────────────────────────────────────────────────────────────

class FakeEnv:
    """Minimal env object for physics hooks."""
    def __init__(self, particle, trigger_state=None, rng=None,
                 total_damping_rate=0.0):
        self.particle = particle
        self.trigger_state = trigger_state or {}
        self.rng = rng or np.random.default_rng(42)
        self.total_damping_rate = total_damping_rate


PARTICLE = {
    "radius_m": 83e-9,
    "density_kgm3": 2200.0,
    "charge_e": 100,
    "mass_kg": 2200.0 * (4.0 / 3.0) * math.pi * (83e-9) ** 3,
}


# ── EpsteinDrag ────────────────────────────────────────────────────────────────

class TestEpsteinDrag:
    def test_damping_rate_formula(self):
        """γ = (8π/3) · r² · P / (m · c̄) · 1e-6  [µs⁻¹]"""
        P = 100.0  # Pa
        T = 300.0
        M_gas = 28.0
        drag = EpsteinDrag(pressure_pa=P, temperature_k=T, gas_mass_amu=M_gas)

        env = FakeEnv(PARTICLE)
        gamma = drag.damping_rate(0.0, np.zeros(3), np.zeros(3), env)

        # Compute expected value from first principles
        c_bar = math.sqrt(8.0 * KB_J * T / (math.pi * M_gas * AMU_KG))
        r = PARTICLE["radius_m"]
        m = PARTICLE["mass_kg"]
        expected = (8.0 * math.pi / 3.0) * r**2 * P / (m * c_bar) * 1e-6

        assert gamma == pytest.approx(expected, rel=1e-10)

    def test_scale_factor(self):
        drag_full = EpsteinDrag(pressure_pa=100.0, temperature_k=300.0)
        drag_half = EpsteinDrag(pressure_pa=100.0, temperature_k=300.0, scale=0.5)
        env = FakeEnv(PARTICLE)
        g_full = drag_full.damping_rate(0.0, np.zeros(3), np.zeros(3), env)
        g_half = drag_half.damping_rate(0.0, np.zeros(3), np.zeros(3), env)
        assert g_half == pytest.approx(0.5 * g_full)

    def test_pressure_ramp(self):
        ramp = {"trigger": "release", "p_final_pa": 10.0, "duration_us": 1000.0}
        drag = EpsteinDrag(pressure_pa=100.0, temperature_k=300.0,
                           pressure_ramp=ramp)
        # Before trigger fires
        assert drag.pressure_at(500.0, {"release": None}) == 100.0
        # Midway through ramp (t_fire=200, evaluate at t=700 → elapsed=500)
        p_mid = drag.pressure_at(700.0, {"release": 200.0})
        expected_mid = 100.0 + 0.5 * (10.0 - 100.0)  # = 55
        assert p_mid == pytest.approx(expected_mid)
        # After ramp completes
        p_end = drag.pressure_at(5000.0, {"release": 200.0})
        assert p_end == pytest.approx(10.0)


# ── Gravity ────────────────────────────────────────────────────────────────────

class TestGravity:
    def test_default_minus_y(self):
        g = Gravity()
        env = FakeEnv(PARTICLE)
        a = g.accel(0.0, np.zeros(3), np.zeros(3), env)
        assert a[0] == 0.0
        assert a[1] == pytest.approx(-G_STANDARD_MM_US2)
        assert a[2] == 0.0

    def test_plus_z(self):
        g = Gravity(axis="+z")
        env = FakeEnv(PARTICLE)
        a = g.accel(0.0, np.zeros(3), np.zeros(3), env)
        assert a[2] == pytest.approx(G_STANDARD_MM_US2)
        assert a[0] == 0.0
        assert a[1] == 0.0

    def test_custom_magnitude(self):
        g = Gravity(g_mm_us2=1.0, axis="-x")
        env = FakeEnv(PARTICLE)
        a = g.accel(0.0, np.zeros(3), np.zeros(3), env)
        np.testing.assert_allclose(a, [-1.0, 0.0, 0.0])

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis must be one of"):
            Gravity(axis="up")


# ── Langevin ───────────────────────────────────────────────────────────────────

class TestLangevin:
    def test_kick_variance(self):
        """Over many samples, kick variance should match the FDT prediction."""
        T = 300.0
        gamma = 1e-4  # 1/µs
        dt = 100.0    # µs
        m = PARTICLE["mass_kg"]

        lang = Langevin(temperature_k=T)
        rng = np.random.default_rng(123)
        env = FakeEnv(PARTICLE, total_damping_rate=gamma, rng=rng)

        n_samples = 50_000
        kicks = np.array([
            lang.kick(dt, 0.0, np.zeros(3), np.zeros(3), env)
            for _ in range(n_samples)
        ])

        # Expected variance per component in (mm/µs)²
        var_mps2 = (KB_J * T / m) * (1.0 - math.exp(-2.0 * gamma * dt))
        expected_var = var_mps2 * 1e-6  # (mm/µs)²

        measured_var = np.var(kicks, axis=0)
        for i in range(3):
            assert measured_var[i] == pytest.approx(expected_var, rel=0.05)

    def test_zero_damping_gives_zero_kick(self):
        lang = Langevin(temperature_k=300.0)
        env = FakeEnv(PARTICLE, total_damping_rate=0.0)
        kick = lang.kick(100.0, 0.0, np.zeros(3), np.zeros(3), env)
        np.testing.assert_array_equal(kick, np.zeros(3))
