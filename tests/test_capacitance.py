"""Tests for trapsim.capacitance — Maxwell capacitance matrix extraction.

Exercises the pure-math core (`_capacitance_from_arrays` and
`_node_eps_from_cell_eps`) directly with synthesised arrays, avoiding any
dependency on the SOR solver, the C++ build, or trimesh.

The parallel-plate test uses analytically constructed phi fields that exactly
satisfy the discrete Laplace equation, so the discrete capacitance recovers
ε₀ A / d at machine precision rather than at solver-convergence tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from trapsim.capacitance import (
    EPS0_F_PER_M,
    _capacitance_from_arrays,
    _node_eps_from_cell_eps,
    mutual_capacitance,
    self_capacitance,
)


# ── Parallel-plate construction ───────────────────────────────────────────────
# Two thin perfect-conductor plates spanning the full xy cross-section at
# z-layers k1 and k2 (k1 < k2).  With floating outer-face BCs and no other
# conductors, the unit-V solution is piecewise linear in z:
#   plate 1 at 1 V:   φ = 1   for k ≤ k1
#                     φ = (k2 - k) / (k2 - k1)   for k1 < k < k2
#                     φ = 0   for k ≥ k2
# plate 2 at 1 V is symmetric (φ = 1 - φ_plate1_at_1V).

def _build_parallel_plate_arrays(NX=12, NY=12, NZ=24, k1=6, k2=18):
    mask_stack = np.zeros((2, NZ, NY, NX), dtype=bool)
    mask_stack[0, k1, :, :] = True
    mask_stack[1, k2, :, :] = True

    z = np.arange(NZ, dtype=np.float64)
    phi1_1d = np.where(z <= k1, 1.0,
              np.where(z >= k2, 0.0,
                       (k2 - z) / (k2 - k1)))
    phi_stack = np.empty((2, NZ, NY, NX), dtype=np.float64)
    phi_stack[0] = phi1_1d[:, None, None]
    phi_stack[1] = 1.0 - phi_stack[0]
    return phi_stack, mask_stack


class TestParallelPlates:
    """Closed-form sanity check: two fully-overlapping plates separated by
    N_gap voxels have C = ε₀ ε_r A / d, with A = (NX·NY) · dx² and
    d = (k2 - k1) · dx."""

    def test_vacuum_self_capacitance(self):
        NX, NY, NZ = 12, 12, 24
        k1, k2 = 6, 18
        dx_m = 0.5e-3                                # 0.5 mm
        d_m   = (k2 - k1) * dx_m
        area  = NX * NY * dx_m**2
        expected = EPS0_F_PER_M * area / d_m

        phi_stack, mask_stack = _build_parallel_plate_arrays(NX, NY, NZ, k1, k2)
        eps_node = np.ones((NZ, NY, NX), dtype=np.float64)
        C = _capacitance_from_arrays(phi_stack, mask_stack, eps_node, dx_m)

        assert C[0, 0] == pytest.approx(expected, rel=1e-12)
        assert C[1, 1] == pytest.approx(expected, rel=1e-12)

    def test_off_diagonal_equals_negative_self(self):
        """For the fully-overlapping two-plate geometry with no leakage
        to ground (every field line ends on the other plate), the strict
        identity C[0,0] + C[0,1] = 0 must hold."""
        NX, NY, NZ = 10, 10, 22
        k1, k2 = 6, 16
        dx_m = 1e-4
        phi_stack, mask_stack = _build_parallel_plate_arrays(NX, NY, NZ, k1, k2)
        eps_node = np.ones((NZ, NY, NX), dtype=np.float64)
        C = _capacitance_from_arrays(phi_stack, mask_stack, eps_node, dx_m)

        assert C[0, 0] > 0
        assert C[1, 1] > 0
        assert C[0, 1] < 0
        assert C[1, 0] < 0
        # reciprocity (exact for symmetric ε and symmetric construction)
        assert C[0, 1] == pytest.approx(C[1, 0], rel=1e-12)
        # all-at-1V gives zero charge → row sums vanish
        assert (C[0, 0] + C[0, 1]) == pytest.approx(0.0, abs=1e-22)
        assert (C[1, 0] + C[1, 1]) == pytest.approx(0.0, abs=1e-22)

    def test_epsilon_scales_capacitance(self):
        """C ∝ ε_r (filled-dielectric limit) when the relative permittivity
        is uniform everywhere."""
        NX, NY, NZ = 10, 10, 22
        k1, k2 = 6, 16
        dx_m = 1e-4
        phi_stack, mask_stack = _build_parallel_plate_arrays(NX, NY, NZ, k1, k2)
        eps_uniform = np.ones((NZ, NY, NX), dtype=np.float64)
        C1 = _capacitance_from_arrays(phi_stack, mask_stack, eps_uniform, dx_m)
        C4 = _capacitance_from_arrays(phi_stack, mask_stack, 4.0 * eps_uniform, dx_m)
        np.testing.assert_allclose(C4, 4.0 * C1, rtol=1e-12)

    def test_distance_scaling(self):
        """Halving the gap doubles the capacitance."""
        NX, NY, NZ = 10, 10, 30
        dx_m = 1e-4
        eps_node = np.ones((NZ, NY, NX), dtype=np.float64)

        # near plates: gap = 5 layers
        phi_a, mask_a = _build_parallel_plate_arrays(NX, NY, NZ, 10, 15)
        Ca = _capacitance_from_arrays(phi_a, mask_a, eps_node, dx_m)
        # far plates: gap = 10 layers
        phi_b, mask_b = _build_parallel_plate_arrays(NX, NY, NZ, 10, 20)
        Cb = _capacitance_from_arrays(phi_b, mask_b, eps_node, dx_m)

        assert Ca[0, 0] == pytest.approx(2.0 * Cb[0, 0], rel=1e-12)


# ── Node-eps interpolation ───────────────────────────────────────────────────

class TestNodeEpsInterpolation:
    """`_node_eps_from_cell_eps` must reproduce the build_coef_node logic in
    `_solver/laplace.cpp`: arithmetic mean of the up-to-8 cells touching
    each node."""

    def test_uniform_returns_uniform(self):
        eps_cell = 2.7 * np.ones((4, 5, 6), dtype=np.float64)
        NX, NY, NZ = 7, 6, 5
        eps_node = _node_eps_from_cell_eps(eps_cell, NX, NY, NZ)
        assert eps_node.shape == (NZ, NY, NX)
        np.testing.assert_allclose(eps_node, 2.7, rtol=1e-15, atol=0)

    def test_interior_node_is_8_cell_mean(self):
        NX, NY, NZ = 5, 5, 5
        NXc, NYc, NZc = 4, 4, 4
        eps_cell = np.arange(NXc * NYc * NZc, dtype=np.float64
                             ).reshape(NZc, NYc, NXc)
        eps_node = _node_eps_from_cell_eps(eps_cell, NX, NY, NZ)
        # node (k=2, j=2, i=2) is the corner of cells {1,2} × {1,2} × {1,2}
        expected = eps_cell[1:3, 1:3, 1:3].mean()
        assert eps_node[2, 2, 2] == pytest.approx(expected, rel=0, abs=0)

    def test_corner_node_is_single_cell(self):
        NX, NY, NZ = 4, 4, 4
        NXc, NYc, NZc = 3, 3, 3
        eps_cell = np.arange(NXc * NYc * NZc, dtype=np.float64
                             ).reshape(NZc, NYc, NXc)
        eps_node = _node_eps_from_cell_eps(eps_cell, NX, NY, NZ)
        # node (0,0,0) touches only cell (0,0,0)
        assert eps_node[0, 0, 0] == eps_cell[0, 0, 0]
        # node (NZ-1, NY-1, NX-1) touches only cell (NZc-1, NYc-1, NXc-1)
        assert eps_node[NZ - 1, NY - 1, NX - 1] == eps_cell[NZc - 1, NYc - 1, NXc - 1]

    def test_rejects_bad_shape(self):
        eps_cell = np.ones((3, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="shape"):
            _node_eps_from_cell_eps(eps_cell, NX=5, NY=4, NZ=4)


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_mutual_capacitance(self):
        C = np.array([[2.0, -0.5], [-0.5, 3.0]])
        assert mutual_capacitance(C, 0, 1) == 0.5
        assert mutual_capacitance(C, 1, 0) == 0.5

    def test_mutual_capacitance_averages_asymmetric(self):
        C = np.array([[2.0, -0.6], [-0.4, 3.0]])
        assert mutual_capacitance(C, 0, 1) == pytest.approx(0.5)

    def test_mutual_capacitance_self_raises(self):
        with pytest.raises(ValueError):
            mutual_capacitance(np.eye(2), 1, 1)

    def test_self_capacitance(self):
        C = np.array([[2.0, -0.5, -0.5],
                      [-0.5, 3.0, -1.0],
                      [-0.5, -1.0, 4.0]])
        assert self_capacitance(C, 0) == 1.0
        assert self_capacitance(C, 1) == 1.5
        assert self_capacitance(C, 2) == 2.5
