"""trapsim.capacitance  –  Maxwell capacitance matrix from PA files.

After ``trapsim refine`` has solved the per-electrode unit-voltage potentials
(``solver/field.pa<j>``), the Maxwell capacitance matrix C ∈ ℝ^{N×N} is a
pure post-processing step:

    Q_i^(j) = C_ij · 1 V

where Q_i^(j) is the total charge on electrode i in the j-th unit-V solution.
Applying the discrete divergence theorem on the voxel grid gives:

    C_ij = -ε₀ · Δx · Σ_{faces ∂Ω_i} ε_face · (φ_j,outside − φ_j,inside)

summed over every face between an electrode-i node and a non-electrode-i
neighbour, with the per-face ε_face taken as the arithmetic mean of the two
adjacent node-centred ε values — the same convention used by the SOR stencil
in ``_solver/laplace.cpp``, so the post-processed flux is numerically
consistent with the discretised PDE the solver actually solved.

Sign convention follows the Maxwell capacitance matrix:
    C[i, i] > 0      (self-capacitance)
    C[i, j] ≤ 0      (i ≠ j, capacitive coupling)
    mutual_capacitance(C, i, j) = -C[i, j]
    self_capacitance(C, i)      = Σ_j C[i, j]   (to ground at infinity)

Usage:
    python -m trapsim.capacitance               # geometry.yaml in CWD, solver/
    python -m trapsim.capacitance --units pF

Or programmatically:
    from trapsim import load_geometry
    from trapsim.capacitance import capacitance_matrix, mutual_capacitance
    geo = load_geometry("geometry.yaml")
    C   = capacitance_matrix(geo, "solver")
    Cm  = mutual_capacitance(C, 0, 1)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .config import GeometryConfig, load_geometry
from .io.pa import load_phi_stack

EPS0_F_PER_M = 8.8541878128e-12       # vacuum permittivity, F/m


# ── ε interpolation: cells → nodes ───────────────────────────────────────────

def _node_eps_from_cell_eps(eps_cell: np.ndarray,
                            NX: int, NY: int, NZ: int) -> np.ndarray:
    """Interpolate cell-centred ε_r ``(NZ-1, NY-1, NX-1)`` to nodes
    ``(NZ, NY, NX)`` by arithmetic mean of the up-to-8 cells that touch each
    node.

    Matches ``build_coef_node`` in ``_solver/laplace.cpp`` so that downstream
    face-ε averages reproduce the stencil weights used by the SOR solver.
    """
    NXc, NYc, NZc = NX - 1, NY - 1, NZ - 1
    if eps_cell.shape != (NZc, NYc, NXc):
        raise ValueError(
            f"eps_cell shape {eps_cell.shape} != expected ({NZc},{NYc},{NXc})")
    eps_sum = np.zeros((NZ, NY, NX), dtype=np.float64)
    eps_cnt = np.zeros((NZ, NY, NX), dtype=np.float64)
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                eps_sum[dk:dk + NZc, dj:dj + NYc, di:di + NXc] += eps_cell
                eps_cnt[dk:dk + NZc, dj:dj + NYc, di:di + NXc] += 1.0
    return eps_sum / eps_cnt


def _load_eps_node(solver_dir: str, NX: int, NY: int, NZ: int) -> np.ndarray:
    """Load ``epsilon.raw`` and interpolate to nodes; return uniform 1.0 if
    the file is absent (geometries without dielectrics)."""
    path = os.path.join(solver_dir, "epsilon.raw")
    if not os.path.exists(path):
        return np.ones((NZ, NY, NX), dtype=np.float64)
    NXc, NYc, NZc = NX - 1, NY - 1, NZ - 1
    n_cells = NXc * NYc * NZc
    eps_cell = np.fromfile(path, dtype=np.float64, count=n_cells)
    if eps_cell.size != n_cells:
        raise IOError(
            f"{path}: read {eps_cell.size} values, expected {n_cells}")
    return _node_eps_from_cell_eps(
        eps_cell.reshape(NZc, NYc, NXc), NX, NY, NZ)


# ── Electrode-mask stack ─────────────────────────────────────────────────────

def _load_mask_stack(geometry: GeometryConfig, solver_dir: str,
                     NX: int, NY: int, NZ: int) -> np.ndarray:
    """Load every ``mask_<id>.raw`` into a single ``(N, NZ, NY, NX)`` bool
    array, ordered by ``geometry.electrodes`` declaration order."""
    n_pts = NX * NY * NZ
    out = np.zeros((geometry.n_electrodes, NZ, NY, NX), dtype=bool)
    for k, elec in enumerate(geometry.electrodes):
        path = os.path.join(solver_dir, f"mask_{elec.electrode_id}.raw")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path}: electrode mask file missing.  "
                f"Run `trapsim refine` (or `python -m trapsim.voxelize`) first.")
        m = np.fromfile(path, dtype=np.uint8, count=n_pts)
        if m.size != n_pts:
            raise IOError(
                f"{path}: read {m.size} bytes, expected {n_pts}")
        out[k] = m.reshape(NZ, NY, NX).astype(bool)
    return out


# ── Core: discrete Gauss-law flux on the voxel grid ──────────────────────────

def _capacitance_from_arrays(phi_stack: np.ndarray,
                             mask_stack: np.ndarray,
                             eps_node: np.ndarray,
                             dx_m: float) -> np.ndarray:
    """Compute the Maxwell capacitance matrix from in-memory arrays.

    Parameters
    ----------
    phi_stack : ndarray, shape (N, NZ, NY, NX)
        ``phi_stack[j]`` = potential everywhere when electrode j is at 1 V
        and all other electrodes are at 0 V.
    mask_stack : ndarray, shape (N, NZ, NY, NX), bool
        ``mask_stack[i]`` is True at every electrode-i node.
    eps_node : ndarray, shape (NZ, NY, NX)
        Node-centred relative permittivity.
    dx_m : float
        Grid spacing in metres.

    Returns
    -------
    C : ndarray, shape (N, N), float64
        ``C[i, j]`` = charge induced on electrode i when electrode j is at
        1 V (all others grounded), in farads.
    """
    if phi_stack.shape != mask_stack.shape:
        raise ValueError(
            f"phi_stack {phi_stack.shape} vs mask_stack {mask_stack.shape}")
    if phi_stack.shape[1:] != eps_node.shape:
        raise ValueError(
            f"phi_stack grid {phi_stack.shape[1:]} vs eps_node {eps_node.shape}")

    N = phi_stack.shape[0]
    C = np.zeros((N, N), dtype=np.float64)
    prefactor = -EPS0_F_PER_M * dx_m

    for axis in range(3):
        # Pairs of adjacent nodes along this axis: "lo" at index k, "hi" at
        # k+1.  A face contributes flux only when exactly one of (lo, hi) is
        # in the electrode mask — purely-interior and purely-exterior faces
        # are skipped automatically.
        sl_lo: list[slice] = [slice(None)] * 3
        sl_hi: list[slice] = [slice(None)] * 3
        sl_lo[axis] = slice(None, -1)
        sl_hi[axis] = slice(1, None)
        tlo, thi = tuple(sl_lo), tuple(sl_hi)
        eps_face = 0.5 * (eps_node[tlo] + eps_node[thi])

        for i in range(N):
            m_lo = mask_stack[i][tlo]
            m_hi = mask_stack[i][thi]
            elec_is_lo = m_lo & ~m_hi      # electrode-i on the "lo" side
            elec_is_hi = m_hi & ~m_lo      # electrode-i on the "hi" side
            if not (elec_is_lo.any() or elec_is_hi.any()):
                continue
            for j in range(N):
                phi_lo = phi_stack[j][tlo]
                phi_hi = phi_stack[j][thi]
                # outward = component of (phi_outside - phi_inside) along
                # the face normal pointing *out* of electrode i
                outward = (np.where(elec_is_lo, phi_hi - phi_lo, 0.0)
                           + np.where(elec_is_hi, phi_lo - phi_hi, 0.0))
                C[i, j] += prefactor * np.sum(eps_face * outward)
    return C


# ── Public API ────────────────────────────────────────────────────────────────

def capacitance_matrix(geometry: GeometryConfig,
                       solver_dir: str,
                       *,
                       verbose: bool = False) -> np.ndarray:
    """Compute the Maxwell capacitance matrix from PA files in ``solver_dir``.

    Reads ``field.pa<id>``, ``mask_<id>.raw`` for every electrode in
    ``geometry``, and (if present) ``epsilon.raw``.  Returns an
    ``(N_electrodes, N_electrodes)`` array of capacitances in farads.

    Sign convention: ``C[i, i] > 0``; ``C[i, j] ≤ 0`` for ``i ≠ j``.  See
    :func:`mutual_capacitance` and :func:`self_capacitance` for the usual
    "between-pair" and "to-ground" reductions.
    """
    phi_stack, grid = load_phi_stack(geometry, solver_dir, verbose=verbose)
    NX, NY, NZ, dx_mm = grid["NX"], grid["NY"], grid["NZ"], grid["dx"]
    mask_stack = _load_mask_stack(geometry, solver_dir, NX, NY, NZ)
    eps_node   = _load_eps_node(solver_dir, NX, NY, NZ)
    dx_m = dx_mm * 1e-3
    return _capacitance_from_arrays(phi_stack, mask_stack, eps_node, dx_m)


def mutual_capacitance(C: np.ndarray, i: int, j: int) -> float:
    """Pair-wise mutual capacitance.

    By the Maxwell convention ``C[i, j]`` and ``C[j, i]`` should be equal
    and non-positive.  Returns the symmetric average to absorb any small
    grid-induced asymmetry: ``C_m = -½ (C[i,j] + C[j,i])``.
    """
    if i == j:
        raise ValueError("mutual_capacitance requires i != j")
    return float(-0.5 * (C[i, j] + C[j, i]))


def self_capacitance(C: np.ndarray, i: int) -> float:
    """Self-capacitance of electrode i to ground at infinity = Σ_j C[i, j]."""
    return float(C[i, :].sum())


# ── CLI ──────────────────────────────────────────────────────────────────────

_UNIT_SCALES = {"F": 1.0, "mF": 1e3, "uF": 1e6, "nF": 1e9,
                "pF": 1e12, "fF": 1e15, "aF": 1e18}


def _format_matrix(C: np.ndarray, names: list[str], unit: str = "fF") -> str:
    """Pretty-print the capacitance matrix labelled by electrode names."""
    scale = _UNIT_SCALES[unit]
    M = C * scale
    name_w = max(10, max(len(n) for n in names))
    cell_w = 13
    head = " " * (name_w + 2) + "".join(f"{n:>{cell_w}s}" for n in names)
    lines = [head]
    for i, ni in enumerate(names):
        row = f"{ni:>{name_w}s}  " + "".join(
            f"{M[i, j]:>{cell_w}.4g}" for j in range(len(names)))
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Compute the Maxwell capacitance matrix from PA files.")
    ap.add_argument("--geometry", default="geometry.yaml",
                    help="Path to geometry.yaml (default: ./geometry.yaml)")
    ap.add_argument("--solver-dir", default="solver",
                    help="Directory with mask_<id>.raw and field.pa<id> "
                         "(default: ./solver)")
    ap.add_argument("--units", default="fF",
                    choices=tuple(_UNIT_SCALES.keys()),
                    help="Display units (default: fF)")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-PA-file progress lines")
    args = ap.parse_args()

    geo = load_geometry(args.geometry)
    if not geo.electrodes:
        raise SystemExit(f"{args.geometry}: no electrodes — nothing to compute.")

    print(f"Reading PA files from {args.solver_dir}/  "
          f"({geo.n_electrodes} electrodes)")
    C = capacitance_matrix(geo, args.solver_dir, verbose=not args.quiet)
    names = geo.electrode_names()

    print(f"\nMaxwell capacitance matrix [{args.units}]:")
    print(_format_matrix(C, names, unit=args.units))

    asym = np.abs(C - C.T)
    rel_asym = asym.max() / np.abs(C).max() if np.abs(C).max() > 0 else 0.0
    flag = "OK" if rel_asym < 0.05 else "WARNING: grid bounds may be too tight"
    print(f"\nReciprocity:   max |C - Cᵀ| / max |C| = {rel_asym:.2e}   ({flag})")

    scale = _UNIT_SCALES[args.units]
    print(f"\nSelf-capacitance to ground-at-infinity (row sums) [{args.units}]:")
    for i, n in enumerate(names):
        print(f"  {n:>20s}: {C[i, :].sum() * scale:>12.4g}")


if __name__ == "__main__":
    main()
