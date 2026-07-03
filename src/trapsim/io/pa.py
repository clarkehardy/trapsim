"""trapsim.io.pa  –  SIMION-compatible Potential Array (PA) reader.

The PA binary format is a 56-byte header followed by NX·NY·NZ float64 values
in [k][j][i] order (k = z slowest, i = x fastest).  Electrode-surface voxels
are encoded with two special flags:

  raw < 0                 → "other electrode"  (trapsim solver writes -1.0)
  raw > 1.5 · scale_ref   → "this electrode"   (= 2·scale_ref + electrode_id)
  otherwise               → φ = raw / scale_ref   (volts per unit drive)

For splat detection, prefer the voxel masks written by voxelize.py
(`<solver_dir>/mask_<id>.raw`, `<solver_dir>/dielectric_mask.raw`) over
the PA encoding: SOR can leave free-space cells with small negative
residuals that overlap the magnitude of the -1.0 sentinel.  See
`load_splat_mask`.
"""

from __future__ import annotations

import os
import struct
import time
from typing import Tuple

import numpy as np

HEADER_BYTES = 56


def read_pa(path: str) -> Tuple[np.ndarray, int, int, int, float]:
    """Load a SIMION PA file as a unit-potential array.

    Returns
    -------
    phi : ndarray, shape (NZ, NY, NX), float64
        Potential at each grid node when 1 V is applied to this electrode
        and 0 V to all others.  Electrode-surface voxels are clipped to
        their nominal value (1.0 for this electrode, 0.0 for others).
    NX, NY, NZ : int
    dx : float (mm)
    """
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        raw_bytes = f.read()

    scale_ref = struct.unpack_from("<d", hdr, 8)[0]    # typically 1e5
    NX        = struct.unpack_from("<i", hdr, 16)[0]
    NY        = struct.unpack_from("<i", hdr, 20)[0]
    NZ        = struct.unpack_from("<i", hdr, 24)[0]
    dx        = struct.unpack_from("<d", hdr, 32)[0]   # mm

    n_pts = NX * NY * NZ
    expected = HEADER_BYTES + n_pts * 8
    if fsize != expected:
        raise IOError(
            f"{path}: file size {fsize} != expected {expected} "
            f"for ({NX}, {NY}, {NZ}) grid")

    raw = np.frombuffer(raw_bytes, dtype="<f8", count=n_pts).copy()

    other_mask = raw < 0
    self_mask  = raw > 1.5 * scale_ref

    phi = np.abs(raw) / scale_ref
    phi[self_mask]  = 1.0
    phi[other_mask] = 0.0

    return phi.reshape(NZ, NY, NX), NX, NY, NZ, dx


def load_splat_mask(geometry, solver_dir: str
                    ) -> tuple[np.ndarray, dict[int, str]] | None:
    """Return `(labels, names)` built from the electrode + dielectric voxel
    masks in `<solver_dir>` (uint8 files written by `trapsim.voxelize`).

    `labels` is a (NZ, NY, NX) integer array: 0 for free space, the
    electrode_id (1..N in declaration order) for each electrode's voxels,
    and N+i for the i-th dielectric's voxels.  `names` maps each nonzero
    label to a human-readable name for splat reporting.  Where masks
    overlap, the later declaration wins — the union is unchanged either
    way.

    Dielectric masks are read from dielectric_mask_<i>.raw (one file per
    body, i = 1..M in declaration order).  Solver dirs voxelized before
    per-body masks existed have only the combined dielectric_mask.raw;
    that is accepted as a fallback, labelled N+1 and named "dielectric"
    (re-run the voxelizer to get per-body names).

    Used for splat detection: a particle is terminated when its nearest
    grid voxel is nonzero, and the label says what it hit.  Dielectric
    bodies are treated as solid obstacles even though the field solver
    only sees their permittivity.

    Returns None if any required electrode mask file is missing — callers
    should treat this as "splat detection unavailable" rather than an
    error, since PA files can exist without the voxelizer's work files.
    Dielectric masks are optional (geometries without dielectrics never
    have them).
    """
    NX, NY, NZ = geometry.grid.shape
    n_pts = NX * NY * NZ
    n_elec = len(geometry.electrodes)
    max_label = n_elec + max(1, len(geometry.dielectrics))
    dtype = np.uint8 if max_label <= np.iinfo(np.uint8).max else np.int16
    labels = np.zeros((NZ, NY, NX), dtype=dtype)
    names: dict[int, str] = {}

    def _read_mask(path):
        m = np.fromfile(path, dtype=np.uint8, count=n_pts)
        if m.size != n_pts:
            raise IOError(
                f"{path}: read {m.size} bytes, expected {n_pts} "
                f"({NX}×{NY}×{NZ})")
        return m.reshape(NZ, NY, NX).astype(bool)

    for elec in geometry.electrodes:
        path = os.path.join(solver_dir, f"mask_{elec.electrode_id}.raw")
        if not os.path.exists(path):
            return None
        labels[_read_mask(path)] = elec.electrode_id
        names[elec.electrode_id] = elec.name

    per_body = [os.path.join(solver_dir, f"dielectric_mask_{i}.raw")
                for i in range(1, len(geometry.dielectrics) + 1)]
    if per_body and all(os.path.exists(p) for p in per_body):
        for i, (diel, path) in enumerate(zip(geometry.dielectrics, per_body),
                                         start=1):
            labels[_read_mask(path)] = n_elec + i
            names[n_elec + i] = diel.name
    else:
        combined = os.path.join(solver_dir, "dielectric_mask.raw")
        if os.path.exists(combined):
            labels[_read_mask(combined)] = n_elec + 1
            names[n_elec + 1] = "dielectric"
    return labels, names


def load_phi_stack(geometry, base_dir: str, verbose: bool = True
                   ) -> tuple[np.ndarray, dict]:
    """Load every electrode's PA file into a stacked array.

    Parameters
    ----------
    geometry : GeometryConfig
        Electrode declaration order determines stacking order; PA files are
        read from `<base_dir>/field.pa<electrode_id>`.
    base_dir : str
        Directory containing the field.pa<N> files.
    verbose : bool
        Print per-file progress.

    Returns
    -------
    phi_stack : ndarray, shape (N_electrodes, NZ, NY, NX)
    grid : dict with keys NX, NY, NZ, dx
    """
    phi_list = []
    grid = None
    for elec in geometry.electrodes:
        path = os.path.join(base_dir, f"field.pa{elec.electrode_id}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PA file for electrode {elec.electrode_id} ({elec.name}): {path}")
        t0 = time.perf_counter()
        phi, NX, NY, NZ, dx = read_pa(path)
        if verbose:
            print(f"  pa{elec.electrode_id:>2} ({elec.name:<20s}): "
                  f"{NX}×{NY}×{NZ}  dx={dx:.3g} mm  "
                  f"({time.perf_counter()-t0:.1f} s)", flush=True)
        if grid is None:
            grid = {"NX": NX, "NY": NY, "NZ": NZ, "dx": dx}
        else:
            if (NX, NY, NZ) != (grid["NX"], grid["NY"], grid["NZ"]):
                raise ValueError(
                    f"{path}: grid mismatch ({NX},{NY},{NZ}) vs "
                    f"({grid['NX']},{grid['NY']},{grid['NZ']})")
        phi_list.append(phi)
    return np.stack(phi_list, axis=0), grid
