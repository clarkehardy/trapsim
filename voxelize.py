"""trapsim.voxelize  –  STL → per-electrode voxel masks + ε_r array.

Driven entirely by a `GeometryConfig` from trapsim.config.  Writes
`solver/mask_<id>.raw`, `solver/epsilon.raw`, and `solver/grid.txt`.

The on-disk format matches the legacy voxelize.py output so the existing
C++ Laplace solver can consume it unchanged.

Mask file:    flat uint8, shape NZ×NY×NX, 1 = inside the electrode.
Epsilon file: flat float64, shape (NZ-1)×(NY-1)×(NX-1), ε_r per cell-centre.
Grid file:    one line "NX NY NZ DX TX TY TZ" where (TX,TY,TZ) is the
              positive GEM offset (so a Fusion-world coord x equals
              i*DX - TX for grid index i).
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np

try:
    import trimesh
except ImportError as e:
    raise SystemExit(
        "trimesh is required.\n  Install: ~/.venvs/mesh/bin/pip install trimesh"
    ) from e

from .config import GeometryConfig, load_geometry


def _voxelize_to_nodes(mesh, grid_shape, dx, world_offset, label=""):
    """Voxelize `mesh` onto the *node* grid (shape NX×NY×NZ).

    Returns flat uint8, shape NZ*NY*NX, 1 = node inside mesh.
    Processes one Z-slice at a time to bound memory.
    """
    NX, NY, NZ = grid_shape
    tx, ty, tz = (-world_offset[0], -world_offset[1], -world_offset[2])

    mask   = np.zeros(NZ * NY * NX, dtype=np.uint8)
    x_node = np.arange(NX) * dx - tx
    y_node = np.arange(NY) * dx - ty
    z_node = np.arange(NZ) * dx - tz

    bb_lo, bb_hi = mesh.bounds
    ix_lo = max(0, int(np.floor((bb_lo[0] + tx) / dx)))
    ix_hi = min(NX, int(np.ceil( (bb_hi[0] + tx) / dx)) + 1)
    iy_lo = max(0, int(np.floor((bb_lo[1] + ty) / dx)))
    iy_hi = min(NY, int(np.ceil( (bb_hi[1] + ty) / dx)) + 1)
    iz_lo = max(0, int(np.floor((bb_lo[2] + tz) / dx)))
    iz_hi = min(NZ, int(np.ceil( (bb_hi[2] + tz) / dx)) + 1)

    xi = x_node[ix_lo:ix_hi]
    yi = y_node[iy_lo:iy_hi]
    nx_sub, ny_sub = len(xi), len(yi)
    if nx_sub == 0 or ny_sub == 0:
        print(f"    {label}: bbox outside grid — skipped")
        return mask

    XX, YY = np.meshgrid(xi, yi, indexing="ij")
    pts = np.empty((nx_sub * ny_sub, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    n_inside = 0
    for iz in range(iz_lo, iz_hi):
        pts[:, 2] = z_node[iz]
        inside = mesh.contains(pts)
        if not inside.any():
            continue
        inside_2d = inside.reshape(nx_sub, ny_sub)
        ix_hits, iy_hits = np.where(inside_2d)
        flat_idx = iz * NY * NX + (iy_lo + iy_hits) * NX + (ix_lo + ix_hits)
        mask[flat_idx] = 1
        n_inside += int(inside.sum())

    print(f"    {label}: bbox {bb_lo} → {bb_hi}  |  {n_inside} voxels")
    return mask


def _voxelize_to_cells(mesh, grid_shape, dx, world_offset, label=""):
    """Voxelize `mesh` onto the *cell-centre* grid (shape (NX-1)×(NY-1)×(NZ-1)).

    Returns flat bool, shape NZc*NYc*NXc.
    """
    NX, NY, NZ = grid_shape
    NXc, NYc, NZc = NX - 1, NY - 1, NZ - 1
    tx, ty, tz = (-world_offset[0], -world_offset[1], -world_offset[2])

    inside_arr = np.zeros(NZc * NYc * NXc, dtype=bool)
    x_c = (np.arange(NXc) + 0.5) * dx - tx
    y_c = (np.arange(NYc) + 0.5) * dx - ty
    z_c = (np.arange(NZc) + 0.5) * dx - tz

    bb_lo, bb_hi = mesh.bounds
    ix_lo = max(0, int(np.floor((bb_lo[0] + tx) / dx - 1)))
    ix_hi = min(NXc, int(np.ceil( (bb_hi[0] + tx) / dx + 1)))
    iy_lo = max(0, int(np.floor((bb_lo[1] + ty) / dx - 1)))
    iy_hi = min(NYc, int(np.ceil( (bb_hi[1] + ty) / dx + 1)))
    iz_lo = max(0, int(np.floor((bb_lo[2] + tz) / dx - 1)))
    iz_hi = min(NZc, int(np.ceil( (bb_hi[2] + tz) / dx + 1)))

    xi = x_c[ix_lo:ix_hi]
    yi = y_c[iy_lo:iy_hi]
    nx_sub, ny_sub = len(xi), len(yi)
    if nx_sub == 0 or ny_sub == 0:
        return inside_arr

    XX, YY = np.meshgrid(xi, yi, indexing="ij")
    pts = np.empty((nx_sub * ny_sub, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    n_inside = 0
    for iz in range(iz_lo, iz_hi):
        pts[:, 2] = z_c[iz]
        inside = mesh.contains(pts)
        if not inside.any():
            continue
        inside_2d = inside.reshape(nx_sub, ny_sub)
        ix_hits, iy_hits = np.where(inside_2d)
        flat_idx = iz * NYc * NXc + (iy_lo + iy_hits) * NXc + (ix_lo + ix_hits)
        inside_arr[flat_idx] = True
        n_inside += int(inside.sum())

    print(f"    {label}: {n_inside} cells inside")
    return inside_arr


def write_grid_txt(geometry: GeometryConfig, out_path: str) -> None:
    NX, NY, NZ = geometry.grid.shape
    dx = geometry.grid.dx_mm
    tx, ty, tz = (-c for c in geometry.grid.world_offset_mm)
    with open(out_path, "w") as f:
        f.write(f"{NX} {NY} {NZ} {dx} {tx} {ty} {tz}\n")


def build_electrode_masks(geometry: GeometryConfig, out_dir: str) -> None:
    """Write solver/mask_<id>.raw for each electrode in `geometry`."""
    NX, NY, NZ = geometry.grid.shape
    dx         = geometry.grid.dx_mm
    world_off  = geometry.grid.world_offset_mm

    print("\nVoxelizing electrodes ...")
    for elec in geometry.electrodes:
        print(f"  Electrode {elec.electrode_id} ({elec.name}): "
              f"{len(elec.stls)} STL(s)")
        mask = np.zeros(NX * NY * NZ, dtype=np.uint8)
        for stl_path in elec.stls:
            mesh = trimesh.load_mesh(stl_path)
            sub  = _voxelize_to_nodes(
                mesh, (NX, NY, NZ), dx, world_off,
                label=os.path.basename(stl_path))
            mask |= sub
        path = os.path.join(out_dir, f"mask_{elec.electrode_id}.raw")
        mask.tofile(path)
        print(f"  → {path}  ({int(mask.sum())} voxels set)")


def build_epsilon(geometry: GeometryConfig, out_dir: str) -> None:
    """Write solver/epsilon.raw — per-cell ε_r.  Overlapping dielectrics
    take the maximum ε_r (loud overlap rather than silent averaging)."""
    NX, NY, NZ = geometry.grid.shape
    NXc, NYc, NZc = NX - 1, NY - 1, NZ - 1
    dx         = geometry.grid.dx_mm
    world_off  = geometry.grid.world_offset_mm

    epsilon = np.ones(NZc * NYc * NXc, dtype=np.float64)

    if geometry.dielectrics:
        print("\nVoxelizing dielectrics ...")
        for diel in geometry.dielectrics:
            print(f"  Dielectric {diel.name} (ε_r = {diel.epsilon_r}):")
            mesh   = trimesh.load_mesh(diel.stl)
            inside = _voxelize_to_cells(
                mesh, (NX, NY, NZ), dx, world_off,
                label=os.path.basename(diel.stl))
            np.maximum(epsilon, np.where(inside, diel.epsilon_r, 1.0),
                       out=epsilon)
    else:
        print("\nNo dielectrics defined — writing uniform ε_r = 1.")

    eps_path = os.path.join(out_dir, "epsilon.raw")
    epsilon.tofile(eps_path)
    n_diel = int((epsilon > 1.0).sum())
    print(f"→ {eps_path}  ({n_diel} cells with ε_r > 1)")


def voxelize(geometry: GeometryConfig, out_dir: str) -> None:
    """Full voxelization: grid.txt + per-electrode masks + epsilon."""
    os.makedirs(out_dir, exist_ok=True)

    NX, NY, NZ = geometry.grid.shape
    dx         = geometry.grid.dx_mm
    tx, ty, tz = (-c for c in geometry.grid.world_offset_mm)
    print(f"Grid: {NX}×{NY}×{NZ}  dx={dx} mm  world_offset=({-tx},{-ty},{-tz})")

    grid_path = os.path.join(out_dir, "grid.txt")
    write_grid_txt(geometry, grid_path)
    print(f"Written {grid_path}")

    build_electrode_masks(geometry, out_dir)
    build_epsilon(geometry, out_dir)
    print("\nVoxelization complete.")


def main():
    ap = argparse.ArgumentParser(
        description="Voxelize geometry from YAML for the Laplace solver.")
    ap.add_argument("--geometry", default="geometry.yaml")
    ap.add_argument("--out-dir",  default="solver")
    args = ap.parse_args()

    geo = load_geometry(args.geometry)
    voxelize(geo, args.out_dir)


if __name__ == "__main__":
    main()
