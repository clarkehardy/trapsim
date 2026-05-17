"""trapsim.voxelize  –  STL → per-electrode voxel masks + ε_r array.

Driven entirely by a `GeometryConfig` from trapsim.config.  Writes
`solver/mask_<id>.raw`, `solver/epsilon.raw`, `solver/grid.txt`, plus
(when magnetic bodies are present) `solver/mu.raw`,
`solver/magnetic_source.raw`, and `solver/magnetic_material_mask.raw`.

The on-disk format matches the legacy voxelize.py output so the existing
C++ Laplace solver can consume it unchanged.

Mask file:           flat uint8, shape NZ×NY×NX, 1 = inside electrode.
Epsilon file:        flat float64, shape (NZ-1)×(NY-1)×(NX-1), ε_r per cell-centre.
Mu file:             flat float64, shape (NZ-1)×(NY-1)×(NX-1), μ_r per cell-centre.
Magnetic source:     flat float64, shape NZ×NY×NX, RHS of ∇²ψ = -source
                     on the *node* grid.  Non-zero only in a 1-voxel-thick
                     external shell around each magnet, where it stores
                     (Br · n̂) / dx_mm — the discretised surface "magnetic
                     charge" distributed over a shell of thickness dx_mm.
Magnetic mat. mask:  flat uint8, shape NZ×NY×NX, union of all magnet +
                     magnetic_material voxels, for splat detection.
Grid file:           one line "NX NY NZ DX TX TY TZ" where (TX,TY,TZ) is the
                     positive GEM offset (so a Fusion-world coord x equals
                     i*DX - TX for grid index i).
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np

from .config import GeometryConfig, load_geometry


def _require_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise SystemExit(
            "trimesh is required for voxelization.\n"
            "  Install: ~/.venvs/mesh/bin/pip install trimesh"
        ) from e


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
    trimesh = _require_trimesh()
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


def build_dielectric_mask(geometry: GeometryConfig, out_dir: str) -> None:
    """Write solver/dielectric_mask.raw — union of all dielectric bodies on
    the *node* grid (same shape as electrode masks), for splat detection.

    Skipped (no file written) if there are no dielectrics.
    """
    if not geometry.dielectrics:
        return
    trimesh = _require_trimesh()
    NX, NY, NZ = geometry.grid.shape
    dx         = geometry.grid.dx_mm
    world_off  = geometry.grid.world_offset_mm

    print("\nVoxelizing dielectric bodies (for splat mask) ...")
    mask = np.zeros(NX * NY * NZ, dtype=np.uint8)
    for diel in geometry.dielectrics:
        print(f"  Dielectric {diel.name}:")
        mesh = trimesh.load_mesh(diel.stl)
        sub  = _voxelize_to_nodes(
            mesh, (NX, NY, NZ), dx, world_off,
            label=os.path.basename(diel.stl))
        mask |= sub
    path = os.path.join(out_dir, "dielectric_mask.raw")
    mask.tofile(path)
    print(f"  → {path}  ({int(mask.sum())} voxels set)")


def build_mu(geometry: GeometryConfig, out_dir: str) -> None:
    """Write solver/mu.raw — per-cell μ_r.  Overlapping magnetic materials
    take the maximum μ_r (loud overlap rather than silent averaging).

    Skipped (no file written) if there are no magnetic materials AND no
    magnets — the magnetic solve isn't run in that case.  If only magnets
    are present (no high-μ pole pieces), writes a uniform μ_r=1 array
    because the C++ solver expects the file.
    """
    if not geometry.magnetic_materials and not geometry.magnets:
        return
    NX, NY, NZ = geometry.grid.shape
    NXc, NYc, NZc = NX - 1, NY - 1, NZ - 1
    dx         = geometry.grid.dx_mm
    world_off  = geometry.grid.world_offset_mm

    mu = np.ones(NZc * NYc * NXc, dtype=np.float64)

    if geometry.magnetic_materials:
        trimesh = _require_trimesh()
        print("\nVoxelizing magnetic materials (for mu) ...")
        for mat in geometry.magnetic_materials:
            print(f"  Magnetic material {mat.name} (μ_r = {mat.mu_r}):")
            mesh   = trimesh.load_mesh(mat.stl)
            inside = _voxelize_to_cells(
                mesh, (NX, NY, NZ), dx, world_off,
                label=os.path.basename(mat.stl))
            np.maximum(mu, np.where(inside, mat.mu_r, 1.0), out=mu)
    else:
        print("\nNo magnetic materials — writing uniform μ_r = 1.")

    mu_path = os.path.join(out_dir, "mu.raw")
    mu.tofile(mu_path)
    n_mag = int((mu != 1.0).sum())
    print(f"→ {mu_path}  ({n_mag} cells with μ_r ≠ 1)")


def build_magnetic_source(geometry: GeometryConfig, out_dir: str) -> None:
    """Write solver/magnetic_source.raw — RHS of ∇²ψ = -source on the node grid.

    For each magnet, locate the 1-voxel-thick *external* shell of nodes
    adjacent to the magnet's interior.  At each shell node, query the
    nearest STL surface point to get the outward unit normal n̂, then
    store (Br · n̂) / dx_mm.  The 1/dx factor distributes the surface
    "magnetic charge" σ_M = M·n̂ over the shell as a volume source.

    Skipped (no file written) if there are no magnets.
    """
    if not geometry.magnets:
        return
    trimesh = _require_trimesh()
    from trimesh.proximity import closest_point  # heavy, import on demand

    NX, NY, NZ = geometry.grid.shape
    dx         = geometry.grid.dx_mm
    wox, woy, woz = geometry.grid.world_offset_mm

    source = np.zeros((NZ, NY, NX), dtype=np.float64)

    print("\nVoxelizing magnets (computing surface source σ_M = Br·n̂/dx) ...")
    for mag in geometry.magnets:
        print(f"  Magnet {mag.magnet_id} ({mag.name})  Br_T = {mag.Br_T}")
        mesh = trimesh.load_mesh(mag.stl)
        sub  = _voxelize_to_nodes(
            mesh, (NX, NY, NZ), dx, (wox, woy, woz),
            label=os.path.basename(mag.stl))
        interior = sub.reshape(NZ, NY, NX).astype(bool)

        # 1-voxel external shell: any node that is *not* interior but has
        # an immediate 6-neighbour that is interior.
        shell = np.zeros_like(interior)
        # Shift interior by ±1 along each axis (zeroing the wrap plane).
        for axis, length in ((0, NZ), (1, NY), (2, NX)):
            for direction in (-1, +1):
                shifted = np.roll(interior, shift=direction, axis=axis)
                if direction == -1:
                    if axis == 0: shifted[length - 1, :, :] = False
                    elif axis == 1: shifted[:, length - 1, :] = False
                    else:          shifted[:, :, length - 1] = False
                else:
                    if axis == 0: shifted[0, :, :] = False
                    elif axis == 1: shifted[:, 0, :] = False
                    else:          shifted[:, :, 0] = False
                shell |= shifted
        shell &= ~interior

        # World coords of every shell node
        iz_s, iy_s, ix_s = np.where(shell)
        if iz_s.size == 0:
            print(f"    [warn] magnet {mag.name!r} has no exterior shell "
                  f"voxels — bbox may be outside the grid")
            continue
        world_pts = np.column_stack([
            ix_s * dx + wox,
            iy_s * dx + woy,
            iz_s * dx + woz,
        ])
        # Outward normal at the nearest surface point
        _close_pts, _dists, tri_ids = closest_point(mesh, world_pts)
        n_hat = mesh.face_normals[tri_ids]      # (N, 3), unit
        Br = np.asarray(mag.Br_T, dtype=np.float64)
        sigma = (n_hat @ Br) / dx               # (N,) T/mm
        # Accumulate (multiple magnets may overlap on shared shell voxels)
        source[iz_s, iy_s, ix_s] += sigma
        print(f"    {iz_s.size} shell voxels  "
              f"σ range [{sigma.min():+.3f}, {sigma.max():+.3f}] T/mm")

    path = os.path.join(out_dir, "magnetic_source.raw")
    source.tofile(path)
    n_nz = int((source != 0.0).sum())
    print(f"→ {path}  ({n_nz} non-zero source voxels)")


def build_magnetic_material_mask(geometry: GeometryConfig, out_dir: str) -> None:
    """Write solver/magnetic_material_mask.raw — union of all magnet +
    magnetic_material voxels on the *node* grid, for splat detection.

    Skipped (no file written) if there are no magnets and no magnetic
    materials.  Magnets are included because they are solid objects too.
    """
    if not geometry.magnets and not geometry.magnetic_materials:
        return
    trimesh = _require_trimesh()
    NX, NY, NZ = geometry.grid.shape
    dx         = geometry.grid.dx_mm
    world_off  = geometry.grid.world_offset_mm

    print("\nVoxelizing magnetic bodies (for splat mask) ...")
    mask = np.zeros(NX * NY * NZ, dtype=np.uint8)
    for body in list(geometry.magnets) + list(geometry.magnetic_materials):
        print(f"  {body.name}:")
        mesh = trimesh.load_mesh(body.stl)
        sub  = _voxelize_to_nodes(
            mesh, (NX, NY, NZ), dx, world_off,
            label=os.path.basename(body.stl))
        mask |= sub
    path = os.path.join(out_dir, "magnetic_material_mask.raw")
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
        trimesh = _require_trimesh()
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
    build_dielectric_mask(geometry, out_dir)
    build_epsilon(geometry, out_dir)
    build_mu(geometry, out_dir)
    build_magnetic_source(geometry, out_dir)
    build_magnetic_material_mask(geometry, out_dir)
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
