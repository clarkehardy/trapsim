"""trapsim.refine  –  Orchestrate voxelization + C++ Laplace solve.

Replaces the SIMION Refine step.  Driven by geometry.yaml.

Usage:
    python -m trapsim.refine [--geometry geometry.yaml] [--out-dir .]
                             [--force-voxelize] [--omega 1.99]
                             [--max-iter 3000] [--tol 1e-5]

Steps:
  1. Voxelize STLs if mask files are stale (or --force-voxelize).
  2. Compile solver/laplace if absent or older than laplace.cpp.
  3. Run the C++ solver once per electrode → paulTrap.pa<id>.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .config import GeometryConfig, load_geometry
from .voxelize import voxelize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER_DIR_DEFAULT = os.path.join(BASE, "solver")


def _newest_mtime(paths) -> float:
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)),
               default=0.0)


def masks_stale(geometry: GeometryConfig, solver_dir: str) -> bool:
    """True if any mask file is missing or older than its source STLs."""
    for elec in geometry.electrodes:
        mask = os.path.join(solver_dir, f"mask_{elec.electrode_id}.raw")
        if not os.path.exists(mask):
            return True
        stl_t = _newest_mtime(elec.stls)
        if os.path.getmtime(mask) < stl_t:
            return True
    # Dielectric epsilon
    eps = os.path.join(solver_dir, "epsilon.raw")
    stl_t = _newest_mtime([d.stl for d in geometry.dielectrics])
    if geometry.dielectrics and (not os.path.exists(eps) or
                                  os.path.getmtime(eps) < stl_t):
        return True
    if not os.path.exists(os.path.join(solver_dir, "grid.txt")):
        return True
    return False


def ensure_compiled(solver_dir: str) -> None:
    src = os.path.join(solver_dir, "laplace.cpp")
    exe = os.path.join(solver_dir, "laplace")
    if (not os.path.exists(exe) or
            os.path.getmtime(exe) < os.path.getmtime(src)):
        print("Compiling solver/laplace ...")
        result = subprocess.run(["make", "-C", solver_dir],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            sys.exit(f"ERROR: compile failed (rc={result.returncode})")
        print("  Compiled OK.")
    else:
        print("solver/laplace is up-to-date.")


def refine(geometry: GeometryConfig, *,
           out_dir: str = BASE,
           solver_dir: str = SOLVER_DIR_DEFAULT,
           force_voxelize: bool = False,
           omega: float = 1.99,
           max_iter: int = 3000,
           tol: float = 1e-5) -> None:
    """Run the full refine pipeline for `geometry`."""

    # ── Step 1: voxelize ────────────────────────────────────────────────
    if force_voxelize or masks_stale(geometry, solver_dir):
        print("─── Voxelizing STL meshes ───")
        t0 = time.time()
        voxelize(geometry, solver_dir)
        print(f"Voxelization done in {time.time()-t0:.1f} s\n")
    else:
        print("Mask files are current — skipping voxelization "
              "(use --force-voxelize to override)\n")

    # ── Step 2: compile ─────────────────────────────────────────────────
    ensure_compiled(solver_dir)
    print()

    # ── Step 3: solve ───────────────────────────────────────────────────
    print(f"─── Running Laplace solver ({geometry.n_electrodes} electrodes) ───")
    grid_file = os.path.join(solver_dir, "grid.txt")
    eps_file  = os.path.join(solver_dir, "epsilon.raw")
    exe       = os.path.join(solver_dir, "laplace")
    mask_args = [os.path.join(solver_dir, f"mask_{e.electrode_id}.raw")
                 for e in geometry.electrodes]

    for f in [grid_file, eps_file] + mask_args:
        if not os.path.exists(f):
            sys.exit(f"ERROR: required file not found: {f}")

    cmd = [exe, grid_file, eps_file, out_dir,
           str(omega), str(max_iter), str(tol)] + mask_args
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: laplace solver exited with code {result.returncode}")
    print(f"\nSolver finished in {time.time()-t0:.1f} s")

    # ── Size check ──────────────────────────────────────────────────────
    NX, NY, NZ = geometry.grid.shape
    expected = 56 + NX * NY * NZ * 8
    all_ok = True
    for elec in geometry.electrodes:
        pa = os.path.join(out_dir, f"paulTrap.pa{elec.electrode_id}")
        if not os.path.exists(pa):
            print(f"  WARNING: {pa} not found")
            all_ok = False
            continue
        sz = os.path.getsize(pa)
        status = "OK" if sz == expected else (
            f"SIZE MISMATCH (got {sz}, expected {expected})")
        print(f"  pa{elec.electrode_id:>2} ({elec.name:<20s}): {sz:,} bytes  {status}")
        if sz != expected:
            all_ok = False

    print("\n─── Refine complete ───" if all_ok else
          "\n─── Refine completed with warnings ───")


def main():
    ap = argparse.ArgumentParser(description="Refine potential arrays from geometry.yaml.")
    ap.add_argument("--geometry",       default=os.path.join(BASE, "geometry.yaml"))
    ap.add_argument("--out-dir",        default=BASE)
    ap.add_argument("--solver-dir",     default=SOLVER_DIR_DEFAULT)
    ap.add_argument("--force-voxelize", action="store_true")
    ap.add_argument("--omega",          type=float, default=1.99)
    ap.add_argument("--max-iter",       type=int,   default=3000)
    ap.add_argument("--tol",            type=float, default=1e-5)
    args = ap.parse_args()

    geo = load_geometry(args.geometry)
    refine(geo,
           out_dir=args.out_dir,
           solver_dir=args.solver_dir,
           force_voxelize=args.force_voxelize,
           omega=args.omega,
           max_iter=args.max_iter,
           tol=args.tol)


if __name__ == "__main__":
    main()
