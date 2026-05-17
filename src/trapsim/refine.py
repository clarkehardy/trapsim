"""trapsim.refine  –  Orchestrate voxelization + C++ Laplace solve.

Replaces the SIMION Refine step.  Driven by geometry.yaml.

Usage:
    python -m trapsim.refine [--geometry geometry.yaml] [--out-dir .]
                             [--force-voxelize] [--omega 1.99]
                             [--max-iter 3000] [--tol 1e-5]

Steps:
  1. Voxelize STLs if mask files are stale (or --force-voxelize).
  2. Compile <solver_dir>/laplace from the bundled C++ source.  If
     <solver_dir>/laplace.cpp exists locally, prefer that (lets you hack
     the SOR loop without forking the package).
  3. Run the solver once per electrode → <out_dir>/field.pa<id>.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from importlib.resources import files as _pkg_files

from .config import GeometryConfig, load_geometry
from .voxelize import voxelize


def _default_solver_dir() -> str:
    """Work dir for masks, epsilon, grid, the compiled binary, and PA files."""
    return os.path.join(os.getcwd(), "solver")


def _default_out_dir() -> str:
    """PA files are written alongside the work files by default."""
    return _default_solver_dir()


def _solver_source(solver_dir: str) -> str:
    """Return the laplace.cpp path to compile.

    Prefer a local copy at <solver_dir>/laplace.cpp (for users who want to
    hack the solver); otherwise compile directly from the package source
    in site-packages.
    """
    local = os.path.join(solver_dir, "laplace.cpp")
    if os.path.exists(local):
        return local
    return str(_pkg_files("trapsim") / "_solver" / "laplace.cpp")


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
    # Magnetic source + mu
    if geometry.magnets:
        src = os.path.join(solver_dir, "magnetic_source.raw")
        stl_t = _newest_mtime([m.stl for m in geometry.magnets])
        if not os.path.exists(src) or os.path.getmtime(src) < stl_t:
            return True
    if geometry.magnetic_materials or geometry.magnets:
        mu = os.path.join(solver_dir, "mu.raw")
        stl_t = _newest_mtime([m.stl for m in geometry.magnetic_materials])
        if not os.path.exists(mu) or os.path.getmtime(mu) < stl_t:
            return True
    if not os.path.exists(os.path.join(solver_dir, "grid.txt")):
        return True
    return False


def ensure_compiled(solver_dir: str) -> None:
    """Build <solver_dir>/laplace from the bundled (or local) C++ source."""
    os.makedirs(solver_dir, exist_ok=True)
    src = _solver_source(solver_dir)
    exe = os.path.join(solver_dir, "laplace")
    if os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(src):
        print(f"{exe} is up-to-date.")
        return

    cxx      = os.environ.get("CXX", "clang++")
    cxxflags = os.environ.get("CXXFLAGS", "-O3 -std=c++17 -Wall -Wextra").split()
    ldflags  = os.environ.get("LDFLAGS", "").split()
    print(f"Compiling solver from {src} ...")
    cmd = [cxx, *cxxflags, "-o", exe, src, *ldflags]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"ERROR: compile failed (rc={result.returncode})")
    print("  Compiled OK.")


def refine(geometry: GeometryConfig, *,
           out_dir: str | None = None,
           solver_dir: str | None = None,
           force_voxelize: bool = False,
           omega: float = 1.99,
           max_iter: int = 3000,
           tol: float = 1e-5) -> None:
    """Run the full refine pipeline for `geometry`.

    `out_dir` defaults to CWD; `solver_dir` defaults to `<CWD>/solver`.
    """
    if out_dir is None:
        out_dir = _default_out_dir()
    if solver_dir is None:
        solver_dir = _default_solver_dir()

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

    # ── Step 3a: electric solve ─────────────────────────────────────────
    grid_file = os.path.join(solver_dir, "grid.txt")
    eps_file  = os.path.join(solver_dir, "epsilon.raw")
    exe       = os.path.join(solver_dir, "laplace")
    mask_args = [os.path.join(solver_dir, f"mask_{e.electrode_id}.raw")
                 for e in geometry.electrodes]

    if geometry.n_electrodes > 0:
        print(f"─── Running Laplace solver: electric "
              f"({geometry.n_electrodes} electrodes) ───")
        for f in [grid_file, eps_file] + mask_args:
            if not os.path.exists(f):
                sys.exit(f"ERROR: required file not found: {f}")
        cmd = [exe, "electric", grid_file, eps_file, out_dir,
               str(omega), str(max_iter), str(tol)] + mask_args
        t0 = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"ERROR: laplace solver (electric) exited with "
                     f"code {result.returncode}")
        print(f"\nElectric solve finished in {time.time()-t0:.1f} s")

    # ── Step 3b: magnetic solve (if magnets present) ────────────────────
    if geometry.magnets:
        mu_file  = os.path.join(solver_dir, "mu.raw")
        src_file = os.path.join(solver_dir, "magnetic_source.raw")
        print(f"\n─── Running Laplace solver: magnetic "
              f"({geometry.n_magnets} magnets, "
              f"{geometry.n_magnetic_materials} magnetic materials) ───")
        for f in [grid_file, mu_file, src_file]:
            if not os.path.exists(f):
                sys.exit(f"ERROR: required file not found: {f}")
        cmd = [exe, "magnetic", grid_file, mu_file, out_dir,
               str(omega), str(max_iter), str(tol), src_file]
        t0 = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"ERROR: laplace solver (magnetic) exited with "
                     f"code {result.returncode}")
        print(f"\nMagnetic solve finished in {time.time()-t0:.1f} s")

    # ── Size check ──────────────────────────────────────────────────────
    NX, NY, NZ = geometry.grid.shape
    expected = 56 + NX * NY * NZ * 8
    all_ok = True
    for elec in geometry.electrodes:
        pa = os.path.join(out_dir, f"field.pa{elec.electrode_id}")
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
    if geometry.magnets:
        pa = os.path.join(out_dir, "magfield.pa")
        if not os.path.exists(pa):
            print(f"  WARNING: {pa} not found")
            all_ok = False
        else:
            sz = os.path.getsize(pa)
            status = "OK" if sz == expected else (
                f"SIZE MISMATCH (got {sz}, expected {expected})")
            print(f"  magfield.pa: {sz:,} bytes  {status}")
            if sz != expected:
                all_ok = False

    print("\n─── Refine complete ───" if all_ok else
          "\n─── Refine completed with warnings ───")


def main():
    ap = argparse.ArgumentParser(description="Refine potential arrays from geometry.yaml.")
    ap.add_argument("--geometry",       default=os.path.join(os.getcwd(), "geometry.yaml"))
    ap.add_argument("--out-dir",        default=_default_out_dir())
    ap.add_argument("--solver-dir",     default=_default_solver_dir(),
                    help="Work dir for the binary, masks, epsilon, grid "
                         "(default: ./solver/).")
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
